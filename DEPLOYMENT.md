# 多币种汇率预测系统 — 部署文档 v6

> **与 v5 的主要区别**：由单一 USD/CNY 扩展为 **10 个货币对**；新增 `**src/services/`** 服务模块；**定时预测**（工作日 09:30 / 11:30 / 15:00，上海时区）；**本地 JSON 持久化**（实时汇率、预测结果、宏观缓存、设置）；Web **设置页**管理各数据源 API Key 与连通性测试；情绪分析可选（OpenAI 兼容接口 + 新闻）。  
> USD/CNY 的 30/60/90 天仍使用原有 `models/*.pkl` 集成模型。  
> 目标服务器示例仍为 Ubuntu；端口仍为 **9091**。

---

## 1. 环境要求


| 项目     | 要求                                                |
| ------ | ------------------------------------------------- |
| 操作系统   | Ubuntu 20.04 / 22.04 / 24.04（生产推荐）；Windows 亦可本地开发 |
| Python | **3.11**（与 v5 一致）                                 |
| 内存     | ≥ 4GB（全量预测、宏观拉取时更从容）                              |
| 磁盘     | ≥ 3GB（venv、依赖、随时间增长的 `data/*.json`）               |
| 网络     | 需访问 CFETS / HKMA / Yahoo / 各央行或 FRED 等（依你启用的渠道）   |


---

## 2. 目录结构（部署时必须包含）

将项目放到例如 `/opt/forecast-system/`（路径可自定义）：

```
/opt/forecast-system/
├── src/
│   ├── app.py                 ← Flask 入口（请从项目根目录执行 python src/app.py）
│   ├── train_model_v4.py      ← 仅训练 USD/CNY 旧流水线时用
│   └── services/              ← 【v6 新增】业务模块包（勿漏）
│       ├── __init__.py
│       ├── holidays.py
│       ├── storage.py
│       ├── settings_manager.py
│       ├── data_fetcher.py
│       ├── sentiment.py
│       ├── predictor.py
│       └── scheduler.py
├── frontend/
│   └── index.html
├── data/                      ← 运行时自动写入 JSON；可仅建空目录
│   ├── （运行后生成）rates_realtime.json, predictions.json, settings.json 等
│   └── download_data.py       ← 若仍保留历史下载脚本
├── models/
│   ├── final_models.pkl
│   ├── scalers.pkl
│   └── feat_cols.pkl
├── exchange_rate_data.csv     ← 旧版加载逻辑仍可能读取
├── macro_data.csv             ← 可选；存在时旧接口仍可用
├── requirements.txt           ← 【v6 推荐】依赖清单
└── DEPLOYMENT.md
```

> **不要**上传 `venv/`、`__pycache__/`。  
> **必须**上传整个 `src/services/`，否则导入失败。

---

## 3. 部署步骤（Ubuntu 生产）

### 3.1 安装 Python 3.11

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.11 python3.11-venv python3.11-dev
python3.11 --version
```

### 3.2 创建虚拟环境并安装依赖

```bash
cd /home/xiaoyangyang/Forecast-system-web
python3.11 -m venv venv
source venv/bin/activate  
pip install --upgrade pip
pip install -r requirements.txt
```

与 v5 的「手写 pip 一行」相比，v6 **请以 `requirements.txt` 为准**（含 `apscheduler`、`openai`、`yfinance` 等）。

验证：

```bash
python -c "import flask, apscheduler, openai, yfinance, lightgbm; print('OK')"
```

### 3.3 手动启动（仅测试用，前台运行）

> ⚠️ **此方式在前台运行，关闭终端即停止，仅适合临时测试。生产环境请使用下方的 3.5 systemd 方式实现后台运行与开机自启。**

**工作目录必须是项目根目录**（含 `src/`、`frontend/`、`models/`、`data/` 的那一层）：

```bash
cd /home/xiaoyangyang/Forecast-system-web
source venv/bin/activate 
//windows: .\venv\Scripts\activate
python src/app.py
```

日志中应出现「多币种汇率预测系统 v6 启动」、监听 `0.0.0.0:9091`。首次启动会在后台线程中做冷启动回填与一次全量预测，**约 1～3 分钟**内 CPU/网络偏高属正常。

`Ctrl+C` 停止。

### 3.4 防火墙

```bash
sudo ufw allow 9091/tcp
sudo ufw reload
```

### 3.5 systemd 生产部署（开机自启 + 后台运行）

使用 systemd 管理服务，可实现：

| 特性 | 说明 |
|------|------|
| **后台运行** | 服务以守护进程方式运行，不依赖终端，关闭 SSH 会话也不会中断 |
| **开机自启** | `systemctl enable` 后，服务器重启自动拉起服务 |
| **崩溃重启** | `Restart=always` 确保进程异常退出后 15 秒自动重启 |
| **日志持久化** | 通过 `journalctl` 查看服务日志，重启后仍可回溯 |

查看端口占用：sudo ss -tulnp | grep ':9091'

#### 3.5.1 创建服务单元文件

建议使用与多币种一致的单元名（以下为示例，可与旧服务并存时注意勿重复占用 9091）：

```bash
sudo tee /etc/systemd/system/Forecast-system-web.service << 'EOF'
[Unit]
Description=Multi-currency Forecast-system-web
After=network.target

[Service]
Type=simple
User=aigan
WorkingDirectory=/home/xiaoyangyang/Forecast-system-web
Environment=ACCESS_PASSWORD=aigan123
ExecStart=/home/xiaoyangyang/Forecast-system-web/venv/bin/python src/app.py
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
```

**关键参数说明：**

| 参数 | 作用 |
|------|------|
| `Type=simple` | 前台进程模式，systemd 通过 PID 追踪进程状态 |
| `Restart=always` | 无论退出码是什么，始终自动重启 |
| `RestartSec=15` | 重启前等待 15 秒，避免频繁重启消耗资源 |
| `WantedBy=multi-user.target` | 多用户模式加载时自动启动（即开机自启） |
| `StandardOutput=journal` / `StandardError=journal` | 标准输出和错误写入 systemd 日志 |

#### 3.5.2 启用并启动服务

```bash
sudo systemctl daemon-reload
sudo systemctl enable Forecast-system-web
sudo systemctl start Forecast-system-web
```

#### 3.5.3 常用管理命令

```bash
# 查看服务状态
sudo systemctl status Forecast-system-web

# 启动 / 停止 / 重启
sudo systemctl start Forecast-system-web
sudo systemctl stop Forecast-system-web
sudo systemctl restart Forecast-system-web

# 重新加载配置（修改 .service 文件后）
sudo systemctl daemon-reload
sudo systemctl restart Forecast-system-web

# 禁用开机自启
sudo systemctl disable Forecast-system-web

# 查看实时日志
journalctl -u Forecast-system-web -f

# 查看最近 100 行日志
journalctl -u Forecast-system-web -n 100

# 查看今天的日志
journalctl -u Forecast-system-web --since today
```

#### 3.5.4 验证部署

```bash
# 1. 确认服务正在运行
sudo systemctl is-active Forecast-system-web
# 输出应为: active

# 2. 确认开机自启已启用
sudo systemctl is-enabled Forecast-system-web
# 输出应为: enabled

# 3. 确认端口在监听
sudo ss -tlnp | grep 9091

# 4. 测试接口
curl -s http://localhost:9091/ | head -c 200
```

#### 3.5.5 故障排查

| 现象 | 排查方法 |
|------|----------|
| 服务无法启动 | `journalctl -u Forecast-system-web -n 50` 查看错误日志 |
| 端口被占用 | `sudo lsof -i :9091` 查找占用进程 |
| 服务反复重启 | `journalctl -u Forecast-system-web --since "5 min ago"` 分析重启原因 |
| Python 模块找不到 | 确认 `WorkingDirectory` 正确、venv 路径有效 |

> 若沿用旧单元名 `usdcny-forecast`，只需将 `ExecStart` 与 `WorkingDirectory` 指到新代码路径，并 **同一时间只运行一个实例**，否则会出现多个进程监听 9091、接口数据不一致。
>
> 建议生产环境将 `User=root` 改为专用非 root 用户（如 `www-data`），并确保该用户对 `/opt/forecast-system/` 有读写权限，特别是 `data/` 目录。

---

## 4. 访问与权限

- 浏览器：`http://<服务器IP>:9091/`
- **内网网段**（192.168.x.x / 10.x.x.x / 172.16–31.x.x / 127.0.0.1）访问 **HTML 与大部分 API** 无需 HTTP Basic 密码。
- **外网** 访问非设置类接口时，若 IP 不在上述范围，浏览器会弹出 **Basic 认证**（密码为环境变量 `ACCESS_PASSWORD`，默认 `admin123`）。

**设置页中的 API Key（FRED、OpenAI、NewsAPI、Banxico 等）** 与 **界面登录密码** 由应用内 `settings.json`（`data/settings.json`）管理，与 `ACCESS_PASSWORD` 不同：


| 用途                | 说明                                     |
| ----------------- | -------------------------------------- |
| `ACCESS_PASSWORD` | 外网访问静态页/API 的 HTTP Basic 密码            |
| 设置页密码             | 首次默认为 `admin123`，可在界面修改；用于保护密钥保存与连通性测试 |


---

## 5. 配置说明（v6 新增）

1. **API Key（可选但推荐）**
  在界面「设置」中配置：宏观数据（如 FRED）、墨西哥 Banxico、新闻 NewsAPI、大模型（OpenAI 兼容 Base URL + Key + 模型名）等。不设 key 时部分功能会降级或跳过，不影响服务启动。
2. **定时任务**
  调度器使用代码内 **Asia/Shanghai**，工作日 **09:30、11:30、15:00** 自动全量预测；周末与国内节假日跳过（见 `services/holidays.py` 日历）。
3. **数据落盘**
  `data/` 下 JSON 建议纳入备份：`**predictions.json`、`rates_realtime.json`、`settings.json`** 及宏观缓存文件。
4. **数据源**
  优先 CFETS / HKMA 等；失败时使用 Yahoo 等备用（如 MXN/CNY 由交叉汇率推算）。生产环境请保证出口网络可用。

---

## 6. 与 v5 部署差异速查


| 项目          | v5                        | v6                                    |
| ----------- | ------------------------- | ------------------------------------- |
| 代码包         | `src/app.py` + `frontend` | 增加 `**src/services/*.py`**            |
| 依赖          | 手写一行 pip                  | `**pip install -r requirements.txt**` |
| 数据目录        | 主要是 CSV                   | **CSV + `data/*.json` 持久化**           |
| 定时          | 无（或自研）                    | **APScheduler 三次自动预测**                |
| 配置          | 环境变量为主                    | **环境变量 + 界面 API Key**                 |
| systemd 示例名 | `usdcny-forecast`         | 建议 `**forecast-system`**（可自定）         |


---

## 7. 模型更新（USD/CNY 训练流水线）

若仍使用 `train_model_v4.py` 训练并覆盖 `models/`：

```bash
cd /opt/forecast-system
source venv/bin/activate
python src/train_model_v4.py
sudo systemctl restart forecast-system
```

---

## 8. 常见问题


| 现象                              | 处理                                                 |
| ------------------------------- | -------------------------------------------------- |
| `ModuleNotFoundError: services` | 在**项目根**执行 `python src/app.py`，且存在 `src/services/` |
| 端口被占用或数据乱跳                      | `sudo lsof -i :9091` 查 PID，**只保留一个**应用进程           |
| CFETS/HKMA 超时                   | 日志会有 WARNING；通常走 Yahoo 备用，检查网络与防火墙                 |
| 首次访问很慢                          | 等待冷启动与首次全量预测结束后再看接口                                |
| `perf_event_open` 等内核提示         | 可忽略，或按内核文档调整 `perf_event_paranoid`                 |
| systemd 服务状态显示 failed           | `journalctl -u forecast-system -n 50` 查看错误原因，常见于 venv 路径错误或 Python 版本不匹配 |
| 服务器重启后服务未自动拉起                    | `systemctl is-enabled forecast-system` 确认是否为 `enabled`，若非则执行 `systemctl enable` |
| 日志文件过大                           | `journalctl --vacuum-size=500M` 限制日志总大小，或配置 `SystemMaxUse` |


---

## 9. Windows 本地开发（可选）

```powershell
cd D:\Work\Forecast-system
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python src\app.py
```

浏览器访问 `http://127.0.0.1:9091/`。

---

## 10. 生产环境进阶（可选）

当前 `app.run()` 为 Flask 内置服务器，适合内网低并发。若需更高并发，可使用 **gunicorn** 等 WSGI 服务器；需注意 **APScheduler 与单实例** 的配合（多 worker 会重复触发定时任务，一般应保持 **单进程** 或把调度独立为单独进程）。需在项目中单独评估后再改启动方式。