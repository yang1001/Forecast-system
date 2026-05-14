"""
本地 JSON 持久化服务
存储路径: <project>/data/
"""
import os
import json
import threading
from datetime import datetime, date

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

_locks: dict = {}
_global_lock = threading.Lock()


def _get_lock(filename: str) -> threading.Lock:
    with _global_lock:
        if filename not in _locks:
            _locks[filename] = threading.Lock()
        return _locks[filename]


def _path(filename: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, filename)


def _json_serial(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def read_json(filename: str, default=None):
    """读取 JSON 文件，文件不存在时返回 default"""
    filepath = _path(filename)
    lock = _get_lock(filename)
    with lock:
        if not os.path.exists(filepath):
            return default if default is not None else {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return default if default is not None else {}


def write_json(filename: str, data):
    """写入 JSON 文件（原子写：先写临时文件再替换）"""
    filepath = _path(filename)
    tmp_path = filepath + '.tmp'
    lock = _get_lock(filename)
    with lock:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_json_serial)
        os.replace(tmp_path, filepath)


def append_rate_record(pair: str, rate_value: float, source: str = 'cfets', ts: str = None):
    """追加实时汇率记录到 rates_realtime.json"""
    data = read_json('rates_realtime.json', default={})
    if pair not in data:
        data[pair] = []
    data[pair].append({
        'ts': ts or datetime.now().isoformat(),
        'rate': rate_value,
        'source': source
    })
    # 保留最近 2000 条
    if len(data[pair]) > 2000:
        data[pair] = data[pair][-2000:]
    write_json('rates_realtime.json', data)


def get_rate_history(pair: str, days: int = 30) -> list:
    """获取指定货币对最近 N 天的历史汇率记录（按自然日，优先从 CSV 读取）"""
    import pandas as pd
    from datetime import timedelta
    csv_path = _path('exchange_rates_10y.csv')
    result = []
    
    cutoff_date = datetime.now() - timedelta(days=days)
    cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
    cutoff_ts = cutoff_date.timestamp()
    
    # 1. 尝试从 CSV 读取历史数据
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, index_col='date')
            df.index = pd.to_datetime(df.index)
            if pair in df.columns:
                # 按自然日过滤
                recent_df = df[df.index >= cutoff_date_str][[pair]].dropna()
                for date_obj, row in recent_df.iterrows():
                    date_str = date_obj.strftime('%Y-%m-%d')
                    result.append({
                        'ts': f"{date_str}T00:00:00",
                        'rate': float(row[pair]),
                        'source': 'csv_history'
                    })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"读取 CSV 历史数据失败: {e}")
            
    # 2. 从 realtime.json 补充最新数据
    data = read_json('rates_realtime.json', default={})
    records = data.get(pair, [])
    if records:
        # 提取 CSV 中已有的日期
        existing_dates = {r['ts'][:10] for r in result}
        
        for r in records:
            # 忽略旧的 backfill 数据，因为 CSV 已经包含了完整的历史
            if r.get('source') == 'yahoo_backfill':
                continue
            try:
                date_str = r['ts'][:10]
                if date_str not in existing_dates:
                    ts = datetime.fromisoformat(r['ts']).timestamp()
                    if ts >= cutoff_ts:
                        result.append(r)
                        existing_dates.add(date_str)
            except Exception:
                pass
                
    # 确保按时间排序
    result.sort(key=lambda x: x['ts'])
    return result


def get_rate_at_days_ago(pair: str, days: int) -> float | None:
    """获取 N 天前（或之后最近的第一个交易日）的汇率值"""
    from datetime import timedelta
    # get_rate_history 已经按自然日过滤了 >= (now - days) 的数据，并按时间正序排列
    records = get_rate_history(pair, days)
    if not records:
        return None
        
    target_date_str = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # 寻找第一个日期 >= 目标日期的记录（即目标日当天，或节后第一天）
    for r in records:
        if r['ts'][:10] >= target_date_str:
            return r.get('rate')
            
    return records[-1].get('rate')


def save_prediction(prediction_record: dict):
    """保存一次预测结果"""
    data = read_json('predictions.json', default={'history': []})
    data['history'].append(prediction_record)
    # 保留最近 1000 次预测记录
    if len(data['history']) > 1000:
        data['history'] = data['history'][-1000:]
    data['latest'] = prediction_record
    write_json('predictions.json', data)
    
    # 将预测结果保存到 CSV 用于回测
    try:
        import pandas as pd
        import logging
        logger = logging.getLogger(__name__)
        
        csv_path = _path('prediction_history.csv')
        excel_path = _path('prediction_history.xlsx')
        
        # 提取当前日期
        today = datetime.now().strftime('%Y-%m-%d')
        
        # 构建扁平化的数据字典
        flat_data = {
            'timestamp': prediction_record.get('timestamp'),
            'session': prediction_record.get('session'),
            'triggered_by': prediction_record.get('triggered_by')
        }
        
        preds = prediction_record.get('predictions', {})
        for pair, p_data in preds.items():
            if not p_data:
                continue
            flat_data[f"{pair}_current"] = p_data.get('current_rate')
            horizons = p_data.get('horizons', {})
            for h in ['1w', '2w', '1m', '2m', '3m', '6m']:
                if h in horizons and horizons[h]:
                    flat_data[f"{pair}_{h}"] = horizons[h].get('predicted_price')
                    
        # 读取或创建 DataFrame
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col='date')
        else:
            df = pd.DataFrame()
            df.index.name = 'date'
            
        # 确保列存在
        for col in flat_data.keys():
            if col not in df.columns:
                df[col] = None
                
        # 更新或追加当天数据
        for col, val in flat_data.items():
            df.loc[today, col] = val
            
        df.sort_index(inplace=True)
        
        df.to_csv(csv_path)
        try:
            df.to_excel(excel_path)
        except Exception as e:
            logger.debug(f"保存预测结果 Excel 失败: {e}")
            
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"保存预测结果到 CSV 失败: {e}")


def get_latest_prediction() -> dict | None:
    """获取最新一次预测结果"""
    data = read_json('predictions.json', default={})
    return data.get('latest')


def get_prediction_history(limit: int = 50) -> list:
    """获取历史预测记录"""
    data = read_json('predictions.json', default={'history': []})
    return data.get('history', [])[-limit:]


def cache_macro(date_str: str, country: str, indicators: dict):
    """缓存宏观经济数据（按日期+国家）"""
    data = read_json('macro_cache.json', default={})
    key = f"{date_str}_{country}"
    data[key] = {
        'date': date_str,
        'country': country,
        'indicators': indicators,
        'cached_at': datetime.now().isoformat()
    }
    # 保留最近 5000 条
    if len(data) > 5000:
        keys = sorted(data.keys())
        for k in keys[:len(data) - 5000]:
            del data[k]
    write_json('macro_cache.json', data)


def get_cached_macro(date_str: str, country: str) -> dict | None:
    """读取缓存的宏观数据"""
    data = read_json('macro_cache.json', default={})
    key = f"{date_str}_{country}"
    entry = data.get(key)
    if entry:
        return entry.get('indicators')
    return None


def append_daily_rates_to_csv(rates: dict | None = None):
    """
    将当日各货币对汇率写入 data/exchange_rates_10y.csv（并同步 xlsx）。
    rates: 若传入则直接写入（与网页展示同源）；为 None 时再请求 fetch_all_realtime_rates。
    """
    import pandas as pd
    import logging
    from .data_fetcher import fetch_all_realtime_rates

    logger = logging.getLogger(__name__)
    csv_path = _path('exchange_rates_10y.csv')
    excel_path = _path('exchange_rates_10y.xlsx')

    if rates is None:
        rates = fetch_all_realtime_rates()
    if not rates:
        logger.warning("无法获取实时汇率，跳过写入 exchange_rates_10y.csv")
        return

    today = datetime.now().strftime('%Y-%m-%d')

    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col='date', parse_dates=True)
        else:
            df = pd.DataFrame()
            df.index.name = 'date'

        # 统一索引为日期字符串，便于与 today 对齐
        if len(df.index) > 0:
            df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')

        for pair in rates.keys():
            if pair not in df.columns:
                df[pair] = None

        for pair, rate in rates.items():
            if rate is not None and rate != '':
                try:
                    df.loc[today, pair] = float(rate)
                except (TypeError, ValueError):
                    pass

        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.index = df.index.strftime('%Y-%m-%d')
        df.index.name = 'date'

        df.to_csv(csv_path)
        try:
            df.to_excel(excel_path)
        except Exception as e:
            logger.debug(f"保存 Excel 失败: {e}")

        logger.info(f"已同步 {today} 共 {len(rates)} 个货币对到 exchange_rates_10y.csv")
    except Exception as e:
        logger.error(f"写入 exchange_rates_10y.csv 失败: {e}")
