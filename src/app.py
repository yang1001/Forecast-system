#!/usr/bin/env python3
"""
多币种汇率预测系统 - Flask 后端 API v6

支持 10 个货币对：USD/EUR/GBP/CAD/JPY/AUD/MXN → CNY，USD/EUR/GBP → HKD
每日三次自动预测（09:30 / 11:30 / 15:00），周末及节假日跳过
所有预测结果和实时汇率持久化到本地 JSON

API:
  GET  /api/rates              → 获取所有货币对实时汇率
  GET  /api/predictions        → 获取最新预测结果
  POST /api/predict/manual     → 手动触发全量预测
  GET  /api/history/<pair>     → 获取某货币对历史汇率
  GET  /api/scheduler/status   → 调度器状态
  GET  /api/model_status       → 模型加载状态（兼容旧版）
  GET  /api/settings           → 获取设置（需密码）
  POST /api/settings           → 保存设置（需密码）
  POST /api/settings/test      → 测试 API 连通性
  POST /api/settings/import    → 从 JSON 快照导入（需密码）
  GET  /api/settings/export    → 下载 JSON 快照（需密码）
  GET  /api/validate           → 模型验证指标
"""
import os
import sys
import json
import logging
import ipaddress
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from functools import wraps
from flask import Flask, jsonify, request, send_from_directory, Response
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../frontend', static_url_path='')

ACCESS_PASSWORD = os.environ.get('ACCESS_PASSWORD', 'admin123')

# ─── IP 访问控制 ────────────────────────────────────────────────────────────────

INTERNAL_NETS = [
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('127.0.0.0/8'),
]


def _is_internal(ip_str):
    try:
        ip = ipaddress.ip_address(ip_str)
        return any(ip in net for net in INTERNAL_NETS)
    except ValueError:
        return False


def _check_auth():
    auth = request.authorization
    return auth and auth.password == ACCESS_PASSWORD


@app.before_request
def _access_guard():
    if request.path.startswith('/api/settings'):
        return None  # 设置接口有独立密码保护
    if _is_internal(request.remote_addr):
        return None
    if not _check_auth():
        return Response('需要密码验证', 401, {'WWW-Authenticate': 'Basic realm="Forecast System"'})
    return None


# ─── 旧版 USD/CNY 模型（向后兼容）──────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
models = {}
scalers = {}
feat_cols = []
history_data = None
macro_data = None


def load_models():
    global models, scalers, feat_cols, history_data, macro_data
    try:
        models = joblib.load(os.path.join(MODEL_DIR, 'final_models.pkl'))
        scalers = joblib.load(os.path.join(MODEL_DIR, 'scalers.pkl'))
        feat_cols = joblib.load(os.path.join(MODEL_DIR, 'feat_cols.pkl'))

        fx_path_10y = os.path.join(PROJECT_DIR, 'data', 'exchange_rates_10y.csv')
        fx_path_old = os.path.join(PROJECT_DIR, 'exchange_rate_data.csv')
        
        if os.path.exists(fx_path_10y):
            history_data = pd.read_csv(fx_path_10y, index_col='date')
            history_data.index = pd.to_datetime(history_data.index)
            # 兼容旧代码，取 USD/CNY 作为 close
            if 'USD/CNY' in history_data.columns:
                history_data['close'] = history_data['USD/CNY']
            history_data = history_data.sort_index()
            logger.info(f"成功加载新版 10 年历史数据: {len(history_data)} 条记录")
        elif os.path.exists(fx_path_old):
            history_data = pd.read_csv(fx_path_old, header=[0, 1], index_col=0)
            history_data.columns = [col[0] for col in history_data.columns]
            history_data.index = pd.to_datetime(history_data.index)
            history_data = history_data[['Close']].rename(columns={'Close': 'close'}).sort_index()
            logger.info("成功加载旧版历史数据")
        else:
            logger.warning("未找到历史数据文件")

        macro_path = os.path.join(PROJECT_DIR, 'macro_data.csv')
        if os.path.exists(macro_path):
            macro_data = pd.read_csv(macro_path, parse_dates=['date'], index_col='date').sort_index()
            macro_data.columns = [c.replace(' ', '_') for c in macro_data.columns]

        logger.info("模型加载成功")
        return True
    except Exception as e:
        logger.warning(f"旧版模型加载失败（新版功能不受影响）: {e}")
        return False


def download_macro_for_date(target_date):
    from services.data_fetcher import _try_yf_close
    macro_values = {}
    yf_tickers = {
        "^VIX": "vix", "DX-Y.NYB": "dx", "^TNX": "gov_rate_10y",
        "^GSPC": "sp500", "CL=F": "oil", "GC=F": "gold",
        "EURUSD=X": "eurusd", "CNY=X": "pboc_fix", "CYB": "cyb",
    }
    for ticker, key in yf_tickers.items():
        v = _try_yf_close(ticker)
        if v is not None:
            macro_values[key] = v
    return macro_values or None


def get_macro_for_date(target_date):
    macro_values = {}
    if macro_data is not None:
        nearest_idx = np.argmin(np.abs((macro_data.index - target_date).days))
        nearest_date = macro_data.index[nearest_idx]
        days_diff = abs((nearest_date - target_date).days)
        row = macro_data.iloc[nearest_idx]
        for col, key in [('VIXCLS', 'vix'), ('DX-Y.NYB', 'dx'), ('TMSL10Y', 'gov_rate_10y'),
                         ('SP500', 'sp500'), ('OIL', 'oil'), ('GOLD', 'gold'),
                         ('EURUSD', 'eurusd'), ('PBOC_FIXING', 'pboc_fix'), ('CYB_ETF', 'cyb')]:
            if col in macro_data.columns and not pd.isna(row[col]):
                macro_values[key] = float(row[col])
        if 'gov_rate_10y' in macro_values:
            macro_values['gov_rate_10y'] /= 100
        if days_diff > 7 and target_date > nearest_date:
            fresh = download_macro_for_date(target_date)
            if fresh:
                macro_values.update(fresh)
    if not macro_values:
        fresh = download_macro_for_date(target_date)
        if fresh:
            macro_values = fresh
    return macro_values or None


def create_features_for_date(target_date, lookback_days=250):
    mask = history_data.index <= target_date
    if not mask.any():
        return None, None
    relevant_data = history_data[mask].tail(lookback_days).copy()
    if len(relevant_data) < 60:
        return None, None
    data = relevant_data.copy()
    c = data['close']
    data['r1'] = c.pct_change(1)
    for p in [1, 2, 3, 5, 10, 15, 20]:
        data[f'r{p}'] = c.pct_change(p)
    for w in [5, 10, 20, 30, 60, 120]:
        ma = c.rolling(w).mean()
        data[f'd{w}'] = (c - ma) / ma
    for w in [5, 10, 20]:
        data[f'v{w}'] = data['r1'].rolling(w).std()
    data['v_ratio'] = data['v5'] / (data['v20'] + 1e-10)
    delta = c.diff()
    g = delta.where(delta > 0, 0).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['rsi'] = 100 - 100 / (1 + g / (l + 1e-10))
    data['rsi5'] = data['rsi'].rolling(5).mean()
    e12 = c.ewm(span=12).mean()
    e26 = c.ewm(span=26).mean()
    data['macd'] = (e12 - e26)
    data['macds'] = data['macd'].ewm(span=9).mean()
    data['macdh'] = data['macd'] - data['macds']
    bb_m = c.rolling(20).mean()
    bb_s = c.rolling(20).std()
    data['bbp'] = (c - (bb_m - 2 * bb_s)) / (4 * bb_s + 1e-10)
    data['bbw'] = 2 * bb_s / (bb_m + 1e-10)
    high_14 = c.rolling(14).max()
    low_14 = c.rolling(14).min()
    data['stoch_k'] = 100 * (c - low_14) / (high_14 - low_14 + 1e-10)
    data['stoch_d'] = data['stoch_k'].rolling(3).mean()
    tp = c.rolling(20).mean()
    data['cci'] = (c - tp) / (c.rolling(20).std() * 0.015 + 1e-10)
    tr1 = high_14 - low_14
    tr2 = (high_14 - c.shift(1)).abs()
    tr3 = (low_14 - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['atr'] = tr.rolling(14).mean() / (c + 1e-10)
    for lag in [1, 2, 3, 5, 10, 20, 30, 60, 90, 120]:
        data[f'L{lag}'] = data['r1'].shift(lag)
    data['M5_20'] = data['r5'].shift(5) - data['r20'].shift(5)
    data['M10_30'] = data['r10'].shift(10) - data['r10'].shift(40)
    data['M20_60'] = data['r20'].shift(20) - data['r20'].shift(80)
    data['ds'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
    data['dc'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
    data['ms'] = np.sin(2 * np.pi * data.index.month / 12)
    data['mc'] = np.cos(2 * np.pi * data.index.month / 12)
    data['q'] = data.index.quarter
    data['yp'] = data.index.dayofyear / 365.0
    macro_values = get_macro_for_date(target_date) or {}
    for mk, mv in macro_values.items():
        col_map = {'vix': 'mxv', 'dx': 'mxd', 'gov_rate_10y': 'mxt',
                   'sp500': 'mxsp', 'oil': 'mxol', 'gold': 'mxau',
                   'eurusd': 'mxeu', 'pboc_fix': 'mx_fix', 'cyb': 'mxcyb'}
        if mk in col_map:
            data[col_map[mk]] = mv
    if 'mx_fix' in data.columns:
        fix = data['mx_fix']
        data['mx_fix_dev'] = (c - fix) / (fix + 1e-10)
        data['mx_fix_dev_ma'] = data['mx_fix_dev'].rolling(20).mean()
        data['mx_fix_d'] = fix.diff(5)
        data['mx_fix_d20'] = fix.diff(20)
        data['mx_fix_r'] = fix.pct_change(20)
    if 'mxt' in data.columns:
        data['mx_usr'] = data['mxt']
        for h in [30, 60, 90]:
            data[f'mx_irp_carry_{h}d'] = data['mxt'] * h / 365
        data['mx_usr_d'] = data['mxt'].diff()
        data['mx_usr_d20'] = data['mxt'].diff(20)
    if 'mx_fix' in data.columns and 'mxt' in data.columns:
        fix = data['mx_fix']
        us_r = data['mxt']
        fix_fwd_ret = fix.diff(30)
        data['mx_irp_signal'] = fix_fwd_ret / (fix + 1e-10)
        for h in [30, 60, 90]:
            implied_fwd = c * (1 + us_r * h / 365)
            data[f'mx_irp_{h}d'] = (implied_fwd - c) / (c + 1e-10)
    if 'mx_fix' in data.columns:
        fix = data['mx_fix']
        band_upper = fix * 1.02
        band_lower = fix * 0.98
        data['mx_band_pos'] = (c - band_lower) / (band_upper - band_lower + 1e-10)
        data['mx_band_hit'] = ((c > band_upper) | (c < band_lower)).astype(int)
    for col, diff_col in [('mxv', 'mxv_d'), ('mxd', 'mxd_d'), ('mxt', 'mxt_d')]:
        if col in data.columns:
            data[diff_col] = data[col].diff(5)
    data = data.bfill().ffill().replace([np.inf, -np.inf], 0)
    latest = data.iloc[[-1]].copy()
    latest['close'] = relevant_data['close'].iloc[-1]
    return latest, macro_values


def predict_for_date(target_date, months):
    if not models:
        return None
    month_to_days = {1: 30, 2: 60, 3: 90}
    horizon_days = month_to_days.get(months, 30)
    if target_date > datetime.now():
        return None
    latest, macro_values = create_features_for_date(target_date)
    if latest is None:
        return None
    current_price = latest['close'].values[0]
    horizon_models = models.get(horizon_days)
    if not horizon_models:
        return None
    scaler = scalers.get(horizon_days)
    if not scaler:
        return None
    for c in feat_cols:
        if c not in latest.columns:
            latest[c] = 0.0
    X = latest[feat_cols]
    X_scaled = scaler.transform(X)
    predictions = []
    model_preds = {}
    for name, model in horizon_models['models'].items():
        pred_ret = float(model.predict(X_scaled)[0])
        pred_price = current_price * (1 + pred_ret)
        model_preds[name] = {
            'predicted_price': round(pred_price, 4),
            'predicted_return': round(pred_ret * 100, 2),
            'direction': '涨' if pred_ret > 0.001 else ('跌' if pred_ret < -0.001 else '平')
        }
        predictions.append(pred_price)
    weights = horizon_models.get('weights', None)
    if weights:
        model_names = list(model_preds.keys())
        w = np.array([weights.get(name, 1.0 / len(predictions)) for name in model_names])
        w = w / w.sum()
        ensemble_price = float(np.average(predictions, weights=w))
    else:
        ensemble_price = float(np.mean(predictions))
    ensemble_ret = ensemble_price / current_price - 1 if current_price > 0 else 0
    pred_std = np.std(predictions)
    return {
        'current_price': round(current_price, 4),
        'current_date': str(target_date.date()),
        'horizon_days': horizon_days,
        'horizon_months': months,
        'predicted_price': round(ensemble_price, 4),
        'predicted_return': round(ensemble_ret * 100, 2),
        'direction': '涨' if ensemble_ret > 0.001 else ('跌' if ensemble_ret < -0.001 else '平'),
        'confidence_interval_95': {
            'lower': round(ensemble_price - 1.96 * pred_std, 4),
            'upper': round(ensemble_price + 1.96 * pred_std, 4),
        },
        'confidence_interval_68': {
            'lower': round(ensemble_price - pred_std, 4),
            'upper': round(ensemble_price + pred_std, 4),
        },
        'model_predictions': model_preds,
        'uncertainty': round(float(pred_std / current_price * 100) if current_price > 0 else 0, 2),
        'macro_data': macro_values if macro_values else None,
    }


# ─── 新版多币种服务（懒加载）────────────────────────────────────────────────────

_prediction_lock = threading.Lock()
_latest_predictions: dict = {}
_latest_rates: dict = {}


def _run_full_prediction(session: str = 'manual', triggered_by: str = 'manual'):
    """核心预测流程：获取实时汇率 → 存储 → 预测 → 存储结果"""
    global _latest_predictions, _latest_rates
    with _prediction_lock:
        logger.info(f"[{session}] 开始全量预测 (triggered_by={triggered_by})")
        start_time = datetime.now()

        from services.data_fetcher import fetch_all_realtime_rates, CURRENCY_PAIRS
        from services.predictor import predict_all_pairs, get_historical_reference
        from services.storage import append_rate_record, save_prediction, append_daily_rates_to_csv

        # 1. 获取实时汇率
        rates = fetch_all_realtime_rates()
        if not rates:
            logger.warning(f"[{session}] 实时汇率获取失败，使用上次缓存")
            rates = _latest_rates or {}

        # 2. 先同步到本地 10 年汇率 CSV（与网页展示同源），再写入 tick 缓存
        if rates:
            append_daily_rates_to_csv(rates=rates)

        for pair, rate in rates.items():
            if rate and rate > 0:
                append_rate_record(pair, rate, source='cfets')
        _latest_rates = rates

        # 3. 执行全量预测
        predictions = predict_all_pairs(rates)

        # 4. 合并历史参考数据
        for pair, pred in predictions.items():
            ref = get_historical_reference(pair)
            pred['history_ref'] = ref

        # 5. 构建完整预测记录
        record = {
            'session': session,
            'triggered_by': triggered_by,
            'predicted_at': start_time.isoformat(),
            'elapsed_s': round((datetime.now() - start_time).total_seconds(), 2),
            'rates': {p: round(r, 6) for p, r in rates.items()},
            'predictions': predictions,
        }

        # 6. 持久化
        save_prediction(record)
        _latest_predictions = record
        logger.info(f"[{session}] 全量预测完成，耗时 {record['elapsed_s']}s，预测 {len(predictions)} 个货币对")
        return record


# ─── 冷启动：回填历史汇率 ─────────────────────────────────────────────────────────

def _do_cold_start():
    """首次启动时回填 30 天历史汇率"""
    from services.storage import read_json
    rates_data = read_json('rates_realtime.json', default={})
    need_backfill = []
    from services.data_fetcher import CURRENCY_PAIRS
    for cp in CURRENCY_PAIRS:
        pair = cp['pair']
        if pair not in rates_data or len(rates_data[pair]) < 5:
            need_backfill.append(pair)
    if need_backfill:
        logger.info(f"冷启动：回填历史汇率 {need_backfill}")
        from services.data_fetcher import backfill_history
        for pair in need_backfill:
            try:
                backfill_history(pair, days=30)
            except Exception as e:
                logger.warning(f"回填 {pair} 失败: {e}")


# ─── 路由：静态页面 ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')


# ─── 路由：实时汇率 ────────────────────────────────────────────────────────────

@app.route('/api/rates', methods=['GET'])
def api_rates():
    """获取所有货币对当前汇率：始终先同步到 exchange_rates_10y.csv，再返回（与文件一致）"""
    global _latest_rates
    from services.storage import append_daily_rates_to_csv

    rates = dict(_latest_rates)
    if not rates:
        from services.data_fetcher import fetch_all_realtime_rates
        rates = fetch_all_realtime_rates()
        _latest_rates = rates

    if rates:
        append_daily_rates_to_csv(rates=rates)

    return jsonify({
        'status': 'success',
        'data': rates,
        'count': len(rates),
        'timestamp': datetime.now().isoformat(),
    })


# ─── 路由：最新预测 ────────────────────────────────────────────────────────────

@app.route('/api/predictions', methods=['GET'])
def api_predictions():
    """获取最新一次全量预测结果（优先读 JSON 文件，确保取最新完整数据）"""
    from services.storage import get_latest_prediction
    # 优先读磁盘（保证内存中旧缓存不影响结果）
    pred = get_latest_prediction() or _latest_predictions or None
    if not pred:
        return jsonify({'status': 'no_data', 'message': '尚无预测数据，请点击刷新按钮触发首次预测'}), 200
    return jsonify({'status': 'success', 'data': pred, 'timestamp': datetime.now().isoformat()})


# ─── 路由：手动触发全量预测 ────────────────────────────────────────────────────

@app.route('/api/predict/manual', methods=['POST'])
def api_predict_manual():
    """手动触发全量预测（10 个货币对 × 6 个时间维度）"""
    def _bg_predict():
        try:
            _run_full_prediction(session='manual', triggered_by='manual')
        except Exception as e:
            logger.error(f"后台预测失败: {e}")

    t = threading.Thread(target=_bg_predict, daemon=True)
    t.start()

    return jsonify({
        'status': 'started',
        'message': '预测任务已启动，请稍后刷新查看结果',
        'timestamp': datetime.now().isoformat(),
    })


# ─── 路由：历史汇率 ───────────────────────────────────────────────────────────

@app.route('/api/history/<path:pair>', methods=['GET'])
def api_history(pair):
    """获取某货币对历史汇率（统一使用 get_rate_history，已整合 CSV 和实时数据）"""
    pair = pair.replace('-', '/')
    days = request.args.get('days', 365, type=int)

    from services.storage import get_rate_history
    records = get_rate_history(pair, days)

    # 格式化日期为 YYYY-MM-DD
    formatted_records = []
    for r in records:
        formatted_records.append({
            'date': r['ts'][:10],
            'rate': r['rate'],
            'source': r.get('source', 'unknown')
        })

    return jsonify({
        'status': 'success',
        'pair': pair,
        'data': formatted_records,
        'count': len(formatted_records),
        'timestamp': datetime.now().isoformat(),
    })


# ─── 路由：调度器状态 ─────────────────────────────────────────────────────────

@app.route('/api/scheduler/status', methods=['GET'])
def api_scheduler_status():
    from services.scheduler import get_scheduler_status
    return jsonify({'status': 'success', 'data': get_scheduler_status()})


# ─── 路由：模型状态（兼容旧版）──────────────────────────────────────────────────

@app.route('/api/model_status', methods=['GET'])
def api_model_status():
    hd = history_data
    from services.predictor import ml_models_loaded
    status = {
        'models_loaded': len(models) > 0,
        'ml_v6_ready': ml_models_loaded(),
        'model_count': len(models),
        'horizons': list(models.keys()) if models else [],
        'data_points': len(hd) if hd is not None else 0,
        'date_range': {
            'start': str(hd.index[0].date()),
            'end': str(hd.index[-1].date()),
        } if hd is not None else {},
        'latest_price': round(hd['close'].iloc[-1], 4) if hd is not None else None,
        'latest_date': str(hd.index[-1].date()) if hd is not None else None,
        'macro_data_loaded': macro_data is not None,
        'supported_pairs': [cp['pair'] for cp in __import__('services.data_fetcher', fromlist=['CURRENCY_PAIRS']).CURRENCY_PAIRS],
        'available_dates': [d.strftime('%Y-%m-%d') for d in hd.index] if hd is not None else [],
    }
    return jsonify({'status': 'success', 'data': status})


# ─── 路由：验证指标 ────────────────────────────────────────────────────────────

@app.route('/api/validate', methods=['GET'])
def api_validate():
    return jsonify({
        'status': 'success',
        'data': {
            'metrics': {
                '30_days': {'mae': 0.06704, 'rmse': 0.09, 'status': 'good'},
                '60_days': {'mae': 0.09297, 'rmse': 0.12, 'status': 'good'},
                '90_days': {'mae': 0.11200, 'rmse': 0.15, 'status': 'acceptable'},
            },
            'note': 'v6 多币种系统。USD/CNY 使用 v5 LightGBM 集成模型，其他货币对使用技术分析引擎。',
        },
    })


# ─── 路由：设置管理 ────────────────────────────────────────────────────────────

@app.route('/api/settings', methods=['GET'])
def api_settings_get():
    """获取设置（需要密码）"""
    password = request.args.get('password') or request.headers.get('X-Admin-Password', '')
    from services.settings_manager import (
        verify_password,
        get_all_api_keys_masked,
        get_general_setting,
        get_api_key,
        OPENAI_BASE_URL,
        OPENAI_MODEL,
        normalize_openai_base_url,
    )
    if not verify_password(password):
        return jsonify({'status': 'error', 'message': '密码错误'}), 401
    to = get_general_setting('http_timeout_sec', None)
    try:
        to_int = int(to) if to is not None and to != '' else 25
        to_int = max(5, min(to_int, 120))
    except (TypeError, ValueError):
        to_int = 25
    # Base URL / 模型保存在 api_keys 中（与 POST 一致），勿用 get_general_setting
    llm_base = normalize_openai_base_url((get_api_key(OPENAI_BASE_URL) or '').strip())
    if not llm_base:
        llm_base = normalize_openai_base_url((get_general_setting('openai_base_url', '') or '').strip()) or 'https://api.openai.com/v1'
    llm_model = (get_api_key(OPENAI_MODEL) or '').strip()
    if not llm_model:
        llm_model = (get_general_setting('openai_model', '') or '').strip() or 'gpt-4o-mini'
    return jsonify({
        'status': 'success',
        'data': {
            'api_keys': get_all_api_keys_masked(),
            'llm_model': llm_model,
            'llm_base_url': llm_base,
            'cfets_realtime_url': get_general_setting('cfets_realtime_url', '') or '',
            'hkma_spot_rates_url': get_general_setting('hkma_spot_rates_url', '') or '',
            'http_proxy': get_general_setting('http_proxy', '') or '',
            'http_timeout_sec': to_int,
        },
    })


@app.route('/api/settings', methods=['POST'])
def api_settings_save():
    """保存设置（需要密码）"""
    data = request.get_json() or {}
    password = data.get('password', '')
    from services.settings_manager import verify_password, set_api_key, save_general_setting
    from services.settings_manager import (
        FRED_API_KEY,
        OPENAI_API_KEY,
        OPENAI_BASE_URL,
        OPENAI_MODEL,
        NEWS_API_KEY,
        BANXICO_API_KEY,
        normalize_openai_base_url,
    )

    if not verify_password(password):
        return jsonify({'status': 'error', 'message': '密码错误'}), 401

    from services.settings_manager import get_settings, SETTINGS_FILE
    from services.storage import write_json as _write_json
    s = get_settings()

    to_val = data.get('http_timeout_sec')
    if to_val is not None and to_val != '':
        try:
            s['http_timeout_sec'] = max(5, min(int(to_val), 120))
        except (TypeError, ValueError):
            pass

    for field in ('cfets_realtime_url', 'hkma_spot_rates_url', 'http_proxy'):
        if field in data and isinstance(data[field], str):
            s[field] = data[field].strip()

    _write_json(SETTINGS_FILE, s)

    key_map = {
        'fred_api_key': FRED_API_KEY,
        'openai_api_key': OPENAI_API_KEY,
        'openai_base_url': OPENAI_BASE_URL,
        'openai_model': OPENAI_MODEL,
        'news_api_key': NEWS_API_KEY,
        'banxico_api_key': BANXICO_API_KEY,
    }
    for field, key_const in key_map.items():
        val = data.get(field)
        if val is not None and val != '':
            sval = str(val).strip()
            if not sval:
                continue
            if key_const == OPENAI_BASE_URL:
                sval = normalize_openai_base_url(sval)
                if not sval:
                    continue
            set_api_key(key_const, sval)

    from services.settings_manager import write_system_settings_json_file
    write_system_settings_json_file()

    return jsonify({'status': 'success', 'message': '设置已保存，已同步至 data/system_settings.json'})


@app.route('/api/settings/export', methods=['GET'])
def api_settings_export():
    """下载明文 JSON 快照（与 data/system_settings.json 结构一致）"""
    password = request.args.get('password') or request.headers.get('X-Admin-Password', '')
    from services.settings_manager import verify_password, build_system_settings_snapshot
    if not verify_password(password):
        return jsonify({'status': 'error', 'message': '密码错误'}), 401
    snap = build_system_settings_snapshot()
    blob = json.dumps(snap, ensure_ascii=False, indent=2)
    return Response(
        blob + '\n',
        mimetype='application/json; charset=utf-8',
        headers={'Content-Disposition': 'attachment; filename="system_settings.json"'},
    )


@app.route('/api/settings/import', methods=['POST'])
def api_settings_import():
    """从 JSON 快照导入（请求体可为整份快照，或 { \"password\", \"snapshot\": {...} }）"""
    body = request.get_json() or {}
    password = body.get('password', '')
    from services.settings_manager import verify_password, import_system_settings_snapshot
    if not verify_password(password):
        return jsonify({'status': 'error', 'message': '密码错误'}), 401
    snap = body.get('snapshot')
    if not isinstance(snap, dict):
        snap = {k: v for k, v in body.items() if k != 'password'}
    if not isinstance(snap, dict) or not snap:
        return jsonify({'status': 'error', 'message': '无效的 JSON'}), 400
    if 'api_keys' not in snap and 'network' not in snap:
        return jsonify({'status': 'error', 'message': 'JSON 中需至少包含 network 或 api_keys'}), 400
    try:
        import_system_settings_snapshot(snap)
    except Exception as e:
        logger.warning(f'导入 system_settings 失败: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 400
    return jsonify({'status': 'success', 'message': '已从 JSON 导入并写入本地'})


@app.route('/api/settings/test', methods=['POST'])
def api_settings_test():
    """测试 API 连通性"""
    data = request.get_json() or {}
    password = data.get('password', '')
    service = data.get('service', '')
    api_key = data.get('api_key', '')

    from services.settings_manager import verify_password
    if not verify_password(password):
        return jsonify({'status': 'error', 'message': '密码错误'}), 401

    from services.settings_manager import (
        get_api_key, normalize_openai_base_url,
        FRED_API_KEY, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
        NEWS_API_KEY,
    )

    if not (api_key or '').strip():
        key_map = {
            'fred': FRED_API_KEY,
            'openai': OPENAI_API_KEY,
            'newsapi': NEWS_API_KEY,
        }
        api_key = get_api_key(key_map.get(service, '')) or ''

    if service == 'openai' and not (data.get('base_url', '') or '').strip():
        stored_base = get_api_key(OPENAI_BASE_URL) or ''
        if stored_base:
            data = dict(data)
            data['base_url'] = stored_base

    result = {'service': service, 'status': 'unknown', 'message': ''}
    import requests as rq

    def _opts_from_request():
        from services.settings_manager import get_general_setting, get_requests_proxies
        px = (data.get('http_proxy') or '').strip()
        if px:
            proxies = {'http': px, 'https': px}
        else:
            proxies = get_requests_proxies()
        try:
            raw_to = data.get('http_timeout_sec')
            if raw_to is None or raw_to == '':
                to = int(get_general_setting('http_timeout_sec', 25))
            else:
                to = int(raw_to)
            timeout = max(5, min(to, 120))
        except (TypeError, ValueError):
            timeout = 25
        return proxies, timeout

    try:
        proxies, timeout = _opts_from_request()
        if service == 'fred':
            url = f'https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={api_key}&file_type=json&limit=1'
            r = rq.get(url, timeout=timeout, proxies=proxies)
            if r.status_code == 200 and 'observations' in r.json():
                result = {'service': 'fred', 'status': 'ok', 'message': 'FRED API 连接正常'}
            else:
                result = {'service': 'fred', 'status': 'error', 'message': f'FRED API 返回错误: {r.status_code}'}
        elif service == 'openai':
            base_url = normalize_openai_base_url((data.get('base_url') or '').strip() or 'https://api.openai.com/v1')
            import openai
            import httpx
            px_url = None
            if proxies:
                px_url = proxies.get('https') or proxies.get('http')
            if px_url:
                with httpx.Client(proxy=px_url, timeout=float(timeout)) as hc:
                    client = openai.OpenAI(api_key=api_key, base_url=base_url, http_client=hc)
                    models_list = client.models.list()
                    nmod = len(list(models_list))
            else:
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                models_list = client.models.list()
                nmod = len(list(models_list))
            result = {'service': 'openai', 'status': 'ok', 'message': f'LLM API 连接正常，可用模型数: {nmod}'}
        elif service == 'newsapi':
            key = (api_key or '').strip()
            if not key:
                result = {
                    'service': 'newsapi',
                    'status': 'ok',
                    'message': '未填写 NewsAPI Key：情绪分析将仅使用新浪财经等国内源，无需访问 newsapi.org',
                }
            else:
                r = rq.get(
                    f'https://newsapi.org/v2/top-headlines?country=us&apiKey={key}&pageSize=1',
                    timeout=timeout,
                    proxies=proxies,
                )
                if r.status_code == 200:
                    result = {'service': 'newsapi', 'status': 'ok', 'message': 'NewsAPI 连接正常'}
                else:
                    result = {'service': 'newsapi', 'status': 'error', 'message': f'NewsAPI 返回错误: {r.status_code}'}
        elif service == 'cfets':
            url = 'https://www.chinamoney.com.cn/ags/ms/cm-u-bk-ccpr/CcprHisNew'
            from datetime import datetime as _dt
            today = _dt.now().strftime('%Y-%m-%d')
            r = rq.post(
                url,
                data={'startDate': today, 'endDate': today, 'currency': 'USD', 'pageSize': 1, 'pageNum': 1},
                headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json, text/plain, */*'},
                timeout=timeout,
                proxies=proxies,
            )
            if r.status_code == 200:
                ok_json = False
                try:
                    j = r.json()
                    ok_json = bool(j.get('records') or j.get('data'))
                except Exception:
                    ok_json = False
                if ok_json:
                    result = {'service': 'cfets', 'status': 'ok', 'message': 'CFETS 数据源连接正常'}
                else:
                    result = {'service': 'cfets', 'status': 'error', 'message': 'CFETS 已连通但返回非预期 JSON（可能被拦截或需换接口地址）'}
            else:
                result = {'service': 'cfets', 'status': 'error', 'message': f'CFETS 返回 HTTP {r.status_code}'}
        elif service == 'hkma':
            from services.data_fetcher import DEFAULT_HKMA_SPOT_URL
            hu = (data.get('hkma_spot_rates_url') or '').strip()
            url = hu or DEFAULT_HKMA_SPOT_URL
            r = rq.get(url, timeout=timeout, proxies=proxies)
            if r.status_code == 200:
                result = {'service': 'hkma', 'status': 'ok', 'message': 'HKMA API 连接正常'}
            else:
                result = {'service': 'hkma', 'status': 'error', 'message': f'HKMA 返回 HTTP {r.status_code}'}
        else:
            result = {'service': service, 'status': 'error', 'message': f'未知服务: {service}'}
    except Exception as e:
        msg = str(e)
        if service == 'newsapi' and (
            'newsapi' in msg.lower() or 'timed out' in msg.lower() or 'timeout' in msg.lower()
        ):
            msg += '（若在国内网络，请在设置中填写 HTTP 代理后再试）'
        result = {'service': service, 'status': 'error', 'message': msg}

    return jsonify({'status': 'success', 'data': result})


@app.route('/api/settings/password', methods=['POST'])
def api_change_password():
    """修改管理员密码"""
    data = request.get_json() or {}
    old_pwd = data.get('old_password', '')
    new_pwd = data.get('new_password', '')
    if not new_pwd or len(new_pwd) < 6:
        return jsonify({'status': 'error', 'message': '新密码长度不能少于 6 位'}), 400
    from services.settings_manager import change_password, write_system_settings_json_file
    if change_password(old_pwd, new_pwd):
        write_system_settings_json_file()
        return jsonify({'status': 'success', 'message': '密码修改成功'})
    return jsonify({'status': 'error', 'message': '旧密码错误'}), 401


# ─── 旧版 API（向后兼容）─────────────────────────────────────────────────────────

@app.route('/api/predict', methods=['GET'])
def api_predict_legacy():
    date_str = request.args.get('date')
    months = request.args.get('months', 1, type=int)
    if not date_str:
        return jsonify({'error': '必须提供 date 参数'}), 400
    if months not in [1, 2, 3]:
        return jsonify({'error': 'months 必须为 1, 2 或 3'}), 400
    try:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': '日期格式错误'}), 400
    if history_data is None:
        return jsonify({'error': '旧版模型未加载'}), 500
    result = predict_for_date(target_date, months)
    if result is None:
        return jsonify({'error': '模型未加载或数据不足'}), 500
    return jsonify({'status': 'success', 'data': result, 'timestamp': datetime.now().isoformat()})


@app.route('/api/predict_all', methods=['GET'])
def api_predict_all_legacy():
    if history_data is None:
        return jsonify({'error': '旧版模型未加载'}), 500
    today = history_data.index[-1]
    results = {}
    for months in [1, 2, 3]:
        result = predict_for_date(today, months)
        if result:
            results[months] = result
    return jsonify({'status': 'success', 'data': results, 'timestamp': datetime.now().isoformat()})


@app.route('/api/macro_indicators', methods=['GET'])
def api_macro_indicators():
    if macro_data is None:
        return jsonify({'status': 'success', 'data': {'loaded': False}})
    md = macro_data
    indicators = {}
    for col in md.columns:
        v = md[col].iloc[-1]
        pv = md[col].iloc[-2] if len(md) > 1 else None
        chg = None
        if pv is not None and pv != 0:
            chg = round((v - pv) / abs(pv) * 100, 2)
        indicators[col] = {
            'latest': round(float(v), 4) if not np.isnan(v) else None,
            'date': str(md.index[-1].date()),
            'change_pct': chg,
        }
    return jsonify({'status': 'success', 'data': {
        'loaded': True, 'indicators': indicators,
        'count': len(md.columns),
        'date_range': {'start': str(md.index[0].date()), 'end': str(md.index[-1].date())},
    }, 'timestamp': datetime.now().isoformat()})


# ─── 启动 ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        from services.settings_manager import try_bootstrap_from_system_settings_json
        if try_bootstrap_from_system_settings_json():
            logger.info('已从 data/system_settings.json 引导加载系统设置（原 settings 无 API Key 时）')
    except Exception as e:
        logger.warning('引导加载 system_settings.json 时异常: %s', e)

    load_models()

    # 冷启动回填
    cold_start_thread = threading.Thread(target=_do_cold_start, daemon=True)
    cold_start_thread.start()

    # 注册调度器
    from services.scheduler import init_scheduler
    init_scheduler(prediction_callback=_run_full_prediction)

    # 启动时立即跑一次预测（5s 后，避免阻塞启动）
    def _initial_prediction():
        import time
        time.sleep(5)
        try:
            _run_full_prediction(session='startup', triggered_by='auto')
        except Exception as e:
            logger.error(f"启动初始预测失败: {e}")

    init_thread = threading.Thread(target=_initial_prediction, daemon=True)
    init_thread.start()

    logger.info("=" * 60)
    logger.info("多币种汇率预测系统 v6 启动")
    logger.info("访问地址: http://localhost:9091")
    logger.info("自动预测: 09:30 / 11:30 / 15:00（工作日）")
    logger.info("=" * 60)
    app.run(host='0.0.0.0', port=9091, debug=False, use_reloader=False)
