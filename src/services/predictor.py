"""
多币种汇率预测引擎

预测维度：未来 1周/2周/1个月/2个月/3个月/6个月
预测逻辑（按优先级）：
  1. USD/CNY：1周/2周由 30 日 ML 按 √(T/30) 衔接；30/60/90 天 LightGBM 集成；6 个月由 90 日 ML 阻尼外推，全期限同一模型族、避免期限折点
  2. USD/HKD：预测价夹紧于联系汇率常见参考区间（约 7.75–7.85）
  3. EUR/CNY：在技术分析/ML 基础上按美元指数强弱做轻度全期限一致性微调
  4. AUD/CAD/MXN 兑 CNY：长端按 √(T/180) 限幅，抑制纯技术外推过大
  5. EUR/HKD、GBP/HKD：与 EUR(CNY)/GBP(CNY) 及 USD 交叉软校准（偏差大时向隐含交叉价收敛）
  6. 其余期限：技术分析 + 宏观 + 情绪

宏观数据时间对齐：
  预测时使用「目标日期之前最近可得」的宏观数据。
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from .storage import get_rate_history, get_rate_at_days_ago
from .data_fetcher import (
    fetch_macro_for_pair, fetch_cfets_history, CURRENCY_PAIRS
)
from .sentiment import analyze_sentiment_for_pair

logger = logging.getLogger(__name__)

# 预测时间维度（天）
HORIZONS = {
    '1w':  7,
    '2w':  14,
    '1m':  30,
    '2m':  60,
    '3m':  90,
    '6m':  180,
}

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')

# ─── 旧模型加载（USD/CNY 专用）──────────────────────────────────────────────────

_ml_models = {}
_ml_scalers = {}
_ml_feat_cols = []
_usdcny_history: Optional[pd.DataFrame] = None
_macro_df: Optional[pd.DataFrame] = None


def load_ml_models():
    """加载现有 USD/CNY 的 LightGBM 集成模型"""
    global _ml_models, _ml_scalers, _ml_feat_cols, _usdcny_history, _macro_df
    try:
        import joblib
        _ml_models = joblib.load(os.path.join(MODEL_DIR, 'final_models.pkl'))
        _ml_scalers = joblib.load(os.path.join(MODEL_DIR, 'scalers.pkl'))
        _ml_feat_cols = joblib.load(os.path.join(MODEL_DIR, 'feat_cols.pkl'))

        fx_10y = os.path.join(PROJECT_DIR, 'data', 'exchange_rates_10y.csv')
        fx_path = os.path.join(PROJECT_DIR, 'exchange_rate_data.csv')
        if os.path.exists(fx_10y):
            hd = pd.read_csv(fx_10y, index_col='date', parse_dates=True)
            if 'USD/CNY' in hd.columns:
                _usdcny_history = hd[['USD/CNY']].rename(columns={'USD/CNY': 'close'}).sort_index()
        elif os.path.exists(fx_path):
            hd = pd.read_csv(fx_path, header=[0, 1], index_col=0)
            hd.columns = [col[0] for col in hd.columns]
            hd.index = pd.to_datetime(hd.index)
            _usdcny_history = hd[['Close']].rename(columns={'Close': 'close'}).sort_index()

        macro_path = os.path.join(PROJECT_DIR, 'macro_data.csv')
        if os.path.exists(macro_path):
            _macro_df = pd.read_csv(macro_path, parse_dates=['date'], index_col='date').sort_index()
            _macro_df.columns = [c.replace(' ', '_') for c in _macro_df.columns]

        logger.info(f"ML 模型加载成功，horizons={list(_ml_models.keys())}")
        return True
    except Exception as e:
        logger.warning(f"ML 模型加载失败: {e}")
        return False


def ml_models_loaded() -> bool:
    return bool(_ml_models)


# ─── 技术特征计算 ───────────────────────────────────────────────────────────────

def _build_price_series(pair: str, current_rate: float) -> pd.Series:
    """
    优先从本地存储获取历史汇率构建价格序列；
    不足时从 Yahoo Finance 补充；
    再不足时合并 USD/CNY 历史作为 fallback（仅 USD/CNY）。
    """
    records = get_rate_history(pair, days=300)
    records.sort(key=lambda x: x['ts'])

    prices = []
    seen_dates = set()
    for r in records:
        d = r['ts'][:10]
        if d not in seen_dates:
            seen_dates.add(d)
            prices.append(r['rate'])

    if len(prices) >= 20:
        series = pd.Series(prices)
    else:
        # 从 CFETS 拉取历史
        history = fetch_cfets_history(pair, days=250)
        if history:
            series = pd.Series([h['rate'] for h in history])
        elif pair == 'USD/CNY' and _usdcny_history is not None:
            series = _usdcny_history['close'].tail(250)
        else:
            series = pd.Series(prices if prices else [current_rate])

    # 追加当前价格（确保包含最新值）
    if len(series) == 0 or abs(series.iloc[-1] - current_rate) / current_rate > 0.05:
        series = pd.concat([series, pd.Series([current_rate])], ignore_index=True)

    return series


def _compute_technical_features(prices: pd.Series) -> dict:
    """计算技术指标特征"""
    c = prices
    n = len(c)
    feats = {}

    if n >= 5:
        feats['r5'] = float(c.pct_change(min(5, n-1)).iloc[-1] or 0)
    if n >= 20:
        feats['r20'] = float(c.pct_change(min(20, n-1)).iloc[-1] or 0)

    for w in [5, 10, 20, 60]:
        if n >= w:
            ma = c.rolling(w).mean().iloc[-1]
            feats[f'ma_dev_{w}'] = float((c.iloc[-1] - ma) / (ma + 1e-10))

    if n >= 14:
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rsi = 100 - 100 / (1 + gain / (loss + 1e-10))
        feats['rsi'] = float(rsi)
    else:
        feats['rsi'] = 50.0

    if n >= 26:
        e12 = c.ewm(span=12).mean().iloc[-1]
        e26 = c.ewm(span=26).mean().iloc[-1]
        feats['macd'] = float(e12 - e26)

    if n >= 20:
        bb_m = c.rolling(20).mean().iloc[-1]
        bb_s = c.rolling(20).std().iloc[-1]
        feats['bb_position'] = float((c.iloc[-1] - (bb_m - 2 * bb_s)) / (4 * bb_s + 1e-10))

    if n >= 5:
        returns = c.pct_change().dropna()
        feats['vol_5d'] = float(returns.tail(5).std() or 0)
    if n >= 20:
        feats['vol_20d'] = float(returns.tail(20).std() or 0)

    if n >= 20:
        x = np.arange(20)
        y = c.tail(20).values
        slope, _ = np.polyfit(x, y, 1)
        feats['trend_slope'] = float(slope / (c.iloc[-1] + 1e-10))

    return feats


# ─── 技术分析预测（通用，所有货币对）──────────────────────────────────────────────

def _predict_technical(
    prices: pd.Series,
    current_rate: float,
    horizon_days: int,
    macro: dict,
    sentiment_score: float = 0.0,
    pair: str = '',
) -> dict:
    """
    技术分析 + 宏观基本面 + 情绪三因子融合预测
    """
    feats = _compute_technical_features(prices)
    n = len(prices)

    # ── 因子1：线性趋势 ──
    lookback = min(30, n)
    if lookback >= 5:
        x = np.arange(lookback)
        y = prices.tail(lookback).values
        slope, _ = np.polyfit(x, y, 1)
        trend_daily = slope / (current_rate + 1e-10)
    else:
        trend_daily = 0.0

    # ── 因子2：RSI 均值回归信号 ──
    rsi = feats.get('rsi', 50.0)
    rsi_return = -(rsi - 50) / 50 * 0.015  # 最大 1.5% 回归

    # ── 因子3：布林带位置 ──
    bb_pos = feats.get('bb_position', 0.5)
    bb_return = -(bb_pos - 0.5) * 0.015  # 最大 1.5% 回归

    # ── 因子4：宏观利差（利率平价简化）──
    macro_daily = _compute_macro_signal(macro)

    # ── 因子5：情绪 ──
    sentiment_return = sentiment_score * 0.01  # 最大 1% 影响

    # ── 加权合并总收益率 ──
    # 趋势和宏观随时间累积，但有衰减；RSI、布林带和情绪是固定幅度的均值回归/冲击
    decay = np.exp(-horizon_days / 180)
    trend_return = trend_daily * horizon_days * (0.5 + 0.5 * decay)
    macro_return = macro_daily * horizon_days

    projected_return = (
        0.30 * trend_return
        + 0.15 * rsi_return
        + 0.15 * bb_return
        + 0.30 * macro_return
        + 0.10 * sentiment_return
    )

    # 商品货币 / 新兴市场：长端技术外推上限（锁汇视角防虚假大波段）
    if pair in ('AUD/CNY', 'CAD/CNY', 'GBP/CNY', 'MXN/CNY'):
        if horizon_days >= 180:
            lim = 0.032
        elif horizon_days >= 90:
            lim = 0.024
        elif horizon_days >= 60:
            lim = 0.018
        else:
            lim = None
        if lim is not None:
            projected_return = max(-lim, min(lim, projected_return))

    predicted_price = current_rate * (1 + projected_return)

    # ── 置信区间（基于历史波动率）──
    if n >= 3:
        daily_returns = prices.pct_change().dropna()
        daily_vol = float(daily_returns.tail(min(60, len(daily_returns))).std())
        if np.isnan(daily_vol) or daily_vol == 0:
            daily_vol = 0.005
    else:
        daily_vol = 0.005

    horizon_vol = daily_vol * np.sqrt(horizon_days)

    ci_95_low = predicted_price * (1 - 1.96 * horizon_vol)
    ci_95_high = predicted_price * (1 + 1.96 * horizon_vol)
    ci_68_low = predicted_price * (1 - horizon_vol)
    ci_68_high = predicted_price * (1 + horizon_vol)

    direction = '涨' if projected_return > 0.002 else ('跌' if projected_return < -0.002 else '平')

    return {
        'predicted_price': round(float(predicted_price), 6),
        'predicted_return': round(float(projected_return * 100), 3),
        'direction': direction,
        'confidence_interval_95': {
            'lower': round(float(ci_95_low), 6),
            'upper': round(float(ci_95_high), 6),
        },
        'confidence_interval_68': {
            'lower': round(float(ci_68_low), 6),
            'upper': round(float(ci_68_high), 6),
        },
        'uncertainty': round(float(horizon_vol * 100), 3),
        'factors': {
            'trend': round(float(trend_daily * horizon_days * 100), 3),
            'macro': round(float(macro_daily * horizon_days * 100), 3),
            'sentiment': round(float(sentiment_score), 3),
            'rsi': round(float(rsi), 2),
        },
        'method': 'technical_analysis',
    }


def _compute_macro_signal(macro: dict) -> float:
    """
    从宏观经济数据中提取单一利差信号（利率平价简化）
    正值 → 卖出货币相对升值压力
    返回的是日化收益率 drift
    """
    signal = 0.0

    # 中美利差（美联储利率 vs PBOC/中国基准）
    us_rate = macro.get('fed_funds_rate', macro.get('treasury_10y', None))
    cn_rate = macro.get('pboc_fix', macro.get('pboc_fix_usd', None))
    boc_rate = macro.get('boc_rate', None)
    boj_rate = macro.get('boj_rate', None)
    rba_rate = macro.get('rba_rate', None)
    ecb_rate = macro.get('ecb_rate', None)
    boe_rate = macro.get('boe_rate', None)
    banxico_rate = macro.get('banxico_rate', None)

    # VIX 风险情绪（高 VIX → 美元 / 港币避险需求）
    vix = macro.get('vix', None)
    if vix is not None:
        signal += (vix - 20) / 100 * 0.05 / 365  # VIX 超过 20 对应小幅利多美元

    # 美元指数趋势
    dx = macro.get('dx_index', None)
    if dx is not None:
        dx_signal = (dx - 100) / 100  # DXY > 100 表示美元偏强
        signal += dx_signal * 0.05 / 365

    # 利率差：高利率货币通常升值（简化），年化利率转日化
    for foreign_rate in [us_rate, boc_rate, boe_rate, ecb_rate]:
        if foreign_rate is not None:
            signal += (foreign_rate / 100) / 365

    return max(-0.0005, min(0.0005, signal))


# ─── USD/CNY 专用 ML 模型预测 ──────────────────────────────────────────────────

def _predict_usdcny_ml(current_rate: float, horizon_days: int, macro: dict) -> Optional[dict]:
    """
    直接调用现有 LightGBM 集成模型预测 USD/CNY（30/60/90 天）
    特征工程和预测逻辑内联，避免循环导入
    """
    if not ml_models_loaded() or _usdcny_history is None:
        return None
    if horizon_days not in _ml_models:
        return None

    try:
        target_date = datetime.now()
        relevant = _usdcny_history[_usdcny_history.index <= target_date].tail(250).copy()
        if len(relevant) < 60:
            return None

        c = relevant['close']
        data = relevant.copy()
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
        l_val = (-delta.where(delta < 0, 0)).rolling(14).mean()
        data['rsi'] = 100 - 100 / (1 + g / (l_val + 1e-10))
        data['rsi5'] = data['rsi'].rolling(5).mean()
        e12 = c.ewm(span=12).mean()
        e26 = c.ewm(span=26).mean()
        data['macd'] = e12 - e26
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

        # 注入宏观特征
        col_map = {'vix': 'mxv', 'dx': 'mxd', 'gov_rate_10y': 'mxt',
                   'sp500': 'mxsp', 'oil': 'mxol', 'gold': 'mxau',
                   'eurusd': 'mxeu', 'pboc_fix': 'mx_fix', 'cyb': 'mxcyb',
                   'dx_index': 'mxd', 'vix': 'mxv', 'fed_funds_rate': 'mxt'}
        for mk, mv in macro.items():
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
            data['mx_irp_signal'] = fix.diff(30) / (fix + 1e-10)
            for h in [30, 60, 90]:
                implied_fwd = c * (1 + us_r * h / 365)
                data[f'mx_irp_{h}d'] = (implied_fwd - c) / (c + 1e-10)
        if 'mx_fix' in data.columns:
            fix = data['mx_fix']
            data['mx_band_pos'] = (c - fix * 0.98) / (fix * 0.04 + 1e-10)
            data['mx_band_hit'] = ((c > fix * 1.02) | (c < fix * 0.98)).astype(int)
        for col_name in ['mxv', 'mxd', 'mxt']:
            if col_name in data.columns:
                data[f'{col_name}_d'] = data[col_name].diff(5)
                data[f'{col_name}_d20'] = data[col_name].diff(20)

        data = data.bfill().ffill().replace([np.inf, -np.inf], 0)
        latest = data.iloc[[-1]].copy()
        latest['close'] = relevant['close'].iloc[-1]
        current_price = float(latest['close'].values[0])

        horizon_models = _ml_models[horizon_days]
        scaler = _ml_scalers.get(horizon_days)
        if not scaler:
            return None

        for col_name in _ml_feat_cols:
            if col_name not in latest.columns:
                latest[col_name] = 0.0

        X = latest[_ml_feat_cols].values  # 转为 numpy 数组，避免特征名称警告
        X_scaled = scaler.transform(X)

        predictions_list = []
        model_preds = {}
        for name, model in horizon_models['models'].items():
            pred_ret = float(model.predict(X_scaled)[0])
            pred_price = current_price * (1 + pred_ret)
            model_preds[name] = {
                'predicted_price': round(pred_price, 4),
                'predicted_return': round(pred_ret * 100, 2),
            }
            predictions_list.append(pred_price)

        weights = horizon_models.get('weights')
        if weights:
            model_names = list(model_preds.keys())
            w = np.array([weights.get(n, 1.0 / len(predictions_list)) for n in model_names])
            w = w / w.sum()
            ensemble_price = float(np.average(predictions_list, weights=w))
        else:
            ensemble_price = float(np.mean(predictions_list))

        ensemble_ret = ensemble_price / current_price - 1
        
        pred_std = float(np.std(predictions_list))

        return {
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
            'uncertainty': round(pred_std / current_price * 100, 2),
            'model_predictions': model_preds,
            'method': 'lightgbm_ensemble',
        }
    except Exception as e:
        logger.warning(f"ML 模型预测失败，回退技术分析: {e}", exc_info=True)
    return None


def _clone_horizon_at_new_return(template: dict, current: float, ret: float, method: str) -> dict:
    """在保持置信区间相对宽度的前提下，按新收益率重定中心价（用于 USD/CNY 6m 与 ML 对齐）。"""
    p = current * (1 + ret)
    old_p = float(template.get('predicted_price') or 0)
    if old_p <= 0:
        old_p = current

    def _scale(ci_key: str) -> dict:
        lo = float(template[ci_key]['lower'])
        hi = float(template[ci_key]['upper'])
        return {
            'lower': round(p * (lo / old_p), 6),
            'upper': round(p * (hi / old_p), 6),
        }

    rd = '涨' if ret > 0.001 else ('跌' if ret < -0.001 else '平')
    out = {
        'predicted_price': round(p, 6),
        'predicted_return': round(ret * 100, 3),
        'direction': rd,
        'confidence_interval_95': _scale('confidence_interval_95'),
        'confidence_interval_68': _scale('confidence_interval_68'),
        'uncertainty': round(float(template.get('uncertainty', 0)) * 1.15, 3),
        'factors': template.get('factors', {}),
        'method': method,
    }
    if template.get('model_predictions'):
        out['model_predictions'] = template['model_predictions']
    return out


def _usdcny_blend_short_horizons_with_ml(
    results: dict, current_rate: float, macro: dict
) -> None:
    """
    1周 / 2周 与 30 日 ML 同一模型族：按 √(T/30) 缩放 30 日预测收益并夹紧，
    避免短端纯技术分析与 1m–3m LightGBM 结果割裂。
    """
    ml30 = _predict_usdcny_ml(current_rate, 30, macro)
    if ml30 is None:
        return
    r30 = ml30['predicted_return'] / 100.0
    r7 = float(np.clip(r30 * np.sqrt(7 / 30.0), -0.018, 0.018))
    r14 = float(np.clip(r30 * np.sqrt(14 / 30.0), -0.022, 0.022))
    if '1w' in results:
        results['1w'] = _clone_horizon_at_new_return(
            ml30, current_rate, r7, 'lightgbm_bridge_7d'
        )
    if '2w' in results:
        results['2w'] = _clone_horizon_at_new_return(
            ml30, current_rate, r14, 'lightgbm_bridge_14d'
        )


def _usdcny_align_six_month_with_ml(
    results: dict,
    current_rate: float,
    macro: dict,
) -> None:
    """
    6 个月与 3 个月（90 日 ML）同一模型族：由 90 日收益率 sqrt(180/90) 外推并夹紧，
    且不与 3 个月方向出现「长端反而更乐观」的倒挂。
    """
    ml90 = _predict_usdcny_ml(current_rate, 90, macro)
    if ml90 is None or '6m' not in results or '3m' not in results:
        return
    r90 = ml90['predicted_return'] / 100.0
    r180 = float(np.clip(r90 * (180 / 90) ** 0.5, -0.045, 0.045))
    r3m = results['3m']['predicted_return'] / 100.0
    if r3m < -0.002 and r180 > r3m + 0.0005:
        r180 = min(r180, r3m)
    elif r3m > 0.002 and r180 < r3m - 0.0005:
        r180 = max(r180, r3m)
    results['6m'] = _clone_horizon_at_new_return(ml90, current_rate, r180, 'lightgbm_extrapolated_180d')


def _clamp_usd_hkd_band(results: dict, current_rate: float) -> None:
    """联系汇率制度：USD/HKD 预测价夹在香港常见弱方/强方参考区间内（工程近似 7.75–7.85）。"""
    lo, hi = 7.75, 7.85
    for _k, pred in results.items():
        if not pred:
            continue
        px = float(pred['predicted_price'])
        if px < lo or px > hi:
            px = max(lo, min(hi, px))
            pred['predicted_price'] = round(px, 6)
            pred['predicted_return'] = round((px / current_rate - 1) * 100, 3)
            pred['direction'] = (
                '涨' if pred['predicted_return'] > 0.2 else (
                    '跌' if pred['predicted_return'] < -0.2 else '平'))
            pred['hkd_band_clamped'] = True


def _factor_summary_zh(macro: dict, sentiment_score: float) -> str:
    parts = []
    vix = macro.get('vix')
    if vix is not None:
        try:
            v = float(vix)
            if v > 24:
                parts.append('风险情绪(VIX)偏高')
            elif v < 16:
                parts.append('风险情绪(VIX)偏低')
            else:
                parts.append('风险情绪(VIX)中性')
        except (TypeError, ValueError):
            pass
    if macro.get('dx_index') is not None:
        parts.append('美元指数入模')
    if macro.get('fed_funds_rate') is not None or macro.get('treasury_10y') is not None:
        parts.append('利差/利率因子入模')
    if sentiment_score > 0.15:
        parts.append('新闻情绪偏正面')
    elif sentiment_score < -0.15:
        parts.append('新闻情绪偏负面')
    return '、'.join(parts[:5]) if parts else '宏观与情绪因子权重已融合（详见 macro_summary）'


_MAX_ABS_RET_AT_180D = {'AUD/CNY': 0.028, 'CAD/CNY': 0.024, 'MXN/CNY': 0.038}


def _horizon_disclaimer_zh(pair: str, days: int, method: str) -> str:
    if pair == 'MXN/CNY':
        return 'MXN/CNY 多为交叉合成报价，实际成交点差与流动性风险高于直盘；以下为模型参考价。'
    if pair == 'JPY/CNY':
        return '标价为每 100 日元兑人民币；涨跌幅为相对即期价的百分比，非日元本身波动幅度。'
    if pair in ('USD/HKD', 'EUR/HKD', 'GBP/HKD'):
        return '港币为联系汇率制度，USD/HKD 已按常见参考区间夹紧；EUR/HKD、GBP/HKD 主要反映非美货币腿波动。'
    if pair == 'USD/CNY' and 'lightgbm_extrapolated_180d' in str(method):
        return 'USD/CNY 6 个月由 90 日 ML 结果阻尼外推，与 1m/2m/3m 同属集成模型族，期限结构已对齐。'
    if pair == 'USD/CNY' and 'lightgbm_bridge_7d' in str(method):
        return '1 周点位由 30 日 ML 按时间尺度衔接，与 1m–3m 同属集成模型族。'
    if pair == 'USD/CNY' and 'lightgbm_bridge_14d' in str(method):
        return '2 周点位由 30 日 ML 按时间尺度衔接，与 1m–3m 同属集成模型族。'
    if method and str(method).endswith('_commodity_cap'):
        return '该期限预测已按商品货币/新兴市场波动做长端限幅，避免技术外推过大。'
    if pair == 'EUR/CNY' and method and 'dx_damp' in str(method):
        return '已按美元指数强弱对欧元兑人民币全期限预测做轻度一致性微调。'
    return ''


def _enrich_horizon_output(
    pair: str, days: int, pred: dict, macro: dict, sentiment_score: float, spot: float
) -> None:
    if not pred:
        return
    method = pred.get('method', '')
    pred['factor_summary_zh'] = _factor_summary_zh(macro, sentiment_score)
    pred['disclaimer_zh'] = _horizon_disclaimer_zh(pair, days, method)
    if 'JPY' in pair:
        pred['quote_convention_zh'] = '每 100 日元兑人民币（常见标价习惯）'
        pred['abs_move_per_100jpy'] = round(float(pred['predicted_price']) - spot, 6)
    if pair == 'MXN/CNY':
        pred['quote_convention_zh'] = '墨西哥比索兑人民币（多为间接合成）'


def _cap_high_beta_fx_horizons(pair: str, results: dict, current_rate: float) -> None:
    """商品货币 / 新兴市场：按 √(T/180) 缩放长端可接受幅度，抑制纯技术外推过大的 6m 等期限。"""
    max180 = _MAX_ABS_RET_AT_180D.get(pair)
    if not max180:
        return
    for label, days in HORIZONS.items():
        pred = results.get(label)
        if not pred:
            continue
        ret = pred['predicted_return'] / 100.0
        lim = max180 * np.sqrt(days / 180.0)
        if abs(ret) <= lim:
            continue
        new_ret = float(np.clip(ret, -lim, lim))
        base_method = pred.get('method', 'technical_analysis')
        nm = base_method if str(base_method).endswith('_commodity_cap') else f'{base_method}_commodity_cap'
        results[label] = _clone_horizon_at_new_return(pred, current_rate, new_ret, nm)
        results[label]['commodity_cap_applied'] = True


def _eur_cny_dx_damp(results: dict, current_rate: float, macro: dict) -> None:
    """美元指数偏强时略压欧元兑人民币全期限单边升值斜率，反之略抬，弱化「全期限同向机械外推」。"""
    dx = macro.get('dx_index')
    if dx is None:
        return
    try:
        dxv = float(dx)
    except (TypeError, ValueError):
        return
    if 100.0 <= dxv <= 103.0:
        return
    damp = 0.92 if dxv > 105.0 else (0.96 if dxv > 103.0 else (1.04 if dxv < 97.0 else (1.02 if dxv < 100.0 else 1.0)))
    for label in HORIZONS:
        pred = results.get(label)
        if not pred:
            continue
        old_ret = pred['predicted_return'] / 100.0
        new_ret = float(np.clip(old_ret * damp, -0.055, 0.055))
        if abs(new_ret - old_ret) < 1e-6:
            continue
        base_method = pred.get('method', 'technical_analysis')
        nm = f'{base_method}_dx_damp'
        results[label] = _clone_horizon_at_new_return(pred, current_rate, new_ret, nm)


def _apply_pair_postprocess(
    pair: str,
    results: dict,
    current_rate: float,
    macro: dict,
    prices: pd.Series,
    sentiment_score: float,
) -> None:
    if pair == 'USD/CNY':
        _usdcny_blend_short_horizons_with_ml(results, current_rate, macro)
        _usdcny_align_six_month_with_ml(results, current_rate, macro)
    if pair == 'USD/HKD':
        _clamp_usd_hkd_band(results, current_rate)
    if pair == 'EUR/CNY':
        _eur_cny_dx_damp(results, current_rate, macro)
    if pair in _MAX_ABS_RET_AT_180D:
        _cap_high_beta_fx_horizons(pair, results, current_rate)
    spot = float(current_rate)
    for label, days in HORIZONS.items():
        pred = results.get(label)
        if not pred:
            continue
        _enrich_horizon_output(pair, days, pred, macro, sentiment_score, spot)


# ─── 主预测入口 ────────────────────────────────────────────────────────────────

def predict_pair(pair: str, current_rate: float, obs_date: str = None) -> dict:
    """
    对单个货币对进行全维度预测（6 个时间节点）

    返回:
    {
      'pair': 'USD/CNY',
      'current_rate': 7.21,
      'predicted_at': '2026-05-11T09:30:00',
      'horizons': {
        '1w':  {'predicted_price': ..., 'direction': ..., ...},
        '2w':  {...},
        '1m':  {...},
        '2m':  {...},
        '3m':  {...},
        '6m':  {...},
      },
      'sentiment': {...},
      'macro': {...},
    }
    """
    if obs_date is None:
        obs_date = datetime.now().strftime('%Y-%m-%d')

    logger.info(f"开始预测 {pair} @ {current_rate} (date={obs_date})")

    prices = _build_price_series(pair, current_rate)
    macro = fetch_macro_for_pair(pair, obs_date)
    try:
        sentiment = analyze_sentiment_for_pair(pair)
        sentiment_score = sentiment.get('sentiment_score', 0.0)
    except Exception:
        sentiment = {'sentiment_score': 0.0, 'sentiment_label': '中性'}
        sentiment_score = 0.0

    results = {}
    for label, days in HORIZONS.items():
        pred = None

        # USD/CNY 的 30/60/90 天优先用 ML 模型
        if pair == 'USD/CNY' and days in [30, 60, 90]:
            pred = _predict_usdcny_ml(current_rate, days, macro)

        # fallback：技术分析
        if pred is None:
            pred = _predict_technical(prices, current_rate, days, macro, sentiment_score, pair)

        results[label] = pred

    _apply_pair_postprocess(pair, results, current_rate, macro, prices, sentiment_score)

    risk_note_zh = ''
    if pair == 'MXN/CNY':
        risk_note_zh = 'MXN/CNY 多为间接合成，点差与流动性风险高于直盘；锁汇请以银行可成交价为准。'
    elif 'JPY' in pair:
        risk_note_zh = '日元对人民币为小数标价；涨跌幅为相对即期百分比，请结合「每100日元绝对变动」理解。'

    framework_note_zh = ''
    if pair == 'USD/CNY':
        framework_note_zh = (
            'USD/CNY 全期限：1周/2周由30日ML按√(T/30)与上下限衔接；'
            '1月/2月/3月为LightGBM集成；6个月由90日ML阻尼外推并与3月方向协调，避免短端纯技术与中段ML割裂。'
        )

    return {
        'pair': pair,
        'current_rate': current_rate,
        'predicted_at': datetime.now().isoformat(),
        'obs_date': obs_date,
        'horizons': results,
        'sentiment': sentiment,
        'macro_summary': {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in list(macro.items())[:10]
        },
        'risk_note_zh': risk_note_zh,
        'framework_note_zh': framework_note_zh,
    }


def predict_all_pairs(rates: dict) -> dict:
    """
    预测全部 10 个货币对
    rates: {'USD/CNY': 7.21, 'EUR/CNY': 7.82, ...}
    """
    today = datetime.now().strftime('%Y-%m-%d')
    predictions = {}
    for cp in CURRENCY_PAIRS:
        pair = cp['pair']
        rate = rates.get(pair)
        if rate is None or rate <= 0:
            logger.warning(f"跳过 {pair}：无有效实时汇率")
            continue
        try:
            predictions[pair] = predict_pair(pair, rate, today)
        except Exception as e:
            logger.error(f"预测 {pair} 失败: {e}")
    try:
        _soft_align_eur_hkd_from_cross(predictions)
        _soft_align_gbp_hkd_from_cross(predictions)
    except Exception as e:
        logger.debug(f"HKD 交叉校验跳过: {e}")
    return predictions


def _soft_align_gbp_hkd_from_cross(predictions: dict) -> None:
    pairs_req = ('GBP/CNY', 'USD/CNY', 'USD/HKD', 'GBP/HKD')
    if not all(p in predictions for p in pairs_req):
        return
    for label in HORIZONS:
        try:
            gu = float(predictions['GBP/CNY']['horizons'][label]['predicted_price'])
            us = float(predictions['USD/CNY']['horizons'][label]['predicted_price'])
            uh = float(predictions['USD/HKD']['horizons'][label]['predicted_price'])
            gh = float(predictions['GBP/HKD']['horizons'][label]['predicted_price'])
        except (KeyError, TypeError, ValueError):
            continue
        if us <= 0 or uh <= 0 or gh <= 0:
            continue
        implied = (gu / us) * uh
        if implied <= 0:
            continue
        rel_dev = abs(implied - gh) / gh
        if rel_dev > 0.02:
            blended = gh * 0.45 + implied * 0.55
            cur_spot = float(predictions['GBP/HKD'].get('current_rate') or gh)
            predictions['GBP/HKD']['horizons'][label]['predicted_price'] = round(blended, 4)
            predictions['GBP/HKD']['horizons'][label]['predicted_return'] = round(
                (blended / cur_spot - 1) * 100, 3)
            predictions['GBP/HKD']['horizons'][label]['cross_aligned'] = True
            cr = predictions['GBP/HKD']['horizons'][label]['predicted_return']
            predictions['GBP/HKD']['horizons'][label]['direction'] = (
                '涨' if cr > 0.15 else ('跌' if cr < -0.15 else '平'))


def _soft_align_eur_hkd_from_cross(predictions: dict) -> None:
    """
    用 EUR/CNY ÷ USD/CNY × USD/HKD 得到隐含 EUR/HKD，与模型值偏差过大时做轻度向交叉价收敛（工程一致性）。
    """
    pairs_req = ('EUR/CNY', 'USD/CNY', 'USD/HKD', 'EUR/HKD')
    if not all(p in predictions for p in pairs_req):
        return
    for label in HORIZONS:
        try:
            eu = float(predictions['EUR/CNY']['horizons'][label]['predicted_price'])
            us = float(predictions['USD/CNY']['horizons'][label]['predicted_price'])
            uh = float(predictions['USD/HKD']['horizons'][label]['predicted_price'])
            eh = float(predictions['EUR/HKD']['horizons'][label]['predicted_price'])
        except (KeyError, TypeError, ValueError):
            continue
        if us <= 0 or uh <= 0 or eh <= 0:
            continue
        implied = (eu / us) * uh
        if implied <= 0:
            continue
        rel_dev = abs(implied - eh) / eh
        if rel_dev > 0.02:
            blended = eh * 0.45 + implied * 0.55
            cur_spot = float(predictions['EUR/HKD'].get('current_rate') or eh)
            predictions['EUR/HKD']['horizons'][label]['predicted_price'] = round(blended, 4)
            predictions['EUR/HKD']['horizons'][label]['predicted_return'] = round(
                (blended / cur_spot - 1) * 100, 3)
            predictions['EUR/HKD']['horizons'][label]['cross_aligned'] = True
            cr = predictions['EUR/HKD']['horizons'][label]['predicted_return']
            predictions['EUR/HKD']['horizons'][label]['direction'] = (
                '涨' if cr > 0.15 else ('跌' if cr < -0.15 else '平'))


# ─── 历史参考数据（过去1周/1个月）────────────────────────────────────────────────

def get_historical_reference(pair: str) -> dict:
    """
    从本地存储读取历史参考汇率
    返回: {'rate_7d_ago': 7.20, 'rate_30d_ago': 7.18, 'avg_7d': 7.21, 'avg_30d': 7.19}
    """
    ref = {}

    # 7 天前的具体汇率
    rate_7d = get_rate_at_days_ago(pair, 7)
    if rate_7d:
        ref['rate_7d_ago'] = round(rate_7d, 6)

    # 30 天前的具体汇率
    rate_30d = get_rate_at_days_ago(pair, 30)
    if rate_30d:
        ref['rate_30d_ago'] = round(rate_30d, 6)

    # 过去 7 天平均
    records_7d = get_rate_history(pair, 7)
    if records_7d:
        avg_7d = np.mean([r['rate'] for r in records_7d])
        ref['avg_7d'] = round(float(avg_7d), 6)

    # 过去 30 天平均
    records_30d = get_rate_history(pair, 30)
    if records_30d:
        avg_30d = np.mean([r['rate'] for r in records_30d])
        ref['avg_30d'] = round(float(avg_30d), 6)

    return ref


# ─── 初始化 ────────────────────────────────────────────────────────────────────
load_ml_models()
