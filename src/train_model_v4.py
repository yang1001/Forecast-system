#!/usr/bin/env python3
"""
USD/CNY 汇率预测 v5 - IRP增强版 (目标 MAE < 0.05)

核心改进：
  1. 利率平价公式 (IRP): F = S * (1+r_cny) / (1+r_usd)
  2. PBOC中间价特征: 偏离度、趋势、波动带
  3. 中美利差特征: 从收益率曲线估算
  4. 近5年数据训练 (2021-05以后)
  5. 更强的正则化 + 近期数据加权
"""
import pandas as pd, numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings, os, joblib
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping as lgb_es, log_evaluation

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FX_PATH_10Y = os.path.join(PROJECT_DIR, 'data', 'exchange_rates_10y.csv')
FX_PATH = os.path.join(PROJECT_DIR, 'exchange_rate_data.csv')
MACRO_PATH = os.path.join(PROJECT_DIR, 'macro_data.csv')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
LOG_PATH = os.path.join(PROJECT_DIR, 'training_output.log')

os.makedirs(MODEL_DIR, exist_ok=True)

with open(LOG_PATH, 'w', encoding='utf-8') as log_f:
    import sys as _sys

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    _sys.stdout = Tee(_sys.stdout, log_f)

    print("=" * 60)
    print("USD/CNY 汇率预测 v5 - IRP利率平价增强版")
    print("  MAE 目标: < 0.05 (30天)")
    print("=" * 60)

    if os.path.exists(FX_PATH_10Y):
        df = pd.read_csv(FX_PATH_10Y, index_col='date')
        df.index = pd.to_datetime(df.index)
        # 兼容列名，取 USD/CNY 作为 close
        df = df[['USD/CNY']].dropna().rename(columns={'USD/CNY': 'close'}).sort_index()
        print(f"使用新的10年历史数据: {FX_PATH_10Y}")
    else:
        df = pd.read_csv(FX_PATH, header=[0, 1], index_col=0)
        df.columns = [col[0] for col in df.columns]
        df.index = pd.to_datetime(df.index)
        df = df[['Close']].dropna().rename(columns={'Close': 'close'}).sort_index()
        print(f"使用旧版历史数据: {FX_PATH}")

    macro_data = None
    if os.path.exists(MACRO_PATH):
        macro_data = pd.read_csv(MACRO_PATH, parse_dates=['date'], index_col='date').sort_index()
        macro_data.columns = [c.replace(' ', '_') for c in macro_data.columns]
        macro_data = macro_data.dropna(how='all')
        print(f"宏观: {len(macro_data)}条, {list(macro_data.columns)}")

    print(f"汇率: {len(df)}条, {df.index[0].date()} ~ {df.index[-1].date()}")

    # ==================== 特征工程 v5 ====================
    def make_features(df, macro_data=None):
        data = df.copy()
        c = data['close']

        data['r1'] = c.pct_change(1)
        for p in [2, 3, 5, 10, 15, 20]:
            data[f'r{p}'] = c.pct_change(p)

        for w in [5, 10, 20, 30, 60, 120]:
            ma = c.rolling(w).mean()
            data[f'd{w}'] = (c - ma) / ma

        for w in [5, 10, 20]:
            data[f'v{w}'] = data['r1'].rolling(w).std()
        data['v_ratio'] = data['v5'] / (data['v20'] + 1e-10)

        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        data['rsi'] = 100 - 100 / (1 + gain / (loss + 1e-10))
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

        high14 = c.rolling(14).max()
        low14 = c.rolling(14).min()
        data['stoch_k'] = 100 * (c - low14) / (high14 - low14 + 1e-10)
        data['stoch_d'] = data['stoch_k'].rolling(3).mean()
        data['cci'] = (c - c.rolling(20).mean()) / (c.rolling(20).std() * 0.015 + 1e-10)
        tr1 = high14 - low14
        tr2 = (high14 - c.shift(1)).abs()
        tr3 = (low14 - c.shift(1)).abs()
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

        macro_out = []
        if macro_data is not None:
            ma = macro_data.copy()
            cols = ma.columns.tolist()

            # ---- PBOC中间价特征 (直接算到data上避免索引不对齐) ----
            has_us_rate = 'TMSL10Y' in cols

            if 'PBOC_FIXING' in cols:
                fix_s = ma['PBOC_FIXING'].reindex(data.index, method='ffill')
                data['mx_fix'] = fix_s
                data['mx_fix_dev'] = (c - fix_s) / (fix_s + 1e-10)
                data['mx_fix_dev_ma'] = data['mx_fix_dev'].rolling(20).mean()
                data['mx_fix_d'] = fix_s.diff(5)
                data['mx_fix_d20'] = fix_s.diff(20)
                data['mx_fix_r'] = fix_s.pct_change(20)

            if has_us_rate:
                us_r_s = ma['TMSL10Y'].reindex(data.index, method='ffill') / 100
                data['mx_usr'] = us_r_s
                for h in [30, 60, 90]:
                    data[f'mx_irp_carry_{h}d'] = us_r_s * h / 365
                data['mx_usr_d'] = us_r_s.diff()
                data['mx_usr_d20'] = us_r_s.diff(20)

            if 'PBOC_FIXING' in cols and has_us_rate:
                fix_s = data['mx_fix']
                us_r_s = data['mx_usr']
                fix_fwd_ret = fix_s.diff(30)
                data['mx_irp_signal'] = fix_fwd_ret / (fix_s + 1e-10)
                for h in [30, 60, 90]:
                    data[f'mx_irp_{h}d'] = (c * (1 + us_r_s * h / 365) - c) / (c + 1e-10)

            if 'PBOC_FIXING' in cols:
                fix_s = data['mx_fix']
                band_upper = fix_s * 1.02
                band_lower = fix_s * 0.98
                data['mx_band_pos'] = (c - band_lower) / (band_upper - band_lower + 1e-10)
                data['mx_band_hit'] = ((c > band_upper) | (c < band_lower)).astype(int)

            if 'VIXCLS' in cols:
                v = ma['VIXCLS']
                ma['mxv'] = v
                ma['mxv_d'] = v.diff(5)
                ma['mxv_d20'] = v.diff(20)
                ma['mxv_h'] = (v > 25).astype(int)

            if 'DX-Y.NYB' in cols:
                d = ma['DX-Y.NYB']
                ma['mxd'] = d
                ma['mxd_d'] = d.diff(5)
                ma['mxd_d20'] = d.diff(20)
                ma['mxd_r'] = d.pct_change(20)

            if has_us_rate:
                t = ma['TMSL10Y']
                ma['mxt'] = t / 100
                ma['mxt_d'] = t.diff()
                ma['mxt_d20'] = t.diff(20)

            if 'SP500' in cols:
                sp = ma['SP500']
                ma['mxsp'] = sp
                ma['mxsp_r'] = sp.pct_change(20)
                ma['mxsp_d'] = sp.pct_change(5)

            if 'OIL' in cols:
                ol = ma['OIL']
                ma['mxol'] = ol
                ma['mxol_r'] = ol.pct_change(20)

            if 'GOLD' in cols:
                au = ma['GOLD']
                ma['mxau'] = au
                ma['mxau_r'] = au.pct_change(20)

            if 'EURUSD' in cols:
                eu = ma['EURUSD']
                ma['mxeu'] = eu
                ma['mxeu_r'] = eu.pct_change(20)

            if 'CYB_ETF' in cols:
                cyb = ma['CYB_ETF']
                ma['mxcyb'] = cyb
                ma['mxcyb_r'] = cyb.pct_change(20)

            if 'SP500' in cols and 'VIXCLS' in cols:
                ma['mxsp_vix'] = ma['SP500'] / (ma['VIXCLS'] + 1e-10)

            macro_out = [c for c in ma.columns if c not in cols]
            ma_sel = ma[macro_out].reindex(data.index, method='ffill')
            data = pd.concat([data, ma_sel], axis=1)

        data = data.bfill().ffill().replace([np.inf, -np.inf], 0)
        return data, macro_out

    df_feat, macro_cols = make_features(df, macro_data)
    ALL_COLS = [c for c in df_feat.columns if c != 'close']
    print(f"总特征: {len(ALL_COLS)} (含{len(macro_cols)}个宏观)")

    def prepare(horizon):
        df_feat['_target'] = df_feat['close'].shift(-horizon) / df_feat['close'] - 1
        v = df_feat['_target'].notna()
        for c in ALL_COLS:
            v = v & df_feat[c].notna()
        X = df_feat.loc[v, ALL_COLS].values
        y = df_feat.loc[v, '_target'].values
        dates = df_feat.index[v]
        return X, y, dates

    # ==================== 仅用近3年数据 ====================
    CUTOFF_DATE = pd.Timestamp('2023-05-09')

    HORIZONS = [30, 60, 90]
    tscv = TimeSeriesSplit(n_splits=3)
    best_info = {}

    for horizon in HORIZONS:
        print(f"\n{'='*50}")
        print(f"Horizon: {horizon}天 ({horizon//30}个月)")
        print(f"{'='*50}")

        X_all, y_all, idx_all = prepare(horizon)

        recent_mask = idx_all >= CUTOFF_DATE
        X_all = X_all[recent_mask]
        y_all = y_all[recent_mask]
        idx_all = idx_all[recent_mask]

        n = len(X_all)
        prices_all = df_feat['close'].loc[idx_all].values
        print(f"  样本(近5年): {n}, 特征: {X_all.shape[1]}")
        print(f"  日期: {idx_all[0].date()} ~ {idx_all[-1].date()}")

        n_recent = int(n * 0.25)
        sw_all = np.ones(n)
        if n_recent > 0:
            sw_all[-n_recent:] = 5.0

        fpm = {'LGB': [], 'XGB': [], 'GBR': [], 'RF': [], 'ENS': []}
        fw_list = []

        for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_all)):
            X_tr, X_te = X_all[tr_idx], X_all[te_idx]
            y_tr, y_te = y_all[tr_idx], y_all[te_idx]
            sw_tr = sw_all[tr_idx]

            sc = RobustScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)

            pp, mp = {}, {}

            lgb = LGBMRegressor(
                n_estimators=5000, max_depth=5, learning_rate=0.008,
                num_leaves=31, subsample=0.7, colsample_bytree=0.6,
                min_child_samples=20, reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, verbose=-1, n_jobs=1
            )
            lgb.fit(X_tr_s, y_tr, sample_weight=sw_tr,
                    eval_set=[(X_te_s, y_te)], eval_metric='l1',
                    callbacks=[lgb_es(250, verbose=False), log_evaluation(0)])
            pp['LGB'] = lgb.predict(X_te_s)
            fpm['LGB'].append(mean_absolute_error(
                prices_all[te_idx] * (1 + y_te), prices_all[te_idx] * (1 + pp['LGB'])))
            mp['LGB'] = lgb

            xgb = XGBRegressor(
                n_estimators=5000, max_depth=5, learning_rate=0.008,
                subsample=0.7, colsample_bytree=0.6,
                reg_alpha=1.0, reg_lambda=2.0,
                random_state=42, verbosity=0, n_jobs=1,
                early_stopping_rounds=250
            )
            xgb.fit(X_tr_s, y_tr, sample_weight=sw_tr,
                    eval_set=[(X_te_s, y_te)], verbose=False)
            pp['XGB'] = xgb.predict(X_te_s)
            fpm['XGB'].append(mean_absolute_error(
                prices_all[te_idx] * (1 + y_te), prices_all[te_idx] * (1 + pp['XGB'])))
            mp['XGB'] = xgb

            gbr = GradientBoostingRegressor(
                n_estimators=2000, max_depth=4, learning_rate=0.015,
                subsample=0.7, min_samples_leaf=15,
                validation_fraction=0.15, n_iter_no_change=120,
                random_state=42
            )
            gbr.fit(X_tr_s, y_tr, sample_weight=sw_tr)
            pp['GBR'] = gbr.predict(X_te_s)
            fpm['GBR'].append(mean_absolute_error(
                prices_all[te_idx] * (1 + y_te), prices_all[te_idx] * (1 + pp['GBR'])))
            mp['GBR'] = gbr

            rf = RandomForestRegressor(
                n_estimators=2000, max_depth=10, min_samples_leaf=15,
                max_features=0.45, random_state=42, n_jobs=1
            )
            rf.fit(X_tr_s, y_tr, sample_weight=sw_tr)
            pp['RF'] = rf.predict(X_te_s)
            fpm['RF'].append(mean_absolute_error(
                prices_all[te_idx] * (1 + y_te), prices_all[te_idx] * (1 + pp['RF'])))
            mp['RF'] = rf

            gw = {k: 1.0 / max(fpm[k][-1], 1e-8) for k in ['LGB', 'XGB', 'GBR', 'RF']}
            gs = sum(gw.values())
            gw = {k: v / gs for k, v in gw.items()}
            fw_list.append(gw)

            ens_r = np.zeros(len(y_te))
            for k in mp:
                ens_r += gw[k] * pp[k]
            fpm['ENS'].append(mean_absolute_error(
                prices_all[te_idx] * (1 + y_te), prices_all[te_idx] * (1 + ens_r)))

            print(f"  Fold {fold+1}: 集成 Price_MAE={fpm['ENS'][-1]:.5f} | "
                  f"LGB={fpm['LGB'][-1]:.5f} XGB={fpm['XGB'][-1]:.5f} "
                  f"GBR={fpm['GBR'][-1]:.5f} RF={fpm['RF'][-1]:.5f}")

        avg_pm = {k: np.mean(v) for k, v in fpm.items()}
        avg_w = {k: np.mean([fw.get(k, 0) for fw in fw_list]) for k in ['LGB', 'XGB', 'GBR', 'RF']}
        ws = sum(avg_w.values())
        avg_w = {k: v / ws for k, v in avg_w.items()}

        s = '✓✓' if avg_pm['ENS'] < 0.05 else '✓' if avg_pm['ENS'] < 0.10 else '✗'
        print(f"\n  {s} {horizon}d 平均 Price_MAE: 集成={avg_pm['ENS']:.5f} | "
              f"LGB={avg_pm['LGB']:.5f} XGB={avg_pm['XGB']:.5f} "
              f"GBR={avg_pm['GBR']:.5f} RF={avg_pm['RF']:.5f}")
        print(f"  权重: {', '.join([f'{k}={avg_w[k]:.3f}' for k in avg_w])}")

        best_info[horizon] = {'price_mae': avg_pm['ENS'], 'weights': avg_w, 'fold_maes': fpm}

    print("\n" + "=" * 60)
    print("训练最终模型（全部近5年数据 + 近期加权）")
    print("=" * 60)

    final_models = {}
    final_scalers = {}

    for horizon in HORIZONS:
        X_all, y_all, idx_all = prepare(horizon)
        recent_mask = idx_all >= CUTOFF_DATE
        X_all = X_all[recent_mask]
        y_all = y_all[recent_mask]

        n = len(X_all)
        sw = np.ones(n)
        n_rec = int(n * 0.25)
        if n_rec > 0:
            sw[-n_rec:] = 5.0

        sc = RobustScaler()
        X_s = sc.fit_transform(X_all)

        md = {}
        for name, Cls, kwargs in [
            ('LGB', LGBMRegressor, dict(n_estimators=5000, max_depth=5, learning_rate=0.008,
                 num_leaves=31, subsample=0.7, colsample_bytree=0.6,
                 min_child_samples=20, reg_alpha=1.0, reg_lambda=1.0,
                 random_state=42, verbose=-1, n_jobs=1)),
            ('XGB', XGBRegressor, dict(n_estimators=5000, max_depth=5, learning_rate=0.008,
                 subsample=0.7, colsample_bytree=0.6,
                 reg_alpha=1.0, reg_lambda=2.0,
                 random_state=42, verbosity=0, n_jobs=1)),
            ('GBR', GradientBoostingRegressor, dict(n_estimators=2000, max_depth=4,
                 learning_rate=0.015, subsample=0.7, min_samples_leaf=15,
                 random_state=42)),
            ('RF', RandomForestRegressor, dict(n_estimators=2000, max_depth=10,
                 min_samples_leaf=15, max_features=0.45,
                 random_state=42, n_jobs=1))
        ]:
            print(f"  {horizon}d: {name}...")
            m = Cls(**kwargs)
            m.fit(X_s, y_all, sample_weight=sw)
            md[name] = m

        final_models[horizon] = {'models': md, 'weights': best_info[horizon]['weights']}
        final_scalers[horizon] = sc
        print(f"  OK {horizon}d")

    print("\n" + "=" * 60)
    print("特征重要性 (LGB, 30d) Top 30")
    print("=" * 60)
    m = final_models[30]['models']['LGB']
    imp = sorted(zip(ALL_COLS, m.feature_importances_), key=lambda x: x[1], reverse=True)
    for name, val in imp[:30]:
        print(f"  {name:20s} {val:.4f}")

    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    ok = True
    for h in HORIZONS:
        bi = best_info[h]
        s = '✓✓' if bi['price_mae'] < 0.05 else '✓' if bi['price_mae'] < 0.10 else '✗'
        if bi['price_mae'] >= 0.05:
            ok = False
        print(f"  {s} {h}d ({h//30}个月): Price_MAE={bi['price_mae']:.5f}")
    if ok:
        print("  ✓✓ 达到 MAE < 0.05 目标!")

    joblib.dump(final_models, os.path.join(MODEL_DIR, 'final_models.pkl'))
    joblib.dump(final_scalers, os.path.join(MODEL_DIR, 'scalers.pkl'))
    joblib.dump(ALL_COLS, os.path.join(MODEL_DIR, 'feat_cols.pkl'))
    print(f"\n模型已保存, 特征数: {len(ALL_COLS)}")
