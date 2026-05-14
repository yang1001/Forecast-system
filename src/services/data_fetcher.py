"""
数据采集服务
- 实时汇率：中国外汇交易中心 CFETS（chinamoney.com.cn）
- HKD 汇率：香港金管局 HKMA API
- 历史汇率：Yahoo Finance（备用 / 冷启动回填）
- 宏观数据：
    美国 → FRED (Federal Reserve Bank of St. Louis)
    欧元区 → ECB Statistical Data Warehouse
    英国 → Bank of England (BoE)
    加拿大 → Bank of Canada
    日本 → Bank of Japan
    澳大利亚 → Reserve Bank of Australia
    墨西哥 → Banco de Mexico (Banxico)
    中国 → 中国人民银行 / 国家统计局
    香港 → HKMA
"""
import requests
import logging
from datetime import datetime, timedelta, date
from .settings_manager import (
    get_api_key,
    FRED_API_KEY,
    BANXICO_API_KEY,
    get_general_setting,
    get_requests_proxies,
    get_http_timeout,
)
from .storage import cache_macro, get_cached_macro

logger = logging.getLogger(__name__)

DEFAULT_CFETS_REALTIME_URL = 'https://www.chinamoney.com.cn/ags/ms/cm-u-bk-ccpr/CcprRealTime'
DEFAULT_HKMA_SPOT_URL = (
    'https://api.hkma.gov.hk/public/market-data-and-statistics/'
    'monthly-statistical-bulletin/exchange-fund/spot-exchange-rates?'
    'offset=0&choose=end_of_period'
)


def _rq_extras(timeout: int | None = None) -> dict:
    """requests 公共参数：超时 + 代理"""
    t = get_http_timeout(timeout if timeout is not None else 20)
    kw = {'timeout': t}
    px = get_requests_proxies()
    if px:
        kw['proxies'] = px
    return kw

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'https://www.chinamoney.com.cn/',
}

# 10 个货币对定义
CURRENCY_PAIRS = [
    {'sell': 'USD', 'buy': 'CNY', 'pair': 'USD/CNY'},
    {'sell': 'EUR', 'buy': 'CNY', 'pair': 'EUR/CNY'},
    {'sell': 'GBP', 'buy': 'CNY', 'pair': 'GBP/CNY'},
    {'sell': 'CAD', 'buy': 'CNY', 'pair': 'CAD/CNY'},
    {'sell': 'JPY', 'buy': 'CNY', 'pair': 'JPY/CNY'},
    {'sell': 'AUD', 'buy': 'CNY', 'pair': 'AUD/CNY'},
    {'sell': 'MXN', 'buy': 'CNY', 'pair': 'MXN/CNY'},
    {'sell': 'USD', 'buy': 'HKD', 'pair': 'USD/HKD'},
    {'sell': 'EUR', 'buy': 'HKD', 'pair': 'EUR/HKD'},
    {'sell': 'GBP', 'buy': 'HKD', 'pair': 'GBP/HKD'},
]


# ─── CFETS 实时汇率 ───────────────────────────────────────────────────────────

def fetch_cfets_realtime_cny() -> dict:
    """从 CFETS CcprHisNew 获取当日人民币中间价（仅返回目标 7 个 CNY 对）"""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        url = 'https://www.chinamoney.com.cn/ags/ms/cm-u-bk-ccpr/CcprHisNew'
        payload = {
            'startDate': today,
            'endDate': today,
            'currency': '',
            'pageSize': 50,
            'pageNum': 1,
            'lang': 'CN',
        }
        resp = requests.post(url, data=payload, headers=HEADERS, **_rq_extras(25))
        data = resp.json()
        head = (data.get('data') or {}).get('head', [])
        records = data.get('records', [])
        if not records or not head:
            return {}
        r = records[0]
        vals = r.get('values', [])
        target_map = {
            'USD/CNY': 'USD/CNY',
            'EUR/CNY': 'EUR/CNY',
            'GBP/CNY': 'GBP/CNY',
            '100JPY/CNY': 'JPY/CNY',
            'CAD/CNY': 'CAD/CNY',
            'AUD/CNY': 'AUD/CNY',
            'MXN/CNY': 'MXN/CNY',
        }
        result = {}
        for i, h in enumerate(head):
            if i < len(vals) and vals[i] and h in target_map:
                try:
                    val = float(vals[i])
                    if h == '100JPY/CNY':
                        val = val / 100
                    result[target_map[h]] = val
                except (ValueError, TypeError):
                    pass
        if result:
            logger.info(f"CFETS 中间价获取成功: {list(result.keys())}")
        return result
    except Exception as e:
        logger.warning(f"CFETS 中间价获取失败: {e}")
        return {}


def fetch_cfets_history_cny(currency: str, start_date: str, end_date: str) -> list:
    try:
        url = 'https://www.chinamoney.com.cn/ags/ms/cm-u-bk-ccpr/CcprHisNew'
        payload = {
            'startDate': start_date,
            'endDate': end_date,
            'currency': currency,
            'pageSize': 500,
            'pageNum': 1,
            'lang': 'CN',
        }
        resp = requests.post(url, data=payload, headers=HEADERS, **_rq_extras(25))
        text = (resp.text or '').strip()
        if not text or resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except ValueError:
            return []
        records = data.get('records', data.get('data', []))
        result = []
        for r in records:
            d = r.get('date', r.get('txDt', ''))
            mid = r.get('middleRatTx', r.get('midRat', r.get('mid', '')))
            if d and mid:
                try:
                    result.append({'date': d, 'rate': float(str(mid).replace(',', ''))})
                except (ValueError, TypeError):
                    pass
        result.sort(key=lambda x: x['date'])
        return result
    except Exception as e:
        logger.warning(f"CFETS 历史数据获取失败 ({currency}): {e}")
        return []


# ─── HKMA 实时及历史汇率 ──────────────────────────────────────────────────────

def fetch_hkma_usd_hkd() -> float | None:
    """从 HKMA 获取 USD/HKD 实时汇率"""
    try:
        url = (get_general_setting('hkma_spot_rates_url', '') or '').strip() or DEFAULT_HKMA_SPOT_URL
        resp = requests.get(url, **_rq_extras(25))
        data = resp.json()
        records = data.get('result', {}).get('dataSet', [])
        if records:
            latest = records[0]
            val = latest.get('hkd_usd_spot')
            if val:
                # HKMA 报的是 HKD per USD，与我们的 USD/HKD 一致
                return float(val)
    except Exception as e:
        logger.warning(f"HKMA USD/HKD 获取失败: {e}")
    return None


# ─── 新浪财经外汇实时数据（国内可访问）────────────────────────────────────────

SINA_FOREX_TICKERS = {
    'USD/CNY': 'fx_susdcny',
    'EUR/CNY': 'fx_seurcny',
    'GBP/CNY': 'fx_sgbpcny',
    'JPY/CNY': 'fx_sjpycny',
    'CAD/CNY': 'fx_scadcny',
    'AUD/CNY': 'fx_saudcny',
    'MXN/CNY': 'fx_smxncny',
    'USD/HKD': 'fx_susdhkd',
    'EUR/HKD': 'fx_seurhkd',
    'GBP/HKD': 'fx_sgbphkd',
}

SINA_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://finance.sina.com.cn',
}


def _sina_forex_current(pair: str) -> float | None:
    """从新浪财经获取实时汇率（国内可访问）"""
    code = SINA_FOREX_TICKERS.get(pair)
    if not code:
        return None
    try:
        url = f'https://hq.sinajs.cn/list={code}'
        resp = requests.get(url, headers=SINA_HEADERS, timeout=10)
        text = (resp.text or '').strip()
        if 'hq_str' not in text:
            logger.debug(f'新浪 {pair}: 无数据')
            return None
        data = text.split('"')[1]
        parts = data.split(',')
        if len(parts) > 3 and parts[3]:
            return float(parts[3])
    except Exception as e:
        logger.debug(f'新浪 {pair}: {e}')
    return None


def fetch_sina_all_rates() -> dict:
    """批量从新浪获取全部汇率"""
    codes = [SINA_FOREX_TICKERS[cp['pair']] for cp in CURRENCY_PAIRS]
    url = f'https://hq.sinajs.cn/list={",".join(codes)}'
    try:
        resp = requests.get(url, headers=SINA_HEADERS, timeout=10)
        text = (resp.text or '').strip()
        rates = {}
        for pair, code in SINA_FOREX_TICKERS.items():
            mark = f'var hq_str_{code}='
            if mark not in text:
                continue
            start = text.index(mark) + len(mark) + 1
            end = text.index('"', start)
            data = text[start:end]
            parts = data.split(',')
            if len(parts) > 3 and parts[3]:
                rates[pair] = float(parts[3])
        if rates:
            logger.info(f'新浪实时汇率获取成功: {list(rates.keys())}')
        return rates
    except Exception as e:
        logger.warning(f'新浪实时汇率获取失败: {e}')
        return {}


# ─── CFETS 每日中间价（历史数据）────────────────────────────────────────────────

def fetch_cfets_history(pair: str, days: int = 250) -> list:
    """从 CFETS CcprHisNew 获取历史日线中间价"""
    currency_map = {
        'USD/CNY': 'USD', 'EUR/CNY': 'EUR', 'GBP/CNY': 'GBP',
        'CAD/CNY': 'CAD', 'JPY/CNY': 'JPY', 'AUD/CNY': 'AUD',
        'MXN/CNY': 'MXN',
    }
    currency = currency_map.get(pair, '')
    if not currency:
        return []
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime('%Y-%m-%d')
        url = 'https://www.chinamoney.com.cn/ags/ms/cm-u-bk-ccpr/CcprHisNew'
        payload = {
            'startDate': start_date,
            'endDate': end_date,
            'currency': '',
            'pageSize': 500,
            'pageNum': 1,
            'lang': 'CN',
        }
        resp = requests.post(url, data=payload, headers=HEADERS, **_rq_extras(25))
        data = resp.json()
        head = (data.get('data') or {}).get('head', [])
        records = data.get('records', [])
        pair_key = f'{currency}/CNY'
        col_idx = None
        for i, h in enumerate(head):
            if h == pair_key:
                col_idx = i
                break
        if col_idx is None:
            return []
        result = []
        for r in records:
            vals = r.get('values', [])
            if col_idx < len(vals) and vals[col_idx]:
                try:
                    result.append({'date': r['date'], 'rate': float(vals[col_idx])})
                except (ValueError, TypeError):
                    pass
        result.sort(key=lambda x: x['date'])
        return result[-days:]
    except Exception as e:
        logger.debug(f'CFETS 历史数据获取失败 ({pair}): {e}')
        return []


# ─── 宏观数据辅助（新浪财经国内接口 + yfinance 外网降级）─────────────────────────

_SINA_MACRO_MAP = {
    'vix':      'gb_vix',       # VIX（国内不可用，返回空）
    'dx_index': 'gb_diniw',     # 美元指数（国内不可用，返回空）
    'sp500':    'gb_$inx',      # S&P 500 ✅
    'nasdaq':   'gb_$ixic',     # NASDAQ ✅
    'dji':      'gb_$dji',      # 道琼斯 ✅
    'gold':     'hf_XAU',       # 黄金 ✅
    'oil':      'hf_CL',        # 原油 ✅
    'nikkei225':'gb_$n225',     # 日经225（国内不可用）
    'iron_ore': 'hf_I',         # 铁矿石
}

_YF_MACRO_MAP = {
    '^VIX':       'vix',
    'DX-Y.NYB':   'dx_index',
    '^GSPC':      'sp500',
    '^IXIC':      'nasdaq',
    '^DJI':       'dji',
    'GC=F':       'gold',
    'CL=F':        'oil',
    '^N225':      'nikkei225',
    'IRON.AX':    'iron_ore',
}

def _sina_macro(key: str) -> float | None:
    """从新浪财经获取宏观指标（国内可用）"""
    code = _SINA_MACRO_MAP.get(key)
    if not code:
        return None
    try:
        url = f'https://hq.sinajs.cn/list={code}'
        resp = requests.get(url, headers=SINA_HEADERS, timeout=10)
        text = (resp.text or '').strip()
        if 'hq_str' not in text or len(text) < 50:
            return None
        data = text.split('"')[1]
        parts = data.split(',')
        if not parts or not parts[0]:
            return None
        # gb_* 格式: name, price, change, ... → parts[1]
        # hf_* 格式: open, price, ... → parts[0] 或 parts[3]
        if code.startswith('gb_'):
            if len(parts) > 1 and parts[1]:
                return float(parts[1])
        else:
            if len(parts) > 3 and parts[3]:
                return float(parts[3])
            if len(parts) > 1 and parts[1]:
                return float(parts[1])
    except Exception:
        pass
    return None

def _try_yf_close(ticker: str) -> float | None:
    """尝试从新浪获取宏观指标，不可用时尝试 yfinance 降级（外网）"""
    macro_key = _YF_MACRO_MAP.get(ticker)
    if macro_key:
        v = _sina_macro(macro_key)
        if v is not None:
            return v
    try:
        import yfinance as yf
        from .storage import DATA_DIR
        yf.set_tz_cache_location(DATA_DIR)
        t = yf.Ticker(ticker)
        hist = t.history(period='5d')
        if hist is not None and not hist.empty:
            closes = hist['Close'].dropna()
            if len(closes) > 0:
                return float(closes.iloc[-1])
    except Exception:
        pass
    return None


# ─── 综合实时汇率获取（优先 CFETS/HKMA，备用新浪）─────────────────────────────

def fetch_all_realtime_rates() -> dict:
    """
    获取全部 10 个货币对的实时汇率
    优先级: 新浪实时价 > HKMA > CFETS 每日中间价(仅填补空缺)
    返回: {'USD/CNY': 7.2100, 'EUR/CNY': 7.8200, ..., 'USD/HKD': 7.7800, ...}
    """
    rates = {}

    # 1. 新浪实时价（优先，盘中真实交易数据）
    sina_rates = fetch_sina_all_rates()
    rates.update(sina_rates)

    # 2. HKMA USD/HKD
    usd_hkd = fetch_hkma_usd_hkd()
    if usd_hkd:
        rates['USD/HKD'] = usd_hkd

    # 3. CFETS 官方中间价填补新浪未覆盖的 CNY 对
    cfets_rates = fetch_cfets_realtime_cny()
    for pair in ['USD/CNY', 'EUR/CNY', 'GBP/CNY', 'CAD/CNY', 'AUD/CNY']:
        if pair in cfets_rates and pair not in rates:
            rates[pair] = cfets_rates[pair]

    logger.info(f"实时汇率获取完成，共 {len(rates)} 个货币对")
    return rates


# ─── 宏观数据：各国央行官方 API ────────────────────────────────────────────────

def fetch_fred_indicator(series_id: str, api_key: str, obs_date: str = None) -> float | None:
    """从 FRED 获取单个指标"""
    if not api_key:
        return None
    try:
        params = {
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json',
            'sort_order': 'desc',
            'limit': 10,
        }
        if obs_date:
            params['observation_end'] = obs_date
        url = 'https://api.stlouisfed.org/fred/series/observations'
        resp = requests.get(url, params=params, **_rq_extras(20))
        data = resp.json()
        for obs in data.get('observations', []):
            val = obs.get('value', '.')
            if val != '.':
                return float(val)
    except Exception as e:
        logger.debug(f"FRED {series_id} 获取失败: {e}")
    return None


def fetch_us_macro(obs_date: str = None) -> dict:
    """获取美国宏观经济数据（FRED 官方 API）"""
    api_key = get_api_key(FRED_API_KEY)
    indicators = {}
    fred_series = {
        'fed_funds_rate': 'FEDFUNDS',      # 联邦基金利率
        'treasury_10y': 'DGS10',           # 10年期国债
        'cpi_yoy': 'CPIAUCSL',             # CPI（同比需自算）
        'unemployment': 'UNRATE',          # 失业率
        'pmi_manufacturing': 'MANEMP',     # 制造业就业（PMI代理）
    }
    if api_key:
        for key, series in fred_series.items():
            val = fetch_fred_indicator(series, api_key, obs_date)
            if val is not None:
                indicators[key] = val
    # 备用：新浪宏观指标（国内可用）
    sina_macro = {
        'vix': 'vix',
        'dx_index': 'dx_index',
        'sp500': 'sp500',
        'gold': 'gold',
        'oil': 'oil',
    }
    for key, macro_key in sina_macro.items():
        if key not in indicators:
            v = _sina_macro(macro_key)
            if v is not None:
                indicators[key] = v
    # CFETS 人民币中间价（国内可用，ML 模型 mx_fix 核心特征）
    if 'pboc_fix' not in indicators:
        _pboc = fetch_pboc_middle_rate('USD')
        if _pboc:
            indicators['pboc_fix'] = _pboc
    return indicators


def fetch_ecb_indicator(series_key: str) -> float | None:
    """从 ECB Statistical Data Warehouse 获取数据"""
    try:
        url = f'https://data-api.ecb.europa.eu/service/data/{series_key}?format=jsondata&lastNObservations=3'
        resp = requests.get(url, timeout=15)
        data = resp.json()
        obs = data.get('dataSets', [{}])[0].get('series', {})
        if obs:
            first_series = next(iter(obs.values()))
            observations = first_series.get('observations', {})
            if observations:
                last_key = max(observations.keys(), key=lambda k: int(k))
                val = observations[last_key][0]
                return float(val)
    except Exception as e:
        logger.debug(f"ECB {series_key} 获取失败: {e}")
    return None


def fetch_eu_macro(obs_date: str = None) -> dict:
    """获取欧元区宏观经济数据（ECB 官方）"""
    indicators = {}
    ecb_series = {
        'ecb_rate': 'FM/B.U2.EUR.RT.MM.EURIBOR3MD_.HSTA',   # EURIBOR 3个月
        'eu_cpi': 'ICP/M.U2.N.000000.4.ANR',                  # CPI 年率
    }
    for key, series in ecb_series.items():
        val = fetch_ecb_indicator(series)
        if val is not None:
            indicators[key] = val
    return indicators


def fetch_boe_rate() -> float | None:
    """英格兰银行基准利率"""
    try:
        url = 'https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/2024&Dateto=now&SeriesCodes=IUMABEDR&CSVF=TN&UsingCodes=Y'
        resp = requests.get(url, timeout=15)
        lines = resp.text.strip().split('\n')
        for line in reversed(lines):
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    return float(parts[-1].strip().strip('"'))
                except ValueError:
                    continue
    except Exception as e:
        logger.debug(f"BoE 利率获取失败: {e}")
    return None


def fetch_uk_macro(obs_date: str = None) -> dict:
    """英国宏观经济数据"""
    indicators = {}
    rate = fetch_boe_rate()
    if rate is not None:
        indicators['boe_rate'] = rate
    return indicators


def fetch_boc_rate() -> float | None:
    """加拿大央行利率"""
    try:
        url = 'https://www.bankofcanada.ca/valet/observations/YCCA/json?recent=10'
        resp = requests.get(url, timeout=15)
        data = resp.json()
        obs = data.get('observations', [])
        if obs:
            val = obs[-1].get('YCCA', {}).get('v')
            if val:
                return float(val)
    except Exception as e:
        logger.debug(f"BoC 利率获取失败: {e}")
    return None


def fetch_canada_macro(obs_date: str = None) -> dict:
    """加拿大宏观经济数据"""
    indicators = {}
    rate = fetch_boc_rate()
    if rate is not None:
        indicators['boc_rate'] = rate
    v = _sina_macro('oil')
    if v is not None:
        indicators['oil_price'] = v
    return indicators


def fetch_boj_rate() -> float | None:
    """日本央行政策利率（BoJ）"""
    try:
        url = 'https://www.stat-search.boj.or.jp/ssi/mtshtml/ir01_m_1_en.html'
        resp = requests.get(url, headers=HEADERS, timeout=15)
        import re
        matches = re.findall(r'[-]?\d+\.\d+', resp.text[:5000])
        if matches:
            for m in matches[:5]:
                val = float(m)
                if -1.0 <= val <= 3.0:
                    return val
    except Exception as e:
        logger.debug(f"BoJ 利率获取失败: {e}")
    return None


def fetch_japan_macro(obs_date: str = None) -> dict:
    """日本宏观经济数据"""
    indicators = {}
    rate = fetch_boj_rate()
    if rate is not None:
        indicators['boj_rate'] = rate
    v = _sina_macro('nikkei225')
    if v is not None:
        indicators['nikkei225'] = v
    return indicators


def fetch_rba_rate() -> float | None:
    """澳洲联储政策利率"""
    try:
        url = 'https://www.rba.gov.au/statistics/tables/csv/f01hist.csv'
        resp = requests.get(url, timeout=15)
        lines = [l for l in resp.text.split('\n') if l.strip()]
        # 找到含数据的最后一行
        for line in reversed(lines):
            parts = line.split(',')
            if len(parts) >= 5:
                try:
                    return float(parts[1].strip())
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        logger.debug(f"RBA 利率获取失败: {e}")
    return None


def fetch_australia_macro(obs_date: str = None) -> dict:
    """澳大利亚宏观经济数据"""
    indicators = {}
    rate = fetch_rba_rate()
    if rate is not None:
        indicators['rba_rate'] = rate
    v = _sina_macro('iron_ore')
    if v is not None:
        indicators['iron_ore'] = v
    return indicators


def fetch_banxico_rate() -> float | None:
    """墨西哥央行隔夜利率（Banxico）"""
    api_key = get_api_key(BANXICO_API_KEY)
    if not api_key:
        return None
    try:
        url = f'https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF61745/datos/oportuno?token={api_key}'
        resp = requests.get(url, timeout=15)
        data = resp.json()
        obs = data.get('bmx', {}).get('series', [{}])[0].get('datos', [])
        if obs:
            return float(obs[-1].get('dato', '0'))
    except Exception as e:
        logger.debug(f"Banxico 利率获取失败: {e}")
    return None


def fetch_mexico_macro(obs_date: str = None) -> dict:
    """墨西哥宏观经济数据"""
    indicators = {}
    rate = fetch_banxico_rate()
    if rate is not None:
        indicators['banxico_rate'] = rate
    return indicators


def fetch_pboc_middle_rate(currency: str = 'USD') -> float | None:
    """
    人民银行中间价（CFETS CcprHisNew API，国内可用）
    CFETS 每日 09:15 公布
    """
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        url = 'https://www.chinamoney.com.cn/ags/ms/cm-u-bk-ccpr/CcprHisNew'
        payload = {
            'startDate': today, 'endDate': today,
            'currency': '',
            'pageSize': 50, 'pageNum': 1, 'lang': 'CN',
        }
        resp = requests.post(url, data=payload, headers=HEADERS, **_rq_extras(25))
        data = resp.json()
        head = (data.get('data') or {}).get('head', [])
        records = data.get('records', [])
        if not records or not head:
            return None
        pair_key = f'{currency}/CNY'
        col_idx = None
        for i, h in enumerate(head):
            if h == pair_key:
                col_idx = i
                break
        if col_idx is None:
            return None
        vals = records[0].get('values', [])
        if col_idx < len(vals) and vals[col_idx]:
            return float(vals[col_idx])
    except Exception as e:
        logger.debug(f"PBOC 中间价获取失败: {e}")
    return None


def fetch_china_macro(obs_date: str = None) -> dict:
    """中国宏观经济数据（人民银行 / 国家统计局）"""
    indicators = {}
    pboc_fix = fetch_pboc_middle_rate('USD')
    if pboc_fix:
        indicators['pboc_fix_usd'] = pboc_fix
        indicators['pboc_fix'] = pboc_fix
    sina_cn = {'shanghai_comp': 'sh000001'}
    for key, code in sina_cn.items():
        try:
            url = f'https://hq.sinajs.cn/list={code}'
            resp = requests.get(url, headers=SINA_HEADERS, timeout=10)
            text = (resp.text or '').strip()
            if 'hq_str' not in text or len(text) < 80:
                continue
            data = text.split('"')[1]
            parts = data.split(',')
            if len(parts) > 3 and parts[3]:
                indicators[key] = float(parts[3])
        except Exception:
            pass
    return indicators


def fetch_hkma_macro(obs_date: str = None) -> dict:
    """香港金管局宏观数据"""
    indicators = {}
    # HIBOR 3个月
    try:
        url = ('https://api.hkma.gov.hk/public/market-data-and-statistics/'
               'monthly-statistical-bulletin/exchange-fund/interbank-rates?offset=0&choose=end_of_period')
        resp = requests.get(url, timeout=15)
        data = resp.json()
        records = data.get('result', {}).get('dataSet', [])
        if records:
            indicators['hibor_3m'] = float(records[0].get('hibor_3m', 0) or 0)
    except Exception as e:
        logger.debug(f"HKMA HIBOR 获取失败: {e}")
    return indicators


# ─── 按国家汇总宏观数据 ────────────────────────────────────────────────────────

COUNTRY_MACRO_FETCHERS = {
    'US': fetch_us_macro,
    'EU': fetch_eu_macro,
    'UK': fetch_uk_macro,
    'CA': fetch_canada_macro,
    'JP': fetch_japan_macro,
    'AU': fetch_australia_macro,
    'MX': fetch_mexico_macro,
    'CN': fetch_china_macro,
    'HK': fetch_hkma_macro,
}

# 货币对所需国家映射
PAIR_COUNTRIES = {
    'USD/CNY': ['US', 'CN'],
    'EUR/CNY': ['EU', 'CN'],
    'GBP/CNY': ['UK', 'CN'],
    'CAD/CNY': ['CA', 'CN'],
    'JPY/CNY': ['JP', 'CN'],
    'AUD/CNY': ['AU', 'CN'],
    'MXN/CNY': ['MX', 'CN'],
    'USD/HKD': ['US', 'HK'],
    'EUR/HKD': ['EU', 'HK'],
    'GBP/HKD': ['UK', 'HK'],
}


def fetch_macro_for_pair(pair: str, obs_date: str = None) -> dict:
    """
    获取货币对所需的宏观经济数据
    obs_date: 目标日期（YYYY-MM-DD），用于缓存 key 和历史查询
    """
    if obs_date is None:
        obs_date = datetime.now().strftime('%Y-%m-%d')

    countries = PAIR_COUNTRIES.get(pair, [])
    combined = {}
    for country in countries:
        # 先查缓存
        cached = get_cached_macro(obs_date, country)
        if cached:
            combined.update(cached)
            continue
        # 实时抓取
        fetcher = COUNTRY_MACRO_FETCHERS.get(country)
        if fetcher:
            try:
                data = fetcher(obs_date)
                if data:
                    combined.update(data)
                    cache_macro(obs_date, country, data)
            except Exception as e:
                logger.warning(f"宏观数据获取失败 ({pair}, {country}): {e}")
    return combined


# ─── 冷启动：批量回填历史汇率 ────────────────────────────────────────────────────

def backfill_history(pair: str, days: int = 30) -> list:
    """
    冷启动时从 CFETS 回填 CNY 历史汇率（HKD 对无历史可用新浪单点）
    同时写入本地存储
    """
    from .storage import append_rate_record
    records = fetch_cfets_history(pair, days)
    if not records:
        v = _sina_forex_current(pair)
        if v:
            today = datetime.now().strftime('%Y-%m-%d')
            records = [{'date': today, 'rate': v}]
    for r in records:
        ts = f"{r['date']}T00:00:00" if 'date' in r else None
        append_rate_record(pair, r['rate'], source='cfets_backfill', ts=ts)
    logger.info(f"冷启动回填 {pair}: {len(records)} 条记录")
    return records
