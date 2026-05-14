"""
情绪分析服务
流程：
  1. 从财经媒体（新浪财经、路透中文、新华财经等）抓取当日外汇相关新闻
  2. 调用 LLM（OpenAI 兼容 API）提取影响汇率的关键信息并量化
  3. 输出 sentiment_score ∈ [-1, 1] 及关键驱动因素

sentiment_score:
  +1.0 → 极度看涨目标货币（卖出方向）
  -1.0 → 极度看跌目标货币
   0.0 → 中性
"""
import re
import logging
import requests
from datetime import datetime
from .settings_manager import (
    get_api_key,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    NEWS_API_KEY,
    get_requests_proxies,
    get_http_timeout,
    get_http_proxy,
    normalize_openai_base_url,
)
from .storage import read_json, write_json

logger = logging.getLogger(__name__)

SENTIMENT_CACHE_FILE = 'sentiment_cache.json'


def _rq_kw(timeout: int = 20):
    kw = {'timeout': get_http_timeout(timeout)}
    px = get_requests_proxies()
    if px:
        kw['proxies'] = px
    return kw

# ─── 新闻抓取 ─────────────────────────────────────────────────────────────────

NEWS_SOURCES = [
    {
        'name': '新浪财经外汇',
        'url': 'https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=1686&k=&num=30&page=1&r=0',
        'type': 'sina',
    },
    {
        'name': '新华财经',
        'url': 'https://www.xinhuanet.com/fortune/forex.htm',
        'type': 'xinhua',
    },
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json, text/html, */*',
    'Referer': 'https://finance.sina.com.cn/',
}


def fetch_sina_finance_news() -> list[dict]:
    """抓取新浪财经外汇频道新闻"""
    news = []
    try:
        url = 'https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=1686&k=&num=30&page=1&r=0'
        resp = requests.get(url, headers=HEADERS, **_rq_kw(20))
        data = resp.json()
        items = data.get('result', {}).get('data', [])
        for item in items[:20]:
            title = item.get('title', '')
            intro = item.get('intro', '')
            if title:
                news.append({
                    'title': title,
                    'summary': intro or title,
                    'source': '新浪财经',
                    'time': item.get('ctime', ''),
                })
    except Exception as e:
        logger.debug(f"新浪财经新闻抓取失败: {e}")
    return news


def fetch_newsapi_forex(pair: str) -> list[dict]:
    """从 NewsAPI 获取外汇相关英文新闻"""
    api_key = get_api_key(NEWS_API_KEY)
    if not api_key:
        return []
    try:
        query_map = {
            'USD/CNY': 'USD CNY exchange rate OR dollar yuan',
            'EUR/CNY': 'EUR CNY exchange rate OR euro yuan',
            'GBP/CNY': 'GBP CNY exchange rate OR pound yuan',
            'CAD/CNY': 'CAD CNY exchange rate OR canadian yuan',
            'JPY/CNY': 'JPY CNY exchange rate OR yen yuan',
            'AUD/CNY': 'AUD CNY exchange rate OR australian yuan',
            'MXN/CNY': 'MXN CNY exchange rate OR peso yuan',
            'USD/HKD': 'USD HKD exchange rate OR dollar hong kong',
            'EUR/HKD': 'EUR HKD exchange rate OR euro hong kong',
            'GBP/HKD': 'GBP HKD exchange rate OR pound hong kong',
        }
        query = query_map.get(pair, f'{pair} exchange rate forex')
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': query,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 10,
            'apiKey': api_key,
        }
        resp = requests.get(url, params=params, **_rq_kw(25))
        data = resp.json()
        news = []
        for article in data.get('articles', []):
            news.append({
                'title': article.get('title', ''),
                'summary': article.get('description', '') or article.get('title', ''),
                'source': article.get('source', {}).get('name', 'NewsAPI'),
                'time': article.get('publishedAt', ''),
            })
        return news
    except Exception as e:
        logger.debug(f"NewsAPI 获取失败: {e}")
        return []


def gather_news_for_pair(pair: str) -> list[dict]:
    """聚合该货币对的所有相关新闻"""
    all_news = []
    all_news.extend(fetch_sina_finance_news())
    all_news.extend(fetch_newsapi_forex(pair))

    sell_currency = pair.split('/')[0]
    buy_currency = pair.split('/')[1]
    keywords = [
        sell_currency.lower(), buy_currency.lower(),
        '汇率', '外汇', 'exchange', 'forex', 'currency',
        sell_currency, buy_currency,
    ]

    filtered = []
    for item in all_news:
        text = (item.get('title', '') + ' ' + item.get('summary', '')).lower()
        if any(kw.lower() in text for kw in keywords):
            filtered.append(item)

    return filtered[:15]


# ─── LLM 量化分析 ─────────────────────────────────────────────────────────────

def build_sentiment_prompt(pair: str, news_items: list[dict]) -> str:
    sell, buy = pair.split('/')
    news_text = '\n'.join([
        f"[{i+1}] {n['source']} | {n['title']}\n    {n['summary'][:150]}"
        for i, n in enumerate(news_items[:10])
    ])
    return f"""你是一位专业的外汇分析师。请分析以下财经新闻对 {sell}/{buy} 汇率的影响。

货币对说明：{sell}/{buy} 表示用 {buy} 购买 {sell}，分析方向为 {sell} 相对 {buy} 的强弱。

当日财经新闻：
{news_text}

请执行以下分析：
1. 识别每条新闻对 {sell}/{buy} 汇率的影响（利多/利空/中性）
2. 综合判断市场情绪倾向
3. 给出情绪分数 sentiment_score，范围 -1.0 到 +1.0：
   - +1.0 = 极度看涨 {sell}（{sell} 将对 {buy} 升值）
   - -1.0 = 极度看跌 {sell}（{sell} 将对 {buy} 贬值）
   - 0.0 = 市场中性

请严格按照以下 JSON 格式输出（不要有额外文字）：
{{
  "sentiment_score": 0.0,
  "sentiment_label": "中性/偏多/偏空/强烈看多/强烈看空",
  "confidence": 0.5,
  "key_drivers": ["驱动因素1", "驱动因素2"],
  "bullish_signals": ["利多信号1"],
  "bearish_signals": ["利空信号1"],
  "summary": "一句话总结"
}}"""


def call_llm_for_sentiment(prompt: str) -> dict | None:
    """调用 LLM API 分析情绪"""
    api_key = get_api_key(OPENAI_API_KEY)
    if not api_key:
        logger.info("未配置 LLM API Key，跳过情绪分析")
        return None

    base_url = normalize_openai_base_url(get_api_key(OPENAI_BASE_URL)) or 'https://api.openai.com/v1'
    model = get_api_key(OPENAI_MODEL) or 'gpt-4o-mini'

    try:
        import openai
        import httpx
        px_url = get_http_proxy()
        if px_url:
            with httpx.Client(proxy=px_url, timeout=get_http_timeout(60)) as http_client:
                client = openai.OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    http_client=http_client,
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {'role': 'system', 'content': '你是专业外汇分析师，只输出 JSON 格式的分析结果。'},
                        {'role': 'user', 'content': prompt},
                    ],
                    max_tokens=500,
                    temperature=0.1,
                )
        else:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': '你是专业外汇分析师，只输出 JSON 格式的分析结果。'},
                    {'role': 'user', 'content': prompt},
                ],
                max_tokens=500,
                temperature=0.1,
            )
        content = response.choices[0].message.content.strip()
        # 提取 JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            import json
            return json.loads(json_match.group())
    except Exception as e:
        logger.warning(f"LLM 情绪分析失败: {e}")
    return None


def analyze_sentiment_for_pair(pair: str) -> dict:
    """
    分析指定货币对的市场情绪
    返回：{'sentiment_score': 0.0, 'confidence': 0.5, ...}
    """
    today = datetime.now().strftime('%Y-%m-%d')
    cache_key = f"{pair}_{today}"

    # 检查今日缓存
    cache = read_json(SENTIMENT_CACHE_FILE, default={})
    if cache_key in cache:
        logger.info(f"使用缓存情绪数据 {pair} {today}")
        return cache[cache_key]

    news = gather_news_for_pair(pair)
    if not news:
        result = _neutral_sentiment(pair, reason='无新闻数据')
        return result

    prompt = build_sentiment_prompt(pair, news)
    llm_result = call_llm_for_sentiment(prompt)

    if llm_result:
        result = {
            'pair': pair,
            'date': today,
            'sentiment_score': float(llm_result.get('sentiment_score', 0.0)),
            'confidence': float(llm_result.get('confidence', 0.5)),
            'sentiment_label': llm_result.get('sentiment_label', '中性'),
            'key_drivers': llm_result.get('key_drivers', []),
            'bullish_signals': llm_result.get('bullish_signals', []),
            'bearish_signals': llm_result.get('bearish_signals', []),
            'summary': llm_result.get('summary', ''),
            'news_count': len(news),
            'source': 'llm_analysis',
        }
    else:
        # LLM 不可用时用关键词简单打分
        result = _keyword_sentiment(pair, news)

    # 写入今日缓存
    cache[cache_key] = result
    if len(cache) > 500:
        oldest_keys = sorted(cache.keys())[:100]
        for k in oldest_keys:
            del cache[k]
    write_json(SENTIMENT_CACHE_FILE, cache)
    return result


def _neutral_sentiment(pair: str, reason: str = '') -> dict:
    return {
        'pair': pair,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'sentiment_score': 0.0,
        'confidence': 0.3,
        'sentiment_label': '中性',
        'key_drivers': [],
        'summary': reason or '情绪数据不可用',
        'source': 'default',
    }


def _keyword_sentiment(pair: str, news: list) -> dict:
    """基于关键词的简单情绪打分（LLM 不可用时的 fallback）"""
    sell = pair.split('/')[0]
    bullish_kw = ['升值', '走强', '上涨', '看涨', '利多', 'rise', 'gain', 'bullish', 'strengthen']
    bearish_kw = ['贬值', '走弱', '下跌', '看跌', '利空', 'fall', 'decline', 'bearish', 'weaken']

    score = 0.0
    for n in news:
        text = (n.get('title', '') + ' ' + n.get('summary', '')).lower()
        for kw in bullish_kw:
            if kw in text:
                score += 0.1
        for kw in bearish_kw:
            if kw in text:
                score -= 0.1

    score = max(-1.0, min(1.0, score))
    label = '偏多' if score > 0.1 else ('偏空' if score < -0.1 else '中性')

    return {
        'pair': pair,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'sentiment_score': round(score, 3),
        'confidence': 0.4,
        'sentiment_label': label,
        'key_drivers': [],
        'summary': f'基于 {len(news)} 条新闻关键词分析',
        'source': 'keyword',
    }


def analyze_all_pairs_sentiment() -> dict:
    """分析所有货币对的情绪（批量）"""
    from .data_fetcher import CURRENCY_PAIRS
    results = {}
    for cp in CURRENCY_PAIRS:
        pair = cp['pair']
        try:
            results[pair] = analyze_sentiment_for_pair(pair)
        except Exception as e:
            logger.warning(f"情绪分析失败 {pair}: {e}")
            results[pair] = _neutral_sentiment(pair)
    return results
