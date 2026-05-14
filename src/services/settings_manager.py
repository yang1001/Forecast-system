"""
设置管理服务
- 支持 API Key 存储（AES 加密）
- 设置页面访问需要密码验证
"""
import os
import hashlib
import hmac
import base64
import secrets
from datetime import datetime
from urllib.parse import urlparse

from .storage import read_json, write_json

SETTINGS_FILE = 'settings.json'
# 明文快照：便于备份、手写编辑与服务迁移；保存设置时会自动刷新
SYSTEM_SETTINGS_JSON = 'system_settings.json'
DEFAULT_ADMIN_PASSWORD = 'admin123'

# ─── 密码哈希 ─────────────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100_000)
    return dk.hex()


def _make_salt() -> str:
    return secrets.token_hex(16)


# ─── 简单对称加密（用于存储 API Key）────────────────────────────────────────────

def _xor_encrypt(text: str, key: str) -> str:
    key_bytes = key.encode()
    text_bytes = text.encode()
    result = bytes([b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(text_bytes)])
    return base64.b64encode(result).decode()


def _xor_decrypt(token: str, key: str) -> str:
    key_bytes = key.encode()
    data = base64.b64decode(token)
    result = bytes([b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(data)])
    return result.decode()


def _get_encryption_key(settings: dict) -> str:
    """用密码哈希的前 32 字符做加密 key"""
    return settings.get('password_hash', 'default_key')[:32].ljust(32, '0')


# ─── 公开接口 ─────────────────────────────────────────────────────────────────

def get_settings() -> dict:
    """读取完整设置（API Key 仍加密）"""
    s = read_json(SETTINGS_FILE, default={})
    if 'password_hash' not in s:
        salt = _make_salt()
        s['password_hash'] = _hash_password(DEFAULT_ADMIN_PASSWORD, salt)
        s['password_salt'] = salt
        write_json(SETTINGS_FILE, s)
    return s


def verify_password(password: str) -> bool:
    """验证管理员密码"""
    s = get_settings()
    salt = s.get('password_salt', '')
    expected = s.get('password_hash', '')
    actual = _hash_password(password, salt)
    return hmac.compare_digest(actual, expected)


def change_password(old_password: str, new_password: str) -> bool:
    """修改密码（同时重新加密所有 API Key）"""
    if not verify_password(old_password):
        return False
    s = get_settings()
    old_key = _get_encryption_key(s)

    # 先解密所有 API key
    api_keys_plain = {}
    for k, v in s.get('api_keys', {}).items():
        try:
            api_keys_plain[k] = _xor_decrypt(v, old_key)
        except Exception:
            api_keys_plain[k] = ''

    # 生成新密码哈希
    salt = _make_salt()
    s['password_hash'] = _hash_password(new_password, salt)
    s['password_salt'] = salt
    new_key = _get_encryption_key(s)

    # 用新 key 重新加密
    s['api_keys'] = {k: _xor_encrypt(v, new_key) for k, v in api_keys_plain.items()}
    write_json(SETTINGS_FILE, s)
    try:
        write_system_settings_json_file()
    except Exception:
        pass
    return True


def set_api_key(name: str, value: str):
    """保存一个 API Key（加密存储）"""
    s = get_settings()
    key = _get_encryption_key(s)
    if 'api_keys' not in s:
        s['api_keys'] = {}
    s['api_keys'][name] = _xor_encrypt(value, key)
    write_json(SETTINGS_FILE, s)


def get_api_key(name: str) -> str:
    """读取一个 API Key（解密）"""
    s = get_settings()
    enc = s.get('api_keys', {}).get(name, '')
    if not enc:
        return os.environ.get(name.upper().replace('-', '_'), '')
    key = _get_encryption_key(s)
    try:
        return _xor_decrypt(enc, key)
    except Exception:
        return ''


def get_all_api_keys_masked() -> dict:
    """返回所有 API Key 的掩码版本（前 4 位 + ***）"""
    s = get_settings()
    key = _get_encryption_key(s)
    result = {}
    for name, enc in s.get('api_keys', {}).items():
        try:
            plain = _xor_decrypt(enc, key)
            if len(plain) > 4:
                result[name] = plain[:4] + '*' * (len(plain) - 4)
            else:
                result[name] = '****'
        except Exception:
            result[name] = '****'
    return result


def save_general_setting(key: str, value):
    """保存普通设置项（非 API Key）"""
    s = get_settings()
    s[key] = value
    write_json(SETTINGS_FILE, s)


def get_general_setting(key: str, default=None):
    """读取普通设置项"""
    s = get_settings()
    return s.get(key, default)


def get_http_proxy() -> str:
    """HTTP(S) 代理，用于访问 NewsAPI 等外网（设置页或系统环境变量）"""
    return (
        (get_general_setting('http_proxy', '') or '').strip()
        or (os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY') or '').strip()
    )


def get_requests_proxies() -> dict | None:
    p = get_http_proxy()
    if not p:
        return None
    return {'http': p, 'https': p}


def get_http_timeout(default: int = 20) -> int:
    """单次 HTTP 超时（秒），范围 5–120"""
    try:
        v = get_general_setting('http_timeout_sec', None)
        if v is None or v == '':
            return default
        return max(5, min(int(v), 120))
    except (TypeError, ValueError):
        return default


def normalize_openai_base_url(url: str) -> str:
    """
    OpenAI 兼容 SDK（含 DeepSeek）通常要求 base_url 指向 …/v1。
    仅填写 https://api.deepseek.com 这类「无路径」时自动补 /v1。
    """
    raw = (url or '').strip()
    if not raw:
        return raw
    raw = raw.rstrip('/')
    if raw.lower().endswith('/v1'):
        return raw
    try:
        parsed = urlparse(raw)
        if not parsed.scheme or not parsed.netloc:
            return raw
        path = (parsed.path or '').strip('/')
        if not path:
            return f'{parsed.scheme}://{parsed.netloc}/v1'
    except Exception:
        return raw
    return raw


# ─── API Key 名称常量 ──────────────────────────────────────────────────────────
FRED_API_KEY = 'fred_api_key'
OPENAI_API_KEY = 'openai_api_key'
OPENAI_BASE_URL = 'openai_base_url'
OPENAI_MODEL = 'openai_model'
NEWS_API_KEY = 'news_api_key'
BANXICO_API_KEY = 'banxico_api_key'

# 官方汇率接口 URL（可选覆盖，默认走常量）
KEY_CFETS_REALTIME_URL = 'cfets_realtime_url'
KEY_HKMA_SPOT_URL = 'hkma_spot_rates_url'
KEY_HTTP_PROXY = 'http_proxy'
KEY_HTTP_TIMEOUT_SEC = 'http_timeout_sec'

_ALL_ENCRYPTED_KEY_NAMES = (
    FRED_API_KEY,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    NEWS_API_KEY,
    BANXICO_API_KEY,
)


def build_system_settings_snapshot() -> dict:
    """组装可导出的明文 JSON（不含管理员密码哈希）。"""
    s = get_settings()
    try:
        to = int(s.get('http_timeout_sec', 25))
        to = max(5, min(to, 120))
    except (TypeError, ValueError):
        to = 25
    api_plain = {}
    for name in _ALL_ENCRYPTED_KEY_NAMES:
        api_plain[name] = get_api_key(name) or ''
    return {
        '_meta': {
            'version': 1,
            'app': 'forecast-system-v6',
            'exported_at': datetime.now().isoformat(),
        },
        'network': {
            'cfets_realtime_url': (s.get('cfets_realtime_url', '') or ''),
            'hkma_spot_rates_url': (s.get('hkma_spot_rates_url', '') or ''),
            'http_proxy': (s.get('http_proxy', '') or ''),
            'http_timeout_sec': to,
        },
        'api_keys': api_plain,
    }


def write_system_settings_json_file() -> None:
    """将当前有效配置写入 data/system_settings.json"""
    write_json(SYSTEM_SETTINGS_JSON, build_system_settings_snapshot())


def import_system_settings_snapshot(snapshot: dict) -> None:
    """
    从明文快照写回加密 settings（不修改管理员密码）。
    snapshot 可来自 system_settings.json 或导出接口。
    """
    network = snapshot.get('network') or {}
    if isinstance(network, dict):
        for k in ('cfets_realtime_url', 'hkma_spot_rates_url', 'http_proxy'):
            if k in network and isinstance(network[k], str):
                save_general_setting(k, network[k].strip())
        if 'http_timeout_sec' in network and network['http_timeout_sec'] is not None and network['http_timeout_sec'] != '':
            try:
                save_general_setting(
                    'http_timeout_sec',
                    max(5, min(int(network['http_timeout_sec']), 120)),
                )
            except (TypeError, ValueError):
                pass

    keys = snapshot.get('api_keys') or {}
    if isinstance(keys, dict):
        for name in _ALL_ENCRYPTED_KEY_NAMES:
            if name not in keys:
                continue
            val = keys[name]
            if val is None:
                continue
            if isinstance(val, str) and val.strip():
                v = val.strip()
                if name == OPENAI_BASE_URL:
                    v = normalize_openai_base_url(v)
                set_api_key(name, v)

    write_system_settings_json_file()


def try_bootstrap_from_system_settings_json() -> bool:
    snap = read_json(SYSTEM_SETTINGS_JSON, default=None)
    if not snap or not isinstance(snap, dict):
        return False
    import_system_settings_snapshot(snap)
    return True
