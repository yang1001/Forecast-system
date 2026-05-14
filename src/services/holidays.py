"""
中国法定节假日日历（2024-2027）
数据来源：国务院办公厅关于节假日放假调休的通知
"""
from datetime import date, timedelta

# 法定节假日（放假日期）
HOLIDAYS = {
    # 2024
    date(2024, 1, 1),   # 元旦
    date(2024, 2, 10), date(2024, 2, 11), date(2024, 2, 12),
    date(2024, 2, 13), date(2024, 2, 14), date(2024, 2, 15),
    date(2024, 2, 16), date(2024, 2, 17),  # 春节
    date(2024, 4, 4), date(2024, 4, 5), date(2024, 4, 6),  # 清明
    date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 3),
    date(2024, 5, 4), date(2024, 5, 5),  # 劳动节
    date(2024, 6, 10),  # 端午
    date(2024, 9, 16), date(2024, 9, 17),  # 中秋
    date(2024, 10, 1), date(2024, 10, 2), date(2024, 10, 3),
    date(2024, 10, 4), date(2024, 10, 7),  # 国庆

    # 2025
    date(2025, 1, 1),  # 元旦
    date(2025, 1, 28), date(2025, 1, 29), date(2025, 1, 30),
    date(2025, 1, 31), date(2025, 2, 1), date(2025, 2, 2),
    date(2025, 2, 3), date(2025, 2, 4),  # 春节
    date(2025, 4, 4), date(2025, 4, 5), date(2025, 4, 6),  # 清明
    date(2025, 5, 1), date(2025, 5, 2), date(2025, 5, 3),
    date(2025, 5, 4), date(2025, 5, 5),  # 劳动节
    date(2025, 5, 31), date(2025, 6, 1), date(2025, 6, 2),  # 端午
    date(2025, 10, 1), date(2025, 10, 2), date(2025, 10, 3),
    date(2025, 10, 4), date(2025, 10, 5), date(2025, 10, 6),
    date(2025, 10, 7), date(2025, 10, 8),  # 国庆+中秋

    # 2026
    date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3),  # 元旦
    date(2026, 2, 17), date(2026, 2, 18), date(2026, 2, 19),
    date(2026, 2, 20), date(2026, 2, 21), date(2026, 2, 22),
    date(2026, 2, 23),  # 春节
    date(2026, 4, 5), date(2026, 4, 6), date(2026, 4, 7),  # 清明
    date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3),
    date(2026, 5, 4), date(2026, 5, 5),  # 劳动节
    date(2026, 6, 19), date(2026, 6, 20), date(2026, 6, 21),  # 端午
    date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 3),
    date(2026, 10, 4), date(2026, 10, 5), date(2026, 10, 6),
    date(2026, 10, 7), date(2026, 10, 8),  # 国庆

    # 2027
    date(2027, 1, 1),  # 元旦
    date(2027, 2, 6), date(2027, 2, 7), date(2027, 2, 8),
    date(2027, 2, 9), date(2027, 2, 10), date(2027, 2, 11),
    date(2027, 2, 12),  # 春节
}

# 补班日期（周末变工作日）
WORKDAY_OVERRIDES = {
    # 2025
    date(2025, 1, 26),  # 补春节
    date(2025, 2, 8),   # 补春节
    date(2025, 4, 27),  # 补劳动节
    date(2025, 9, 28),  # 补国庆
    date(2025, 10, 11), # 补国庆

    # 2026
    date(2026, 2, 15),  # 补春节
    date(2026, 2, 28),  # 补春节
    date(2026, 4, 26),  # 补劳动节
    date(2026, 5, 9),   # 补劳动节
    date(2026, 6, 28),  # 补端午
    date(2026, 9, 27),  # 补国庆
    date(2026, 10, 10), # 补国庆
}


def is_trading_day(d: date) -> bool:
    """判断指定日期是否为交易日（A股）"""
    if isinstance(d, str):
        from datetime import datetime
        d = datetime.strptime(d, '%Y-%m-%d').date()

    # 法定节假日
    if d in HOLIDAYS:
        return False

    # 补班日（周末变工作日）
    if d in WORKDAY_OVERRIDES:
        return True

    # 周末
    if d.weekday() >= 5:
        return False

    return True


def next_trading_day(d: date) -> date:
    """获取下一个交易日"""
    next_d = d + timedelta(days=1)
    while not is_trading_day(next_d):
        next_d += timedelta(days=1)
    return next_d


def prev_trading_day(d: date) -> date:
    """获取上一个交易日"""
    prev_d = d - timedelta(days=1)
    while not is_trading_day(prev_d):
        prev_d -= timedelta(days=1)
    return prev_d


def get_trading_days_ago(d: date, n: int) -> date:
    """获取 n 个交易日之前的日期"""
    cur = d
    for _ in range(n):
        cur = prev_trading_day(cur)
    return cur
