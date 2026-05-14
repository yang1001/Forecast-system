"""
定时调度服务
每日三次自动预测：09:30 / 11:30 / 15:00（A股交易时段）
周末及法定节假日自动跳过
"""
import logging
from datetime import datetime, date

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from .holidays import is_trading_day

logger = logging.getLogger(__name__)

CST = pytz.timezone('Asia/Shanghai')
_scheduler: BackgroundScheduler | None = None
_prediction_callback = None  # 由 app.py 注入


def _should_run_today() -> bool:
    """判断今天是否应该执行预测"""
    today = datetime.now(CST).date()
    return is_trading_day(today)


def _run_scheduled_prediction(session: str):
    """执行一次定时预测"""
    if not _should_run_today():
        logger.info(f"[{session}] 今日为节假日/周末，跳过自动预测")
        return

    now = datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"[{session}] 定时预测触发 @ {now}")

    if _prediction_callback is not None:
        try:
            _prediction_callback(session=session, triggered_by='scheduler')
        except Exception as e:
            logger.error(f"[{session}] 定时预测执行失败: {e}")
    else:
        logger.warning("[scheduler] 预测回调未注册")


def _run_daily_append():
    """每天最后一次刷新后，将当日实时汇率存入历史文件"""
    if not _should_run_today():
        logger.info("[append] 今日为节假日/周末，跳过保存汇率")
        return
    logger.info("[append] 开始将今日汇率追加到历史 CSV 文件...")
    try:
        from .storage import append_daily_rates_to_csv
        append_daily_rates_to_csv()
    except Exception as e:
        logger.error(f"[append] 追加汇率失败: {e}")


def _run_weekly_finetune():
    """每周日晚上 8 点自动微调模型"""
    logger.info("[finetune] 开始每周自动微调模型...")
    try:
        import subprocess
        import os
        import sys
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        script_path = os.path.join(project_dir, 'src', 'train_model_v4.py')
        # 运行训练脚本
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.join(project_dir, 'src')
        result = subprocess.run(
            [sys.executable, script_path], 
            cwd=project_dir, 
            capture_output=True, 
            text=True, 
            env=env
        )
        if result.returncode == 0:
            logger.info("[finetune] 模型微调成功完成")
            # 重新加载模型
            from .predictor import load_ml_models
            load_ml_models()
            logger.info("[finetune] 已重新加载微调后的模型")
        else:
            logger.error(f"[finetune] 模型微调失败，返回码: {result.returncode}\n{result.stderr}")
    except Exception as e:
        logger.error(f"[finetune] 执行微调脚本失败: {e}")


def init_scheduler(prediction_callback):
    """
    初始化并启动调度器
    prediction_callback: callable(session, triggered_by) → None
    """
    global _scheduler, _prediction_callback
    _prediction_callback = prediction_callback

    if _scheduler is not None and _scheduler.running:
        logger.info("调度器已在运行，跳过重新初始化")
        return _scheduler

    _scheduler = BackgroundScheduler(timezone=CST)

    # 09:30 开盘预测
    _scheduler.add_job(
        func=lambda: _run_scheduled_prediction('09:30_open'),
        trigger=CronTrigger(hour=9, minute=30, timezone=CST),
        id='predict_open',
        name='开盘预测',
        replace_existing=True,
        misfire_grace_time=300,
    )

    # 11:30 午休预测
    _scheduler.add_job(
        func=lambda: _run_scheduled_prediction('11:30_noon'),
        trigger=CronTrigger(hour=11, minute=30, timezone=CST),
        id='predict_noon',
        name='午休预测',
        replace_existing=True,
        misfire_grace_time=300,
    )

    # 15:00 收盘预测
    _scheduler.add_job(
        func=lambda: _run_scheduled_prediction('15:00_close'),
        trigger=CronTrigger(hour=15, minute=0, timezone=CST),
        id='predict_close',
        name='收盘预测',
        replace_existing=True,
        misfire_grace_time=300,
    )

    # 15:05 保存今日汇率到 CSV
    _scheduler.add_job(
        func=_run_daily_append,
        trigger=CronTrigger(hour=15, minute=5, timezone=CST),
        id='append_daily_rates',
        name='保存今日汇率',
        replace_existing=True,
        misfire_grace_time=300,
    )

    # 周日 20:00 微调模型
    _scheduler.add_job(
        func=_run_weekly_finetune,
        trigger=CronTrigger(day_of_week='sun', hour=20, minute=0, timezone=CST),
        id='finetune_model',
        name='每周模型微调',
        replace_existing=True,
        misfire_grace_time=3600,
    )

    _scheduler.start()
    logger.info("调度器启动成功，任务：09:30 / 11:30 / 15:00 预测，15:05 保存汇率，周日 20:00 微调模型")
    return _scheduler


def stop_scheduler():
    """停止调度器"""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("调度器已停止")


def get_scheduler_status() -> dict:
    """获取调度器状态"""
    if _scheduler is None:
        return {'running': False, 'jobs': []}

    jobs = []
    for job in _scheduler.get_jobs():
        next_run = job.next_run_time
        jobs.append({
            'id': job.id,
            'name': job.name,
            'next_run': next_run.strftime('%Y-%m-%d %H:%M:%S %Z') if next_run else None,
        })

    today = datetime.now(CST).date()
    return {
        'running': _scheduler.running,
        'is_trading_day': is_trading_day(today),
        'jobs': jobs,
        'timezone': 'Asia/Shanghai',
    }


def trigger_now(session: str = 'manual') -> bool:
    """立即手动触发一次预测（绕过节假日检查）"""
    if _prediction_callback is None:
        logger.warning("预测回调未注册，无法手动触发")
        return False
    try:
        _prediction_callback(session=session, triggered_by='manual')
        return True
    except Exception as e:
        logger.error(f"手动预测失败: {e}")
        return False
