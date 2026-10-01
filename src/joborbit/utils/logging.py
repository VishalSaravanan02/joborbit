"""Logging setup shared by every JobOrbit process."""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler

from joborbit.settings import PROJECT_ROOT, get_settings

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(process_name: str, level: int = logging.INFO) -> None:
    """Send log messages to the terminal and to var/logs/<process_name>.log.

    The file starts fresh at midnight, and the last 14 days are kept.
    """
    log_dir = PROJECT_ROOT / "var" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = TimedRotatingFileHandler(
        log_dir / f"{process_name}.log",
        when="midnight",
        backupCount=14,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Only show warnings from these libraries. Besides cutting noise, this is
    # a security measure: httpx logs every URL it requests, and Telegram's
    # URLs contain the bot token.
    for library in ("httpx", "httpcore", "telegram", "apscheduler"):
        logging.getLogger(library).setLevel(logging.WARNING)

    settings = get_settings()
    logging.getLogger(__name__).info(
        "Logging started for %s (env=%s, dry_run=%s)",
        process_name,
        settings.env,
        settings.dry_run,
    )
