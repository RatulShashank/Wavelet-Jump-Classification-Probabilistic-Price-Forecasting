import structlog
import logging
import sys

def setup_logging():
    """
    Configures structlog for structured JSON or pretty-printed logs.
    Section 14 & 18 — Logging logic.
    """
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Console output configuration
    if sys.stdout.isatty():
        # Terminal: Pretty printing
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        # File/Pipe: JSON printing
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Standard logging bridge
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

if __name__ == "__main__":
    setup_logging()
    logger = structlog.get_logger("test")
    logger.info("Logging system initialized", status="OK")
