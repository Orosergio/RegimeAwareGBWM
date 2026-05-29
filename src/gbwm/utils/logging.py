"""Small logging helper that uses rich when available, stdlib otherwise."""
from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str = "gbwm", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
            fmt = "%(message)s"
        except Exception:  # pragma: no cover
            handler = logging.StreamHandler()
            fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        logging.basicConfig(level=level, format=fmt, handlers=[handler])
        _CONFIGURED = True
    return logging.getLogger(name)
