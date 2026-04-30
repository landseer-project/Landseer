"""Shared Sentry initialization for backend and worker processes."""

from __future__ import annotations

import os
from typing import Optional

from .pylogger import get_logger

logger = get_logger(__name__)


def init_sentry(service_name: str) -> bool:
    """
    Initialize Sentry from environment variables.

    Required:
    - SENTRY_DSN

    Optional:
    - SENTRY_ENVIRONMENT (default: development)
    - SENTRY_RELEASE
    - SENTRY_TRACES_SAMPLE_RATE (default: 0.0)
    - SENTRY_PROFILES_SAMPLE_RATE (default: 0.0)
    """
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        return False

    try:
        import sentry_sdk
    except ImportError:
        logger.warning("SENTRY_DSN is set but sentry-sdk is not installed.")
        return False

    environment = os.getenv("SENTRY_ENVIRONMENT", "development")
    release = os.getenv("SENTRY_RELEASE")
    traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0"))
    profiles_sample_rate = float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.0"))

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=release,
        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=profiles_sample_rate,
    )
    sentry_sdk.set_tag("service", service_name)
    logger.info("Sentry initialized for %s (environment=%s)", service_name, environment)
    return True
