"""Input validation utilities for CLI and data processing."""

import os
from pathlib import Path
from typing import Optional


class ValidationError(ValueError):
    """Custom validation error with detailed message."""
    pass


def validate_file_path(
    path: str | Path,
    max_size_mb: int = 500,
    must_exist: bool = True,
    allowed_dirs: Optional[list[str]] = None,
) -> Path:
    """Validate file path for security and size."""
    p = Path(path).resolve()

    # Check if file exists
    if must_exist and not p.exists():
        raise ValidationError(f"File not found: {path}")

    # Check file size
    if p.exists() and p.is_file():
        size_bytes = p.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValidationError(
                f"File too large: {size_mb:.2f}MB (max {max_size_mb}MB)"
            )

    # Check path is within allowed directories
    if allowed_dirs:
        allowed = [Path(d).resolve() for d in allowed_dirs]
        if not any(p.is_relative_to(a) for a in allowed):
            raise ValidationError(
                f"Path not allowed: {path}. Must be in: {', '.join(allowed_dirs)}"
            )

    return p


def validate_directory_path(path: str | Path, create: bool = False) -> Path:
    """Validate directory path."""
    p = Path(path).resolve()

    if not p.exists():
        if create:
            p.mkdir(parents=True, exist_ok=True)
        else:
            raise ValidationError(f"Directory not found: {path}")
    elif not p.is_dir():
        raise ValidationError(f"Path is not a directory: {path}")

    return p


def validate_region(region: str, allowed: Optional[list[str]] = None) -> str:
    """Validate region parameter against whitelist."""
    if allowed is None:
        allowed = ["us", "cn", "eu", "ap"]

    if region.lower() not in allowed:
        raise ValidationError(
            f"Invalid region '{region}'. Allowed: {', '.join(allowed)}"
        )

    return region.lower()


def validate_timeframe(timeframe: str) -> str:
    """Validate timeframe format (e.g., 1m, 5m, 1h, 1d)."""
    import re

    pattern = r"^\d+[mhdw]$"
    if not re.match(pattern, timeframe.lower()):
        raise ValidationError(
            f"Invalid timeframe '{timeframe}'. Expected format: 1m, 5m, 1h, 1d, etc."
        )

    return timeframe.lower()


def validate_ticker(ticker: str) -> str:
    """Validate stock ticker symbol."""
    import re

    ticker = ticker.upper().strip()
    if not re.match(r"^[A-Z]{1,5}$", ticker):
        raise ValidationError(f"Invalid ticker '{ticker}'. Must be 1-5 uppercase letters.")

    return ticker


def validate_tickers(tickers_str: str) -> list[str]:
    """Validate comma-separated ticker list."""
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    for ticker in tickers:
        try:
            validate_ticker(ticker)
        except ValidationError:
            raise ValidationError(f"Invalid ticker in list: {ticker}")

    if not tickers:
        raise ValidationError("No valid tickers provided")

    return tickers


def validate_channel_id(channel_id: str) -> int:
    """Validate Discord channel ID."""
    try:
        cid = int(channel_id)
        if cid < 0:
            raise ValueError()
        return cid
    except (ValueError, TypeError):
        raise ValidationError(f"Invalid channel ID '{channel_id}'. Must be a positive integer.")


def validate_channel_ids(channels_str: str) -> list[int]:
    """Validate comma-separated channel ID list."""
    channel_ids = []

    for part in channels_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            channel_ids.append(validate_channel_id(part))
        except ValidationError as e:
            raise ValidationError(f"Invalid channel in list: {e}")

    if not channel_ids:
        raise ValidationError("No valid channel IDs provided")

    return channel_ids


def validate_max_rows(max_rows: Optional[int]) -> Optional[int]:
    """Validate max_rows parameter."""
    if max_rows is None:
        return None

    if max_rows < 1:
        raise ValidationError("max_rows must be positive")

    if max_rows > 1000000:
        raise ValidationError("max_rows exceeds maximum (1,000,000)")

    return max_rows


def safe_env_for_subprocess(additional_vars: Optional[dict] = None) -> dict:
    """Get safe environment variables for subprocess execution."""
    safe_keys = ["PATH", "HOME", "TEMP", "TMP", "LANG", "LC_ALL"]

    env = {k: os.environ[k] for k in safe_keys if k in os.environ}

    if additional_vars:
        # Only add explicitly allowed vars
        for key, value in additional_vars.items():
            if value is not None:
                env[key] = str(value)

    return env
