"""Application settings.

Every knob from the plan's Global Constraints lives here with its spec default,
so the whole pipeline reads one bootable config. The app must boot with no
`backend/.env` (spec §11.15); values in `.env` (keys are case-insensitive)
override the defaults below.
"""

import secrets
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env, located relative to this file so Settings boots from any cwd.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Service configuration; all defaults are the Global Constraints / §11.15 values."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- storage ---
    db_path: str = "data/app.db"
    media_dir: str = "media/"

    # --- scan lifecycle (120 s deadline; preloaded models excluded from the clock) ---
    scan_deadline_s: int = 120
    max_concurrent_scans: int = 2
    engine_timeout_s: int = 30
    purge_interval_s: int = 21600  # background snapshot purge every 6 h
    snapshot_ttl_s: int = 2_592_000  # 30 days

    # --- fetcher caps (GATE 1) ---
    fetch_timeout_s: int = 15
    body_cap_mb: int = 10  # streamed body cap, also the gzip-bomb guard limit
    max_redirects: int = 3  # re-validated at every hop
    respect_robots: bool = True

    # --- extractor caps (GATE 2) ---
    max_images: int = 20
    text_cap_bytes: int = 61_440  # ≤ 60 kB extracted text
    block_min_chars: int = 80  # blocks shorter than this are skipped
    chunk_tokens: int = 512  # blocks longer are chunked to this many model tokens
    image_min_px: int = 100  # smaller natural dimension → skipped

    # --- verdict bands (score < band_unclear_min → authentic) ---
    band_unclear_min: int = 35
    band_synthetic_min: int = 65  # ≥ 65 → synthetic

    # --- API politeness ---
    rate_window_s: int = 60
    rate_max_requests: int = 30

    # --- engines (planned winners; confirm/flip after the Task 11 bake-off) ---
    text_engine_id: str = "text-ai-roberta"
    image_engine_id: str = "image-ai-vit"

    # --- secrets ---
    # Boot-generated when unset, so the app runs with no .env. Set IP_SALT in
    # .env for a stable ip_hash across restarts.
    ip_salt: str = Field(default_factory=lambda: secrets.token_hex(16))


settings = Settings()
