"""QuantConnect API client for downloading historical options chain data."""

import hashlib
import io
import json
import logging
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests

from strategylab.config import load_settings

logger = logging.getLogger(__name__)

_QC_BASE_URL = "https://www.quantconnect.com/api/v2"

_OPTIONS_CACHE_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "output" / "data_cache" / "options"
)


class QCClient:
    """QuantConnect API client."""

    def __init__(self, user_id: str | None = None, api_token: str | None = None):
        settings = load_settings()
        qc_config = settings.get("quantconnect", {})

        self.user_id = user_id or os.environ.get(
            qc_config.get("user_id_env_var", "QC_USER_ID"), ""
        )
        self.api_token = api_token or os.environ.get(
            qc_config.get("api_token_env_var", "QC_API_TOKEN"), ""
        )
        self.organization_id = os.environ.get(
            qc_config.get("organization_id_env_var", "QC_ORGANIZATION_ID"),
            qc_config.get("organization_id", ""),
        )

        if not self.user_id or not self.api_token:
            raise ValueError(
                "QuantConnect credentials not configured. "
                "Set QC_USER_ID and QC_API_TOKEN env vars, or configure in settings.yaml."
            )

    def _auth(self) -> tuple[str, str]:
        """Return (user_id, api_hash) for QC API authentication."""
        # QC API uses basic auth: userId as username, api_token as password
        return (self.user_id, self.api_token)

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{_QC_BASE_URL}/{endpoint.lstrip('/')}"
        resp = requests.get(url, auth=self._auth(), params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", True):
            raise RuntimeError(f"QC API error: {data.get('errors', data.get('messages', 'unknown'))}")
        return data

    def _post(self, endpoint: str, payload: dict | None = None) -> dict:
        url = f"{_QC_BASE_URL}/{endpoint.lstrip('/')}"
        resp = requests.post(url, auth=self._auth(), json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", True):
            raise RuntimeError(f"QC API error: {data.get('errors', data.get('messages', 'unknown'))}")
        return data

    def authenticate(self) -> bool:
        """Verify API credentials are valid."""
        try:
            result = self._get("authenticate")
            return result.get("success", False)
        except Exception as e:
            logger.error("QC authentication failed: %s", e)
            return False

    def list_data_files(self, path_prefix: str) -> list[str]:
        """List available data files under a given path prefix."""
        result = self._post("data/list", {"directoryName": path_prefix})
        return result.get("files", result.get("data", []))

    def download_data(self, data_key: str) -> bytes:
        """Download a raw data file from QuantConnect.

        Args:
            data_key: The data path key, e.g.
                      "option/usa/daily/uvxy/20230101_trade_american.zip"
        """
        result = self._post("data/read", {
            "organizationId": self.organization_id,
            "key": data_key,
        })

        # QC returns the data as a base64-encoded string or a download URL
        if "url" in result:
            resp = requests.get(result["url"], timeout=120)
            resp.raise_for_status()
            return resp.content
        elif "data" in result:
            import base64
            return base64.b64decode(result["data"])
        else:
            raise RuntimeError(f"Unexpected data/read response: {list(result.keys())}")


def download_options_chain(
    underlying: str,
    start: str,
    end: str,
    resolution: str = "daily",
    option_type: str = "put",
    client: QCClient | None = None,
) -> pd.DataFrame:
    """Download historical options chain data from QuantConnect.

    Args:
        underlying: Ticker symbol (e.g. "UVXY").
        start: Start date YYYY-MM-DD.
        end: End date YYYY-MM-DD.
        resolution: Data resolution ("daily", "hour", "minute").
        option_type: "put", "call", or "both".
        client: QCClient instance (created from config if None).

    Returns:
        DataFrame with columns: date, expiry, strike, right, open, high, low,
        close, volume, open_interest, underlying_close, implied_volatility
    """
    if client is None:
        client = QCClient()

    cache_path = _options_cache_path(underlying, start, end, resolution, option_type)
    if cache_path.exists():
        logger.info("Loading cached options chain for %s", underlying)
        return pd.read_parquet(cache_path)

    logger.info("Downloading %s options chain from QuantConnect (%s to %s)", underlying, start, end)

    symbol = underlying.lower()
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    # List available data files for this underlying
    data_prefix = f"option/usa/{resolution}/{symbol}"
    try:
        available_files = client.list_data_files(data_prefix)
    except Exception as e:
        logger.error("Failed to list QC data files for %s: %s", underlying, e)
        return pd.DataFrame()

    if not available_files:
        logger.warning("No options data files found for %s at %s", underlying, data_prefix)
        return pd.DataFrame()

    all_records = []

    for file_key in available_files:
        # Filter by date range if filename contains date
        file_date = _extract_date_from_key(file_key)
        if file_date is not None:
            if file_date < start_dt or file_date > end_dt:
                continue

        # Filter by option type
        if option_type != "both":
            # QC files may be named like "20230101_trade_american.zip"
            # or split by put/call - handle both cases
            pass

        try:
            raw_data = client.download_data(file_key)
            records = _parse_options_data(raw_data, file_key, option_type)
            all_records.extend(records)
        except Exception as e:
            logger.warning("Failed to download/parse %s: %s", file_key, e)
            continue

    if not all_records:
        logger.warning("No options records parsed for %s", underlying)
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "expiry", "strike"]).reset_index(drop=True)

    # Cache the result
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info("Cached %d options records for %s", len(df), underlying)

    return df


def _parse_options_data(
    raw_data: bytes, file_key: str, option_type: str
) -> list[dict]:
    """Parse raw QC options data (typically a zip of CSVs)."""
    records = []

    try:
        if file_key.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw_data)) as zf:
                for name in zf.namelist():
                    if name.endswith(".csv"):
                        with zf.open(name) as f:
                            csv_records = _parse_options_csv(
                                f.read().decode("utf-8"), name, option_type
                            )
                            records.extend(csv_records)
        else:
            csv_records = _parse_options_csv(
                raw_data.decode("utf-8"), file_key, option_type
            )
            records.extend(csv_records)
    except Exception as e:
        logger.warning("Parse error for %s: %s", file_key, e)

    return records


def _parse_options_csv(
    csv_text: str, filename: str, option_type: str
) -> list[dict]:
    """Parse a single QC options CSV file.

    QC daily options format (no header):
    timestamp(ms), open*10000, high*10000, low*10000, close*10000, volume, open_interest

    The contract details (expiry, strike, right) are encoded in the filename:
    e.g., "uvxy_20231215_put_15.00.csv"
    """
    records = []

    # Extract contract info from filename
    contract_info = _parse_contract_from_filename(filename)
    if contract_info is None:
        return records

    right = contract_info.get("right", "")
    if option_type != "both" and right != option_type:
        return records

    for line in csv_text.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split(",")
        if len(parts) < 5:
            continue

        try:
            # QC uses millisecond timestamps or date strings
            timestamp = parts[0].strip()
            if timestamp.isdigit() and len(timestamp) > 8:
                date = pd.Timestamp(int(timestamp), unit="ms")
            else:
                date = pd.Timestamp(timestamp)

            # QC prices are stored as price * 10000
            scale = 10000.0
            records.append({
                "date": date,
                "expiry": contract_info.get("expiry"),
                "strike": contract_info.get("strike"),
                "right": right,
                "open": float(parts[1]) / scale,
                "high": float(parts[2]) / scale,
                "low": float(parts[3]) / scale,
                "close": float(parts[4]) / scale,
                "volume": int(parts[5]) if len(parts) > 5 else 0,
                "open_interest": int(parts[6]) if len(parts) > 6 else 0,
            })
        except (ValueError, IndexError) as e:
            continue

    return records


def _parse_contract_from_filename(filename: str) -> dict | None:
    """Extract contract details from QC-style filename.

    Handles formats like:
      - uvxy_20231215_put_15.00.csv
      - UVXY231215P00015000.csv (OCC-style)
      - 20231215/uvxy_put_15.csv
    """
    name = Path(filename).stem.lower()
    parts = name.split("_")

    # Try format: underlying_YYYYMMDD_right_strike
    if len(parts) >= 4:
        try:
            expiry = pd.Timestamp(parts[1])
            right = parts[2] if parts[2] in ("put", "call") else None
            strike = float(parts[3])
            if right:
                return {"expiry": expiry, "right": right, "strike": strike}
        except (ValueError, IndexError):
            pass

    # Try format: underlying_right_strike_YYYYMMDD
    if len(parts) >= 4:
        try:
            right = parts[1] if parts[1] in ("put", "call") else None
            strike = float(parts[2])
            expiry = pd.Timestamp(parts[3])
            if right:
                return {"expiry": expiry, "right": right, "strike": strike}
        except (ValueError, IndexError):
            pass

    # Try OCC-style: UVXY231215P00015000
    import re
    occ_match = re.match(r"[a-z]+(\d{6})([pc])(\d{8})", name)
    if occ_match:
        date_str, right_code, strike_str = occ_match.groups()
        expiry = pd.Timestamp("20" + date_str) if len(date_str) == 6 else pd.Timestamp(date_str)
        right = "put" if right_code == "p" else "call"
        strike = int(strike_str) / 1000.0
        return {"expiry": expiry, "right": right, "strike": strike}

    return None


def _extract_date_from_key(file_key: str) -> pd.Timestamp | None:
    """Try to extract a date from a QC data file path."""
    import re
    match = re.search(r"(\d{8})", file_key)
    if match:
        try:
            return pd.Timestamp(match.group(1))
        except ValueError:
            pass
    return None


def _options_cache_path(
    underlying: str, start: str, end: str, resolution: str, option_type: str
) -> Path:
    """Generate a deterministic cache path for an options query."""
    key = f"{underlying}_{start}_{end}_{resolution}_{option_type}"
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    return _OPTIONS_CACHE_DIR / f"{underlying.upper()}_{digest}.parquet"


def import_options_csv(
    csv_path: str | Path,
    underlying: str,
    date_col: str = "date",
    expiry_col: str = "expiry",
    strike_col: str = "strike",
    right_col: str = "right",
    close_col: str = "close",
    volume_col: str = "volume",
    oi_col: str = "open_interest",
    iv_col: str | None = "implied_volatility",
) -> pd.DataFrame:
    """Import options chain data from a user-provided CSV file.

    This supports exporting data from QC Research notebooks or any other source
    that produces a CSV with the standard columns.

    Args:
        csv_path: Path to the CSV file.
        underlying: Ticker symbol for cache key.
        *_col: Column name mappings for non-standard CSVs.

    Returns:
        Normalized DataFrame cached for future use.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Options CSV not found: {csv_path}")

    logger.info("Importing options data from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Normalize column names
    rename_map = {}
    for target, source in [
        ("date", date_col),
        ("expiry", expiry_col),
        ("strike", strike_col),
        ("right", right_col),
        ("close", close_col),
        ("volume", volume_col),
        ("open_interest", oi_col),
    ]:
        if source in df.columns and source != target:
            rename_map[source] = target

    if iv_col and iv_col in df.columns and iv_col != "implied_volatility":
        rename_map[iv_col] = "implied_volatility"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Normalize types
    df["date"] = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["strike"] = df["strike"].astype(float)
    df["right"] = df["right"].str.lower()
    df["close"] = df["close"].astype(float)

    # Cache
    cache_path = _OPTIONS_CACHE_DIR / f"{underlying.upper()}_imported.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info("Imported and cached %d records for %s", len(df), underlying)

    return df


def load_cached_options(underlying: str) -> pd.DataFrame | None:
    """Load any cached options data for an underlying (API or imported)."""
    if not _OPTIONS_CACHE_DIR.exists():
        return None

    # Check imported first (user-provided data is preferred)
    imported = _OPTIONS_CACHE_DIR / f"{underlying.upper()}_imported.parquet"
    if imported.exists():
        return pd.read_parquet(imported)

    # Check API-downloaded caches
    pattern = f"{underlying.upper()}_*.parquet"
    matches = list(_OPTIONS_CACHE_DIR.glob(pattern))
    if matches:
        # Return the most recently modified
        latest = max(matches, key=lambda p: p.stat().st_mtime)
        return pd.read_parquet(latest)

    return None
