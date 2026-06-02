#!/usr/bin/env python3
"""
StockTwits Indian Market Sentiment Scanner
==========================================
Dynamically fetches trending StockTwits symbols, screens each one for NSE/BSE
listing via Yahoo Finance (no hardcoded stock list), analyses bullish and bearish
message streams, scores with a weighted composite, then prints two ranked tables
and discards all data — nothing is written to disk.

Composite score  :  Frequency 40%  |  Recency 35%  |  Engagement 25%
Recency model    :  exponential decay,  half-life = 6 h
Normalisation    :  min-max across all records before weighting
Rate limit       :  hard cap 60 req/min (sliding window)
                    auto-pause when X-Ratelimit-Remaining ≤ 10
                    one retry on HTTP 429
"""

import math
import re
import sys
import time
import datetime
from collections import deque

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import requests
except ImportError:
    sys.exit("[ERROR] Missing: pip install requests")

try:
    import yfinance as yf
except ImportError:
    sys.exit("[ERROR] Missing: pip install yfinance")

# ── Configuration ─────────────────────────────────────────────────────────────
RATE_LIMIT_RPM       = 60          # hard cap: requests per minute
LOW_LIMIT_THRESHOLD  = 10          # auto-pause when X-Ratelimit-Remaining ≤ this
MAX_429_RETRIES      = 1           # single retry on HTTP 429

WEIGHT_FREQUENCY     = 0.40
WEIGHT_RECENCY       = 0.35
WEIGHT_ENGAGEMENT    = 0.25

RECENCY_HALF_LIFE_H  = 6.0         # exponential-decay half-life in hours
STREAM_MSG_LIMIT     = 30          # messages fetched per symbol

STOCKTWITS_BASE      = "https://api.stocktwits.com/api/2"
INDIAN_EXCHANGES     = {"NSI", "NSE", "BSE", "BOM"}   # yfinance exchange codes

# ── Rate-limit state ──────────────────────────────────────────────────────────
_req_window: deque = deque()   # monotonic timestamps of calls in the last 60 s


def _throttle() -> None:
    """Sliding-window enforcer: blocks until fewer than RATE_LIMIT_RPM calls
    have been made within the last 60 seconds."""
    now = time.monotonic()
    while _req_window and now - _req_window[0] >= 60.0:
        _req_window.popleft()
    if len(_req_window) >= RATE_LIMIT_RPM:
        sleep_s = 60.0 - (now - _req_window[0]) + 0.05
        if sleep_s > 0:
            time.sleep(sleep_s)
    _req_window.append(time.monotonic())


def _get(url: str, params: dict | None = None,
         _retries: int = MAX_429_RETRIES) -> dict | None:
    """Rate-limited GET with X-Ratelimit-Remaining monitoring and 429 retry."""
    _throttle()
    try:
        resp = requests.get(url, params=params, timeout=15)
    except requests.RequestException as exc:
        print(f"  [WARN] Request error: {exc}")
        return None

    # ── X-Ratelimit-Remaining monitoring ──────────────────────────────────────
    remaining_hdr = resp.headers.get("X-Ratelimit-Remaining")
    if remaining_hdr is not None:
        try:
            remaining = int(remaining_hdr)
            if remaining <= LOW_LIMIT_THRESHOLD:
                print(
                    f"  [PAUSE] X-Ratelimit-Remaining={remaining} "
                    f"(≤ {LOW_LIMIT_THRESHOLD}) — waiting 15 s"
                )
                time.sleep(15)
        except ValueError:
            pass

    # ── 429 handling — one retry ──────────────────────────────────────────────
    if resp.status_code == 429:
        if _retries > 0:
            retry_after = int(resp.headers.get("Retry-After", 20))
            print(
                f"  [429] Rate limited — retrying in {retry_after} s "
                f"(retries remaining: {_retries - 1})"
            )
            time.sleep(retry_after)
            return _get(url, params, _retries - 1)
        print("  [429] Rate limited — retry exhausted, skipping this request")
        return None

    if resp.status_code != 200:
        return None

    return resp.json()


# ── Indian exchange validation — fully dynamic, no hardcoded stock list ───────
_exchange_cache: dict[str, bool] = {}


def _is_indian(symbol: str) -> bool:
    """
    Checks whether `symbol` is traded on NSE or BSE by querying Yahoo Finance
    with both the .NS (NSE) and .BO (BSE) suffixes.  No static ticker list is
    used at any point — every lookup is performed live against Yahoo Finance.

    Returns True if the exchange code or currency confirms an Indian listing.
    """
    if symbol in _exchange_cache:
        return _exchange_cache[symbol]

    for suffix in (".NS", ".BO"):
        try:
            fast = yf.Ticker(symbol + suffix).fast_info
            exch = getattr(fast, "exchange", None)
            if exch and exch.upper() in INDIAN_EXCHANGES:
                _exchange_cache[symbol] = True
                return True
            currency = getattr(fast, "currency", None)
            if currency and currency.upper() == "INR":
                _exchange_cache[symbol] = True
                return True
        except Exception:
            continue

    _exchange_cache[symbol] = False
    return False


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _recency_decay(created_at: str) -> float:
    """
    Exponential decay score in [0, 1]:
        score = 2^(−hours_elapsed / half_life)
    A message posted right now scores 1.0; one posted half_life hours ago
    scores 0.5; the score approaches 0 for very old messages.
    """
    try:
        dt = datetime.datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        dt = dt.replace(tzinfo=datetime.timezone.utc)
        hours_ago = (
            datetime.datetime.now(datetime.timezone.utc) - dt
        ).total_seconds() / 3600.0
        return math.pow(2.0, -hours_ago / RECENCY_HALF_LIFE_H)
    except Exception:
        return 0.0


def _engagement(msg: dict) -> float:
    """Total engagement = likes + reshares (both default to 0 when absent)."""
    likes    = ((msg.get("likes")    or {}).get("total", 0)          or 0)
    reshares = ((msg.get("reshares") or {}).get("reshared_count", 0) or 0)
    return float(likes + reshares)


def _minmax(values: list[float]) -> list[float]:
    """
    Min-max normalise a list to [0, 1].
    If all values are identical, maps every element to 0.5.
    """
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0.0:
        return [0.5] * len(values)
    return [(v - lo) / span for v in values]


# ── StockTwits API helpers ────────────────────────────────────────────────────

def _fetch_trending_symbols() -> list[str]:
    """Return all trending ticker symbols from StockTwits."""
    data = _get(f"{STOCKTWITS_BASE}/trending/symbols.json")
    if not data:
        return []
    return [s["symbol"] for s in data.get("symbols", []) if "symbol" in s]


def _fetch_stream(symbol: str) -> list[dict]:
    """Fetch the most recent messages for a single symbol."""
    data = _get(
        f"{STOCKTWITS_BASE}/streams/symbol/{symbol}.json",
        {"limit": STREAM_MSG_LIMIT},
    )
    return data.get("messages", []) if data else []


def _sentiment(msg: dict) -> str | None:
    """Return 'Bullish', 'Bearish', or None if the message is untagged."""
    try:
        return msg["entities"]["sentiment"]["basic"]
    except (KeyError, TypeError):
        return None


def _clean_body(text: str) -> str:
    """Strip $TICKER cashtags and collapse whitespace for a readable narrative."""
    text = re.sub(r"\$\w+", "", text or "")
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "(no text)"


# ── Output formatting ─────────────────────────────────────────────────────────
_LINE = "=" * 84
_SEP  = (
    "  ────  ────────────  ─────────────  ──────────────  "
    "─────────────────  ────────────────────────────────────────────────────────"
)
_HDR  = (
    f"  {'Rank':<4}  {'Ticker':<12}  {'Mention Count':<13}  "
    f"{'Recency Rating':<14}  {'Engagement Rating':<17}  Key Narrative"
)


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{_LINE}")
    print(f"  {title}")
    print(_LINE)
    print(_SEP)
    print(_HDR)
    print(_SEP)
    if not rows:
        print("  (No data — no Indian tickers were detected in this scan)")
    else:
        for rank, r in enumerate(rows, start=1):
            narrative = r["narrative"]
            if len(narrative) > 52:
                narrative = narrative[:49] + "…"
            print(
                f"  {rank:<4}  {r['symbol']:<12}  {r['freq']:<13}  "
                f"{r['rec_raw']:.4f}          "
                f"  {r['eng_raw']:<17.0f}  {narrative}"
            )
    print(_SEP)


def _empty_tables() -> None:
    for title in [
        "📈  Bullish Opportunities",
        "📉  Bearish / Short Candidates",
    ]:
        _print_table(title, [])


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run() -> None:
    print()
    print(_LINE)
    print("  StockTwits — Indian Market Sentiment Scanner")
    print("  Exchange filter: NSE / BSE only  |  No data written to disk")
    print(
        f"  Weights: Frequency {WEIGHT_FREQUENCY*100:.0f}%  "
        f"Recency {WEIGHT_RECENCY*100:.0f}%  "
        f"Engagement {WEIGHT_ENGAGEMENT*100:.0f}%  "
        f"|  Recency half-life {RECENCY_HALF_LIFE_H:.0f} h  "
        f"|  Rate cap {RATE_LIMIT_RPM} req/min"
    )
    print(_LINE)

    # ── Step 1: Trending symbols ──────────────────────────────────────────────
    print("\n  [1/4] Fetching trending symbols from StockTwits…")
    symbols = _fetch_trending_symbols()

    if not symbols:
        print("  ✗  StockTwits returned no trending symbols.")
        _empty_tables()
        return

    print(f"  →  {len(symbols)} symbol(s) retrieved.")

    # ── Step 2: Screen for NSE/BSE via Yahoo Finance ──────────────────────────
    print("\n  [2/4] Screening for NSE/BSE listings via Yahoo Finance…")
    indian: list[str] = []

    for sym in symbols:
        listed = _is_indian(sym)
        tag    = "✓  NSE/BSE" if listed else "✗  not Indian"
        print(f"        {sym:<16} {tag}")
        if listed:
            indian.append(sym)

    if not indian:
        print(
            "\n  No Indian (NSE/BSE) tickers found among StockTwits trending symbols."
        )
        _empty_tables()
        return

    # ── Step 3: Fetch message streams ─────────────────────────────────────────
    print(
        f"\n  [3/4] Fetching message streams for {len(indian)} Indian stock(s)…"
    )
    raw: list[dict] = []

    for sym in indian:
        messages = _fetch_stream(sym)
        if not messages:
            continue

        for label in ("Bullish", "Bearish"):
            bucket = [m for m in messages if _sentiment(m) == label]
            if not bucket:
                continue

            freq     = len(bucket)
            rec_avg  = (
                sum(_recency_decay(m.get("created_at", "")) for m in bucket) / freq
            )
            eng_tot  = sum(_engagement(m) for m in bucket)

            # Key narrative: body of the highest-engagement message
            top_msg   = max(bucket, key=_engagement)
            narrative = _clean_body(top_msg.get("body", ""))

            raw.append({
                "symbol":    sym,
                "sentiment": label,
                "freq":      freq,
                "rec_raw":   rec_avg,   # mean exponential-decay score ∈ [0, 1]
                "eng_raw":   eng_tot,   # total likes + reshares
                "narrative": narrative,
            })

    if not raw:
        print(
            "  No sentiment-tagged (Bullish/Bearish) messages found "
            "for any Indian stock."
        )
        _empty_tables()
        return

    # ── Step 4: Score and rank ────────────────────────────────────────────────
    print(f"\n  [4/4] Scoring {len(raw)} record(s) and ranking…")

    # Min-max normalise each dimension across ALL records jointly
    freq_n = _minmax([r["freq"]    for r in raw])
    rec_n  = _minmax([r["rec_raw"] for r in raw])
    eng_n  = _minmax([r["eng_raw"] for r in raw])

    for i, r in enumerate(raw):
        r["score"] = (
            WEIGHT_FREQUENCY  * freq_n[i]
            + WEIGHT_RECENCY    * rec_n[i]
            + WEIGHT_ENGAGEMENT * eng_n[i]
        )

    bullish = sorted(
        [r for r in raw if r["sentiment"] == "Bullish"],
        key=lambda x: x["score"],
        reverse=True,
    )
    bearish = sorted(
        [r for r in raw if r["sentiment"] == "Bearish"],
        key=lambda x: x["score"],
        reverse=True,
    )

    _print_table("📈  Bullish Opportunities",      bullish)
    _print_table("📉  Bearish / Short Candidates", bearish)

    print(
        f"\n  Scan complete — {len(raw)} record(s) scored, nothing saved to disk.\n"
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
