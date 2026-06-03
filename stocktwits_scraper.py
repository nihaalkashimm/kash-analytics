#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stocktwits_scraper.py -- Kash Analytics Phase 1B
Pipeline: Fetch -> Analyse -> Display -> Discard
No data is written to disk at any point.

Scoring (Bullish and Bearish records normalised jointly before weighting)
  Frequency  40%  mentions in the sentiment bucket for this ticker
  Recency    35%  mean exponential-decay score across bucket messages
                  decay = exp(-lambda * t),  lambda = ln(2)/21600 s  (6h half-life)
  Engagement 25%  sum of (likes + reshares) across bucket messages

  All three raw scores are min-max normalised across EVERY record
  (both sentiments combined) before weights are applied.

Rate limiting
  Hard cap   : 60 requests inside any rolling 60-second sliding window
  Auto-pause : if X-Ratelimit-Remaining <= 10, sleep until X-Ratelimit-Reset
  429 retry  : one retry after Retry-After (or 30 s fallback)
  User-Agent : 'web:KashAnalytics:v1.0.0 (by /u/Kashimmm)' on every request

Indian stock validation (yfinance, no hardcoded list)
  Try {ticker}.NS (NSE) then {ticker}.BO (BSE)
  Confirmed if yfinance fast_info.currency == 'INR'
"""

import math
import re
import time
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict

import numpy as np
import requests
import yfinance as yf
from tabulate import tabulate


# ========================== CONSTANTS =========================================

STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"
USER_AGENT      = "web:KashAnalytics:v1.0.0 (by /u/Kashimmm)"

HALF_LIFE_S     = 6 * 3600
DECAY_LAMBDA    = math.log(2) / HALF_LIFE_S

RATE_LIMIT      = 60
WINDOW_S        = 60.0
PAUSE_THRESH    = 10

WEIGHTS = {
    "frequency":  0.40,
    "recency":    0.35,
    "engagement": 0.25,
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "this", "that", "it", "its",
    "i", "we", "you", "he", "she", "they", "my", "our", "your", "their",
    "and", "or", "but", "if", "as", "so", "not", "no", "up", "out",
    "just", "about", "stock", "shares", "market", "going", "get",
    "like", "think", "good", "great", "bad", "more", "very", "really",
    "now", "what", "when", "how", "who", "why", "which", "there",
    "than", "then", "into", "also", "still", "look", "see", "know",
    "make", "all", "any", "one", "two", "day", "week", "year", "time",
    "new", "long", "short", "buy", "sell", "hold", "price", "target",
    "hit", "run", "high", "low", "amp", "rt", "via", "https", "http",
}


# ========================== SLIDING-WINDOW RATE LIMITER =======================

class _SlidingWindowLimiter:
    """
    Enforces at most `limit` HTTP calls inside any rolling `window`-second
    window. Blocks by sleeping when the window is full.
    """

    def __init__(self, limit: int = RATE_LIMIT, window: float = WINDOW_S):
        self._limit  = limit
        self._window = window
        self._calls: deque = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > self._window:
            self._calls.popleft()
        if len(self._calls) >= self._limit:
            sleep_for = self._window - (now - self._calls[0]) + 0.05
            print("  [rate-limiter] window full -- sleeping %.1fs ..." % sleep_for)
            time.sleep(sleep_for)
            now = time.monotonic()
            while self._calls and now - self._calls[0] > self._window:
                self._calls.popleft()
        self._calls.append(time.monotonic())


_limiter = _SlidingWindowLimiter()


# ========================== NETWORK LAYER =====================================

def _get(url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
    """
    Rate-limited GET. Enforces:
      - User-Agent on every single request (no exceptions)
      - auto-pause when X-Ratelimit-Remaining <= PAUSE_THRESH
      - exactly one retry on HTTP 429
    Returns the Response, or None on permanent failure.
    """
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(2):
        _limiter.acquire()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=12)
        except requests.RequestException as exc:
            print("  [network error] %s" % exc)
            return None

        # auto-pause on low remaining quota
        remaining_hdr = resp.headers.get("X-Ratelimit-Remaining")
        if remaining_hdr is not None:
            try:
                remaining_int = int(remaining_hdr)
            except ValueError:
                remaining_int = None
            if remaining_int is not None and remaining_int <= PAUSE_THRESH:
                reset_hdr = resp.headers.get("X-Ratelimit-Reset")
                if reset_hdr:
                    wait = max(1, int(reset_hdr) - int(time.time())) + 2
                else:
                    wait = 35
                print(
                    "  [auto-pause] X-Ratelimit-Remaining=%d (<=10) "
                    "-- pausing %ds until quota resets ..." % (remaining_int, wait)
                )
                time.sleep(wait)

        # one retry on 429
        if resp.status_code == 429:
            if attempt == 0:
                retry_after = int(resp.headers.get("Retry-After", 30))
                print(
                    "  [429] Rate limited -- waiting %ds before retry ..."
                    % retry_after
                )
                time.sleep(retry_after)
                continue
            else:
                print("  [429] Retry exhausted -- skipping this request.")
                return None

        return resp

    return None


# ========================== STOCKTWITS API ====================================

def fetch_trending_tickers() -> List[str]:
    """
    Call the StockTwits trending/symbols endpoint.
    Returns raw ticker strings exactly as StockTwits provides them.
    No hardcoded ticker list is used anywhere in this file.
    """
    resp = _get(STOCKTWITS_BASE + "/trending/symbols.json")
    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp is not None else "N/A"
        print("  [error] trending endpoint returned HTTP %s" % code)
        return []
    symbols = resp.json().get("symbols", [])
    tickers = [s["symbol"] for s in symbols if "symbol" in s]
    print("  -> %d tickers returned by StockTwits trending" % len(tickers))
    return tickers


def fetch_messages(ticker: str) -> List[Dict]:
    """
    Fetch the public message stream for `ticker` from StockTwits.
    Returns a (possibly empty) list of raw message dicts.
    """
    resp = _get(STOCKTWITS_BASE + "/streams/symbol/" + ticker + ".json")
    if resp is None or resp.status_code != 200:
        return []
    return resp.json().get("messages", [])


# ========================== INDIAN STOCK VALIDATION ==========================

def validate_indian(ticker: str) -> Tuple[bool, str]:
    """
    Try yfinance with {ticker}.NS (NSE) then {ticker}.BO (BSE).
    Confirms an Indian listing if fast_info.currency == 'INR'.
    Returns (is_indian, qualified_symbol).
    No hardcoded ticker list -- every check is a live yfinance call.
    """
    for suffix in (".NS", ".BO"):
        sym = ticker + suffix
        try:
            fi = yf.Ticker(sym).fast_info
            if getattr(fi, "currency", None) == "INR":
                return True, sym
        except Exception:
            continue
    return False, ""


# ========================== MESSAGE PARSING ===================================

def _parse_message(raw: Dict) -> Dict:
    """Extract and pre-compute fields from a raw StockTwits message."""
    body      = raw.get("body", "")
    sentiment = ((raw.get("entities") or {}).get("sentiment") or {}).get("basic")

    likes    = int((raw.get("likes")    or {}).get("total",          0) or 0)
    reshares = int((raw.get("reshares") or {}).get("reshared_count", 0) or 0)
    engagement = likes + reshares

    created_raw = raw.get("created_at", "")
    try:
        dt    = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        age_s = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except (ValueError, AttributeError):
        age_s = 0.0

    recency_decay = math.exp(-DECAY_LAMBDA * age_s)

    return {
        "body":       body,
        "sentiment":  sentiment,
        "recency":    recency_decay,
        "engagement": engagement,
    }


# ========================== SCORING ENGINE ====================================

def _minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0,1]. All-equal input maps to all-zero."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _key_narrative(messages: List[Dict]) -> str:
    """Top-3 non-stopword words from message bodies, joined by ' | '."""
    words: List[str] = []
    for m in messages:
        tokens = re.findall(r"[A-Za-z]{3,}", m.get("body", ""))
        words.extend(t.lower() for t in tokens if t.lower() not in STOPWORDS)
    if not words:
        return "--"
    return " | ".join(w for w, _ in Counter(words).most_common(3))


def score_and_rank(buckets: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Build raw records for every ticker x sentiment pair, then apply
    min-max normalisation across ALL records jointly (both Bullish and
    Bearish together) before weighting -- satisfying the spec requirement
    'min-max normalisation across all records before weighting'.

    Returns (ranked_bullish, ranked_bearish), each sorted descending by
    composite score with integer ranks assigned from 1.
    """
    # 1. Build raw records for both sentiments
    all_rows: List[Dict] = []
    for b in buckets:
        for sentiment in ("Bullish", "Bearish"):
            msgs = [m for m in b["parsed"] if m["sentiment"] == sentiment]
            if not msgs:
                continue
            freq_raw = float(len(msgs))
            rec_raw  = float(sum(m["recency"]    for m in msgs) / len(msgs))
            eng_raw  = float(sum(m["engagement"] for m in msgs))
            all_rows.append({
                "ticker":        b["ticker"],
                "sentiment":     sentiment,
                "mention_count": int(freq_raw),
                "freq_raw":      freq_raw,
                "rec_raw":       rec_raw,
                "eng_raw":       eng_raw,
                "messages":      msgs,
            })

    if not all_rows:
        return [], []

    # 2. Min-max normalise ACROSS ALL records (both sentiments jointly)
    freq_norm = _minmax(np.array([r["freq_raw"] for r in all_rows], dtype=float))
    rec_norm  = _minmax(np.array([r["rec_raw"]  for r in all_rows], dtype=float))
    eng_norm  = _minmax(np.array([r["eng_raw"]  for r in all_rows], dtype=float))

    for i, row in enumerate(all_rows):
        row["recency_score"]    = float(rec_norm[i])
        row["engagement_score"] = float(eng_norm[i])
        row["composite"] = (
            WEIGHTS["frequency"]  * float(freq_norm[i])
            + WEIGHTS["recency"]  * float(rec_norm[i])
            + WEIGHTS["engagement"] * float(eng_norm[i])
        )

    # 3. Split by sentiment, sort each table independently
    bull_rows = sorted(
        [r for r in all_rows if r["sentiment"] == "Bullish"],
        key=lambda r: r["composite"],
        reverse=True,
    )
    bear_rows = sorted(
        [r for r in all_rows if r["sentiment"] == "Bearish"],
        key=lambda r: r["composite"],
        reverse=True,
    )

    for rank, row in enumerate(bull_rows, start=1):
        row["rank"] = rank
    for rank, row in enumerate(bear_rows, start=1):
        row["rank"] = rank

    return bull_rows, bear_rows


# ========================== OUTPUT FORMATTING =================================

_BAR = "=" * 74


def _print_table(ranked: List[Dict], title: str) -> None:
    print("\n" + _BAR)
    print("  " + title)
    print(_BAR)
    if not ranked:
        print("  No Indian tickers detected in this scan.\n")
        return
    headers = [
        "Rank", "Ticker", "Mention Count",
        "Recency Rating", "Engagement Rating", "Key Narrative",
    ]
    table = []
    for row in ranked:
        narrative = _key_narrative(row["messages"])
        table.append([
            row["rank"],
            row["ticker"],
            row["mention_count"],
            "%.4f" % row["recency_score"],
            "%.4f" % row["engagement_score"],
            narrative[:52],
        ])
    print(tabulate(table, headers=headers, tablefmt="rounded_outline"))
    print()


def _empty_tables() -> None:
    """Print clean empty tables with the required no-Indian-tickers message."""
    for title in ("BULLISH OPPORTUNITIES", "BEARISH / SHORT CANDIDATES"):
        print("\n" + _BAR)
        print("  " + title)
        print(_BAR)
        print("  No Indian tickers detected in this scan.\n")


# ========================== MAIN PIPELINE =====================================

def run() -> None:
    bar = "-" * 74
    print("\n" + bar)
    print("  Kash Analytics -- StockTwits Indian Market Screener")
    print("  Phase 1B  |  Fetch -> Analyse -> Display -> Discard")
    print(
        "  Weights: Frequency %.0f%%  Recency %.0f%%  Engagement %.0f%%"
        "  |  Recency half-life 6h  |  Rate cap %d req/min"
        % (
            WEIGHTS["frequency"]  * 100,
            WEIGHTS["recency"]    * 100,
            WEIGHTS["engagement"] * 100,
            RATE_LIMIT,
        )
    )
    print(bar + "\n")

    # Step 1: Trending tickers (no hardcoded list)
    print("[1/4] Fetching trending symbols from StockTwits ...")
    raw_tickers = fetch_trending_tickers()
    if not raw_tickers:
        print("  [abort] StockTwits returned no trending symbols.")
        _empty_tables()
        return

    # Step 2: Filter to Indian stocks via yfinance currency check
    print(
        "\n[2/4] Validating %d ticker(s) against NSE/BSE via yfinance ..."
        % len(raw_tickers)
    )
    indian: List[Tuple[str, str]] = []
    for raw in raw_tickers:
        ok, sym = validate_indian(raw)
        if ok:
            print("  [OK]  %-14s -> %s" % (raw, sym))
            indian.append((raw, sym))

    if not indian:
        print(
            "\n  No Indian (NSE/BSE) tickers found among current "
            "StockTwits trending symbols."
        )
        _empty_tables()
        return

    print("\n  -> %d Indian ticker(s) confirmed." % len(indian))

    # Step 3: Fetch message streams
    print("\n[3/4] Fetching message streams ...")
    buckets: List[Dict] = []
    for raw_ticker, qualified in indian:
        raw_msgs = fetch_messages(raw_ticker)
        if not raw_msgs:
            print("  [skip] %s: no messages returned." % qualified)
            continue
        parsed = [_parse_message(m) for m in raw_msgs]
        n_bull = sum(1 for p in parsed if p["sentiment"] == "Bullish")
        n_bear = sum(1 for p in parsed if p["sentiment"] == "Bearish")
        n_neut = len(parsed) - n_bull - n_bear
        print(
            "  [OK]  %-14s  %3d messages  "
            "(Bullish=%d  Bearish=%d  Neutral=%d)"
            % (qualified, len(parsed), n_bull, n_bear, n_neut)
        )
        buckets.append({"ticker": qualified, "parsed": parsed})

    if not buckets:
        print("  No message data returned for any Indian ticker.")
        _empty_tables()
        return

    # Step 4: Normalise across all records (both sentiments), rank, display
    print("\n[4/4] Normalising scores across all records and ranking ...")
    ranked_bull, ranked_bear = score_and_rank(buckets)

    print("  Bullish candidates : %d" % len(ranked_bull))
    print("  Bearish candidates : %d" % len(ranked_bear))

    _print_table(ranked_bull, "BULLISH OPPORTUNITIES")
    _print_table(ranked_bear, "BEARISH / SHORT CANDIDATES")

    print(bar)
    print("  Scan complete -- zero data was stored.")
    print(bar + "\n")


if __name__ == "__main__":
    run()
