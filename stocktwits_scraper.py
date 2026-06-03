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
  User-Agent : web:KashAnalytics:v1.0.0 (by /u/Kashimmm) on every request

Indian stock validation (yfinance, no hardcoded list)
  StockTwits may append its own exchange suffix (e.g. WIPRO.NSE).
  The bare symbol (split on ".") is extracted first, then .NS and .BO
  are tried -- preventing invalid chains like WIPRO.NSE.NS.
  Confirmed if yfinance fast_info.currency == INR.
"""

import math
import re
import time
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import yfinance as yf
from tabulate import tabulate


# ========================== CONSTANTS =========================================

STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"
USER_AGENT      = "web:KashAnalytics:v1.0.0 (by /u/Kashimmm)"

HALF_LIFE_S  = 6 * 3600
DECAY_LAMBDA = math.log(2) / HALF_LIFE_S

RATE_LIMIT   = 60
WINDOW_S     = 60.0
PAUSE_THRESH = 10

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
    """Enforces at most `limit` HTTP calls per rolling `window` seconds."""

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
    Rate-limited GET with User-Agent, auto-pause, and one 429 retry.
    User-Agent is set on every request without exception.
    """
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(2):
        _limiter.acquire()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=12)
        except requests.RequestException as exc:
            print("  [network error] %s" % exc)
            return None

        remaining_hdr = resp.headers.get("X-Ratelimit-Remaining")
        if remaining_hdr is not None:
            try:
                remaining_int = int(remaining_hdr)
            except ValueError:
                remaining_int = None
            if remaining_int is not None and remaining_int <= PAUSE_THRESH:
                reset_hdr = resp.headers.get("X-Ratelimit-Reset")
                wait = max(1, int(reset_hdr) - int(time.time())) + 2 if reset_hdr else 35
                print(
                    "  [auto-pause] X-Ratelimit-Remaining=%d (<=10) "
                    "-- pausing %ds ..." % (remaining_int, wait)
                )
                time.sleep(wait)

        if resp.status_code == 429:
            if attempt == 0:
                retry_after = int(resp.headers.get("Retry-After", 30))
                print("  [429] waiting %ds before retry ..." % retry_after)
                time.sleep(retry_after)
                continue
            print("  [429] retry exhausted -- skipping.")
            return None

        return resp

    return None


# ========================== STOCKTWITS API ====================================

def fetch_trending_tickers() -> List[str]:
    """Return raw ticker strings from StockTwits trending endpoint."""
    resp = _get(STOCKTWITS_BASE + "/trending/symbols.json")
    if resp is None or resp.status_code != 200:
        print("  [error] trending endpoint returned HTTP %s" % (
            resp.status_code if resp is not None else "N/A"))
        return []
    symbols = resp.json().get("symbols", [])
    tickers = [s["symbol"] for s in symbols if "symbol" in s]
    print("  -> %d tickers from StockTwits trending" % len(tickers))
    return tickers


def fetch_messages(ticker: str) -> List[Dict]:
    """Fetch message stream for `ticker` (raw StockTwits symbol)."""
    resp = _get(STOCKTWITS_BASE + "/streams/symbol/" + ticker + ".json")
    if resp is None or resp.status_code != 200:
        return []
    return resp.json().get("messages", [])


# ========================== INDIAN STOCK VALIDATION ==========================

def validate_indian(ticker: str) -> Tuple[bool, str]:
    """
    Determine whether a StockTwits ticker is listed on NSE or BSE.

    StockTwits sometimes appends its own exchange suffix, e.g.:
        WIPRO.NSE  INFY.BSE  RELIANCE.NSI

    We strip everything after the first dot to get the bare symbol,
    then build the yfinance symbol ourselves:
        bare + ".NS"  (NSE)
        bare + ".BO"  (BSE)

    This prevents invalid compound suffixes such as WIPRO.NSE.NS.

    A ticker is confirmed Indian if yfinance reports currency == "INR".
    No hardcoded ticker list is used -- every check is a live API call.
    """
    bare = ticker.split(".")[0]          # "WIPRO.NSE" -> "WIPRO"
    for suffix in (".NS", ".BO"):
        sym = bare + suffix              # "WIPRO.NS" then "WIPRO.BO"
        try:
            fi = yf.Ticker(sym).fast_info
            if getattr(fi, "currency", None) == "INR":
                return True, sym
        except Exception:
            continue
    return False, ""


# ========================== MESSAGE PARSING ===================================

def _parse_message(raw: Dict) -> Dict:
    """Extract sentiment, recency decay, and engagement from one message."""
    body      = raw.get("body", "")
    sentiment = ((raw.get("entities") or {}).get("sentiment") or {}).get("basic")
    likes     = int((raw.get("likes")    or {}).get("total",          0) or 0)
    reshares  = int((raw.get("reshares") or {}).get("reshared_count", 0) or 0)

    created_raw = raw.get("created_at", "")
    try:
        dt    = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        age_s = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())
    except (ValueError, AttributeError):
        age_s = 0.0

    return {
        "body":       body,
        "sentiment":  sentiment,           # "Bullish" | "Bearish" | None
        "recency":    math.exp(-DECAY_LAMBDA * age_s),
        "engagement": likes + reshares,
    }


# ========================== SCORING ENGINE ====================================

def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _key_narrative(messages: List[Dict]) -> str:
    words: List[str] = []
    for m in messages:
        tokens = re.findall(r"[A-Za-z]{3,}", m.get("body", ""))
        words.extend(t.lower() for t in tokens if t.lower() not in STOPWORDS)
    if not words:
        return "--"
    return " | ".join(w for w, _ in Counter(words).most_common(3))


def score_and_rank(buckets: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Compute raw Frequency/Recency/Engagement for every ticker x sentiment pair.
    Min-max normalise across ALL records jointly (both sentiments together) then
    apply weights. Split and rank each sentiment table independently.
    """
    all_rows: List[Dict] = []
    for b in buckets:
        for sentiment in ("Bullish", "Bearish"):
            msgs = [m for m in b["parsed"] if m["sentiment"] == sentiment]
            if not msgs:
                continue
            all_rows.append({
                "ticker":        b["ticker"],
                "sentiment":     sentiment,
                "mention_count": len(msgs),
                "freq_raw":      float(len(msgs)),
                "rec_raw":       float(sum(m["recency"]    for m in msgs) / len(msgs)),
                "eng_raw":       float(sum(m["engagement"] for m in msgs)),
                "messages":      msgs,
            })

    if not all_rows:
        return [], []

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

    bull = sorted([r for r in all_rows if r["sentiment"] == "Bullish"],
                  key=lambda r: r["composite"], reverse=True)
    bear = sorted([r for r in all_rows if r["sentiment"] == "Bearish"],
                  key=lambda r: r["composite"], reverse=True)

    for rank, row in enumerate(bull, 1):
        row["rank"] = rank
    for rank, row in enumerate(bear, 1):
        row["rank"] = rank

    return bull, bear


# ========================== OUTPUT FORMATTING =================================

_BAR = "=" * 74


def _print_table(ranked: List[Dict], title: str) -> None:
    print("\n" + _BAR)
    print("  " + title)
    print(_BAR)
    if not ranked:
        print("  No Indian tickers detected in this scan.\n")
        return
    headers = ["Rank", "Ticker", "Mention Count",
                "Recency Rating", "Engagement Rating", "Key Narrative"]
    table = [
        [r["rank"], r["ticker"], r["mention_count"],
         "%.4f" % r["recency_score"], "%.4f" % r["engagement_score"],
         _key_narrative(r["messages"])[:52]]
        for r in ranked
    ]
    print(tabulate(table, headers=headers, tablefmt="rounded_outline"))
    print()


def _empty_tables() -> None:
    for title in ("BULLISH OPPORTUNITIES", "BEARISH / SHORT CANDIDATES"):
        print("\n" + _BAR)
        print("  " + title)
        print(_BAR)
        print("  No Indian tickers detected in this scan.\n")


# ========================== MAIN PIPELINE =====================================

def run() -> None:
    sep = "-" * 74
    print("\n" + sep)
    print("  Kash Analytics -- StockTwits Indian Market Screener")
    print("  Phase 1B  |  Fetch -> Analyse -> Display -> Discard")
    print("  Weights: Frequency %.0f%%  Recency %.0f%%  Engagement %.0f%%"
          "  |  Half-life 6h  |  Cap %d req/min" % (
              WEIGHTS["frequency"] * 100, WEIGHTS["recency"] * 100,
              WEIGHTS["engagement"] * 100, RATE_LIMIT))
    print(sep + "\n")

    print("[1/4] Fetching trending symbols from StockTwits ...")
    raw_tickers = fetch_trending_tickers()
    if not raw_tickers:
        print("  [abort] no trending symbols returned.")
        _empty_tables()
        return

    print("\n[2/4] Validating %d ticker(s) against NSE/BSE ..." % len(raw_tickers))
    indian: List[Tuple[str, str]] = []
    for raw in raw_tickers:
        ok, sym = validate_indian(raw)
        if ok:
            print("  [OK]  %-16s -> %s" % (raw, sym))
            indian.append((raw, sym))

    if not indian:
        print("\n  No Indian tickers found in StockTwits trending.")
        _empty_tables()
        return
    print("\n  -> %d Indian ticker(s) confirmed." % len(indian))

    print("\n[3/4] Fetching message streams ...")
    buckets: List[Dict] = []
    for raw_ticker, qualified in indian:
        msgs = fetch_messages(raw_ticker)
        if not msgs:
            print("  [skip] %s -- no messages returned." % qualified)
            continue
        parsed = [_parse_message(m) for m in msgs]
        n_b = sum(1 for p in parsed if p["sentiment"] == "Bullish")
        n_s = sum(1 for p in parsed if p["sentiment"] == "Bearish")
        print("  [OK]  %-16s  %3d msgs  (Bull=%d Bear=%d Neutral=%d)" % (
            qualified, len(parsed), n_b, n_s, len(parsed) - n_b - n_s))
        buckets.append({"ticker": qualified, "parsed": parsed})

    if not buckets:
        print("  No message data for any Indian ticker.")
        _empty_tables()
        return

    print("\n[4/4] Normalising and ranking ...")
    ranked_bull, ranked_bear = score_and_rank(buckets)
    print("  Bullish: %d  Bearish: %d" % (len(ranked_bull), len(ranked_bear)))

    _print_table(ranked_bull, "BULLISH OPPORTUNITIES")
    _print_table(ranked_bear, "BEARISH / SHORT CANDIDATES")

    print(sep)
    print("  Scan complete -- zero data was stored.")
    print(sep + "\n")


if __name__ == "__main__":
    run()