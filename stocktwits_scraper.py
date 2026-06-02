#!/usr/bin/env python3
"""
StockTwits Signal Scraper — Kash Analytics Phase 1B
=====================================================
Fetches trending tickers and recent messages from StockTwits.
Filters to Indian equities (NSE / BSE) only.
Extracts bullish/bearish sentiment and ranks by composite score:

    Frequency 40%  |  Recency 35%  |  Engagement 25%

Outputs two ranked tables:
    📈  Bullish Opportunities
    📉  Bearish / Short Candidates

Each row: Rank | Ticker | Mention Count | Recency Rating
           | Engagement Rating | Key Narrative

Policy: NO data storage — fetch → analyse → display → discard.
Rate policy: ≤ 60 req/min (StockTwits free-tier limit).
             Monitors X-RateLimit-Remaining header and auto-pauses.
"""

import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import requests


# ── Configuration ──────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "KashAnalytics/1.0 Phase1B-SignalScraper "
        "(github.com/nihaalkashimji/kash-analytics; "
        "contact: nihaalkashimji@gmail.com)"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

BASE_URL           = "https://api.stocktwits.com/api/2"
RATE_LIMIT_DELAY   = 1.0     # baseline seconds between calls (60 req/min ceiling)
RATE_LIMIT_PAUSE   = 30.0    # seconds to pause when header signals ≤ 10 remaining
RATE_LIMIT_WARN_AT = 10      # X-RateLimit-Remaining threshold for pause
MAX_TICKERS        = 30      # trending tickers to fetch (before Indian filter)
MSG_LIMIT          = 30      # messages per ticker (StockTwits max per request)
TOP_N              = 10      # rows in each output table
RECENCY_HALF_LIFE  = 8.0     # hours — recency score halves every N hours

WEIGHTS = {"frequency": 0.40, "recency": 0.35, "engagement": 0.25}

# ── Indian equity filter ───────────────────────────────────────────────────────
# StockTwits identifies Indian stocks via region, exchange, and MIC codes.
INDIAN_REGION       = "IN"
INDIAN_EXCHANGES    = {"NSE", "BSE", "XNSE", "XBOM"}
INDIAN_MIC_SUFFIXES = {".XNSE", ".XBOM"}  # MIC: XNSE = NSE, XBOM = BSE

# ── Stop words for narrative extraction ───────────────────────────────────────
_STOP = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "to","of","in","on","at","by","for","with","about","as","up","out","if","this",
    "that","these","those","it","its","and","or","but","not","no","so","just","very",
    "also","only","more","most","new","now","all","my","your","their","our","we",
    "they","he","she","i","you","me","him","her","us","them","what","which","who",
    "how","when","where","why","stock","shares","market","price","trade","trading",
    "buy","sell","long","short","call","put","bullish","bearish","hold","going",
    "get","getting","see","look","think","still","good","bad","big","today","week",
    "month","year","day","time","high","low","after","before","like","here","there",
    "than","then","from","into","over","under","through","down","up","im","dont",
    "cant","wont","ive","already","let","make","need","want","way","even","back",
    "much","well","got","right","too","every","another","set","amp","lol","gonna",
    "really","something","nothing","someone","anyone","re","ve","ll","s","t","m","d",
    "nse","bse","inr","rs","rupee","rupees","indian","india","sensex","nifty",
}


# ── Rate-limit-aware HTTP GET ──────────────────────────────────────────────────

def _get(url: str, params: dict | None = None) -> dict | None:
    """
    HTTP GET with:
     - Mandatory baseline delay before every call.
     - Response header monitoring: pauses if X-RateLimit-Remaining ≤ threshold.
     - Automatic single retry on 429 (Too Many Requests) after a long pause.
     - Graceful error handling (returns None on any failure).
    """
    time.sleep(RATE_LIMIT_DELAY)
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=12)

        # ── Rate-limit header monitoring ──
        remaining = resp.headers.get("X-RateLimit-Remaining")
        limit     = resp.headers.get("X-RateLimit-Limit")
        if remaining is not None:
            rem_int = int(remaining)
            if rem_int <= RATE_LIMIT_WARN_AT:
                reset_ts = resp.headers.get("X-RateLimit-Reset")
                wait = RATE_LIMIT_PAUSE
                if reset_ts:
                    try:
                        reset_dt = datetime.fromtimestamp(int(reset_ts), tz=timezone.utc)
                        wait = max(
                            5.0,
                            (reset_dt - datetime.now(timezone.utc)).total_seconds() + 2,
                        )
                    except Exception:
                        pass
                print(
                    f"  [RATE] {remaining}/{limit} requests remaining — "
                    f"pausing {wait:.0f} s …",
                    file=sys.stderr,
                )
                time.sleep(wait)

        # ── 429 retry ──
        if resp.status_code == 429:
            print(
                f"  [WARN] 429 Too Many Requests — pausing {RATE_LIMIT_PAUSE:.0f} s …",
                file=sys.stderr,
            )
            time.sleep(RATE_LIMIT_PAUSE)
            resp = requests.get(url, headers=HEADERS, params=params, timeout=12)

        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.Timeout:
        print(f"  [WARN] Timeout: {url}", file=sys.stderr)
    except requests.exceptions.HTTPError as e:
        print(f"  [WARN] HTTP {e.response.status_code}: {url}", file=sys.stderr)
    except requests.exceptions.RequestException as e:
        print(f"  [WARN] Request error: {e}", file=sys.stderr)
    return None


# ── Indian equity check ────────────────────────────────────────────────────────

def _is_indian(symbol_data: dict) -> bool:
    """Return True if the symbol_data dict represents an Indian equity (NSE/BSE)."""
    region   = symbol_data.get("region", "")
    exchange = symbol_data.get("exchange", "")
    mic      = symbol_data.get("symbol_mic", "")
    return (
        region == INDIAN_REGION
        or exchange in INDIAN_EXCHANGES
        or any(mic.endswith(sfx) for sfx in INDIAN_MIC_SUFFIXES)
    )


# ── API fetch functions ────────────────────────────────────────────────────────

def fetch_indian_trending() -> list[str]:
    """
    Fetch global trending tickers and return only Indian equities (NSE/BSE).
    Returns whatever StockTwits organically surfaces — no supplementation.
    """
    data = _get(f"{BASE_URL}/trending/symbols.json")
    if not data:
        return []

    all_symbols = data.get("symbols", [])[:MAX_TICKERS]
    return [s["symbol"] for s in all_symbols if _is_indian(s) and "symbol" in s]


def fetch_messages(ticker: str) -> list[dict]:
    """Fetch the most recent public messages for a ticker symbol."""
    data = _get(
        f"{BASE_URL}/streams/symbol/{ticker}.json",
        params={"limit": MSG_LIMIT},
    )
    return data.get("messages", []) if data else []


# ── Signal extraction ──────────────────────────────────────────────────────────

def _sentiment(msg: dict) -> str | None:
    """Return 'Bullish', 'Bearish', or None from message entities."""
    s = msg.get("entities", {}).get("sentiment")
    return s.get("basic") if isinstance(s, dict) else None


def _recency_score(created_at: str) -> float:
    """
    Exponential decay: 1.0 at post time, halves every RECENCY_HALF_LIFE hours.
    Score = exp(−age_h × ln2 / half_life)
    """
    try:
        ts    = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
        return math.exp(-age_h * math.log(2) / RECENCY_HALF_LIFE)
    except Exception:
        return 0.0


def _engagement(msg: dict) -> float:
    """Likes + reshares as a raw engagement count."""
    likes    = (msg.get("likes") or {}).get("total", 0) or 0
    reshares = msg.get("reshares_count", 0) or 0
    return float(likes + reshares)


def _keywords(bodies: list[str], top_k: int = 5) -> str:
    """
    Extract the top-k meaningful words from message bodies.
    Strips $TICKER tags, URLs, punctuation, and stop words.
    """
    words: list[str] = []
    for body in bodies:
        text = re.sub(r"\$[A-Z]{1,10}", "", body)     # remove $TICKER cashtags
        text = re.sub(r"https?://\S+", "", text)        # remove URLs
        text = re.sub(r"[^\w\s]", " ", text)            # strip punctuation
        for w in text.lower().split():
            if len(w) > 2 and w not in _STOP and not w.isdigit():
                words.append(w)
    top = Counter(words).most_common(top_k)
    return ", ".join(w for w, _ in top) if top else "—"


# ── Per-ticker aggregation ─────────────────────────────────────────────────────

def _aggregate(msgs: list[dict]) -> dict:
    """Roll up a list of same-sentiment messages into aggregate metrics."""
    count    = len(msgs)
    mean_rec = sum(_recency_score(m["created_at"]) for m in msgs) / count
    total_eng = sum(_engagement(m) for m in msgs)
    return {
        "count":     count,
        "rec":       mean_rec,
        "eng":       total_eng,
        "narrative": _keywords([m.get("body", "") for m in msgs]),
    }


def analyse_ticker(ticker: str) -> dict | None:
    """
    Fetch + analyse messages for one ticker.
    Returns a dict {ticker, bullish, bearish} or None if fetch failed /
    no sentiment-tagged messages exist.
    """
    msgs = fetch_messages(ticker)
    if not msgs:
        return None

    bull = [m for m in msgs if _sentiment(m) == "Bullish"]
    bear = [m for m in msgs if _sentiment(m) == "Bearish"]

    if not bull and not bear:
        return None   # all messages untagged

    return {
        "ticker":  ticker,
        "bullish": _aggregate(bull) if bull else None,
        "bearish": _aggregate(bear) if bear else None,
    }


# ── Scoring & ranking ──────────────────────────────────────────────────────────

def _normalise(values: list[float]) -> list[float]:
    """Min-max normalise to [0, 1]. Returns all-1 if values are constant."""
    mn, mx = min(values), max(values)
    if mx == mn:
        return [1.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def build_ranked_table(signals: list[dict], side: str) -> list[dict]:
    """
    Compute composite score for every signal on the given side,
    normalise each dimension, apply weights, return top-N rows.

    Composite = 0.40 × norm_frequency
              + 0.35 × norm_recency
              + 0.25 × norm_engagement
    """
    rows = [s for s in signals if s.get(side)]
    if not rows:
        return []

    nf = _normalise([r[side]["count"] for r in rows])
    nr = _normalise([r[side]["rec"]   for r in rows])
    ne = _normalise([r[side]["eng"]   for r in rows])

    scored = [
        {
            "ticker":    r["ticker"],
            "count":     r[side]["count"],
            "norm_rec":  nr[i],
            "norm_eng":  ne[i],
            "score":     (
                WEIGHTS["frequency"]  * nf[i] +
                WEIGHTS["recency"]    * nr[i] +
                WEIGHTS["engagement"] * ne[i]
            ),
            "narrative": r[side]["narrative"],
        }
        for i, r in enumerate(rows)
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:TOP_N]


# ── Rendering ──────────────────────────────────────────────────────────────────

def _stars(norm: float, n: int = 5) -> str:
    """Convert a normalised float [0, 1] to a star string (★☆)."""
    return "★" * round(norm * n) + "☆" * (n - round(norm * n))


_WIDTH = 100
_SEP   = "─" * _WIDTH


def _print_table(rows: list[dict], title: str, emoji: str) -> None:
    print(f"\n{emoji}  {title}")
    print(_SEP)
    print(
        f"  {'Rank':<5} {'Ticker':<12} {'Mentions':>10}  "
        f"{'Recency':>12}  {'Engagement':>14}  Key Narrative"
    )
    print(_SEP)
    if not rows:
        print("  (no sentiment-tagged messages found for this side)")
    else:
        for rank, row in enumerate(rows, 1):
            print(
                f"  {rank:<5} {row['ticker']:<12} {row['count']:>10}  "
                f"{_stars(row['norm_rec']):>12}  {_stars(row['norm_eng']):>14}  "
                f"{row['narrative']}"
            )
    print(_SEP)
    w = WEIGHTS
    print(
        f"  Weights: Frequency {w['frequency']*100:.0f}% | "
        f"Recency {w['recency']*100:.0f}% | "
        f"Engagement {w['engagement']*100:.0f}% | "
        f"Recency half-life: {RECENCY_HALF_LIFE:.0f} h"
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * _WIDTH)
    print("  KASH ANALYTICS — StockTwits Signal Scraper  │  Phase 1B")
    print(f"  Run: {run_ts}  │  Scope: Indian equities (NSE / BSE)")
    print(f"  Policy: No data stored — fetch → analyse → display → discard")
    print("=" * _WIDTH)

    # ── 1. Trending Indian tickers ────────────────────────────────────────────
    print(f"\n[1/3] Fetching trending tickers (Indian equities only) …")
    tickers = fetch_indian_trending()
    if not tickers:
        print("  No Indian equities trending on StockTwits right now.")
        print("  Only organically mentioned stocks are surfaced — nothing to analyse.")
        print(f"\n  ✓ Run complete at {run_ts}. No data persisted.\n")
        return
    print(f"  Tickers ({len(tickers)}): {', '.join(tickers)}")

    # ── 2. Fetch messages + extract sentiment ─────────────────────────────────
    est_s = len(tickers) * RATE_LIMIT_DELAY
    print(f"\n[2/3] Fetching messages per ticker (~{est_s:.0f} s at {RATE_LIMIT_DELAY} s/req) …")
    print(f"  {'Ticker':<12} {'Bullish':>8}  {'Bearish':>8}")
    print(f"  {'─'*12} {'─'*8}  {'─'*8}")

    signals: list[dict] = []
    for ticker in tickers:
        result = analyse_ticker(ticker)
        bull_n = result["bullish"]["count"] if result and result["bullish"] else 0
        bear_n = result["bearish"]["count"] if result and result["bearish"] else 0
        flag   = "" if (bull_n or bear_n) else "  (no tagged msgs)"
        print(f"  {ticker:<12} {bull_n:>8}  {bear_n:>8}{flag}")
        if result:
            signals.append(result)

    if not signals:
        print("\n  No sentiment-tagged messages across any trending Indian ticker.")
        print(f"\n  ✓ Run complete at {run_ts}. No data persisted.\n")
        return

    tagged = sum(1 for s in signals if s["bullish"] or s["bearish"])
    print(f"\n  {tagged}/{len(tickers)} tickers had sentiment-tagged messages.")

    # ── 3. Score, rank, display ───────────────────────────────────────────────
    print(
        f"\n[3/3] Scoring (Freq {WEIGHTS['frequency']*100:.0f}% | "
        f"Rec {WEIGHTS['recency']*100:.0f}% | "
        f"Eng {WEIGHTS['engagement']*100:.0f}%) and ranking …"
    )

    _print_table(build_ranked_table(signals, "bullish"), "BULLISH OPPORTUNITIES",       "📈")
    _print_table(build_ranked_table(signals, "bearish"), "BEARISH / SHORT CANDIDATES",  "📉")

    print(f"\n  ✓ Analysis complete at {run_ts}. No data persisted.\n")


if __name__ == "__main__":
    main()
