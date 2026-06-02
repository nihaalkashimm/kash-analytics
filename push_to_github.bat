@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  Kash Analytics — one-time git setup + push to GitHub
REM  Double-click this file from Windows Explorer, or run in Command Prompt.
REM  Requires: Git for Windows (https://git-scm.com/download/win)
REM ─────────────────────────────────────────────────────────────────────────────

SET REPO_DIR=%~dp0
SET REMOTE=https://github.com/nihaalkashimji/kash-analytics.git

echo.
echo === Kash Analytics : Git Setup ===
echo Folder : %REPO_DIR%
echo Remote : %REMOTE%
echo.

cd /d "%REPO_DIR%"

REM ── Init (safe to run on an existing repo) ─────────────────────────────────
git init -b main
git config user.email "nihaalkashimji@gmail.com"
git config user.name "Kashimm"

REM ── Add remote (skip if already exists) ───────────────────────────────────
git remote get-url origin >nul 2>&1
IF ERRORLEVEL 1 (
    git remote add origin %REMOTE%
    echo Remote added.
) ELSE (
    echo Remote already set.
)

REM ── Stage and commit ───────────────────────────────────────────────────────
git add stocktwits_scraper.py
git commit -m "feat(phase-1b): add StockTwits signal scraper

- Fetches trending tickers + recent messages via StockTwits public API
- Filters to Indian equities (NSE / BSE) only
- Extracts bullish/bearish sentiment from message entities
- Ranks by composite score: Frequency 40% | Recency 35% | Engagement 25%
- Outputs two ranked tables: bullish opportunities + bearish/short candidates
- Rate-limited to 60 req/min; monitors X-RateLimit-Remaining header
- No data storage: fetch -> analyse -> display -> discard"

REM ── Push ──────────────────────────────────────────────────────────────────
echo.
echo Pushing to GitHub ...
echo (A browser window or credential prompt may open for authentication.)
echo.
git push -u origin main

echo.
IF ERRORLEVEL 1 (
    echo [ERROR] Push failed. Check your GitHub credentials and repo permissions.
) ELSE (
    echo [OK] Pushed successfully to %REMOTE%
)

pause
