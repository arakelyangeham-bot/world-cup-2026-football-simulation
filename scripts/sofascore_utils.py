# sofascore_utils.py
from curl_cffi import requests
from time import sleep
from pathlib import Path

BASE_URL = "https://api.sofascore.com/api/v1"
HOME_URL = "https://www.sofascore.com/"

MAX_RETRIES = 3
MAX_CONSECUTIVE_BLOCKS = 8
RETRYABLE_CODES = {403, 429, 500, 502, 503, 504}
PERMANENT_SKIP_CODES = {400, 401, 404}

# Pick a fingerprint that actually exists in your installed curl_cffi build.
# "chrome146" is not a real shipped target and may silently fall back to a
# generic/default fingerprint, which is easier for Cloudflare to flag.
# Run print_available_impersonations() once to confirm what's valid, then
# hardcode the latest real one here.
IMPERSONATE_TARGET = "chrome131"

COMMON_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
    "x-requested-with": "XMLHttpRequest",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

consecutive_blocks = 0

# A single persistent session carries Cloudflare cookies (e.g. __cf_bm,
# cf_clearance) across requests, the same way a real browser tab does.
# Creating a fresh requests.get() per call, as before, throws those cookies
# away every time, which is one of the strongest signals Cloudflare's bot
# management uses to tell scripts apart from browsers.
_session = requests.Session(impersonate=IMPERSONATE_TARGET)
_warmed_up = False


def print_available_impersonations():
    """
    Diagnostic helper: prints what curl_cffi actually supports in this
    environment. Run this once locally and pick a real, recent Chrome
    version for IMPERSONATE_TARGET above.
    """
    try:
        from curl_cffi.const import CurlSslVersion  # noqa: F401
    except Exception:
        pass
    try:
        from curl_cffi.requests.impersonate import BrowserType
        print([b.name for b in BrowserType])
    except Exception as e:
        print(f"Could not introspect impersonation list directly: {e}")
        print("Check curl_cffi's CHANGELOG/docs for the version you have installed.")


def warm_up_session(force=False):
    """
    Loads the public Sofascore homepage first so the session picks up
    Cloudflare's cookies (e.g. __cf_bm, cf_clearance) before any API calls
    are made. This mimics what a real browser does: load the page, then
    fire XHR requests using cookies set during that page load.

    Without this, requests can look "stateless" to Cloudflare's bot
    management even with a correct TLS fingerprint, which is consistent
    with seeing 403s on the API while the website itself loads fine.
    """
    global _warmed_up
    if _warmed_up and not force:
        return

    r = _session.get(
        HOME_URL,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=20,
    )
    print(f"[warm-up] {r.status_code} {HOME_URL} (cookies: {len(_session.cookies)})")
    _warmed_up = True
    sleep(1.5)


def get_json(url):
    global consecutive_blocks

    warm_up_session()

    for attempt in range(1, MAX_RETRIES + 1):
        r = _session.get(
            url,
            headers=COMMON_HEADERS,
            timeout=20,
        )
        print(f"[Attempt {attempt}] {r.status_code} {url}")

        if r.status_code in PERMANENT_SKIP_CODES:
            raise Exception(f"Permanent HTTP {r.status_code}")

        if r.status_code in {403, 429}:
            consecutive_blocks += 1
            if consecutive_blocks >= MAX_CONSECUTIVE_BLOCKS:
                raise SystemExit(f"Stopping: {consecutive_blocks} consecutive blocks")

            # A fresh 403 after a previously-working session often means the
            # Cloudflare cookie expired or got flagged mid-run. Re-warm and
            # retry once before giving up on this attempt.
            if r.status_code == 403:
                print("[403] Re-warming session and retrying...")
                warm_up_session(force=True)
        else:
            consecutive_blocks = 0

        if r.status_code in RETRYABLE_CODES:
            if attempt == MAX_RETRIES:
                raise Exception(f"Retryable HTTP {r.status_code} after {MAX_RETRIES} attempts")
            sleep(attempt * 5)
            continue

        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    # Quick standalone diagnostic. Run: python sofascore_utils.py
    print_available_impersonations()
    test_url = f"{BASE_URL}/sport/football/scheduled-events/2026-06-19"
    try:
        data = get_json(test_url)
        print("SUCCESS — sample keys:", list(data.keys())[:5])
    except Exception as e:
        print(f"FAILED: {e}")