"""
FIFA Men's World Ranking scraper
---------------------------------
Fetches the current FIFA Men's World Ranking from inside.fifa.com and saves
it to CSV.

Why a headless browser?
FIFA moved the rankings page to inside.fifa.com and the table is now
populated by client-side JavaScript after the initial page load (the data
is not present in the raw HTML, so a plain `requests.get()` won't see it).
This script uses Playwright to load the page like a real browser, wait for
the table to populate, then parse it.

Setup:
    pip install playwright beautifulsoup4 --break-system-packages
    playwright install chromium

Usage:
    python fifa_rankings_scraper.py [-o output.csv] [--debug]

Notes:
- Be a good citizen: this script loads the page once. Don't hammer the site
  with rapid repeated requests, and check fifa.com's Terms of Use / robots.txt
  before using scraped data commercially.
- If FIFA changes their markup again, update the CSS selectors in
  `extract_rows()` below. Use --debug to dump a screenshot + HTML snapshot
  to help find the new selectors.
"""

import argparse
import csv
import re
import sys
from datetime import datetime, timezone

from playwright.sync_api import sync_playwright

from pathlib import Path

RANKING_URL = "https://inside.fifa.com/fifa-world-ranking/men"


def fetch_table_html(debug: bool = False) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page.goto(RANKING_URL, wait_until="networkidle", timeout=30000)

        # The table has a header row with no data rows until JS populates it,
        # so wait for at least one real row to show up.
        try:
            page.wait_for_selector("table tbody tr", timeout=15000)
        except Exception:
            if debug:
                page.screenshot(path="fifa_debug.png", full_page=True)
                with open("fifa_debug.html", "w", encoding="utf-8") as f:
                    f.write(page.content())
            browser.close()
            raise RuntimeError(
                "Timed out waiting for ranking rows to load. "
                "Run with --debug to save a screenshot/HTML snapshot for inspection."
            )

        html = page.content()
        if debug:
            page.screenshot(path="fifa_debug.png", full_page=True)
            with open("fifa_debug.html", "w", encoding="utf-8") as f:
                f.write(html)
        browser.close()
        return html


def extract_rows(html: str) -> list:
    """Parse ranking rows out of the rendered page HTML using BeautifulSoup."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("table tbody tr")
    if not rows:
        raise RuntimeError("No table rows found in rendered HTML.")

    results = []
    for tr in rows:
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if not cells:
            continue
        # Expected columns on the live site: Rank | Team | Last result | +/- | Points
        rank = cells[0] if len(cells) > 0 else ""
        team = cells[1] if len(cells) > 1 else ""
        points = cells[-1] if len(cells) > 1 else ""

        rank_match = re.search(r"\d+", rank)
        rank = rank_match.group() if rank_match else rank

        if not team:
            continue

        results.append({"rank": rank, "team": team, "points": points})

    return results


def save_csv(rows: list, path: str):
    fieldnames = ["rank", "team", "points"]

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Scrape the FIFA Men's World Ranking")
    parser.add_argument(
        "-o", "--output",
        default="data/external/fifa_rankings.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save a screenshot and HTML snapshot of the rendered page for troubleshooting",
    )
    args = parser.parse_args()

    try:
        html = fetch_table_html(debug=args.debug)
        rows = extract_rows(html)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("No ranking data found.", file=sys.stderr)
        sys.exit(1)

    save_csv(rows, args.output)
    print(f"Saved {len(rows)} teams to {args.output}")
    print("\nTop 10:")
    for r in rows[:10]:
        print(f"  {r['rank']:>3}  {r['team']:<25} {r['points']} pts")


if __name__ == "__main__":
    main()