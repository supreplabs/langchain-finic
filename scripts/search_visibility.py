"""Collect a minimal, aggregate Search Console report for the tracking issue."""

import json
import os
import sys
from datetime import date, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

SITE = "https://supreplabs.github.io/langchain-finic/"
SITEMAP = SITE + "sitemap.xml"
PAGES = {
    "Homepage": SITE,
    "Host-model tutorial": SITE + "tutorials/finic-langgraph-host-model/",
}
API = "https://www.googleapis.com"


def request_json(path, token, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        path if path.startswith("https://") else API + path,
        data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def safe_call(path, token, payload=None):
    try:
        return request_json(path, token, payload), None
    except HTTPError as error:
        return None, f"Google API returned HTTP {error.code}"
    except (URLError, TimeoutError, ValueError, OSError):
        return None, "Google API request failed"


def report(token, today):
    end = today - timedelta(days=3)
    start = end - timedelta(days=27)
    site = quote(SITE, safe="")
    lines = [
        "## Daily search visibility",
        "",
        f"Search Analytics window: {start.isoformat()} to {end.isoformat()} (28 days; Web search).",
        "Figures are aggregate and may be delayed or incomplete.",
        "",
    ]
    result, error = safe_call(
        f"/webmasters/v3/sites/{site}/searchAnalytics/query",
        token,
        {"startDate": start.isoformat(), "endDate": end.isoformat(), "type": "web"},
    )
    if error:
        lines.append(f"- Search performance: unavailable ({error}).")
    elif not result.get("rows"):
        lines.append("- Search performance: no rows returned; metrics may be unavailable or zero.")
    else:
        row = result["rows"][0]
        lines.append(
            f"- Search performance: {row['clicks']:.0f} clicks; {row['impressions']:.0f} impressions."
        )
    sitemap, error = safe_call(f"/webmasters/v3/sites/{site}/sitemaps/{quote(SITEMAP, safe='')}", token)
    if error:
        lines.append(f"- Sitemap: unavailable ({error}).")
    else:
        lines.append(
            f"- Sitemap: pending={sitemap.get('isPending', 'unknown')}; "
            f"errors={sitemap.get('errors', 'unknown')}; warnings={sitemap.get('warnings', 'unknown')}; "
            f"last downloaded={sitemap.get('lastDownloaded', 'unavailable')}."
        )
    for label, url in PAGES.items():
        result, error = safe_call(
            "https://searchconsole.googleapis.com/v1/urlInspection/index:inspect",
            token,
            {"inspectionUrl": url, "siteUrl": SITE},
        )
        if error:
            lines.append(f"- {label}: unavailable ({error}).")
        else:
            status = result.get("inspectionResult", {}).get("indexStatusResult", {})
            lines.append(
                f"- {label}: {status.get('coverageState', 'unavailable')}; "
                f"verdict={status.get('verdict', 'unavailable')}; "
                f"last crawl={status.get('lastCrawlTime', 'unavailable')}."
            )
    lines.extend(["", "AI Mode and manual search results are not measured by this report."])
    return "\n".join(lines) + "\n"


def main():
    token = os.environ.get("SEARCH_CONSOLE_ACCESS_TOKEN")
    if not token:
        print("Search Console access token is missing.", file=sys.stderr)
        return 1
    print(report(token, date.today()), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
