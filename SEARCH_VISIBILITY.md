# Search visibility tracking

This is a manual measurement log for the public documentation property
<https://supreplabs.github.io/langchain-finic/>. Review weekly for the first
month, then monthly. No automatic monitoring or notifications are configured.
Do not record account details, personal information, search histories, or API keys.

## Baseline: 1 November 2026

- URL Inspection: homepage indexed; Googlebot smartphone last crawled it on
  1 November at 15:19:37, fetched it successfully, and selected the inspected URL
  as canonical. The report did not identify a referring sitemap.
- Sitemap report: `/sitemap.xml` submitted on 1 November; status “Couldn't fetch,”
  zero discovered pages. A normal HTTP request returned the sitemap successfully;
  that does not establish what Googlebot received.
- Search performance: Search Console displayed “Processing data, please check again
  in a day or so”; clicks, impressions, and query metrics are **unavailable**, not zero.
- Manual Google search, query `finic langchain`: no package result observed on the
  first results page in a signed-in India session. AI Mode did not mention or cite
  the package; it discussed Finic's older Psychic tutorial. One observation is not
  a representative ranking or evidence of what other users see.

## Weekly check

1. In Search Console, choose Web search and a fixed 28-day window; record the date
   range, total impressions and clicks, top relevant queries, and top documentation
   pages. Treat delayed or unavailable data as unavailable. Google can omit some
   query data, so the displayed list may not be complete.
2. Record indexed-page counts from the Pages report, plus homepage and tutorial URL
   Inspection status and last crawl date. Record the sitemap status and last-read
   date separately; don't infer causation from these statuses.
3. In a consistent browser setup, search `finic langchain`, `finic grayson langchain`,
   and `langchain-finic`. Record date, country, signed-in or signed-out state, first
   page presence, and whether AI Mode mentioned or cited the package. Results vary
   across sessions; absence from the first page is not a site-wide ranking.
4. Record package downloads separately from [PyPI Stats](https://pypistats.org/packages/langchain-finic),
   including the period and metric used. Download counts are not unique users.
5. Compare equivalent periods after enough data accumulates. Note major releases,
   docs changes, and external listings, but do not attribute changes to them without
   additional evidence.

| Checked on | Search data range | Impressions / clicks | Indexed pages | Sitemap status / last read | Manual results | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-01 | Unavailable | Unavailable | Homepage indexed; total unavailable | Couldn't fetch; 2026-11-01 | `finic langchain`: not on first page; AI Mode did not mention or cite package | Initial baseline; signed-in India session |
