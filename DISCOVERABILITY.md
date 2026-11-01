# Discoverability maintainer checklist

The documentation is published at <https://supreplabs.github.io/langchain-finic/>
with a generated sitemap at <https://supreplabs.github.io/langchain-finic/sitemap.xml>.
The GitHub Pages site shares the `supreplabs.github.io` origin: a `robots.txt`
under `/langchain-finic/` would not control crawling of this project site.

After publishing meaningful documentation changes:

1. In the verified Search Console URL-prefix property for the documentation site,
   inspect the homepage and important guides, submit the sitemap, and request
   indexing where appropriate. Confirm that Search Console reports the request was
   accepted; a live test alone does not establish that it was queued. See Google's
   [URL Inspection guide](https://support.google.com/webmasters/answer/9012289)
   and [sitemap guidance](https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap).
   Submission and inspection do not guarantee indexing or ranking.
2. Record impressions, clicks, queries, and indexed URLs in Search Console over time;
   compare PyPI download trends separately. Results may take time, and do not prove
   causality. Use [SEARCH_VISIBILITY.md](SEARCH_VISIBILITY.md) for monitoring setup and
   the tracking issue for observations. Do not add analytics or expose customer data.
3. Consider submitting the independent integration through the current
   [LangChain listing process](https://docs.langchain.com/oss/python/contributing/publish-langchain);
   see [LANGCHAIN_LISTING.md](LANGCHAIN_LISTING.md). A Finic documentation listing
   is also subject to Finic's review. Neither listing is implied by this checklist.
4. Include README and metadata updates in the next release; PyPI displays metadata
   from the published distribution, not the latest repository branch.

If the sitemap report says **Couldn't fetch**, inspect the exact submitted URL and
its last-read date, check that the public sitemap returns HTTP 200 and contains the
expected canonical URLs, and review Google's
[sitemap troubleshooting guidance](https://support.google.com/webmasters/answer/7451001).
A successful local HTTP request does not prove Googlebot could fetch it. Check the
report again after processing; avoid repeatedly submitting an unchanged sitemap.
For a homepage reported as unknown to Google, use URL Inspection to check its
index status and test the live URL. Record the reported reason rather than guessing
whether the cause is crawling, indexing, or ranking.

No listing acceptance, ranking, or AI citation is claimed by this file. Google recommends its regular
[search best practices for AI features](https://developers.google.com/search/docs/appearance/ai-features).
