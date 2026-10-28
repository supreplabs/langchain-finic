# Discoverability maintainer checklist

The documentation is published at <https://supreplabs.github.io/langchain-finic/>
with a generated sitemap at <https://supreplabs.github.io/langchain-finic/sitemap.xml>.
The GitHub Pages site shares the `supreplabs.github.io` origin: a `robots.txt`
under `/langchain-finic/` would not control crawling of this project site.

After publishing meaningful documentation changes:

1. An owner with the appropriate Google account can verify a Search Console URL-prefix
   property for the documentation site. Inspect the homepage and important guides,
   submit the sitemap, and request indexing where appropriate. See Google's
   [URL Inspection guide](https://support.google.com/webmasters/answer/9012289)
   and [sitemap guidance](https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap).
   Submission and inspection do not guarantee indexing or ranking.
2. Record impressions, clicks, queries, and indexed URLs in Search Console over time;
   compare PyPI download trends separately. Results may take time, and do not prove
   causality. Do not add analytics or expose customer data for this purpose.
3. Consider submitting the independent integration through the current
   [LangChain listing process](https://docs.langchain.com/oss/python/contributing/publish-langchain);
   see [LANGCHAIN_LISTING.md](LANGCHAIN_LISTING.md). A Finic documentation listing
   is also subject to Finic's review. Neither listing is implied by this checklist.
4. Include README and metadata updates in the next release; PyPI displays metadata
   from the published distribution, not the latest repository branch.

No Search Console verification, listing submission, ranking, or AI citation is
claimed by this file. Google recommends its regular
[search best practices for AI features](https://developers.google.com/search/docs/appearance/ai-features).
