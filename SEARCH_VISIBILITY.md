# Search visibility monitoring

The [tracking issue](https://github.com/supreplabs/langchain-finic/issues/56)
contains the baseline and weekly observations. The Search visibility workflow runs
at 05:23 UTC on Mondays and can also be started with `workflow_dispatch`. Its
comments are public: it posts aggregate clicks and impressions, sitemap metadata,
and index status for the homepage and tutorial, never raw queries or credentials.

## Configure access

The workflow uses GitHub OIDC and Google Cloud Workload Identity Federation to
obtain a short-lived token with the `webmasters.readonly` scope. To activate it:

1. In a Google Cloud project, enable the Search Console API, IAM Service Account
   Credentials API, and Security Token Service API as needed for your federation
   configuration. Create a dedicated service account.
2. Create a workload identity pool and GitHub OIDC provider; restrict its attribute
   condition to the `supreplabs/langchain-finic` repository and trusted branch or
   workflow, and allow that identity to impersonate the dedicated service account.
   Follow [Google's deployment pipeline guide](https://docs.cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines)
   and [GitHub's OIDC guide](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-google-cloud-platform).
3. In Search Console's Users and permissions for the URL-prefix property
   `https://supreplabs.github.io/langchain-finic/`, add that service account as a
   user with the least access that permits the three read operations. Confirm URL
   Inspection, sitemap, and Search Analytics access with a manual run; do not grant
   owner access merely for convenience.
4. In GitHub repository **Settings → Secrets and variables → Actions → Variables**,
   set `SEARCH_CONSOLE_WIF_PROVIDER` to the full workload identity provider resource
   name and `SEARCH_CONSOLE_SERVICE_ACCOUNT` to the service account email.
5. Run **Search visibility → Run workflow** on `main`. Verify that the run succeeds
   and a report appears on the tracking issue. Google may return no performance rows
   while data is processing; this is recorded as ambiguous, not as zero traffic.

These resources and property permissions must be configured by an authorized Google
Cloud and Search Console administrator. Without both GitHub variables, the workflow
skips the API and comment steps and records the missing configuration in the run
summary. Authentication failures fail the run. Individual API failures are reported
as unavailable without exposing response bodies. Schedule runs may be delayed.

The API reads the indexed version of a page; it does not perform a live URL test or
request indexing. It does not return the Pages report's total indexed count. Search
Analytics may omit data and be delayed. The workflow does not measure AI Mode or
Google search rankings; maintainers can add clearly labeled, occasional manual
observations to the tracking issue. Use consistent query, region, and signed-in
status, and do not treat a single result as representative. Compare similar time
windows, and don't infer causality from simultaneous changes. See Google's
[AI search guidance](https://developers.google.com/search/docs/appearance/ai-features).
