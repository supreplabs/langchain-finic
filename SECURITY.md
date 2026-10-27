# Security

During alpha and beta, security fixes target the latest published prerelease;
older prereleases are not maintained. Once a final `0.1.x` series is published,
fixes target its latest patch while that series is maintained. Users should upgrade
to that patch; fixes are not backported to every earlier patch or prerelease.

Changes to the supported series will be announced in the changelog. There is no
fixed LTS period or response-time SLA. See the
[compatibility and deprecation policy](docs/compatibility.md) for API-change notices.

Use [GitHub private vulnerability reporting](https://github.com/supreplabs/langchain-finic/security/advisories/new).
Include affected versions, reproduction steps, impact, and a minimal example with fake
credentials and synthetic data. Do not disclose vulnerabilities through public issues.

The library sends request state and questions to Finic. Host-enabled LangSmith tracing
can also record inputs and responses. The application controls consent, access checks,
redaction, tracing, retention, and environment loading. The configured API key is excluded from
configuration representations and serialized values. Request state and caller metadata
are not automatically redacted. Load serialized objects only from
trusted sources. Tool screening is probabilistic and does not grant authorization.
