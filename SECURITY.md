# Security

Security fixes target the latest published alpha. Older pre-releases are not maintained.

Use [GitHub private vulnerability reporting](https://github.com/supreplabs/langchain-finic/security/advisories/new).
Include affected versions, reproduction steps, impact, and a minimal example with fake
credentials and synthetic data. Do not disclose vulnerabilities through public issues.

The library sends request state and questions to Finic. Host-enabled LangSmith tracing
can also record inputs and responses. The application controls consent, access checks,
redaction, tracing, retention, and environment loading. Secrets are excluded from
configuration representations and serialized values. Load serialized objects only from
trusted sources. Tool screening is probabilistic and does not grant authorization.
