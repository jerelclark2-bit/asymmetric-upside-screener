# Security policy

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's private vulnerability-reporting feature for this repository instead.

Include the affected file or component, reproduction steps, potential impact, and any suggested mitigation. Reports will be reviewed as quickly as practical.

## Scope

The application is designed as local research software. Its bundled launcher binds Streamlit to `127.0.0.1`. Users who expose the server to a network are responsible for adding appropriate authentication, TLS, request limits, and network controls.

