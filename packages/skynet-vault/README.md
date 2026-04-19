# skynet-vault

Kubernetes-ServiceAccount-authenticated Vault client for Skynet
services. Replaces the ~50-70 LOC of copy-pasted `hvac` boilerplate
that previously lived in `skynet-agent/vault_secrets.py` and
`skynet-movies/vault_client.py`.

## What it does

- Authenticates to Vault using the pod's K8s ServiceAccount JWT
  (`/var/run/secrets/kubernetes.io/serviceaccount/token`) via the
  `kubernetes` auth method.
- Reads static KV-v2 secrets (`secret/openrouter`,
  `secret/matrix/bot`, etc.).
- Issues dynamic PostgreSQL/database credentials and caches each
  lease per role, refreshing when within 5 min of expiry.
- Re-authenticates once on HTTP 403, retries once, then gives up.

## Install

```
pip install --no-cache-dir \
    --extra-index-url http://nexus.nexus.svc:8081/repository/pypi-group/simple/ \
    --trusted-host nexus.nexus.svc \
    skynet-vault
```

## Environment variables

| Var | Default | Required? |
|-----|---------|-----------|
| `VAULT_ADDR` | `http://vault.vault.svc:8200` | no |
| `VAULT_ROLE` | (none) | yes, for module-level helpers |
| `VAULT_K8S_TOKEN_PATH` | `/var/run/secrets/kubernetes.io/serviceaccount/token` | no |

## Usage -- explicit client

```python
from skynet_vault import VaultClient

vault = VaultClient(addr="http://vault.vault.svc:8200", role="skynet-movies")
vault.authenticate()

api_key = vault.read_kv("secret/openrouter", "api_key")
matrix_token = vault.read_kv("secret/matrix/bot", "access_token")

creds = vault.get_db_creds("skynet-movies-role")
# creds.username, creds.password, creds.lease_duration, creds.expires_at
if creds.needs_renewal():
    creds = vault.get_db_creds("skynet-movies-role", force_refresh=True)
```

## Usage -- module-level (env-driven singleton)

```python
from skynet_vault import get_secret, get_dynamic_db_creds

api_key = get_secret("secret/openrouter", "api_key")
creds = get_dynamic_db_creds("skynet-movies-role")
```

The singleton reads `VAULT_ADDR` and `VAULT_ROLE` from the process
environment. Use the explicit `VaultClient` constructor when one
process needs multiple Vault roles.

## Exceptions

All errors derive from `VaultError`:

- `VaultAuthError` -- SA token missing, Vault rejected the JWT, etc.
- `VaultSecretNotFound` -- KV path or key does not exist.
- `VaultDBCredsError` -- database engine refused to issue creds.
- `VaultConfigError` -- required env var (`VAULT_ROLE`) missing.

No secret material is ever embedded in exception messages or logs.
