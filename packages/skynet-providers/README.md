# skynet-providers

One call site for every Skynet service that talks to an OpenAI-compatible
`/chat/completions` endpoint. Routes the API key from Vault based on the URL
so the caller never hard-codes provider-specific key lookups.

```python
from skynet_providers import chat_completion

text = chat_completion(
    prompt="привіт",
    model="mistral-large-latest",
    api_url="https://api.mistral.ai/v1",
)
```

Provider registry (URL substring → Vault KV path):

| URL contains | Vault path | secret key |
|---|---|---|
| `api.mistral.ai` | `mistral/api-key` | `api_key` |
| `openrouter.ai` | `openrouter` | `api_key` |
| `skynet-cache` | `openrouter` | `api_key` |
| `localhost` / `127.0.0.1` / `100.64.0.*` / `:11434` | _no auth_ | — |

Callers that already have a key (tests, CLI scripts) pass `api_key=...` and
skip Vault entirely.
