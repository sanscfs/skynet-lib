"""Exception hierarchy for skynet-providers.

ProviderError
└── ProviderAuthError  -- Vault lookup missing / 401 from upstream
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base exception for any skynet-providers failure."""


class ProviderAuthError(ProviderError):
    """No API key available for the resolved provider.

    Raised when the URL dispatch returned an empty key AND the endpoint
    is not a known local (Ollama) URL. Callers typically cannot recover
    from this without a config change or Vault fix.
    """
