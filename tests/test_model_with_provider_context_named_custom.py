"""Tests for model_with_provider_context() on NAMED custom providers.

A model picked from a named custom provider that is not the active one (e.g.
``custom:omni`` while ``model.provider`` is ``custom:llm-proxy``) must keep its
``@custom:<slug>:`` qualifier even when the model id contains '/'. Without it
the bare id falls through to the ACTIVE provider and 400s with
"Invalid model format or no credentials for provider: <bare id>".
"""
import api.config as config
import pytest


PROXY_CFG = {
    "default": "x-ai/grok-4.5",
    "provider": "custom:llm-proxy",
    "base_url": "https://proxy.example/v1",
}

CUSTOM_PROVIDERS = [
    {
        "name": "llm-proxy",
        "base_url": "https://proxy.example/v1",
        "key_env": "LLM_PROXY_API_KEY",
    },
    {
        "name": "omni",
        "base_url": "https://omni.example/v1",
        "key_env": "OMNI_API_KEY",
    },
]


def _with_custom_provider_config(fn):
    """Run ``fn()`` against the named-custom-provider config, then restore cfg."""
    old_cfg = dict(config.cfg)
    config.cfg["model"] = dict(PROXY_CFG)
    config.cfg["custom_providers"] = [dict(e) for e in CUSTOM_PROVIDERS]
    config.cfg.pop("providers", None)
    try:
        return fn()
    finally:
        config.cfg.clear()
        config.cfg.update(old_cfg)


def test_slash_id_on_non_active_named_custom_provider_keeps_qualifier():
    """#7346: slash id + non-active named custom provider must stay routable."""
    runtime_model, resolved = _with_custom_provider_config(
        lambda: (
            config.model_with_provider_context("infrex/glm-5.3-flash", "custom:omni"),
            config.resolve_model_provider(
                config.model_with_provider_context("infrex/glm-5.3-flash", "custom:omni")
            ),
        )
    )

    assert runtime_model == "@custom:omni:infrex/glm-5.3-flash"

    model, provider, base_url = resolved
    assert model == "infrex/glm-5.3-flash"
    assert provider == "custom:omni"
    # base_url resolution for a non-active named custom provider is #6516's
    # scope; this test pins the encoding/routing decision only, so it stays
    # green regardless of whether that companion fix has landed.


def test_slash_id_on_active_named_custom_provider_stays_bare():
    """The active provider keeps its bare passthrough (base_url/proxy settings)."""
    runtime_model = _with_custom_provider_config(
        lambda: config.model_with_provider_context("x-ai/grok-4.5", "custom:llm-proxy")
    )

    assert runtime_model == "x-ai/grok-4.5"


def test_slash_id_with_unknown_custom_slug_stays_bare():
    """Fail closed: an unmatched slug must not get a fabricated qualifier."""
    runtime_model = _with_custom_provider_config(
        lambda: config.model_with_provider_context("vendor/model-x", "custom:ghost")
    )

    assert runtime_model == "vendor/model-x"


def test_slash_id_colliding_custom_slug_still_raises_in_resolver():
    """A colliding slug keeps its qualifier here, so resolve_model_provider
    stays the single fail-closed boundary — pinned on the slash-id path this
    fix newly makes reachable (#7346)."""
    old_cfg = dict(config.cfg)
    config.cfg["model"] = dict(PROXY_CFG)
    config.cfg["custom_providers"] = [
        {"name": "Foo Bar", "base_url": "https://a.example/v1"},
        {"name": "foo-bar", "base_url": "https://b.example/v1"},
    ]
    config.cfg.pop("providers", None)
    try:
        encoded = config.model_with_provider_context("vendor/m", "custom:foo-bar")

        assert encoded == "@custom:foo-bar:vendor/m"
        with pytest.raises(config.AmbiguousCustomProviderError):
            config.resolve_model_provider(encoded)
    finally:
        config.cfg.clear()
        config.cfg.update(old_cfg)


def test_bare_custom_vendor_proxy_slash_id_unchanged():
    """Bare 'custom' (vendor-routing proxy, #3872) is deliberately excluded from
    the named-provider branch: its slash ids keep the final bare fallback."""
    old_cfg = dict(config.cfg)
    config.cfg["model"] = {
        "default": "vendor/model-x", "provider": "custom",
        "base_url": "https://proxy.example/v1",
    }
    config.cfg["custom_providers"] = [dict(e) for e in CUSTOM_PROVIDERS]
    config.cfg.pop("providers", None)
    try:
        runtime_model = config.model_with_provider_context("vendor/model-x", "custom")
    finally:
        config.cfg.clear()
        config.cfg.update(old_cfg)

    assert runtime_model == "vendor/model-x"


def test_non_custom_provider_slash_id_encoding_unchanged():
    """Configured non-custom providers keep their existing '@provider:' hint."""
    old_cfg = dict(config.cfg)
    config.cfg["model"] = {"provider": "openai-codex", "default": "gpt-5.5"}
    config.cfg["providers"] = {
        "llama-cpp": {"base_url": "http://127.0.0.1:8088/v1", "api_key": "test-key"},
    }
    config.cfg.pop("custom_providers", None)
    try:
        runtime_model = config.model_with_provider_context(
            "unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL",
            "llama-cpp",
        )
    finally:
        config.cfg.clear()
        config.cfg.update(old_cfg)

    assert runtime_model == "@llama-cpp:unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL"


def test_plain_id_on_non_active_named_custom_provider_still_qualifies():
    """Regression guard: no-slash ids already qualified and must keep doing so."""
    runtime_model = _with_custom_provider_config(
        lambda: config.model_with_provider_context("plain-model", "custom:omni")
    )

    assert runtime_model == "@custom:omni:plain-model"
