"""Regression tests for #1806 named custom provider routing.

The WebUI must treat ``model.provider: <custom_providers[].name>`` as the
same provider slug the picker emits: ``custom:<name>``.  Otherwise a stale
agent-side base-url slug such as ``custom:local-(127.0.0.1:11434)`` can win
model selection and send runtime auth down an impossible env-var path.
"""

from __future__ import annotations

import copy
import json
import sys
import types

import pytest

import api.config as config


@pytest.fixture(autouse=True)
def _isolate_models_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_models_cache_path", tmp_path / "models_cache.json")
    config.invalidate_models_cache()
    yield
    config.invalidate_models_cache()


def _with_ollama_local_config():
    old_cfg = dict(config.cfg)
    old_mtime = config._cfg_mtime
    old_path = getattr(config, "_cfg_path", None)
    config.cfg.clear()
    config.cfg.update(
        {
            "model": {
                "default": "carnice-9b:latest",
                "provider": "ollama-local",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "ollama",
            },
            "custom_providers": [
                {
                    "name": "ollama-local",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "api_key": "ollama",
                    "model": "carnice-9b:latest",
                }
            ],
        }
    )
    try:
        config._cfg_mtime = config.Path(config._get_config_path()).stat().st_mtime
    except Exception:
        config._cfg_mtime = 0.0
    config._cfg_path = config._get_config_path()

    def restore():
        config.cfg.clear()
        config.cfg.update(old_cfg)
        config._cfg_mtime = old_mtime
        config._cfg_path = old_path
        config.invalidate_models_cache()

    return restore


def test_model_provider_name_resolves_to_named_custom_slug():
    restore = _with_ollama_local_config()
    try:
        model, provider, base_url = config.resolve_model_provider("carnice-9b:latest")
    finally:
        restore()

    assert model == "carnice-9b:latest"
    assert provider == "custom:ollama-local"
    assert base_url == "http://127.0.0.1:11434/v1"


def test_available_models_drops_base_url_derived_custom_slug(monkeypatch):
    """A stale agent catalog slug must not create a second local custom group."""
    fake_models = types.ModuleType("hermes_cli.models")
    fake_models.list_available_providers = lambda: [
        {"id": "custom:local-(127.0.0.1:11434)", "authenticated": True},
    ]
    fake_auth = types.ModuleType("hermes_cli.auth")
    fake_auth.get_auth_status = lambda _pid: {"key_source": "config_yaml"}
    monkeypatch.setitem(sys.modules, "hermes_cli.models", fake_models)
    monkeypatch.setitem(sys.modules, "hermes_cli.auth", fake_auth)
    monkeypatch.setattr(config, "_get_auth_store_path", lambda: config.Path("/tmp/does-not-exist-auth.json"))
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [])

    class _Resp:
        def read(self):
            return json.dumps(
                {"data": [{"id": "carnice-9b:latest", "name": "carnice-9b:latest"}]}
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())

    restore = _with_ollama_local_config()
    try:
        result = config.get_available_models()
    finally:
        restore()

    assert result["active_provider"] == "custom:ollama-local"
    groups_by_id = {g["provider_id"]: g for g in result["groups"]}
    assert "custom:ollama-local" in groups_by_id
    assert "custom:local-(127.0.0.1:11434)" not in groups_by_id
    assert "ollama-local" not in groups_by_id

    named_models = [m["id"] for m in groups_by_id["custom:ollama-local"]["models"]]
    assert "carnice-9b:latest" in named_models


def _with_multi_custom_provider_config():
    """Active custom provider PLUS a second, non-active named custom provider.

    Mirrors the config.yaml shape that has no ``providers:`` map at all: every
    endpoint lives in ``custom_providers:``, and only one of them is the active
    ``model.provider``.
    """
    old_cfg = dict(config.cfg)
    old_mtime = config._cfg_mtime
    old_path = getattr(config, "_cfg_path", None)
    config.cfg.clear()
    config.cfg.update(
        {
            "model": {
                "default": "active/model",
                "provider": "custom:active",
                "base_url": "https://active.example/v1",
                "api_key": "active-key",
            },
            "custom_providers": [
                {
                    "name": "active",
                    "base_url": "https://active.example/v1",
                    "api_key": "active-key",
                },
                {
                    "name": "omni",
                    "base_url": "https://omni.example/v1",
                    "api_key": "omni-key",
                },
            ],
        }
    )
    try:
        config._cfg_mtime = config.Path(config._get_config_path()).stat().st_mtime
    except Exception:
        config._cfg_mtime = 0.0
    config._cfg_path = config._get_config_path()

    def restore():
        config.cfg.clear()
        config.cfg.update(old_cfg)
        config._cfg_mtime = old_mtime
        config._cfg_path = old_path
        config.invalidate_models_cache()

    return restore


def test_provider_qualified_id_uses_nonactive_custom_provider_base_url():
    """``@custom:omni:<model>`` must route to omni's base_url, not the default.

    ``_get_provider_base_url`` only reads ``providers:`` and the ACTIVE
    ``model.base_url``, so a named custom provider that lives solely in
    ``custom_providers:`` used to resolve to base_url=None. The WebUI then sent
    the bare model to the active endpoint and got back HTTP 400 "Invalid model
    format or no credentials for provider: <bare-model>".
    """
    restore = _with_multi_custom_provider_config()
    try:
        model, provider, base_url = config.resolve_model_provider(
            "@custom:omni:antigravity/gemini-3.7-flash-tiered"
        )
    finally:
        restore()

    assert model == "antigravity/gemini-3.7-flash-tiered"
    assert provider == "custom:omni"
    assert base_url == "https://omni.example/v1"


def test_provider_qualified_unknown_custom_slug_keeps_base_url_none():
    """An UNKNOWN ``custom:`` slug must stay base_url=None -- never a guess.

    Slugs derived from a base-url authority (``custom:local-(127.0.0.1:11434)``)
    or from a provider that is simply not in ``custom_providers:`` have no
    endpoint of their own. Guessing one (e.g. "there is only one custom provider,
    use it" or "fall back to the active ``model.base_url``") would persist a
    stale endpoint for that slug, which is the #4728 regression. Preserve the
    prior behaviour: no unique matching entry -> no base_url.
    """
    restore = _with_multi_custom_provider_config()
    try:
        model, provider, base_url = config.resolve_model_provider(
            "@custom:not-configured:qwen/qwen-1.5b"
        )
    finally:
        restore()

    assert model == "qwen/qwen-1.5b"
    assert provider == "custom:not-configured"
    assert base_url is None


def test_provider_qualified_active_custom_slug_still_resolves():
    """The ACTIVE custom provider keeps resolving to its own endpoint."""
    restore = _with_multi_custom_provider_config()
    try:
        model, provider, base_url = config.resolve_model_provider(
            "@custom:active:antigravity/gemini-3.7-flash-tiered"
        )
    finally:
        restore()

    assert model == "antigravity/gemini-3.7-flash-tiered"
    assert provider == "custom:active"
    assert base_url == "https://active.example/v1"


def test_provider_qualified_non_custom_provider_is_unaffected():
    """A non-``custom:`` @provider hint must not pick up a custom endpoint.

    The custom_providers lookup is gated on the ``custom:`` prefix, so an
    @openrouter route still resolves through _get_provider_base_url() -- None
    here, since openrouter is neither the active provider nor in ``providers:``.
    """
    restore = _with_multi_custom_provider_config()
    try:
        model, provider, base_url = config.resolve_model_provider(
            "@openrouter:anthropic/claude-sonnet-4.6"
        )
    finally:
        restore()

    assert model == "anthropic/claude-sonnet-4.6"
    assert provider == "openrouter"
    assert base_url is None


def _with_keyed_and_list_provider_config():
    """Same slug present BOTH as a ``providers:`` key and a ``custom_providers`` entry.

    Deployments that started on the legacy ``custom_providers:`` list and later
    gained a keyed ``providers:`` map can carry two records for one slug, each
    with its own ``base_url``. The other fixtures in this file deliberately model
    the list-only shape, so this one pins which record wins.
    """
    old_cfg = dict(config.cfg)
    old_mtime = config._cfg_mtime
    old_path = getattr(config, "_cfg_path", None)
    config.cfg.clear()
    config.cfg.update(
        {
            "model": {
                "default": "active/model",
                "provider": "custom:active",
                "base_url": "https://active.example/v1",
                "api_key": "active-key",
            },
            "providers": {
                "custom:omni": {
                    "base_url": "https://omni-keyed.example/v1",
                    "api_key": "omni-keyed-key",
                },
            },
            "custom_providers": [
                {
                    "name": "active",
                    "base_url": "https://active.example/v1",
                    "api_key": "active-key",
                },
                {
                    "name": "omni",
                    "base_url": "https://omni-list.example/v1",
                    "api_key": "omni-list-key",
                },
            ],
        }
    )
    try:
        config._cfg_mtime = config.Path(config._get_config_path()).stat().st_mtime
    except Exception:
        config._cfg_mtime = 0.0
    config._cfg_path = config._get_config_path()

    def restore():
        config.cfg.clear()
        config.cfg.update(old_cfg)
        config._cfg_mtime = old_mtime
        config._cfg_path = old_path
        config.invalidate_models_cache()

    return restore


def test_list_entry_wins_over_keyed_providers_entry_for_same_slug():
    """``custom_providers[]`` outranks a same-slug ``providers:`` key.

    The named entry is the record the picker's ``custom:<slug>`` id is MINTED
    from (``_custom_provider_slug_from_name`` reads ``custom_providers[].name``),
    and it is the one credential resolution scans, so the endpoint must come from
    that same entry -- otherwise a stale keyed ``providers:`` leftover could pair
    entry A's URL with entry B's API key. Guards the precedence of the
    ``custom_base_url if custom_base_url is not None else _get_provider_base_url()``
    ordering: both lookups return a URL here, so a flipped order would silently
    route to ``omni-keyed`` instead.
    """
    restore = _with_keyed_and_list_provider_config()
    try:
        # Sanity: the keyed entry really is resolvable, so this test would fail
        # (not merely pass vacuously on a None fallback) if precedence flipped.
        keyed_base_url = config._get_provider_base_url("custom:omni")
        model, provider, base_url = config.resolve_model_provider(
            "@custom:omni:antigravity/gemini-3.7-flash-tiered"
        )
        conn_api_key, conn_base_url = config.resolve_custom_provider_connection("custom:omni")
    finally:
        restore()

    assert keyed_base_url == "https://omni-keyed.example/v1"
    assert model == "antigravity/gemini-3.7-flash-tiered"
    assert provider == "custom:omni"
    assert base_url == "https://omni-list.example/v1"
    assert conn_api_key == "omni-list-key"
    assert conn_base_url == "https://omni-list.example/v1"
    assert (base_url, conn_api_key) == ("https://omni-list.example/v1", "omni-list-key")


def _with_keyed_and_blank_list_provider_config():
    """Same slug present in ``providers:`` and as a ``custom_providers`` entry with blank base_url.

    The list entry exists and has a distinct API key, but its ``base_url`` is
    empty. The keyed ``providers:`` entry has both a valid ``base_url`` and its
    own distinct API key.
    """
    old_cfg = dict(config.cfg)
    old_mtime = config._cfg_mtime
    old_path = getattr(config, "_cfg_path", None)
    config.cfg.clear()
    config.cfg.update(
        {
            "model": {
                "default": "active/model",
                "provider": "custom:active",
                "base_url": "https://active.example/v1",
                "api_key": "active-key",
            },
            "providers": {
                "custom:omni": {
                    "base_url": "https://omni-keyed.example/v1",
                    "api_key": "omni-keyed-key",
                },
            },
            "custom_providers": [
                {
                    "name": "active",
                    "base_url": "https://active.example/v1",
                    "api_key": "active-key",
                },
                {
                    "name": "omni",
                    "base_url": "",
                    "api_key": "omni-list-key",
                },
            ],
        }
    )
    try:
        config._cfg_mtime = config.Path(config._get_config_path()).stat().st_mtime
    except Exception:
        config._cfg_mtime = 0.0
    config._cfg_path = config._get_config_path()

    def restore():
        config.cfg.clear()
        config.cfg.update(old_cfg)
        config._cfg_mtime = old_mtime
        config._cfg_path = old_path
        config.invalidate_models_cache()

    return restore


def test_blank_list_entry_does_not_fall_through_to_keyed_providers_endpoint():
    """A blank ``custom_providers[].base_url`` must not fall through to a keyed endpoint.

    When a slug exists both in ``custom_providers[]`` (with a blank or empty
    base_url and API key K_list) and in ``providers:`` (with a populated base_url
    and API key K_keyed), resolution must treat the list entry as authoritative.
    Falling through to ``_get_provider_base_url()`` on empty base_url would pair
    the keyed record's endpoint with the list record's API key, violating the
    same-entry invariant between resolve_model_provider() and
    resolve_custom_provider_connection().
    """
    restore = _with_keyed_and_blank_list_provider_config()
    try:
        keyed_base_url = config._get_provider_base_url("custom:omni")
        model, provider, base_url = config.resolve_model_provider(
            "@custom:omni:antigravity/gemini-3.7-flash-tiered"
        )
        conn_api_key, conn_base_url = config.resolve_custom_provider_connection("custom:omni")
    finally:
        restore()

    assert keyed_base_url == "https://omni-keyed.example/v1"
    assert model == "antigravity/gemini-3.7-flash-tiered"
    assert provider == "custom:omni"
    assert base_url is None
    assert conn_base_url is None
    assert conn_api_key == "omni-list-key"
    assert (base_url, conn_api_key) != ("https://omni-keyed.example/v1", "omni-list-key")


def _setup_production_composed_runtime(
    monkeypatch,
    cfg_dict,
    runtime_dict,
    session_id="test-session-1806",
    fail_first=None,
):
    """Compose the production streaming send path around a capturing agent.

    ``fail_first`` drives the two 401 self-heal retry paths that rebuild the
    runtime bundle a second time:

    * ``"returned_error"`` -- the first ``run_conversation`` RETURNS an auth
      error without raising (``api/streaming.py`` returned-error retry).
    * ``"raised"`` -- the first ``run_conversation`` RAISES it
      (``api/streaming.py`` raised-exception retry).

    Both make ``_attempt_credential_self_heal`` hand back the same ambient
    runtime dict production would re-resolve, so the retry sees the identical
    truthy side-field sentinels the initial send did.
    """
    from unittest import mock
    import queue
    import api.streaming as streaming
    import api.oauth

    with config.SESSION_AGENT_CACHE_LOCK:
        config.SESSION_AGENT_CACHE.clear()

    old_cfg = dict(config.cfg)
    old_mtime = config._cfg_mtime
    old_path = getattr(config, "_cfg_path", None)
    config.cfg.clear()
    config.cfg.update(cfg_dict)
    try:
        config._cfg_mtime = config.Path(config._get_config_path()).stat().st_mtime
    except Exception:
        config._cfg_mtime = 0.0
    config._cfg_path = config._get_config_path()

    class FakeSession:
        def __init__(self):
            self.session_id = session_id
            self.title = "Test"
            self.workspace = "/tmp"
            self.model = "test-model"
            self.messages = []
            self.personality = None
            self.input_tokens = 0
            self.output_tokens = 0
            self.estimated_cost = None
            self.tool_calls = []
            self.active_stream_id = None
            self.pending_user_message = None
            self.pending_attachments = []
            self.pending_started_at = None
            self.pending_user_source = None
            self.profile = None

        def save(self, touch_updated_at=True, skip_index=False):
            self._saved = touch_updated_at

        def compact(self):
            return {"session_id": self.session_id, "messages": self.messages}

    captured = {}

    class CapturingAgent:
        # Every constructor-routing field must be a NAMED parameter. The
        # streaming path gates each optional kwarg on
        # ``inspect.signature(AIAgent.__init__)``, so a double that only accepts
        # model/provider/base_url/api_key silently filters the runtime-owned
        # fields out — hiding exactly the stale-authority defect these tests pin.
        def __init__(
            self,
            model=None,
            provider=None,
            base_url=None,
            api_key=None,
            api_mode=None,
            acp_command=None,
            acp_args=None,
            credential_pool=None,
            **kwargs,
        ):
            captured["init_kwargs"] = {
                "model": model,
                "provider": provider,
                "base_url": base_url,
                "api_key": api_key,
                "api_mode": api_mode,
                "acp_command": acp_command,
                "acp_args": acp_args,
                "credential_pool": credential_pool,
                **kwargs,
            }
            captured.setdefault("init_kwargs_history", []).append(
                dict(captured["init_kwargs"])
            )
            self.session_id = kwargs.get("session_id")
            self.context_compressor = None
            self.session_prompt_tokens = 0
            self.session_completion_tokens = 0
            self.session_estimated_cost_usd = None
            self.reasoning_config = None
            self.ephemeral_system_prompt = None
            self._last_error = None

        def run_conversation(self, **kwargs):
            captured["run_kwargs"] = kwargs
            captured["run_calls"] = captured.get("run_calls", 0) + 1
            if fail_first is not None and captured["run_calls"] == 1:
                if fail_first == "raised":
                    raise RuntimeError("401 Unauthorized")
                return {"messages": [], "error": "401 Unauthorized"}
            return {
                "messages": [
                    {"role": "user", "content": kwargs.get("persist_user_message", "")},
                    {"role": "assistant", "content": "ok"},
                ]
            }

        def interrupt(self, _message):
            captured["interrupted"] = _message

    fake_session = FakeSession()
    fake_stream_id = f"stream-{session_id}"
    if fail_first == "returned_error":
        # The returned-error branch lives behind the stale-writeback guard, which
        # only lets the owning worker persist. /api/chat/start stamps
        # ``active_stream_id`` before dispatching the worker, so model that here
        # or the guard returns before the retry is ever reached.
        fake_session.active_stream_id = fake_stream_id
    fake_queue = queue.Queue()

    fake_runtime_module = types.ModuleType("hermes_cli.runtime_provider")
    fake_runtime_module.resolve_runtime_provider = mock.Mock(return_value=dict(runtime_dict))
    fake_hermes_cli = types.ModuleType("hermes_cli")
    fake_hermes_cli.runtime_provider = fake_runtime_module
    fake_hermes_state = types.ModuleType("hermes_state")
    fake_hermes_state.SessionDB = mock.Mock(return_value=object())

    monkeypatch.setitem(sys.modules, "hermes_cli", fake_hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.runtime_provider", fake_runtime_module)
    monkeypatch.setitem(sys.modules, "hermes_state", fake_hermes_state)

    monkeypatch.setattr(
        api.oauth,
        "resolve_runtime_provider_with_anthropic_env_lock",
        lambda resolver, **kwargs: resolver(**kwargs),
    )
    monkeypatch.setattr(streaming, "get_session", lambda _session_id: fake_session)
    monkeypatch.setattr(streaming, "_get_ai_agent", lambda: CapturingAgent)
    monkeypatch.setattr("api.config.get_config", lambda: dict(config.cfg))
    monkeypatch.setattr("api.config._resolve_cli_toolsets", lambda *_args, **_kwargs: [])
    if fail_first is not None:
        # Production re-resolves the ambient runtime provider on a 401 heal, so
        # the retry must see the same truthy side-field sentinels — that is
        # exactly the state in which a partial rebuild leaks them through.
        monkeypatch.setattr(
            streaming,
            "_attempt_credential_self_heal",
            lambda *_args, **_kwargs: dict(runtime_dict),
        )

    def restore():
        with config.SESSION_AGENT_CACHE_LOCK:
            config.SESSION_AGENT_CACHE.clear()
        config.cfg.clear()
        config.cfg.update(old_cfg)
        config._cfg_mtime = old_mtime
        config._cfg_path = old_path
        config.invalidate_models_cache()

    return fake_stream_id, fake_queue, captured, restore


def test_production_composed_nonblank_exact_list_row_yields_list_url_and_list_key(monkeypatch):
    """(a) Nonblank exact list row plus distinct same-slug keyed row yields list URL/list key.

    In production, resolve_runtime_provider() scans providers: first and returns
    the keyed URL and key. When an exact custom_providers[] entry matches, the
    runtime send path must atomically replace both fields so that final AIAgent
    construction receives list endpoint + list key, never keyed key.
    """
    import api.streaming as streaming

    cfg_dict = {
        "model": {"default": "active/model", "provider": "custom:active"},
        "providers": {
            "custom:omni": {
                "base_url": "https://keyed-url-sentinel.example/v1",
                "api_key": "keyed-key-sentinel-abc",
            },
        },
        "custom_providers": [
            {
                "name": "omni",
                "base_url": "https://list-url-sentinel.example/v1",
                "api_key": "list-key-sentinel-xyz",
            },
        ],
    }
    runtime_dict = {
        "provider": "custom:omni",
        "base_url": "https://keyed-url-sentinel.example/v1",
        "api_key": "keyed-key-sentinel-abc",
    }
    stream_id, q, captured, restore = _setup_production_composed_runtime(
        monkeypatch, cfg_dict, runtime_dict, session_id="session-1806-nonblank"
    )
    try:
        streaming.STREAMS[stream_id] = q
        streaming._run_agent_streaming(
            session_id="session-1806-nonblank",
            msg_text="hello",
            model="@custom:omni:antigravity/gemini-3.7-flash-tiered",
            workspace="/tmp",
            stream_id=stream_id,
        )
    finally:
        streaming.STREAMS.pop(stream_id, None)
        streaming.AGENT_INSTANCES.pop(stream_id, None)
        restore()

    init_kwargs = captured["init_kwargs"]
    assert init_kwargs["base_url"] == "https://list-url-sentinel.example/v1"
    assert init_kwargs["api_key"] == "list-key-sentinel-xyz"
    assert init_kwargs["provider"] == "custom"
    assert init_kwargs["base_url"] != "https://keyed-url-sentinel.example/v1"
    assert init_kwargs["api_key"] != "keyed-key-sentinel-abc"


def test_production_composed_blank_exact_list_row_cannot_repopulate_from_keyed_row(monkeypatch):
    """(b) Blank exact list row cannot repopulate from the keyed row.

    When an exact custom_providers[] entry exists with an empty base_url, that
    row is authoritative. Final construction must receive base_url=None and the
    list key; it must NOT fall through to or repopulate the keyed base_url or
    keyed API key.
    """
    import api.streaming as streaming

    cfg_dict = {
        "model": {"default": "active/model", "provider": "custom:active"},
        "providers": {
            "custom:omni": {
                "base_url": "https://keyed-url-sentinel.example/v1",
                "api_key": "keyed-key-sentinel-abc",
            },
        },
        "custom_providers": [
            {
                "name": "omni",
                "base_url": "",
                "api_key": "list-key-sentinel-xyz",
            },
        ],
    }
    runtime_dict = {
        "provider": "custom:omni",
        "base_url": "https://keyed-url-sentinel.example/v1",
        "api_key": "keyed-key-sentinel-abc",
    }
    stream_id, q, captured, restore = _setup_production_composed_runtime(
        monkeypatch, cfg_dict, runtime_dict, session_id="session-1806-blank"
    )
    try:
        streaming.STREAMS[stream_id] = q
        streaming._run_agent_streaming(
            session_id="session-1806-blank",
            msg_text="hello",
            model="@custom:omni:antigravity/gemini-3.7-flash-tiered",
            workspace="/tmp",
            stream_id=stream_id,
        )
    finally:
        streaming.STREAMS.pop(stream_id, None)
        streaming.AGENT_INSTANCES.pop(stream_id, None)
        restore()

    init_kwargs = captured["init_kwargs"]
    assert init_kwargs["base_url"] is None
    assert init_kwargs["api_key"] == "list-key-sentinel-xyz"
    assert init_kwargs["base_url"] != "https://keyed-url-sentinel.example/v1"
    assert init_kwargs["api_key"] != "keyed-key-sentinel-abc"


def test_production_composed_keyed_only_still_yields_keyed_url_and_key(monkeypatch):
    """(c) Keyed-only/no-list still yields the keyed URL/key.

    When no matching row exists in custom_providers[], the keyed providers:
    record is authoritative and supplies both base_url and api_key.
    """
    import api.streaming as streaming

    cfg_dict = {
        "model": {"default": "active/model", "provider": "custom:active"},
        "providers": {
            "custom:omni": {
                "base_url": "https://keyed-url-sentinel.example/v1",
                "api_key": "keyed-key-sentinel-abc",
            },
        },
        "custom_providers": [
            {
                "name": "active",
                "base_url": "https://active.example/v1",
                "api_key": "active-key",
            },
        ],
    }
    runtime_dict = {
        "provider": "custom:omni",
        "base_url": "https://keyed-url-sentinel.example/v1",
        "api_key": "keyed-key-sentinel-abc",
    }
    stream_id, q, captured, restore = _setup_production_composed_runtime(
        monkeypatch, cfg_dict, runtime_dict, session_id="session-1806-keyed"
    )
    try:
        streaming.STREAMS[stream_id] = q
        streaming._run_agent_streaming(
            session_id="session-1806-keyed",
            msg_text="hello",
            model="@custom:omni:antigravity/gemini-3.7-flash-tiered",
            workspace="/tmp",
            stream_id=stream_id,
        )
    finally:
        streaming.STREAMS.pop(stream_id, None)
        streaming.AGENT_INSTANCES.pop(stream_id, None)
        restore()

    init_kwargs = captured["init_kwargs"]
    assert init_kwargs["base_url"] == "https://keyed-url-sentinel.example/v1"
    assert init_kwargs["api_key"] == "keyed-key-sentinel-abc"
    assert init_kwargs["provider"] == "custom"


# ─────────────────────────────────────────────────────────────────────────────
# Full-constructor bundle authority (streaming initial send + both retry paths)
#
# The connection fields are only half the constructor contract. AIAgent also
# takes ``credential_pool``, ``api_mode``, ``acp_command`` and ``acp_args`` from
# the resolved runtime provider, and the ambient provider legitimately reports
# all four (it is what the process is otherwise authenticated as). A send that
# replaces provider/base_url/api_key from an exact ``custom_providers[]`` row but
# keeps those four builds an agent whose credential source, wire protocol and
# transport still point at the previous authority — the custom HTTP endpoint gets
# Anthropic credential pooling and a Claude ACP subprocess command.
#
# ``_AMBIENT_SIDE_FIELD_RUNTIME`` seeds every one of them truthy so a
# pass-through is a hard assertion failure rather than a vacuous ``None == None``.
# ─────────────────────────────────────────────────────────────────────────────

_KEYED_VS_LIST_CFG = {
    "model": {"default": "active/model", "provider": "custom:active"},
    "providers": {
        "custom:omni": {
            "base_url": "https://keyed-url-sentinel.example/v1",
            "api_key": "keyed-key-sentinel-abc",
        },
    },
    "custom_providers": [
        {
            "name": "omni",
            "base_url": "https://list-url-sentinel.example/v1",
            "api_key": "list-key-sentinel-xyz",
        },
    ],
}

_AMBIENT_SIDE_FIELD_RUNTIME = {
    "provider": "custom:omni",
    "base_url": "https://keyed-url-sentinel.example/v1",
    "api_key": "keyed-key-sentinel-abc",
    # Runtime-owned constructor fields belonging to the AMBIENT provider.
    "credential_pool": ["ambient-pool-sentinel-1", "ambient-pool-sentinel-2"],
    "api_mode": "anthropic_messages",
    "command": "claude-code-acp-sentinel",
    "args": ["--acp-arg-sentinel"],
}

_LIST_ROW_URL = "https://list-url-sentinel.example/v1"
_LIST_ROW_KEY = "list-key-sentinel-xyz"


# Case (d): every side field in ``_AMBIENT_SIDE_FIELD_RUNTIME`` is genuinely
# FOREIGN to ``_KEYED_VS_LIST_CFG`` -- the list row owns a different endpoint and
# declares none of these for itself, so provenance proves they belong to the
# ambient provider and every one must be cleared. This is an ownership verdict,
# not a blanket "custom routes never carry side fields" rule: the cases below
# pin records that DO own them, and there the same merge must keep them.
_FOREIGN_AMBIENT_SIDE_FIELDS = {
    "api_mode": None,
    "acp_command": None,
    "acp_args": None,
    "credential_pool": None,
}


def _assert_side_fields(init_kwargs, expected, label):
    """Assert each constructor side field equals the authority that OWNS it."""
    for field, value in expected.items():
        actual = init_kwargs[field]
        if callable(value):
            assert actual is value, f"{label}: {field} lost its owner's value"
        else:
            assert actual == value, f"{label}: {field} is {actual!r}, expected {value!r}"


def _assert_exact_list_row_bundle(init_kwargs, label, side_fields=None):
    """Assert the whole constructor bundle is target-owned, not a mixed one.

    ``side_fields`` names what the exact list row's authority resolves each side
    field to; it defaults to case (d), the all-foreign ambient runtime.
    """
    assert init_kwargs["base_url"] == _LIST_ROW_URL, label
    assert init_kwargs["api_key"] == _LIST_ROW_KEY, label
    assert init_kwargs["provider"] == "custom", label
    # None of the keyed/ambient authority may survive anywhere in the bundle.
    assert init_kwargs["base_url"] != "https://keyed-url-sentinel.example/v1", label
    assert init_kwargs["api_key"] != "keyed-key-sentinel-abc", label
    _assert_side_fields(
        init_kwargs,
        _FOREIGN_AMBIENT_SIDE_FIELDS if side_fields is None else side_fields,
        label,
    )


def test_capturing_agent_exposes_every_runtime_constructor_field(monkeypatch):
    """Guard the guard: the double must not signature-filter the side fields.

    ``_run_agent_streaming`` gates each optional kwarg on
    ``inspect.signature(AIAgent.__init__).parameters``. A double that swallowed
    ``credential_pool`` / ``api_mode`` / ``acp_command`` / ``acp_args`` into
    ``**kwargs`` would make every assertion below pass vacuously, because the
    stale values would never be passed at all.
    """
    import inspect

    import api.streaming as streaming

    _stream_id, _q, _captured, restore = _setup_production_composed_runtime(
        monkeypatch, dict(_KEYED_VS_LIST_CFG), {}
    )
    try:
        params = set(inspect.signature(streaming._get_ai_agent().__init__).parameters)
    finally:
        restore()

    for field in ("api_mode", "acp_command", "acp_args", "credential_pool"):
        assert field in params, f"{field} would be signature-filtered out"


def test_production_composed_initial_send_replaces_full_runtime_bundle(monkeypatch):
    """Initial send: exact list row owns the WHOLE constructor bundle."""
    import api.streaming as streaming

    stream_id, q, captured, restore = _setup_production_composed_runtime(
        monkeypatch,
        dict(_KEYED_VS_LIST_CFG),
        dict(_AMBIENT_SIDE_FIELD_RUNTIME),
        session_id="session-1806-bundle-initial",
    )
    try:
        streaming.STREAMS[stream_id] = q
        streaming._run_agent_streaming(
            session_id="session-1806-bundle-initial",
            msg_text="hello",
            model="@custom:omni:antigravity/gemini-3.7-flash-tiered",
            workspace="/tmp",
            stream_id=stream_id,
        )
    finally:
        streaming.STREAMS.pop(stream_id, None)
        streaming.AGENT_INSTANCES.pop(stream_id, None)
        restore()

    _assert_exact_list_row_bundle(captured["init_kwargs"], "initial send")


def test_production_composed_returned_error_retry_replaces_full_runtime_bundle(monkeypatch):
    """Returned-error 401 retry: the rebuilt bundle is target-owned too.

    The retry path rebuilds agent kwargs from the self-healed runtime dict. When
    it only refreshed provider/key/base_url (plus ``credential_pool`` from the
    heal result), the retry agent was constructed with the ambient provider's
    pool/api_mode/ACP fields even though the endpoint is the custom list row.
    """
    import api.streaming as streaming

    stream_id, q, captured, restore = _setup_production_composed_runtime(
        monkeypatch,
        dict(_KEYED_VS_LIST_CFG),
        dict(_AMBIENT_SIDE_FIELD_RUNTIME),
        session_id="session-1806-bundle-returned",
        fail_first="returned_error",
    )
    try:
        streaming.STREAMS[stream_id] = q
        streaming._run_agent_streaming(
            session_id="session-1806-bundle-returned",
            msg_text="hello",
            model="@custom:omni:antigravity/gemini-3.7-flash-tiered",
            workspace="/tmp",
            stream_id=stream_id,
        )
    finally:
        streaming.STREAMS.pop(stream_id, None)
        streaming.AGENT_INSTANCES.pop(stream_id, None)
        restore()

    history = captured["init_kwargs_history"]
    assert len(history) >= 2, "returned-error retry did not construct a second agent"
    for index, init_kwargs in enumerate(history):
        _assert_exact_list_row_bundle(init_kwargs, f"returned-error construction #{index}")


def test_production_composed_raised_exception_retry_replaces_full_runtime_bundle(monkeypatch):
    """Raised-exception 401 retry: same complete-bundle contract."""
    import api.streaming as streaming

    stream_id, q, captured, restore = _setup_production_composed_runtime(
        monkeypatch,
        dict(_KEYED_VS_LIST_CFG),
        dict(_AMBIENT_SIDE_FIELD_RUNTIME),
        session_id="session-1806-bundle-raised",
        fail_first="raised",
    )
    try:
        streaming.STREAMS[stream_id] = q
        streaming._run_agent_streaming(
            session_id="session-1806-bundle-raised",
            msg_text="hello",
            model="@custom:omni:antigravity/gemini-3.7-flash-tiered",
            workspace="/tmp",
            stream_id=stream_id,
        )
    finally:
        streaming.STREAMS.pop(stream_id, None)
        streaming.AGENT_INSTANCES.pop(stream_id, None)
        restore()

    history = captured["init_kwargs_history"]
    assert len(history) >= 2, "raised-exception retry did not construct a second agent"
    for index, init_kwargs in enumerate(history):
        _assert_exact_list_row_bundle(init_kwargs, f"raised-exception construction #{index}")


def test_agent_cache_signature_tracks_the_resolved_bundle(monkeypatch):
    """The cached-agent signature must be derived from the FINAL bundle.

    If ``_sig_blob`` still read ``api_mode`` / ACP / pool off the raw runtime
    provider, two sends whose resolved bundles differ only in a cleared side
    field would hash identically — so the second send would reuse an agent built
    on the previous authority instead of minting a new one.
    """
    import api.streaming as streaming

    def _run(runtime_dict, session_id):
        stream_id, q, captured, restore = _setup_production_composed_runtime(
            monkeypatch, dict(_KEYED_VS_LIST_CFG), runtime_dict, session_id=session_id
        )
        try:
            streaming.STREAMS[stream_id] = q
            streaming._run_agent_streaming(
                session_id=session_id,
                msg_text="hello",
                model="@custom:omni:antigravity/gemini-3.7-flash-tiered",
                workspace="/tmp",
                stream_id=stream_id,
            )
            with config.SESSION_AGENT_CACHE_LOCK:
                return config.SESSION_AGENT_CACHE[session_id][1]
        finally:
            streaming.STREAMS.pop(stream_id, None)
            streaming.AGENT_INSTANCES.pop(stream_id, None)
            restore()

    # Same resolved bundle, wildly different ambient side fields: because the
    # custom row clears them all, the signature must be identical.
    plain_runtime = {
        "provider": "custom:omni",
        "base_url": "https://keyed-url-sentinel.example/v1",
        "api_key": "keyed-key-sentinel-abc",
    }
    sig_plain = _run(plain_runtime, "session-1806-sig-plain")
    sig_ambient = _run(dict(_AMBIENT_SIDE_FIELD_RUNTIME), "session-1806-sig-ambient")
    assert sig_plain == sig_ambient, (
        "signature still varies with runtime fields the bundle cleared"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ownership-aware side fields
#
# "Clear the runtime-owned side fields whenever a custom record supplies the
# connection" over-corrects. ``api_mode``, ``credential_pool``, ``acp_command``
# and ``acp_args`` are not intrinsically ambient: a ``custom_providers[]`` row is
# free to declare ``api_mode: anthropic_messages``, and a keyed
# ``providers['custom:<slug>']`` record is free to declare a pool and an ACP
# transport. Blanket-clearing them downgrades an Anthropic-protocol row to
# chat-completions and drops a keyed record's own pool/transport.
#
# The rule these tests pin is provenance, not field name:
#
#   * the selected record declares the field  -> the record's value wins;
#   * the runtime resolved the SAME endpoint  -> its value is same-authority, keep;
#   * the runtime resolved a DIFFERENT one    -> proven foreign, clear.
#
# The same distinction gates ``dummy-key``: it is a statement that the endpoint
# is UNAUTHENTICATED, so it may only be substituted once the record's whole
# credential ladder (pool, api_key, key_env, CUSTOM_<SLUG>_API_KEY, key_cmd,
# host-gated env) has come up empty. Substituting it over a ``key_cmd`` or a
# pooled credential turns a working endpoint into a 401.
# ─────────────────────────────────────────────────────────────────────────────


def _run_composed_send(monkeypatch, cfg_dict, runtime_dict, session_id, before_send=None):
    """Drive ONE production-composed streaming send; return the constructor kwargs.

    ``before_send`` runs after the fake ``hermes_cli.runtime_provider`` module is
    installed, so a test can hang extra runtime helpers (the credential-pool
    lookup) off it before resolution happens.
    """
    import api.streaming as streaming

    stream_id, q, captured, restore = _setup_production_composed_runtime(
        monkeypatch, cfg_dict, runtime_dict, session_id=session_id
    )
    try:
        if before_send is not None:
            before_send()
        streaming.STREAMS[stream_id] = q
        streaming._run_agent_streaming(
            session_id=session_id,
            msg_text="hello",
            model="@custom:omni:antigravity/gemini-3.7-flash-tiered",
            workspace="/tmp",
            stream_id=stream_id,
        )
    finally:
        streaming.STREAMS.pop(stream_id, None)
        streaming.AGENT_INSTANCES.pop(stream_id, None)
        restore()
    return captured["init_kwargs"]


def _exact_list_row_cfg(*, drop=(), **row_fields):
    """``_KEYED_VS_LIST_CFG`` with the exact ``custom_providers[]`` row extended."""
    cfg_dict = copy.deepcopy(_KEYED_VS_LIST_CFG)
    for field in drop:
        cfg_dict["custom_providers"][0].pop(field, None)
    cfg_dict["custom_providers"][0].update(row_fields)
    return cfg_dict


def _ambient_runtime(**overrides):
    runtime = copy.deepcopy(_AMBIENT_SIDE_FIELD_RUNTIME)
    runtime.update(overrides)
    return runtime


# ── (a) exact-list-owned Anthropic mode ──────────────────────────────────────


@pytest.mark.parametrize(
    "row_fields,session_suffix",
    [
        ({"api_mode": "anthropic_messages"}, "api-mode"),
        # ``transport:`` is the v12-migration spelling and ``anthropic`` an
        # accepted alias; a hand-edited config using either still owns the mode.
        ({"transport": "anthropic"}, "transport-alias"),
    ],
)
def test_exact_list_row_keeps_its_own_anthropic_api_mode(
    monkeypatch, row_fields, session_suffix
):
    """(a) The row declares its wire protocol, so neither clearing nor the ambient wins.

    The ambient runtime reports ``chat_completions`` for a DIFFERENT endpoint.
    Both failure modes are visible here: passing the ambient value through gives
    ``chat_completions``, blanket-clearing gives ``None``, and only reading the
    row's own declaration gives ``anthropic_messages`` — which is what decides
    whether the send speaks /v1/messages or /v1/chat/completions.
    """
    init_kwargs = _run_composed_send(
        monkeypatch,
        _exact_list_row_cfg(**row_fields),
        _ambient_runtime(api_mode="chat_completions"),
        f"session-1806-owned-{session_suffix}",
    )

    _assert_exact_list_row_bundle(
        init_kwargs,
        f"exact list row owns api_mode ({session_suffix})",
        side_fields={**_FOREIGN_AMBIENT_SIDE_FIELDS, "api_mode": "anthropic_messages"},
    )


# ── (b) exact-list key_cmd / pool credential ─────────────────────────────────


def _fake_command_token_source(monkeypatch, build):
    fake_module = types.ModuleType("agent.command_token_source")
    fake_module.build_command_token_provider = build
    monkeypatch.setitem(sys.modules, "agent.command_token_source", fake_module)


def test_exact_list_row_key_cmd_is_not_replaced_by_dummy_key(monkeypatch):
    """(b) A ``key_cmd`` row mints a real per-request bearer, so it is NOT keyless.

    ``key_cmd`` names a command that prints a short-lived bearer; both wire
    clients accept a callable api_key and mint one per request. Handing the
    endpoint ``dummy-key`` instead — because the row carries no literal
    ``api_key`` — is a guaranteed 401 against an endpoint that does want auth.
    """
    built = {}

    def _token_provider():
        return "minted-bearer-sentinel"

    def _build(key_cmd, name):
        built["key_cmd"] = key_cmd
        built["name"] = name
        return _token_provider

    _fake_command_token_source(monkeypatch, _build)

    init_kwargs = _run_composed_send(
        monkeypatch,
        _exact_list_row_cfg(drop=("api_key",), key_cmd="print-omni-bearer"),
        _ambient_runtime(),
        "session-1806-owned-key-cmd",
    )

    assert built["key_cmd"] == "print-omni-bearer", "key_cmd was never consulted"
    assert built["name"] == "omni"
    assert init_kwargs["api_key"] is _token_provider, "row's key_cmd token source lost"
    assert init_kwargs["api_key"]() == "minted-bearer-sentinel"
    assert init_kwargs["api_key"] != config.KEYLESS_CUSTOM_API_KEY
    assert init_kwargs["api_key"] != "keyed-key-sentinel-abc"
    assert init_kwargs["base_url"] == _LIST_ROW_URL
    assert init_kwargs["provider"] == "custom"
    _assert_side_fields(init_kwargs, _FOREIGN_AMBIENT_SIDE_FIELDS, "exact list row key_cmd")


def test_exact_list_row_unbuildable_key_cmd_still_refuses_dummy_key(monkeypatch):
    """(b) ``dummy-key`` asserts "this endpoint is unauthenticated" — never a guess.

    When the token provider cannot be built (older agent build, broken command
    spec) the endpoint is still an authenticated one whose credential is missing.
    Substituting the keyless placeholder would report that as an opaque 401
    instead of the real cause, so the send goes out with no credential at all.
    """

    def _build(_key_cmd, _name):
        raise RuntimeError("command token source unavailable")

    _fake_command_token_source(monkeypatch, _build)

    init_kwargs = _run_composed_send(
        monkeypatch,
        _exact_list_row_cfg(drop=("api_key",), key_cmd="print-omni-bearer"),
        _ambient_runtime(),
        "session-1806-owned-key-cmd-broken",
    )

    assert init_kwargs["api_key"] is None, "keyless placeholder masked a declared key_cmd"
    assert init_kwargs["api_key"] != "keyed-key-sentinel-abc"
    assert init_kwargs["base_url"] == _LIST_ROW_URL


def test_exact_list_row_pool_credential_and_pool_object_both_survive(monkeypatch):
    """(b) A pooled row keeps the pool's key AND the pool object — from ITS endpoint.

    The credential and the ``credential_pool`` the agent rotates it with come
    from one lookup keyed on the ROW's base_url. Clearing the pool (while keeping
    its key) leaves the agent unable to rotate; keeping the ambient pool points
    rotation at the previous authority; falling back to ``dummy-key`` drops the
    credential entirely.
    """
    pool_sentinel = ["list-row-pool-sentinel"]
    seen = {}

    def _before_send():
        runtime_module = sys.modules["hermes_cli.runtime_provider"]

        def _try_resolve_from_custom_pool(
            base_url, provider_label, api_mode_override=None, provider_name=None
        ):
            seen["base_url"] = base_url
            seen["provider_name"] = provider_name
            if base_url != _LIST_ROW_URL:
                return None
            return {"api_key": "pool-key-sentinel", "credential_pool": pool_sentinel}

        runtime_module._try_resolve_from_custom_pool = _try_resolve_from_custom_pool

    init_kwargs = _run_composed_send(
        monkeypatch,
        _exact_list_row_cfg(drop=("api_key",)),
        _ambient_runtime(),
        "session-1806-owned-pool",
        before_send=_before_send,
    )

    # The pool was looked up for the ROW's endpoint, not the ambient one.
    assert seen["base_url"] == _LIST_ROW_URL
    assert seen["provider_name"] == "omni"
    assert init_kwargs["api_key"] == "pool-key-sentinel"
    assert init_kwargs["api_key"] != config.KEYLESS_CUSTOM_API_KEY
    assert init_kwargs["api_key"] != "keyed-key-sentinel-abc"
    _assert_side_fields(
        init_kwargs,
        {**_FOREIGN_AMBIENT_SIDE_FIELDS, "credential_pool": pool_sentinel},
        "exact list row pool",
    )
    assert init_kwargs["credential_pool"] is pool_sentinel
    assert init_kwargs["credential_pool"] != _AMBIENT_SIDE_FIELD_RUNTIME["credential_pool"]


# ── (c) keyed-only record's own side fields ──────────────────────────────────


_KEYED_ONLY_OWNED_CFG = {
    "model": {"default": "active/model", "provider": "custom:active"},
    "providers": {
        "custom:omni": {
            "base_url": "https://keyed-url-sentinel.example/v1",
            "api_key": "keyed-key-sentinel-abc",
            # Side fields the KEYED record declares for itself.
            "api_mode": "anthropic_messages",
            "credential_pool": ["keyed-pool-sentinel"],
            "acp_command": "keyed-acp-sentinel",
            "acp_args": ["--keyed-arg-sentinel"],
        },
    },
    "custom_providers": [
        {
            "name": "active",
            "base_url": "https://active.example/v1",
            "api_key": "active-key",
        },
    ],
}


def test_keyed_only_record_keeps_the_side_fields_it_owns(monkeypatch):
    """(c) No list row: the keyed record is the authority for ITS side fields too.

    ``providers['custom:omni']`` supplies the endpoint and the credential, so it
    also owns the pool, wire protocol and ACP transport it declares. Clearing
    them because the route is ``custom:<slug>`` throws away the record's own
    configuration; taking the ambient values (all four differ here) routes the
    send through the previous authority.
    """
    init_kwargs = _run_composed_send(
        monkeypatch,
        copy.deepcopy(_KEYED_ONLY_OWNED_CFG),
        _ambient_runtime(
            api_mode="chat_completions",
            command="ambient-acp-sentinel",
            args=["--ambient-arg-sentinel"],
            credential_pool=["ambient-pool-sentinel"],
        ),
        "session-1806-keyed-owned",
    )

    assert init_kwargs["base_url"] == "https://keyed-url-sentinel.example/v1"
    assert init_kwargs["api_key"] == "keyed-key-sentinel-abc"
    assert init_kwargs["provider"] == "custom"
    _assert_side_fields(
        init_kwargs,
        {
            "api_mode": "anthropic_messages",
            "credential_pool": ["keyed-pool-sentinel"],
            "acp_command": "keyed-acp-sentinel",
            "acp_args": ["--keyed-arg-sentinel"],
        },
        "keyed-only owned side fields",
    )


def test_keyed_only_record_without_side_fields_keeps_same_endpoint_runtime(monkeypatch):
    """(c) The clear is provenance-driven, not slug-driven.

    Here the keyed record declares no side fields and the runtime resolved the
    SAME endpoint the record owns, so the runtime's values are same-authority.
    Clearing them would strip a legitimately-pooled Anthropic-protocol endpoint
    of its pool and protocol just because the route is spelled ``custom:<slug>``.
    """
    cfg_dict = copy.deepcopy(_KEYED_ONLY_OWNED_CFG)
    for field in ("api_mode", "credential_pool", "acp_command", "acp_args"):
        cfg_dict["providers"]["custom:omni"].pop(field)

    init_kwargs = _run_composed_send(
        monkeypatch,
        cfg_dict,
        _ambient_runtime(),
        "session-1806-keyed-same-authority",
    )

    assert init_kwargs["base_url"] == "https://keyed-url-sentinel.example/v1"
    _assert_side_fields(
        init_kwargs,
        {
            "api_mode": _AMBIENT_SIDE_FIELD_RUNTIME["api_mode"],
            "credential_pool": _AMBIENT_SIDE_FIELD_RUNTIME["credential_pool"],
            "acp_command": _AMBIENT_SIDE_FIELD_RUNTIME["command"],
            "acp_args": _AMBIENT_SIDE_FIELD_RUNTIME["args"],
        },
        "keyed-only same-authority runtime",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Non-streaming route consumers
#
# Four WebUI consumers build their own AIAgent outside the streaming path.
# Each resolved the endpoint deterministically (``resolve_model_provider`` now
# returns the exact ``custom_providers[]`` row's URL) and then applied the
# custom-provider connection with a FILL-ONLY pattern:
#
#     if not api_key and _cp_key: api_key = _cp_key
#
# Because ``resolve_runtime_provider`` had already supplied a truthy key from the
# same-slug keyed ``providers:`` record, the guard never fired — so the final
# constructor received the LIST row's URL paired with the KEYED row's API key.
# Each test below asserts on the FINAL constructor kwargs, so it fails on that
# mixed bundle and passes only once the row is applied atomically.
# ─────────────────────────────────────────────────────────────────────────────


class _RouteAgentCaptured(Exception):
    """Sentinel raised from the double's __init__ once the bundle is captured.

    All four consumers construct the agent and immediately use it (compress,
    run_conversation, text completion). Stopping at construction keeps each test
    scoped to the routing contract instead of to downstream persistence.
    """


def _setup_route_consumer_runtime(monkeypatch, session_messages=None):
    """Compose the keyed-vs-list config around a capturing route agent.

    Returns ``(captured, fake_session)``. ``captured["init_kwargs"]`` holds the
    FINAL constructor bundle the consumer under test built.
    """
    import types as _types
    from unittest import mock

    import api.config as _config
    import api.oauth
    import api.routes as routes

    monkeypatch.setattr(_config, "cfg", dict(_KEYED_VS_LIST_CFG), raising=False)
    monkeypatch.setattr(_config, "get_config", lambda: dict(_KEYED_VS_LIST_CFG))

    # Production shape: resolve_runtime_provider scans providers: first, so it
    # hands back the KEYED endpoint and the KEYED key for this slug.
    fake_runtime_module = _types.ModuleType("hermes_cli.runtime_provider")
    fake_runtime_module.resolve_runtime_provider = mock.Mock(
        return_value={
            "provider": "custom:omni",
            "base_url": "https://keyed-url-sentinel.example/v1",
            "api_key": "keyed-key-sentinel-abc",
        }
    )
    fake_hermes_cli = _types.ModuleType("hermes_cli")
    fake_hermes_cli.__path__ = []
    fake_hermes_cli.runtime_provider = fake_runtime_module
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.runtime_provider", fake_runtime_module)
    monkeypatch.setattr(
        api.oauth,
        "resolve_runtime_provider_with_anthropic_env_lock",
        lambda resolver, **kwargs: resolver(**kwargs),
    )

    captured = {}

    class CapturingRouteAgent:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = dict(kwargs)
            raise _RouteAgentCaptured("captured")

    monkeypatch.setattr(routes, "require_ai_agent_class", lambda: CapturingRouteAgent)
    monkeypatch.setattr(routes, "ensure_agent_runtime_current", lambda *_a, **_k: None)
    monkeypatch.setattr(routes, "_resolve_cli_toolsets", lambda *_a, **_k: [])
    monkeypatch.setattr(routes, "_session_is_subagent_view_only", lambda *_a, **_k: False)

    class _FakeSession:
        def __init__(self):
            self.session_id = "session-1806-route"
            self.title = "Test"
            self.workspace = "/tmp"
            self.model = "antigravity/gemini-3.7-flash-tiered"
            self.model_provider = "custom:omni"
            self.messages = list(session_messages or [])
            self.context_messages = list(session_messages or [])
            self.active_stream_id = None
            self.pending_user_message = None
            self.pending_attachments = []
            self.pending_started_at = None
            self.pending_user_source = None
            self.profile = None

        def save(self, *_a, **_k):
            return None

    return captured, _FakeSession()


def _assert_list_row_not_keyed(init_kwargs, label):
    """The endpoint AND the credential must both come from the list row."""
    assert init_kwargs["base_url"] == _LIST_ROW_URL, label
    assert init_kwargs["api_key"] == _LIST_ROW_KEY, label
    assert init_kwargs["base_url"] != "https://keyed-url-sentinel.example/v1", label
    assert init_kwargs["api_key"] != "keyed-key-sentinel-abc", label
    assert init_kwargs["provider"] == "custom", label


def _four_route_messages():
    return [
        {"role": "user", "content": "one", "timestamp": 1.0, "_ts": 1.0},
        {"role": "assistant", "content": "two", "timestamp": 2.0, "_ts": 2.0},
        {"role": "user", "content": "three", "timestamp": 3.0, "_ts": 3.0},
        {"role": "assistant", "content": "four", "timestamp": 4.0, "_ts": 4.0},
    ]


def test_sync_chat_route_applies_exact_list_row_atomically(monkeypatch):
    """Consumer 1/4: POST /api/chat."""
    import api.routes as routes

    captured, fake_session = _setup_route_consumer_runtime(
        monkeypatch, session_messages=_four_route_messages()
    )
    monkeypatch.setattr(routes, "get_session", lambda _sid: fake_session)
    monkeypatch.setattr(routes, "_agent_runtime_barrier_response", lambda **_k: None)
    monkeypatch.setattr(routes, "resolve_trusted_workspace", lambda ws: "/tmp")
    monkeypatch.setattr(
        routes, "_get_session_agent_lock", lambda _sid: __import__("contextlib").nullcontext()
    )
    monkeypatch.setattr(
        routes, "_read_profile_model_config", lambda *_a, **_k: (None, None, None)
    )
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda model, provider, **_k: (model, provider),
    )
    monkeypatch.setattr(routes, "j", lambda _handler, payload, **_k: payload)
    monkeypatch.setattr(routes, "bad", lambda _handler, message, *_a, **_k: {"error": message})

    with pytest.raises(_RouteAgentCaptured):
        routes._handle_chat_sync(
            object(), {"session_id": "session-1806-route", "message": "hi"}
        )

    _assert_list_row_not_keyed(captured["init_kwargs"], "sync chat (/api/chat)")


def test_manual_compression_route_applies_exact_list_row_atomically(monkeypatch):
    """Consumer 2/4: POST /api/session/{sid}/compress."""
    import contextlib

    import api.config as _config
    import api.routes as routes

    captured, fake_session = _setup_route_consumer_runtime(
        monkeypatch, session_messages=_four_route_messages()
    )
    monkeypatch.setattr(routes, "get_session", lambda _sid: fake_session)
    monkeypatch.setattr(
        _config, "_get_session_agent_lock", lambda _sid: contextlib.nullcontext()
    )
    monkeypatch.setattr(routes, "j", lambda _handler, payload, **_k: payload)
    monkeypatch.setattr(routes, "bad", lambda _handler, message, *_a, **_k: {"error": message})

    routes._handle_session_compress(object(), {"session_id": "session-1806-route"})

    _assert_list_row_not_keyed(captured["init_kwargs"], "manual compression (/compress)")


def test_commit_message_route_applies_exact_list_row_atomically(monkeypatch):
    """Consumer 3/4: LLM git commit-message generation."""
    import types as _types

    import api.routes as routes

    captured, fake_session = _setup_route_consumer_runtime(monkeypatch)
    # Force the auxiliary-client shortcut to decline so the main-model
    # constructor (the bundle under test) is the one that runs.
    fake_aux = _types.ModuleType("agent.auxiliary_client")
    fake_aux.get_text_auxiliary_client = lambda *_a, **_k: (None, None)
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", fake_aux)

    with pytest.raises(_RouteAgentCaptured):
        routes._llm_git_commit_message("sys", "user", session=fake_session)

    _assert_list_row_not_keyed(captured["init_kwargs"], "git commit message")


def test_handoff_summary_route_applies_exact_list_row_atomically(monkeypatch):
    """Consumer 4/4: on-demand handoff summary."""
    import api.models as models
    import api.routes as routes

    captured, fake_session = _setup_route_consumer_runtime(
        monkeypatch, session_messages=_four_route_messages()
    )
    monkeypatch.setattr(models, "get_session", lambda _sid: fake_session)
    monkeypatch.setattr(
        models,
        "count_conversation_rounds",
        lambda _sid, since=None: models.CONVERSATION_ROUND_THRESHOLD + 1,
    )
    monkeypatch.setattr(
        models, "get_cli_session_messages", lambda _sid: _four_route_messages()
    )
    monkeypatch.setattr(routes, "j", lambda _handler, payload, **_k: payload)
    monkeypatch.setattr(routes, "bad", lambda _handler, message, *_a, **_k: {"error": message})

    routes._handle_handoff_summary(object(), {"session_id": "session-1806-route"})

    _assert_list_row_not_keyed(captured["init_kwargs"], "handoff summary")


def test_update_summary_route_applies_exact_list_row_atomically(monkeypatch):
    """Consumer: update summary (_llm_update_summary)."""
    import types as _types
    import api.routes as routes

    cfg = dict(_KEYED_VS_LIST_CFG)
    cfg["model"] = {
        "default": "@custom:omni:antigravity/gemini-3.7-flash-tiered",
        "provider": "custom:omni",
    }
    captured, fake_session = _setup_route_consumer_runtime(monkeypatch)
    monkeypatch.setattr("api.config.cfg", cfg, raising=False)
    monkeypatch.setattr("api.config.get_config", lambda: dict(cfg))

    fake_aux = _types.ModuleType("agent.auxiliary_client")
    recorded_main_runtime = {}

    def fake_get_aux(task, main_runtime=None):
        recorded_main_runtime.update(main_runtime or {})
        return (None, None)

    fake_aux.get_text_auxiliary_client = fake_get_aux
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", fake_aux)

    with pytest.raises(_RouteAgentCaptured):
        routes._llm_update_summary("sys", "user", active_profile=None)

    _assert_list_row_not_keyed(captured["init_kwargs"], "update summary (AIAgent)")
    assert recorded_main_runtime.get("base_url") == "https://list-url-sentinel.example/v1"
    assert recorded_main_runtime.get("api_key") == "list-key-sentinel-xyz"


def test_production_composed_retry_caches_agent_under_recomputed_signature(monkeypatch):
    """Retry agents are cached under the healed bundle's signature, not the stale initial one."""
    import api.streaming as streaming

    computed_signatures = []
    real_compute_sig = streaming._compute_agent_cache_signature

    def tracking_compute_sig(*args, **kwargs):
        sig = real_compute_sig(*args, **kwargs)
        computed_signatures.append(sig)
        return sig

    monkeypatch.setattr(streaming, "_compute_agent_cache_signature", tracking_compute_sig)

    stream_id, q, captured, restore = _setup_production_composed_runtime(
        monkeypatch,
        dict(_KEYED_VS_LIST_CFG),
        dict(_AMBIENT_SIDE_FIELD_RUNTIME),
        session_id="session-1806-retry-cache-sig",
        fail_first="returned_error",
    )
    # Give self-heal a distinct provider/endpoint so initial and healed signatures differ
    healed_rt = dict(_AMBIENT_SIDE_FIELD_RUNTIME)
    healed_rt["base_url"] = "https://healed-endpoint.example/v1"
    monkeypatch.setattr(
        streaming,
        "_attempt_credential_self_heal",
        lambda *_args, **_kwargs: dict(healed_rt),
    )
    try:
        streaming.STREAMS[stream_id] = q
        streaming._run_agent_streaming(
            session_id="session-1806-retry-cache-sig",
            msg_text="hello",
            model="@custom:omni:antigravity/gemini-3.7-flash-tiered",
            workspace="/tmp",
            stream_id=stream_id,
        )
        assert len(computed_signatures) >= 2, "signature was not recomputed on retry"
        with config.SESSION_AGENT_CACHE_LOCK:
            cached = config.SESSION_AGENT_CACHE.get("session-1806-retry-cache-sig")
            assert cached is not None, "retry agent was not cached"
            cached_agent, cached_sig = cached
            assert cached_agent is not None
            assert cached_sig == computed_signatures[-1]
    finally:
        streaming.STREAMS.pop(stream_id, None)
        streaming.AGENT_INSTANCES.pop(stream_id, None)
        restore()
