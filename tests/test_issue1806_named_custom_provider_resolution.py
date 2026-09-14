"""Regression tests for #1806 named custom provider routing.

The WebUI must treat ``model.provider: <custom_providers[].name>`` as the
same provider slug the picker emits: ``custom:<name>``.  Otherwise a stale
agent-side base-url slug such as ``custom:local-(127.0.0.1:11434)`` can win
model selection and send runtime auth down an impossible env-var path.
"""

from __future__ import annotations

import contextlib
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
    heal_mutate=None,
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

    ``heal_mutate`` is the hook for the state a retry guard actually defends
    against: it runs INSIDE the stubbed self-heal, i.e. after the first agent
    has already been constructed and has already failed with a 401, but BEFORE
    the retry re-resolves its bundle. Mutating ``config.cfg`` there reproduces
    the row being edited, unnamed or drained mid-turn -- the only way a route
    that was routable at first resolution becomes terminal at retry
    construction.
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
            captured.setdefault("instances", []).append(self)
            # Mirror ``agent/agent_init.py:_init_openai_client()``:
            #
            #     if api_key and base_url:
            #         client_kwargs = _explicit_client_kwargs(...)
            #     else:
            #         client_kwargs = _routed_client_kwargs(...)
            #
            # An incomplete connection pair is NOT a refusal at this boundary —
            # it is the signal to resolve a provider all over again, through the
            # centralized router and then the init-time fallback chain. Recording
            # the branch here is what lets a test assert the real defect ("this
            # send would have been re-routed") instead of the weaker proxy
            # ("base_url came back None"), which an unroutable bundle satisfies
            # while still reaching a provider the user never chose.
            if api_key and base_url:
                captured.setdefault("explicit_client_kwargs_calls", []).append(
                    {"api_key": api_key, "base_url": base_url}
                )
            else:
                captured.setdefault("routed_client_kwargs_calls", []).append(
                    {"provider": provider, "api_key": api_key, "base_url": base_url}
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
    # Both the returned-error retry branch AND the outer error emission live
    # behind the stale-writeback guard, which only lets the OWNING worker
    # persist and emit. /api/chat/start stamps ``active_stream_id`` before
    # dispatching the worker, so model that here unconditionally — otherwise the
    # guard returns early and a terminal route verdict reaches the queue as
    # nothing at all, which a refusal test would read as "no controlled failure".
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
        def _fake_self_heal(*_args, **_kwargs):
            # Production re-reads provider state during the heal. Running the
            # mutation here -- not before the send -- is what makes the FIRST
            # resolution routable and only the RETRY resolution terminal.
            if heal_mutate is not None:
                heal_mutate()
            return dict(runtime_dict)

        monkeypatch.setattr(streaming, "_attempt_credential_self_heal", _fake_self_heal)

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
    """(b) A blank exact list row is terminal — it cannot repopulate from the keyed row.

    When an exact ``custom_providers[]`` entry exists with an empty base_url,
    that row is authoritative and the route has no endpoint. Handing AIAgent
    ``base_url=None`` with the list key would NOT refuse: ``_init_openai_client()``
    honours an explicit pair only when both fields are truthy, so it would call
    ``_routed_client_kwargs()`` and re-resolve a provider — reaching the keyed
    row's endpoint by another door. The send must stop before construction.
    """
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

    captured, apperrors = _run_composed_send_expecting_refusal(
        monkeypatch,
        cfg_dict,
        runtime_dict,
        "session-1806-blank",
        model="@custom:omni:antigravity/gemini-3.7-flash-tiered",
    )

    payload = apperrors[-1]
    assert "resolved no endpoint" in payload["message"], payload
    assert "base_url" in payload["hint"], (
        f"the refusal must name the endpoint setting to fix: {payload}"
    )
    # The refusal must not hand the user the keyed row it declined to fall
    # through to — naming it would read as "this endpoint was used".
    assert "keyed-url-sentinel" not in str(payload), payload
    assert not captured.get("explicit_client_kwargs_calls"), (
        "a client was configured for a route with no endpoint"
    )


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


# ─────────────────────────────────────────────────────────────────────────────
# Terminal route verdicts: the send must STOP, not fail closed and continue
#
# "Fail closed" was the wrong shape for an unresolvable named route. Clearing
# ``base_url``/``api_key`` looks terminal in a bundle assertion, but at the
# constructor it is the opposite: ``_init_openai_client()`` honours an explicit
# pair only when BOTH fields are truthy and otherwise calls
# ``_routed_client_kwargs()``, which re-resolves a provider through the
# centralized router and the init-time fallback chain. So the very bundles the
# earlier tests asserted were "safe" are the ones that route the user's prompt —
# and whatever credential init finds — to a provider they never picked.
#
# The helpers below assert the real property: for both terminal shapes (a slug
# nothing owns, and an owned row whose declared credential resolved to nothing)
# NO agent is constructed at all, so ``_routed_client_kwargs()`` is never
# reached, the agent cache is never written, and the turn ends on a controlled
# ``provider_unroutable`` apperror instead.
# ─────────────────────────────────────────────────────────────────────────────


def _drain_apperrors(fake_queue):
    """Return every ``apperror`` payload the worker queued."""
    import queue as _queue

    payloads = []
    while True:
        try:
            item = fake_queue.get_nowait()
        except _queue.Empty:
            break
        if item and item[0] == "apperror":
            payloads.append(item[1])
    return payloads


def _assert_route_refused(captured, apperrors, label, *, expected_reason=None):
    """Assert the send stopped at the route verdict, before any provider routing."""
    routed = captured.get("routed_client_kwargs_calls", [])
    assert not routed, (
        f"{label}: AIAgent was constructed with an incomplete connection pair "
        f"{routed}, so _init_openai_client() fell through to "
        f"_routed_client_kwargs() and re-resolved a provider"
    )
    assert not captured.get("init_kwargs_history"), (
        f"{label}: an agent was constructed for an unroutable route"
    )
    assert not captured.get("run_calls"), f"{label}: the turn was actually sent"

    assert apperrors, f"{label}: no controlled failure was emitted"
    payload = apperrors[-1]
    assert payload["type"] == "provider_unroutable", (
        f"{label}: emitted {payload['type']!r} instead of a provider-route failure"
    )
    assert payload.get("hint"), f"{label}: the failure named no fix"
    if expected_reason is not None:
        assert expected_reason in payload.get("message", "") or expected_reason in payload.get(
            "hint", ""
        ), f"{label}: the failure did not name {expected_reason!r}: {payload}"

    with config.SESSION_AGENT_CACHE_LOCK:
        assert not config.SESSION_AGENT_CACHE, (
            f"{label}: the agent cache was poisoned with an unroutable bundle, so "
            f"every later turn in this session would reuse it"
        )
    return payload


def _run_composed_send_expecting_refusal(
    monkeypatch, cfg_dict, runtime_dict, session_id, *, model
):
    """Drive one composed send whose route is terminal at the FIRST resolution.

    Returns ``(captured, apperrors)``. Deliberately takes no ``fail_first``: the
    verdict lands before any agent exists, so there is no 401 for the self-heal
    retries to act on and threading one through would only produce cases that
    re-run identical code. The retry regions have their own harness,
    :func:`_run_composed_retry_expecting_abandoned_heal`.
    """
    import api.streaming as streaming

    stream_id, q, captured, restore = _setup_production_composed_runtime(
        monkeypatch, cfg_dict, runtime_dict, session_id=session_id
    )
    try:
        streaming.STREAMS[stream_id] = q
        streaming._run_agent_streaming(
            session_id=session_id,
            msg_text="hello",
            model=model,
            workspace="/tmp",
            stream_id=stream_id,
        )
        apperrors = _drain_apperrors(q)
        # Read the cache BEFORE restore() clears it.
        _assert_route_refused(captured, apperrors, session_id)
    finally:
        streaming.STREAMS.pop(stream_id, None)
        streaming.AGENT_INSTANCES.pop(stream_id, None)
        restore()
    return captured, apperrors


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
    The placeholder would report that as an opaque 401, and sending with NO key
    is no safer: an agent built with ``api_key=None`` never reaches an explicit
    client, so ``_routed_client_kwargs()`` re-resolves a provider and the turn
    leaves for whatever credential init finds next. The route is terminal.
    """

    def _build(_key_cmd, _name):
        raise RuntimeError("command token source unavailable")

    _fake_command_token_source(monkeypatch, _build)

    captured, apperrors = _run_composed_send_expecting_refusal(
        monkeypatch,
        _exact_list_row_cfg(drop=("api_key",), key_cmd="print-omni-bearer"),
        _ambient_runtime(),
        "session-1806-owned-key-cmd-broken",
        model="@custom:omni:antigravity/gemini-3.7-flash-tiered",
    )

    payload = apperrors[-1]
    assert "produced no API key" in payload["message"], payload
    assert "key_cmd" in payload["hint"], (
        f"the refusal must name the credential setting to fix: {payload}"
    )
    # Neither the placeholder nor the ambient keyed credential may appear
    # anywhere on the refusal path.
    assert config.KEYLESS_CUSTOM_API_KEY not in str(payload), payload
    assert "keyed-key-sentinel-abc" not in str(payload), payload
    assert not captured.get("explicit_client_kwargs_calls"), (
        "a client was configured for a route whose declared credential failed"
    )


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


# The runtime dict the route consumers see by default: production shape, where
# ``resolve_runtime_provider`` scans ``providers:`` first and so hands back the
# KEYED endpoint and the KEYED key for this slug.
_ROUTE_KEYED_RUNTIME = {
    "provider": "custom:omni",
    "base_url": "https://keyed-url-sentinel.example/v1",
    "api_key": "keyed-key-sentinel-abc",
}


def _setup_route_consumer_runtime(
    monkeypatch, session_messages=None, cfg_dict=None, runtime_dict=None
):
    """Compose the keyed-vs-list config around a capturing route agent.

    Returns ``(captured, fake_session)``. ``captured["init_kwargs"]`` holds the
    FINAL constructor bundle the consumer under test built.

    ``cfg_dict``/``runtime_dict`` override the defaults so a test can hand the
    consumers an exact row that OWNS side fields (``api_mode``,
    ``credential_pool``, ACP transport) and an ambient runtime that reports
    different ones -- the probe for whether the complete bundle, or just its
    three connection fields, reaches the constructor.
    """
    import types as _types
    from unittest import mock

    import api.config as _config
    import api.oauth
    import api.routes as routes

    cfg_dict = copy.deepcopy(_KEYED_VS_LIST_CFG if cfg_dict is None else cfg_dict)
    monkeypatch.setattr(_config, "cfg", dict(cfg_dict), raising=False)
    monkeypatch.setattr(_config, "get_config", lambda: copy.deepcopy(cfg_dict))

    runtime_dict = copy.deepcopy(
        _ROUTE_KEYED_RUNTIME if runtime_dict is None else runtime_dict
    )
    fake_runtime_module = _types.ModuleType("hermes_cli.runtime_provider")
    fake_runtime_module.resolve_runtime_provider = mock.Mock(
        return_value=runtime_dict
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
        # Every constructor-routing field must be a NAMED parameter, for the
        # same reason as the streaming double: the route consumers gate each
        # optional kwarg on ``inspect.signature(AIAgent.__init__)``, so a double
        # that swallowed them into ``**kwargs`` would filter out exactly the
        # fields these tests pin and pass vacuously.
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


# Each driver composes the harness the way its consumer needs it and returns the
# FINAL constructor kwargs. Sharing them keeps the two contracts below --
# "the endpoint and credential are one record's" and "the record's side fields
# reach the constructor too" -- asserted over the SAME five consumers, so a new
# consumer cannot be added to one list and forgotten in the other.


def _drive_sync_chat_route(monkeypatch, cfg_dict=None, runtime_dict=None):
    """Consumer 1/5: POST /api/chat."""
    import api.routes as routes

    captured, fake_session = _setup_route_consumer_runtime(
        monkeypatch,
        session_messages=_four_route_messages(),
        cfg_dict=cfg_dict,
        runtime_dict=runtime_dict,
    )
    monkeypatch.setattr(routes, "get_session", lambda _sid: fake_session)
    monkeypatch.setattr(routes, "_agent_runtime_barrier_response", lambda **_k: None)
    monkeypatch.setattr(routes, "resolve_trusted_workspace", lambda ws: "/tmp")
    monkeypatch.setattr(
        routes, "_get_session_agent_lock", lambda _sid: contextlib.nullcontext()
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
    return captured["init_kwargs"]


def _drive_manual_compression_route(monkeypatch, cfg_dict=None, runtime_dict=None):
    """Consumer 2/5: POST /api/session/{sid}/compress."""
    import api.config as _config
    import api.routes as routes

    captured, fake_session = _setup_route_consumer_runtime(
        monkeypatch,
        session_messages=_four_route_messages(),
        cfg_dict=cfg_dict,
        runtime_dict=runtime_dict,
    )
    monkeypatch.setattr(routes, "get_session", lambda _sid: fake_session)
    monkeypatch.setattr(
        _config, "_get_session_agent_lock", lambda _sid: contextlib.nullcontext()
    )
    monkeypatch.setattr(routes, "j", lambda _handler, payload, **_k: payload)
    monkeypatch.setattr(routes, "bad", lambda _handler, message, *_a, **_k: {"error": message})

    routes._handle_session_compress(object(), {"session_id": "session-1806-route"})
    return captured["init_kwargs"]


def _decline_auxiliary_client(monkeypatch, recorded_main_runtime=None):
    """Force the auxiliary-model shortcut to decline.

    The consumers that have one would otherwise return before constructing the
    main-model agent -- and that constructor is the bundle under test.
    """
    import types as _types

    fake_aux = _types.ModuleType("agent.auxiliary_client")

    def _get_text_auxiliary_client(_task, main_runtime=None):
        if recorded_main_runtime is not None:
            recorded_main_runtime.update(main_runtime or {})
        return (None, None)

    fake_aux.get_text_auxiliary_client = _get_text_auxiliary_client
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", fake_aux)


def _drive_commit_message_route(
    monkeypatch, cfg_dict=None, runtime_dict=None, recorded_main_runtime=None
):
    """Consumer 3/5: LLM git commit-message generation."""
    import api.routes as routes

    captured, fake_session = _setup_route_consumer_runtime(
        monkeypatch, cfg_dict=cfg_dict, runtime_dict=runtime_dict
    )
    _decline_auxiliary_client(monkeypatch, recorded_main_runtime)

    with pytest.raises(_RouteAgentCaptured):
        routes._llm_git_commit_message("sys", "user", session=fake_session)
    return captured["init_kwargs"]


def _drive_handoff_summary_route(monkeypatch, cfg_dict=None, runtime_dict=None):
    """Consumer 4/5: on-demand handoff summary."""
    import api.models as models
    import api.routes as routes

    captured, fake_session = _setup_route_consumer_runtime(
        monkeypatch,
        session_messages=_four_route_messages(),
        cfg_dict=cfg_dict,
        runtime_dict=runtime_dict,
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
    return captured["init_kwargs"]


def _drive_update_summary_route(
    monkeypatch, cfg_dict=None, runtime_dict=None, recorded_main_runtime=None
):
    """Consumer 5/5: update summary (``_llm_update_summary``).

    This one resolves ``get_effective_default_model()`` rather than the session's
    model, so the default has to name the slug under test.
    """
    import api.routes as routes

    cfg_dict = copy.deepcopy(_KEYED_VS_LIST_CFG if cfg_dict is None else cfg_dict)
    cfg_dict["model"] = {
        "default": "@custom:omni:antigravity/gemini-3.7-flash-tiered",
        "provider": "custom:omni",
    }
    captured, _fake_session = _setup_route_consumer_runtime(
        monkeypatch, cfg_dict=cfg_dict, runtime_dict=runtime_dict
    )
    _decline_auxiliary_client(monkeypatch, recorded_main_runtime)

    with pytest.raises(_RouteAgentCaptured):
        routes._llm_update_summary("sys", "user", active_profile=None)
    return captured["init_kwargs"]


_ROUTE_CONSUMER_DRIVERS = [
    (_drive_sync_chat_route, "sync chat (/api/chat)"),
    (_drive_manual_compression_route, "manual compression (/compress)"),
    (_drive_commit_message_route, "git commit message"),
    (_drive_handoff_summary_route, "handoff summary"),
    (_drive_update_summary_route, "update summary"),
]


@pytest.mark.parametrize(
    "driver,label", _ROUTE_CONSUMER_DRIVERS, ids=[d[1] for d in _ROUTE_CONSUMER_DRIVERS]
)
def test_route_consumers_apply_exact_list_row_atomically(monkeypatch, driver, label):
    """Every non-streaming consumer applies the exact row's URL *and* its key."""
    _assert_list_row_not_keyed(driver(monkeypatch), label)


# The two consumers whose auxiliary-client shortcut can answer the request
# outright -- for them ``main_runtime`` is the only carrier of the resolved
# authority, because AIAgent is never built.
_AUXILIARY_ROUTE_DRIVERS = [
    (_drive_commit_message_route, "git commit message"),
    (_drive_update_summary_route, "update summary"),
]


@pytest.mark.parametrize(
    "driver,label",
    _AUXILIARY_ROUTE_DRIVERS,
    ids=[d[1] for d in _AUXILIARY_ROUTE_DRIVERS],
)
def test_auxiliary_routes_hand_the_row_connection_to_the_auxiliary_client(
    monkeypatch, driver, label
):
    """The aux-client shortcut must see the row's connection, not the keyed one.

    It runs BEFORE the main-model constructor, so a consumer that resolved the
    bundle only on the fallback path would still send this request to the keyed
    endpoint.
    """
    recorded_main_runtime = {}
    driver(monkeypatch, recorded_main_runtime=recorded_main_runtime)

    assert recorded_main_runtime.get("base_url") == _LIST_ROW_URL, label
    assert recorded_main_runtime.get("api_key") == _LIST_ROW_KEY, label


# ── The complete bundle, not just its three connection fields ────────────────
#
# ``apply_custom_provider_connection_authority`` returns ``(provider, api_key,
# base_url)``. Every consumer above used to apply exactly that and construct the
# agent from it, so an exact row's ``api_mode`` and ``credential_pool`` -- the
# wire protocol the send speaks and the credential source it rotates -- were
# truncated away before AIAgent ever saw them, while the streaming path carried
# them through. The row below OWNS both, and the ambient runtime reports
# different values for all four side fields, so a truncating consumer fails on
# ``None`` and a pass-through consumer fails on the ambient sentinel.


_ROUTE_ROW_POOL_SENTINEL = ["list-row-pool-sentinel"]

_ROUTE_OWNED_SIDE_FIELD_CFG = _exact_list_row_cfg(
    api_mode="anthropic_messages",
    credential_pool=_ROUTE_ROW_POOL_SENTINEL,
)

# The ambient runtime disagrees on every side field it reports.
_ROUTE_AMBIENT_RUNTIME = {
    **_ROUTE_KEYED_RUNTIME,
    "api_mode": "chat_completions",
    "credential_pool": ["ambient-pool-sentinel"],
    "command": "ambient-acp-sentinel",
    "args": ["--ambient-arg-sentinel"],
}


@pytest.mark.parametrize(
    "driver,label", _ROUTE_CONSUMER_DRIVERS, ids=[d[1] for d in _ROUTE_CONSUMER_DRIVERS]
)
def test_route_consumers_pass_the_exact_rows_side_fields_to_the_constructor(
    monkeypatch, driver, label
):
    """The exact row's ``api_mode``/``credential_pool`` reach the FINAL constructor."""
    init_kwargs = driver(
        monkeypatch,
        cfg_dict=copy.deepcopy(_ROUTE_OWNED_SIDE_FIELD_CFG),
        runtime_dict=copy.deepcopy(_ROUTE_AMBIENT_RUNTIME),
    )

    _assert_list_row_not_keyed(init_kwargs, label)
    _assert_side_fields(
        init_kwargs,
        {
            "api_mode": "anthropic_messages",
            "credential_pool": _ROUTE_ROW_POOL_SENTINEL,
            # The row declares no ACP transport and owns a different endpoint
            # than the runtime, so the ambient subprocess is provably foreign.
            "acp_command": None,
            "acp_args": None,
        },
        f"{label}: exact row side fields",
    )
    assert init_kwargs["api_mode"] != _ROUTE_AMBIENT_RUNTIME["api_mode"], (
        f"{label}: the ambient wire protocol reached the constructor"
    )
    assert init_kwargs["credential_pool"] != _ROUTE_AMBIENT_RUNTIME["credential_pool"], (
        f"{label}: the ambient credential pool reached the constructor"
    )


@pytest.mark.parametrize(
    "driver,label", _ROUTE_CONSUMER_DRIVERS, ids=[d[1] for d in _ROUTE_CONSUMER_DRIVERS]
)
def test_route_consumers_clear_foreign_ambient_side_fields(monkeypatch, driver, label):
    """A row that owns NO side fields still strips the ambient provider's.

    Complement of the test above: "carry the complete bundle" must not degrade
    into "pass the runtime's side fields through". The row here declares none of
    them and owns a different endpoint, so all four are provably foreign.
    """
    init_kwargs = driver(
        monkeypatch, runtime_dict=copy.deepcopy(_ROUTE_AMBIENT_RUNTIME)
    )

    _assert_list_row_not_keyed(init_kwargs, label)
    _assert_side_fields(init_kwargs, _FOREIGN_AMBIENT_SIDE_FIELDS, f"{label}: foreign ambient")


@pytest.mark.parametrize(
    "driver,label",
    _AUXILIARY_ROUTE_DRIVERS,
    ids=[d[1] for d in _AUXILIARY_ROUTE_DRIVERS],
)
def test_auxiliary_routes_hand_the_complete_bundle_to_the_auxiliary_client(
    monkeypatch, driver, label
):
    """``main_runtime`` carries the WHOLE bundle, not its three connection fields.

    When the auxiliary client answers, AIAgent is bypassed entirely, so the
    side fields that never entered ``main_runtime`` were simply lost: the exact
    row's ``api_mode: anthropic_messages`` silently degraded to the aux client's
    default wire protocol, and the credential pool/ACP transport the row owns
    never reached the send. The aux dict must therefore agree with the fallback
    constructor field for field -- same authority, whichever path answers.
    """
    import api.routes as routes

    recorded_main_runtime = {}
    init_kwargs = driver(
        monkeypatch,
        cfg_dict=copy.deepcopy(_ROUTE_OWNED_SIDE_FIELD_CFG),
        runtime_dict=copy.deepcopy(_ROUTE_AMBIENT_RUNTIME),
        recorded_main_runtime=recorded_main_runtime,
    )

    _assert_list_row_not_keyed(recorded_main_runtime, f"{label}: aux main_runtime")
    _assert_side_fields(
        recorded_main_runtime,
        {
            "api_mode": "anthropic_messages",
            "credential_pool": _ROUTE_ROW_POOL_SENTINEL,
            # The row declares no ACP transport and owns a different endpoint
            # than the runtime, so the ambient subprocess is provably foreign.
            "acp_command": None,
            "acp_args": None,
        },
        f"{label}: aux main_runtime side fields",
    )
    assert recorded_main_runtime["api_mode"] != _ROUTE_AMBIENT_RUNTIME["api_mode"], (
        f"{label}: the ambient wire protocol reached the auxiliary client"
    )
    assert (
        recorded_main_runtime["credential_pool"]
        != _ROUTE_AMBIENT_RUNTIME["credential_pool"]
    ), f"{label}: the ambient credential pool reached the auxiliary client"

    for field in ("provider", "model", "base_url", "api_key") + tuple(
        routes._AGENT_BUNDLE_SIDE_FIELDS
    ):
        assert recorded_main_runtime.get(field) == init_kwargs[field], (
            f"{label}: aux {field} disagrees with the fallback constructor, so "
            "which path answers decides the authority"
        )
    assert recorded_main_runtime["model"], f"{label}: aux main_runtime lost the model"


@pytest.mark.parametrize(
    "driver,label",
    _AUXILIARY_ROUTE_DRIVERS,
    ids=[d[1] for d in _AUXILIARY_ROUTE_DRIVERS],
)
def test_auxiliary_routes_clear_foreign_ambient_side_fields(monkeypatch, driver, label):
    """A row owning no side fields strips the ambient provider's from the aux dict too.

    Complement of the test above: "carry the complete bundle into
    ``main_runtime``" must not degrade into "pass the runtime's side fields
    through" on the path where nothing downstream re-resolves them.
    """
    recorded_main_runtime = {}
    driver(
        monkeypatch,
        runtime_dict=copy.deepcopy(_ROUTE_AMBIENT_RUNTIME),
        recorded_main_runtime=recorded_main_runtime,
    )

    _assert_list_row_not_keyed(recorded_main_runtime, f"{label}: aux main_runtime")
    _assert_side_fields(
        recorded_main_runtime,
        _FOREIGN_AMBIENT_SIDE_FIELDS,
        f"{label}: aux foreign ambient",
    )


def test_capturing_route_agent_exposes_every_runtime_constructor_field(monkeypatch):
    """Guard the guard, route edition.

    The consumers gate each optional kwarg on
    ``inspect.signature(AIAgent.__init__).parameters``. A double that swallowed
    ``api_mode`` / ``credential_pool`` / ``acp_command`` / ``acp_args`` into
    ``**kwargs`` would make every assertion above pass vacuously, because the
    fields would never be passed at all.
    """
    import inspect

    import api.routes as routes

    _captured, _fake_session = _setup_route_consumer_runtime(monkeypatch)
    agent_cls = routes.require_ai_agent_class()
    params = set(inspect.signature(agent_cls.__init__).parameters)

    for field in routes._AGENT_BUNDLE_SIDE_FIELDS:
        assert field in params, (
            f"the route double hides {field} behind **kwargs, so the "
            "signature gate would filter it out and the assertions would be vacuous"
        )


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


# ─────────────────────────────────────────────────────────────────────────────
# Identity-owned selection, and the two ways a route can be "keyless"
#
# Selection for a named ``custom:<slug>`` must be owned by that identity: an
# exact normalized ``custom_providers[]`` row, an exact keyed record, or an
# explicitly matching ``model:``/bare-``custom`` authority. "The list happens to
# hold exactly one row, so use it" is NOT ownership — it resolved ``custom:ghost``
# to the endpoint AND credential of a sole unrelated row named ``omni``, sending
# the prompt and that credential to a provider the user never named.
#
# The second half of the same contract is ``dummy-key``. It asserts "this
# endpoint is UNAUTHENTICATED", so it may only appear when the record declares no
# credential source at all. A DECLARED credential that failed to resolve (unset
# ``${ENV}``, ``key_env`` naming a missing variable, a pool that yielded nothing)
# is a misconfiguration to surface, not an unauthenticated endpoint: substituting
# the placeholder there reports it as an opaque 401 from the endpoint instead.
# ─────────────────────────────────────────────────────────────────────────────


_SOLE_UNRELATED_ROW_CFG = {
    "model": {"default": "active/model", "provider": "custom:omni"},
    "custom_providers": [
        {
            "name": "omni",
            "base_url": "https://omni.example/v1",
            "api_key": "omni-key",
        },
    ],
}


def _with_direct_config(monkeypatch, cfg_dict):
    """Point both ``config.cfg`` and ``get_config()`` at ``cfg_dict``."""
    monkeypatch.setattr(config, "get_config", lambda: copy.deepcopy(cfg_dict))
    monkeypatch.setitem(config.cfg, "custom_providers", cfg_dict.get("custom_providers", []))
    monkeypatch.setitem(config.cfg, "model", cfg_dict.get("model", {}))
    monkeypatch.setitem(config.cfg, "providers", cfg_dict.get("providers", {}))


def test_unknown_named_slug_does_not_select_a_sole_unrelated_list_row(monkeypatch):
    """The reviewer's probe: ``custom:ghost`` must not inherit sole row ``omni``.

    ``_select_custom_provider_record`` used to append ``custom_providers[0]``
    whenever the list held exactly one row, then accept it merely because it had
    a key or a URL. Neither test is ownership: ``ghost`` and ``omni`` are
    different identities, so the pair belongs to ``omni`` alone.
    """
    _with_direct_config(monkeypatch, _SOLE_UNRELATED_ROW_CFG)

    assert config.resolve_custom_provider_connection("custom:ghost") == (None, None)
    # The row is still authoritative for its OWN slug.
    assert config.resolve_custom_provider_connection("custom:omni") == (
        "omni-key",
        "https://omni.example/v1",
    )


def test_unknown_named_slug_reports_an_explicit_missing_selection(monkeypatch):
    """The bundle carries the missing verdict instead of an ambiguous ``None``.

    ``keyless`` must be False on it: "nothing owns this route" is not a claim
    that the route is unauthenticated, and it is ``keyless`` alone that gates
    ``dummy-key``.
    """
    _with_direct_config(monkeypatch, _SOLE_UNRELATED_ROW_CFG)

    ghost = config.resolve_custom_provider_bundle("custom:ghost")
    assert ghost is not None, "a named custom route must report its selection outcome"
    assert ghost["status"] == config.CUSTOM_SELECTION_MISSING
    assert ghost["record"] is None
    assert ghost["base_url"] is None
    assert ghost["api_key"] is None
    assert ghost["keyless"] is False, "a missing route must not be reported keyless"
    assert ghost["owned"] == {}

    owned = config.resolve_custom_provider_bundle("custom:omni")
    assert owned["status"] == config.CUSTOM_SELECTION_EXACT
    assert owned["base_url"] == "https://omni.example/v1"


def test_unknown_named_slug_keeps_no_ambient_connection(monkeypatch):
    """The merge refuses the ambient provider's URL/key/pool for an unowned slug."""
    _with_direct_config(monkeypatch, _SOLE_UNRELATED_ROW_CFG)

    bundle = config.merge_custom_provider_runtime_bundle(
        "custom:ghost",
        "ambient-key-sentinel",
        "https://ambient.example/v1",
        dict(_AMBIENT_SIDE_FIELD_RUNTIME),
        lookup_provider="custom:ghost",
    )

    assert bundle["base_url"] is None
    assert bundle["api_key"] is None
    assert bundle["api_key"] != config.KEYLESS_CUSTOM_API_KEY
    # The provider must NOT be rewritten to generic ``custom``: that would
    # present an unresolvable route as a resolved one.
    assert bundle["provider"] == "custom:ghost"
    _assert_side_fields(bundle, _FOREIGN_AMBIENT_SIDE_FIELDS, "unowned named slug")


# A provider LABEL is not proof of ownership. These two cases are how an unowned
# slug could still walk away with the ambient connection:
#
#   1. the runtime dict names ITSELF ``custom:ghost`` -- a self-assigned string
#      on a connection the process authenticated for some other reason; and
#   2. no runtime dict at all, the non-streaming consumers' shape, where
#      ``_connection_identity`` falls back to ``resolved_provider`` -- the very
#      slug being looked up -- so the route would match itself.
#
# Both must fail closed exactly as the differently-labelled ambient runtime does:
# ownership is decided by config records, and here there are none.


_SELF_LABELLED_GHOST_RUNTIME = {
    **_AMBIENT_SIDE_FIELD_RUNTIME,
    "provider": "custom:ghost",
    "base_url": "https://ambient.example/v1",
    "api_key": "ambient-key-sentinel",
}


def _assert_ghost_bundle_failed_closed(bundle, label):
    """No endpoint, no credential, no placeholder, no ambient side fields."""
    assert bundle["base_url"] is None, f"{label}: an unowned slug kept an endpoint"
    assert bundle["api_key"] is None, f"{label}: an unowned slug kept a credential"
    assert bundle["api_key"] != config.KEYLESS_CUSTOM_API_KEY, (
        f"{label}: an unresolvable route was handed the keyless placeholder"
    )
    # Still the NAMED slug: rewriting it to generic ``custom`` would present an
    # unresolvable route as a resolved one.
    assert bundle["provider"] == "custom:ghost", label
    _assert_side_fields(bundle, _FOREIGN_AMBIENT_SIDE_FIELDS, label)


@pytest.mark.parametrize(
    "runtime_dict,label",
    [
        (_SELF_LABELLED_GHOST_RUNTIME, "runtime labels itself custom:ghost"),
        (None, "no runtime dict (non-streaming caller shape)"),
    ],
    ids=["self-labelled-runtime", "no-runtime-dict"],
)
def test_unknown_named_slug_fails_closed_even_when_the_runtime_claims_the_slug(
    monkeypatch, runtime_dict, label
):
    """A matching provider label must not resurrect the ambient connection.

    Nothing in config owns ``custom:ghost``: not an exact ``custom_providers[]``
    row, not a keyed ``providers:`` record, not a ``model:`` authority. The only
    thing pointing at the slug is a string the caller supplied, so keeping the
    URL, key, wire protocol and credential pool that came with it would send the
    user's prompt and that credential to a provider they never configured.
    """
    _with_direct_config(monkeypatch, _SOLE_UNRELATED_ROW_CFG)

    bundle = config.merge_custom_provider_runtime_bundle(
        "custom:ghost",
        "ambient-key-sentinel",
        "https://ambient.example/v1",
        copy.deepcopy(runtime_dict) if runtime_dict else runtime_dict,
        lookup_provider="custom:ghost",
    )

    _assert_ghost_bundle_failed_closed(bundle, label)
    assert bundle["base_url"] != "https://ambient.example/v1", label
    assert bundle["api_key"] != "ambient-key-sentinel", label


def test_unknown_named_slug_connection_view_also_fails_closed(monkeypatch):
    """The three-field view reports the same verdict, and ``custom_owned`` False.

    Callers that genuinely construct nothing else still take this view, so the
    self-labelled runtime must not reach them with a connection either.
    """
    _with_direct_config(monkeypatch, _SOLE_UNRELATED_ROW_CFG)

    provider, api_key, base_url, custom_owned = (
        config.apply_custom_provider_connection_authority(
            "custom:ghost",
            "ambient-key-sentinel",
            "https://ambient.example/v1",
            lookup_provider="custom:ghost",
            runtime_provider=copy.deepcopy(_SELF_LABELLED_GHOST_RUNTIME),
        )
    )

    assert base_url is None
    assert api_key is None
    assert provider == "custom:ghost"
    assert custom_owned is False, "nothing owns the slug, so the record cannot be reported as owning it"


def test_unowned_named_slug_fails_closed_at_the_first_streaming_resolution(monkeypatch):
    """The initial resolution refuses the unrelated row — terminally.

    The ambient runtime seeds a truthy URL, key and pool (plus api_mode and an
    ACP transport), and config holds exactly ONE custom row — named ``omni``,
    with a truthy URL and key of its own. The send asks for ``custom:ghost``,
    which nothing owns.

    This covers the FIRST of the three streaming regions that build an agent,
    and only that one: the verdict is terminal here, so no agent is constructed,
    no turn is sent, and no 401 exists for the two self-heal retries to act on.
    Parametrizing ``fail_first`` over this case would re-run identical code and
    prove nothing about those retry guards — they are exercised directly by
    ``test_retry_abandons_the_heal_*`` below, which start from a ROUTABLE route
    and break it mid-turn.
    """
    label = "initial send"
    session_id = "session-1806-ghost-initial"

    captured, apperrors = _run_composed_send_expecting_refusal(
        monkeypatch,
        copy.deepcopy(_SOLE_UNRELATED_ROW_CFG),
        dict(_AMBIENT_SIDE_FIELD_RUNTIME),
        session_id,
        model="@custom:ghost:antigravity/gemini-3.7-flash-tiered",
    )

    payload = apperrors[-1]
    assert "custom:ghost" in payload["message"], f"{label}: {payload}"
    assert "is not configured" in payload["message"], f"{label}: {payload}"

    # Nothing that could have been routed to may appear on the refusal path:
    # not the unrelated sole row's pair, not the ambient provider's, and never
    # the keyless placeholder.
    blob = str(payload) + str(captured)
    for leaked in (
        "https://omni.example/v1",
        "omni-key",
        _AMBIENT_SIDE_FIELD_RUNTIME["base_url"],
        _AMBIENT_SIDE_FIELD_RUNTIME["api_key"],
        config.KEYLESS_CUSTOM_API_KEY,
    ):
        assert leaked not in blob, f"{label}: {leaked!r} leaked into the refused route"

    assert not captured.get("explicit_client_kwargs_calls"), (
        f"{label}: a client was configured for a slug nothing owns"
    )


# ── declared-but-unresolved credentials are NOT keyless ──────────────────────


_MISSING_ENV_VAR = "HERMES_TEST_1806_UNSET_CREDENTIAL"


def _clear_credential_env(monkeypatch):
    """Remove every env var the credential ladder could still mint a key from."""
    monkeypatch.delenv(_MISSING_ENV_VAR, raising=False)
    monkeypatch.delenv("CUSTOM_OMNI_API_KEY", raising=False)


_UNRESOLVED_CREDENTIAL_ROWS = [
    # A declared ``${ENV}`` reference whose variable is unset.
    ({"api_key": "${" + _MISSING_ENV_VAR + "}"}, "env-reference"),
    # A declared ``key_env`` naming a variable that does not exist.
    ({"key_env": _MISSING_ENV_VAR}, "key-env"),
    # A configured credential pool that yields nothing for this endpoint.
    ({"credential_pool": ["configured-pool-sentinel"]}, "unavailable-pool"),
]


@pytest.mark.parametrize("row_fields,label", _UNRESOLVED_CREDENTIAL_ROWS)
def test_declared_but_unresolved_credential_is_not_keyless(monkeypatch, row_fields, label):
    """``keyless`` means "declares no credential", not "resolved no credential".

    Each row here DECLARES a credential source that produces nothing. Reporting
    that as keyless is what let ``dummy-key`` be substituted for a genuinely
    missing credential, turning a fixable misconfiguration into an opaque 401.
    """
    _clear_credential_env(monkeypatch)
    cfg_dict = _exact_list_row_cfg(drop=("api_key",), **row_fields)
    _with_direct_config(monkeypatch, cfg_dict)

    bundle = config.resolve_custom_provider_bundle("custom:omni")

    assert bundle["api_key"] is None, f"{label}: the credential must not resolve here"
    assert bundle["keyless"] is False, (
        f"{label}: a declared credential source was reported as keyless"
    )


@pytest.mark.parametrize("row_fields,label", _UNRESOLVED_CREDENTIAL_ROWS)
def test_declared_but_unresolved_credential_never_gets_the_dummy_key(
    monkeypatch, row_fields, label
):
    """End to end: the send STOPS — neither the placeholder nor a keyless send.

    The endpoint resolves (the row owns it), but the declared credential source
    produced nothing. Sending anyway with ``api_key=None`` is not the safe
    middle ground it looks like: with only one of the pair truthy, AIAgent falls
    through to ``_routed_client_kwargs()`` and re-resolves a provider, so the
    turn — and whatever credential init then finds — leaves for an endpoint the
    user never chose. The turn ends on the actionable cause instead.
    """
    _clear_credential_env(monkeypatch)

    captured, apperrors = _run_composed_send_expecting_refusal(
        monkeypatch,
        _exact_list_row_cfg(drop=("api_key",), **row_fields),
        _ambient_runtime(),
        f"session-1806-unresolved-{label}",
        model="@custom:omni:antigravity/gemini-3.7-flash-tiered",
    )

    payload = apperrors[-1]
    assert "produced no API key" in payload["message"], f"{label}: {payload}"
    assert "custom:omni" in payload["message"], (
        f"{label}: the refusal did not name the provider that failed: {payload}"
    )
    assert config.KEYLESS_CUSTOM_API_KEY not in str(payload), (
        f"{label}: keyless placeholder masked a declared-but-missing credential"
    )
    assert "keyed-key-sentinel-abc" not in str(payload), (
        f"{label}: the ambient keyed record leaked into the refusal"
    )
    # The row owns its endpoint, so nothing may have been dialled with it.
    assert not captured.get("explicit_client_kwargs_calls"), f"{label}: a client was configured"


def test_row_declaring_no_credential_is_still_keyless(monkeypatch):
    """The negative control: a genuinely unauthenticated endpoint keeps ``dummy-key``.

    Local OpenAI-compatible servers routinely run without auth, and tightening
    the keyless rule must not take that away — otherwise every keyless setup
    regresses into an agent built with no credential at all.
    """
    _clear_credential_env(monkeypatch)

    init_kwargs = _run_composed_send(
        monkeypatch,
        _exact_list_row_cfg(drop=("api_key",)),
        _ambient_runtime(),
        "session-1806-genuinely-keyless",
    )

    assert init_kwargs["api_key"] == config.KEYLESS_CUSTOM_API_KEY
    assert init_kwargs["base_url"] == _LIST_ROW_URL
    assert init_kwargs["provider"] == "custom"


# ── the exact row's credential is never the same-endpoint keyed row's ───────
#
# Every unresolved-credential case above puts the keyed record on a DIFFERENT
# endpoint, so the merge's provenance test already proves the ambient credential
# foreign. That leaves the collision untested: a keyed
# ``providers["custom:<slug>"]`` record may declare the SAME ``base_url`` as the
# exact ``custom_providers[]`` row, and then endpoint equality alone cannot tell
# the row's own resolution from the keyed row's. Inferring "same authority" from
# the URL there hands the keyed row's key to the exact row — the row's endpoint
# married to somebody else's credential, which is the split-authority merge the
# whole module exists to prevent.
#
# An exact row is a COMPLETE record: its credential comes from its own ladder or
# it does not come at all.


_SHARED_ENDPOINT_URL = "https://shared-url-sentinel.example/v1"
_SHARED_ENDPOINT_KEYED_KEY = "shared-endpoint-keyed-sentinel-def"


def _shared_endpoint_cfg(**row_fields):
    """Exact list row and same-slug keyed record on the IDENTICAL ``base_url``."""
    return {
        "model": {"default": "active/model", "provider": "custom:active"},
        "providers": {
            "custom:omni": {
                "base_url": _SHARED_ENDPOINT_URL,
                "api_key": _SHARED_ENDPOINT_KEYED_KEY,
            },
        },
        "custom_providers": [
            {"name": "omni", "base_url": _SHARED_ENDPOINT_URL, **row_fields},
        ],
    }


def _shared_endpoint_runtime():
    """The ambient runtime that resolution of the KEYED record produces.

    Same endpoint as the exact row, carrying the keyed record's credential — the
    state in which URL equality stops being evidence of provenance.
    """
    return _ambient_runtime(
        base_url=_SHARED_ENDPOINT_URL, api_key=_SHARED_ENDPOINT_KEYED_KEY
    )


@pytest.mark.parametrize("row_fields,label", _UNRESOLVED_CREDENTIAL_ROWS)
def test_exact_row_unresolved_credential_ignores_same_endpoint_keyed_key(
    monkeypatch, row_fields, label
):
    """The row DECLARED a credential that produced nothing: the route is terminal.

    The keyed record sharing the endpoint changes nothing about that. If the
    merge accepts ``_rt["api_key"]`` because the URLs match, the bundle silently
    authenticates the row's endpoint with the keyed row's secret and reports
    itself routable, so the user never learns their ``key_env`` is unset.
    """
    _clear_credential_env(monkeypatch)
    _with_direct_config(monkeypatch, _shared_endpoint_cfg(**row_fields))

    resolved = config.resolve_custom_provider_bundle("custom:omni")
    assert resolved["is_exact"] is True, f"{label}: the exact row was not selected"
    assert resolved["keyless"] is False, (
        f"{label}: a declared credential source was reported as keyless"
    )

    bundle = config.merge_custom_provider_runtime_bundle(
        "custom:omni",
        _SHARED_ENDPOINT_KEYED_KEY,
        _SHARED_ENDPOINT_URL,
        _shared_endpoint_runtime(),
        lookup_provider="custom:omni",
    )

    assert bundle["base_url"] == _SHARED_ENDPOINT_URL, f"{label}: the row lost its endpoint"
    assert bundle["api_key"] is None, (
        f"{label}: the exact row borrowed the same-endpoint keyed credential: {bundle!r}"
    )
    assert bundle["api_key"] != _SHARED_ENDPOINT_KEYED_KEY, f"{label}: {bundle!r}"
    assert bundle["api_key"] != config.KEYLESS_CUSTOM_API_KEY, (
        f"{label}: the placeholder masked a declared-but-missing credential"
    )
    verdict = config.custom_provider_route_error(bundle)
    assert verdict, f"{label}: an unroutable bundle reported itself routable: {bundle!r}"
    assert verdict["reason"] == config.CUSTOM_ROUTE_NO_CREDENTIAL, f"{label}: {verdict}"


@pytest.mark.parametrize("row_fields,label", _UNRESOLVED_CREDENTIAL_ROWS)
def test_composed_send_refuses_exact_row_sharing_the_keyed_endpoint(
    monkeypatch, row_fields, label
):
    """End to end: the turn stops instead of dialling with the keyed row's key.

    The production-composed path is where the borrowed credential would actually
    be spent — the endpoint resolves, so nothing downstream would question a
    truthy ``api_key``.
    """
    _clear_credential_env(monkeypatch)

    captured, apperrors = _run_composed_send_expecting_refusal(
        monkeypatch,
        _shared_endpoint_cfg(**row_fields),
        _shared_endpoint_runtime(),
        f"session-1806-shared-endpoint-{label}",
        model="@custom:omni:antigravity/gemini-3.7-flash-tiered",
    )

    payload = apperrors[-1]
    assert "produced no API key" in payload["message"], f"{label}: {payload}"
    assert "custom:omni" in payload["message"], f"{label}: {payload}"
    assert _SHARED_ENDPOINT_KEYED_KEY not in str(payload), (
        f"{label}: the same-endpoint keyed credential leaked into the refusal"
    )
    assert config.KEYLESS_CUSTOM_API_KEY not in str(payload), f"{label}: {payload}"
    assert not captured.get("explicit_client_kwargs_calls"), (
        f"{label}: a client was configured for a route whose credential failed"
    )


def test_exact_row_declaring_no_credential_ignores_the_shared_endpoint_key(monkeypatch):
    """The other half of the rule: a keyless row does not borrow the key either.

    The row declares NO credential, which is its own statement that the endpoint
    is unauthenticated. ``dummy-key`` follows from the row, not from the keyed
    record that happens to point at the same URL — so the route stays routable
    and the keyed secret still never leaves the keyed record.
    """
    _clear_credential_env(monkeypatch)
    _with_direct_config(monkeypatch, _shared_endpoint_cfg())

    bundle = config.merge_custom_provider_runtime_bundle(
        "custom:omni",
        _SHARED_ENDPOINT_KEYED_KEY,
        _SHARED_ENDPOINT_URL,
        _shared_endpoint_runtime(),
        lookup_provider="custom:omni",
    )

    assert bundle["base_url"] == _SHARED_ENDPOINT_URL
    assert bundle["api_key"] == config.KEYLESS_CUSTOM_API_KEY, (
        f"the keyless row did not keep its own credential verdict: {bundle!r}"
    )
    assert bundle["api_key"] != _SHARED_ENDPOINT_KEYED_KEY, bundle
    assert config.custom_provider_route_error(bundle) is None, bundle


# ─────────────────────────────────────────────────────────────────────────────
# The two 401 self-heal RETRIES are route boundaries too
#
# Everything above stops a terminal route at the FIRST resolution, where no agent
# exists yet. The self-heal retries are a different boundary, and a strictly
# worse one: the route WAS routable when the turn started, an agent was built and
# written to ``SESSION_AGENT_CACHE`` under a valid bundle signature, the provider
# answered 401, and only THEN does the re-resolve run. Whatever that second
# resolution returns is what the retry agent gets constructed from.
#
# That window is real state, not a hypothetical: between the first send and the
# heal the owning ``custom_providers[]`` row can be edited, renamed or deleted in
# Settings, and its ``key_env`` / pool can stop yielding a credential. The
# refreshed bundle is then terminal, and building the retry agent from it hands
# ``_init_openai_client()`` an incomplete pair — so the retry, the send that
# actually carries the user's prompt, is the one that re-enters
# ``_routed_client_kwargs()``. It also overwrites a GOOD cache entry with the
# poisoned agent, so every later turn in the session reuses it.
#
# ``heal_mutate`` runs inside the stubbed ``_attempt_credential_self_heal`` —
# after the first agent has already failed with its 401, before the retry
# re-resolves its bundle — which is the only point at which that transition can
# be staged. Each test below asserts the first send really happened (one agent,
# one turn, one cache write), so a regression that makes the route terminal up
# front cannot pass these by the back door.
# ─────────────────────────────────────────────────────────────────────────────


# No ``providers:`` block and a ``model.provider`` naming an UNRELATED slug: the
# single list row is the ONLY authority for ``custom:omni``, so removing or
# renaming it below leaves the slug owned by nothing at all. A same-slug keyed
# record or a ``model:`` block naming ``custom:omni`` would each still be an
# authority and would make the "unowned" cases test something weaker.
_RETRY_OWNED_CFG = {
    "model": {"default": "active/model", "provider": "custom:active"},
    "custom_providers": [
        {
            "name": "omni",
            "base_url": _LIST_ROW_URL,
            "api_key": _LIST_ROW_KEY,
        },
    ],
}

_RETRY_MODEL = "@custom:omni:antigravity/gemini-3.7-flash-tiered"


def _retry_cfg_rows(**row_fields):
    """The owning row with ``row_fields`` applied; ``None`` values drop the field."""
    row = dict(_RETRY_OWNED_CFG["custom_providers"][0])
    for field, value in row_fields.items():
        if value is None:
            row.pop(field, None)
        else:
            row[field] = value
    return [row]


def _drop_the_owning_row():
    """The row is deleted mid-turn — nothing owns ``custom:omni`` any more."""
    config.cfg["custom_providers"] = []


def _rename_the_owning_row():
    """The row is renamed mid-turn, so it now owns a DIFFERENT slug.

    The pair is still sitting in config, which is exactly the shape that made
    the sole-unrelated-row fallback look harmless: ``custom:omni`` must not
    inherit it just because it is the only row left.
    """
    config.cfg["custom_providers"] = _retry_cfg_rows(name="omni-renamed")


def _malform_the_owning_row():
    """The row loses its name, so it names no slug and therefore owns none.

    ``CUSTOM_SELECTION_MALFORMED`` proper describes a bare ``custom:`` lookup
    with no slug behind it, which cannot arise mid-turn — the lookup is fixed at
    the first resolve. A row that stops naming any identity is the reachable
    malformed-record analogue, and it must fail closed the same way.
    """
    config.cfg["custom_providers"] = _retry_cfg_rows(name=None)


_RETRY_UNOWNED_MUTATIONS = [
    (_drop_the_owning_row, "deleted", "row deleted"),
    (_rename_the_owning_row, "renamed", "row renamed to another slug"),
    (_malform_the_owning_row, "unnamed", "row lost its name"),
]


def _retry_credential_mutation(row_fields):
    """Return a heal mutation that keeps the endpoint but breaks the credential."""

    def _mutate():
        # ``api_key`` first so a row that DECLARES one (the ``${ENV}`` shape)
        # overrides the drop, while ``key_env`` / ``credential_pool`` rows keep
        # it dropped. Either way the row still owns its endpoint.
        fields = {"api_key": None, **row_fields}
        config.cfg["custom_providers"] = _retry_cfg_rows(**fields)

    return _mutate


def _run_composed_retry_expecting_abandoned_heal(
    monkeypatch, cfg_dict, runtime_dict, session_id, *, fail_first, heal_mutate
):
    """Drive a send that starts routable and turns terminal at the 401 retry.

    Returns ``(captured, apperrors, cache_at_heal, cache_after)``. Both cache
    views are read while the worker's entries are still live — ``restore()``
    clears ``SESSION_AGENT_CACHE`` — so the caller can compare what the initial
    send cached against what survived the abandoned heal.
    """
    import api.streaming as streaming

    cache_at_heal = {}

    def _stage_the_terminal_retry():
        # Snapshot the entry the FIRST (routable) agent was cached under, taken
        # before the route is broken. The abandoned retry must leave it exactly
        # as it is: replacing it is how one mid-turn config edit would re-route
        # every remaining turn in the session.
        with config.SESSION_AGENT_CACHE_LOCK:
            cache_at_heal.update(config.SESSION_AGENT_CACHE)
        heal_mutate()

    stream_id, q, captured, restore = _setup_production_composed_runtime(
        monkeypatch,
        copy.deepcopy(cfg_dict),
        runtime_dict,
        session_id=session_id,
        fail_first=fail_first,
        heal_mutate=_stage_the_terminal_retry,
    )
    try:
        streaming.STREAMS[stream_id] = q
        streaming._run_agent_streaming(
            session_id=session_id,
            msg_text="hello",
            model=_RETRY_MODEL,
            workspace="/tmp",
            stream_id=stream_id,
        )
        apperrors = _drain_apperrors(q)
        with config.SESSION_AGENT_CACHE_LOCK:
            cache_after = dict(config.SESSION_AGENT_CACHE)
    finally:
        streaming.STREAMS.pop(stream_id, None)
        streaming.AGENT_INSTANCES.pop(stream_id, None)
        restore()
    return captured, apperrors, cache_at_heal, cache_after


def _assert_retry_abandoned(
    captured, apperrors, cache_at_heal, cache_after, session_id, label, *, expected_cause
):
    """Assert the retry stopped at the refreshed route verdict, not at the 401."""
    # (0) The case is only meaningful if the FIRST send was routable and really
    # ran. Without this, a regression that made the route terminal at initial
    # resolution would satisfy every assertion below for the wrong reason.
    history = captured.get("init_kwargs_history", [])
    assert history, f"{label}: the initial send never constructed an agent"
    assert history[0]["base_url"] == _LIST_ROW_URL, (
        f"{label}: the initial send did not resolve the owning row, so no 401 "
        f"retry path was ever reached: {history[0]}"
    )
    assert history[0]["api_key"] == _LIST_ROW_KEY, f"{label}: {history[0]}"
    assert cache_at_heal.get(session_id), (
        f"{label}: the initial agent was never cached, so this case cannot show "
        f"that the abandoned retry left a good cache entry alone"
    )

    # (1) No second agent — the retry construction is what the guard prevents.
    assert len(history) == 1, (
        f"{label}: the retry constructed a second agent on a route that had "
        f"become terminal ({len(history)} constructions: {history})"
    )
    assert len(captured.get("instances", [])) == 1, label

    # (2) _routed_client_kwargs() is never reached. The one explicit call is the
    # initial, complete pair; anything else means an incomplete pair reached
    # _init_openai_client() and re-entered provider routing.
    routed = captured.get("routed_client_kwargs_calls", [])
    assert not routed, (
        f"{label}: the retry agent was built with an incomplete connection pair "
        f"{routed}, so _init_openai_client() fell through to "
        f"_routed_client_kwargs() and re-resolved a provider"
    )
    explicit = captured.get("explicit_client_kwargs_calls", [])
    assert explicit == [{"api_key": _LIST_ROW_KEY, "base_url": _LIST_ROW_URL}], (
        f"{label}: expected only the initial send's explicit pair, got {explicit}"
    )

    # (3) The retry turn was never sent.
    assert captured.get("run_calls") == 1, (
        f"{label}: run_conversation ran {captured.get('run_calls')!r} times — the "
        f"retry turn was sent on a route that no longer resolves"
    )

    # (4) The cache still holds the FIRST agent under the FIRST bundle
    # signature. A retry write here would be the durable half of the defect:
    # later turns reuse the cached agent without re-resolving at all.
    assert list(cache_after) == [session_id], (
        f"{label}: unexpected agent-cache contents {list(cache_after)}"
    )
    assert cache_after[session_id][0] is cache_at_heal[session_id][0], (
        f"{label}: the agent cache was rewritten with a retry agent built on an "
        f"unroutable bundle"
    )
    assert cache_after[session_id][1] == cache_at_heal[session_id][1], (
        f"{label}: a cache entry was written under the invalid bundle's signature"
    )
    assert cache_after[session_id][0] is captured["instances"][0], label

    # (5) The client is told the real cause, not the 401 that triggered the heal.
    assert apperrors, f"{label}: no controlled failure was emitted"
    payload = apperrors[-1]
    assert payload["type"] == "provider_unroutable", (
        f"{label}: emitted {payload['type']!r} instead of a provider-route "
        f"failure — the 401 is the symptom, the unroutable route is the cause"
    )
    assert expected_cause in payload.get("message", ""), (
        f"{label}: the failure did not name {expected_cause!r}: {payload}"
    )
    assert "custom:omni" in payload.get("message", ""), (
        f"{label}: the failure did not name the provider that failed: {payload}"
    )
    assert payload.get("hint"), f"{label}: the failure named no fix"

    # Nothing the abandoned retry could have been re-routed to may appear on the
    # refusal path, and the keyless placeholder may never stand in for it.
    blob = str(payload)
    for leaked in (
        _AMBIENT_SIDE_FIELD_RUNTIME["base_url"],
        _AMBIENT_SIDE_FIELD_RUNTIME["api_key"],
        config.KEYLESS_CUSTOM_API_KEY,
    ):
        assert leaked not in blob, f"{label}: {leaked!r} leaked into the refusal"
    return payload


@pytest.mark.parametrize("fail_first", ["returned_error", "raised"])
@pytest.mark.parametrize(
    "heal_mutate,mutation_slug,mutation_label",
    _RETRY_UNOWNED_MUTATIONS,
    ids=[slug for _fn, slug, _label in _RETRY_UNOWNED_MUTATIONS],
)
def test_retry_abandons_the_heal_when_the_slug_stops_being_owned(
    monkeypatch, fail_first, heal_mutate, mutation_slug, mutation_label
):
    """Case A — the refreshed route is unowned, so the retry must not be built.

    The turn starts on a row that owns ``custom:omni`` outright, so an agent is
    constructed with the row's exact pair and the turn is sent. The provider
    answers 401. Before the heal re-resolves, the row stops owning the slug.

    Building the retry from that bundle is the whole defect: its key and URL are
    empty, which ``_init_openai_client()`` reads as "resolve a provider yourself"
    — and the ambient runtime dict the heal returns still carries a truthy
    endpoint, credential and pool for a provider the user never named.
    """
    label = f"{mutation_label} / {fail_first}"
    session_id = f"session-1806-retry-unowned-{fail_first}-{mutation_slug}"

    captured, apperrors, cache_at_heal, cache_after = (
        _run_composed_retry_expecting_abandoned_heal(
            monkeypatch,
            _RETRY_OWNED_CFG,
            _ambient_runtime(),
            session_id,
            fail_first=fail_first,
            heal_mutate=heal_mutate,
        )
    )

    _assert_retry_abandoned(
        captured,
        apperrors,
        cache_at_heal,
        cache_after,
        session_id,
        label,
        expected_cause="is not configured",
    )


@pytest.mark.parametrize("fail_first", ["returned_error", "raised"])
@pytest.mark.parametrize("row_fields,credential_label", _UNRESOLVED_CREDENTIAL_ROWS)
def test_retry_abandons_the_heal_when_the_refreshed_credential_resolves_to_nothing(
    monkeypatch, fail_first, row_fields, credential_label
):
    """Case B — the row still owns the route, but its credential now yields nothing.

    This is the shape a 401 self-heal is most likely to meet in the wild: the
    401 happened BECAUSE the credential went away, so the re-resolve finds the
    same row with a declared-but-unresolved source. ``keyless`` is False there,
    so no ``dummy-key`` is substituted and the pair stays incomplete — and an
    incomplete pair at the constructor is a re-route, not a refusal.

    The retry therefore has to stop, and the turn has to end naming the
    credential setting rather than the 401 it produced.
    """
    _clear_credential_env(monkeypatch)
    label = f"{credential_label} / {fail_first}"
    session_id = f"session-1806-retry-nocred-{fail_first}-{credential_label}"

    captured, apperrors, cache_at_heal, cache_after = (
        _run_composed_retry_expecting_abandoned_heal(
            monkeypatch,
            _RETRY_OWNED_CFG,
            _ambient_runtime(),
            session_id,
            fail_first=fail_first,
            heal_mutate=_retry_credential_mutation(row_fields),
        )
    )

    _assert_retry_abandoned(
        captured,
        apperrors,
        cache_at_heal,
        cache_after,
        session_id,
        label,
        expected_cause="produced no API key",
    )


def test_retry_still_succeeds_when_the_refreshed_route_is_still_routable(monkeypatch):
    """The negative control: a heal that re-resolves a GOOD route still retries.

    Without this, every assertion above would also pass if the guards simply
    abandoned all self-heals — which would take the #1401 credential-refresh
    retry away entirely. Here the mutation swaps the row's key for a NEW one
    (the refresh a heal exists to pick up), so the retry must be constructed
    with the new pair and the turn must be sent a second time.
    """
    session_id = "session-1806-retry-still-routable"

    def _rotate_the_credential():
        config.cfg["custom_providers"] = _retry_cfg_rows(api_key="rotated-key-sentinel")

    import api.streaming as streaming

    stream_id, q, captured, restore = _setup_production_composed_runtime(
        monkeypatch,
        copy.deepcopy(_RETRY_OWNED_CFG),
        _ambient_runtime(),
        session_id=session_id,
        fail_first="returned_error",
        heal_mutate=_rotate_the_credential,
    )
    try:
        streaming.STREAMS[stream_id] = q
        streaming._run_agent_streaming(
            session_id=session_id,
            msg_text="hello",
            model=_RETRY_MODEL,
            workspace="/tmp",
            stream_id=stream_id,
        )
        apperrors = _drain_apperrors(q)
    finally:
        streaming.STREAMS.pop(stream_id, None)
        streaming.AGENT_INSTANCES.pop(stream_id, None)
        restore()

    history = captured["init_kwargs_history"]
    assert len(history) == 2, f"the routable heal did not retry: {history}"
    assert history[1]["api_key"] == "rotated-key-sentinel"
    assert history[1]["base_url"] == _LIST_ROW_URL
    assert captured["run_calls"] == 2, "the refreshed credential never sent a turn"
    assert not apperrors, f"a routable retry emitted a failure: {apperrors}"


# ─────────────────────────────────────────────────────────────────────────────
# The generic bare-``custom`` authority owns NO named slug
#
# ``providers['custom']`` and a ``model:`` block whose provider is the bare
# string ``custom`` name no slug at all. While they were still eligible
# candidates for every ``custom:<slug>`` lookup, ``custom:ghost`` selected one of
# them and inherited its endpoint, credential, credential pool, ``api_mode`` and
# ACP transport — the same wrong-authority pairing the sole-unrelated-row rule
# above exists to prevent, only sourced from ``providers:``/``model:`` instead of
# ``custom_providers[]``. A named route is identity-owned: exact slug match, or
# nothing.
# ─────────────────────────────────────────────────────────────────────────────


_BARE_CUSTOM_PROVIDERS_RECORD = {
    "base_url": "https://bare-providers-sentinel.example/v1",
    "api_key": "bare-providers-key-sentinel",
    "api_mode": "anthropic_messages",
    "credential_pool": ["bare-pool-sentinel"],
    "acp_command": "bare-acp-sentinel",
    "acp_args": ["--bare-arg-sentinel"],
}

_BARE_CUSTOM_MODEL_BLOCK = {
    "provider": "custom",
    "base_url": "https://bare-model-sentinel.example/v1",
    "api_key": "bare-model-key-sentinel",
}


def _bare_custom_cfg(*, providers_record=True, model_block=True):
    """Config carrying the generic bare-``custom`` authority, but no ``ghost``."""
    cfg_dict = {
        "model": {"default": "ghostly/model"},
        "providers": {},
        "custom_providers": [
            {
                "name": "omni",
                "base_url": "https://omni.example/v1",
                "api_key": "omni-key",
            },
        ],
    }
    if providers_record:
        cfg_dict["providers"]["custom"] = copy.deepcopy(_BARE_CUSTOM_PROVIDERS_RECORD)
    if model_block:
        cfg_dict["model"].update(copy.deepcopy(_BARE_CUSTOM_MODEL_BLOCK))
    return cfg_dict


def _clear_ghost_credential_env(monkeypatch):
    """Keep the ``CUSTOM_<SLUG>_API_KEY`` convention out of these verdicts."""
    monkeypatch.delenv("CUSTOM_GHOST_API_KEY", raising=False)
    monkeypatch.delenv("CUSTOM_OMNI_API_KEY", raising=False)


_BARE_CUSTOM_SHAPES = [
    (True, False, "providers['custom'] only"),
    (False, True, "model.provider: custom only"),
    (True, True, "both bare-custom authorities"),
]


@pytest.mark.parametrize(
    "providers_record,model_block,label",
    _BARE_CUSTOM_SHAPES,
    ids=["providers-custom", "model-provider-custom", "both"],
)
def test_unknown_named_slug_never_claims_the_bare_custom_authority(
    monkeypatch, providers_record, model_block, label
):
    """``custom:ghost`` must fail closed even with a generic ``custom`` record present.

    Neither authority names ``ghost``, so neither owns the route. Selecting one
    would pair the user's prompt with an endpoint and a credential they never
    pointed this slug at.
    """
    _clear_ghost_credential_env(monkeypatch)
    _with_direct_config(
        monkeypatch, _bare_custom_cfg(providers_record=providers_record, model_block=model_block)
    )

    assert config.resolve_custom_provider_connection("custom:ghost") == (None, None), label

    ghost = config.resolve_custom_provider_bundle("custom:ghost")
    assert ghost["status"] == config.CUSTOM_SELECTION_MISSING, label
    assert ghost["source"] == "", label
    assert ghost["record"] is None, f"{label}: a bare-custom record was selected for a named slug"
    assert ghost["base_url"] is None, label
    assert ghost["api_key"] is None, label
    assert ghost["keyless"] is False, f"{label}: an unowned route was reported keyless"
    assert ghost["owned"] == {}, f"{label}: an unowned route claimed side fields"

    # The named row is still authoritative for its OWN slug.
    owned = config.resolve_custom_provider_bundle("custom:omni")
    assert owned["status"] == config.CUSTOM_SELECTION_EXACT, label
    assert owned["base_url"] == "https://omni.example/v1", label


@pytest.mark.parametrize(
    "providers_record,model_block,label",
    _BARE_CUSTOM_SHAPES,
    ids=["providers-custom", "model-provider-custom", "both"],
)
def test_unknown_named_slug_inherits_no_bare_custom_connection_or_side_fields(
    monkeypatch, providers_record, model_block, label
):
    """The merge carries nothing from the bare-``custom`` record onto ``ghost``.

    Not the endpoint, not the credential, and — because a bare record's
    ``api_mode``/pool/ACP transport are as unowned as its URL — none of the side
    fields either.
    """
    _clear_ghost_credential_env(monkeypatch)
    _with_direct_config(
        monkeypatch, _bare_custom_cfg(providers_record=providers_record, model_block=model_block)
    )

    bundle = config.merge_custom_provider_runtime_bundle(
        "custom:ghost",
        "ambient-key-sentinel",
        "https://ambient.example/v1",
        dict(_AMBIENT_SIDE_FIELD_RUNTIME),
        lookup_provider="custom:ghost",
    )

    _assert_ghost_bundle_failed_closed(bundle, label)
    assert bundle["base_url"] != _BARE_CUSTOM_PROVIDERS_RECORD["base_url"], label
    assert bundle["base_url"] != _BARE_CUSTOM_MODEL_BLOCK["base_url"], label
    assert bundle["api_key"] != _BARE_CUSTOM_PROVIDERS_RECORD["api_key"], label
    assert bundle["api_key"] != _BARE_CUSTOM_MODEL_BLOCK["api_key"], label


def test_unknown_named_slug_connection_view_ignores_the_bare_custom_record(monkeypatch):
    """The three-field view reaches the same verdict, and reports ``custom_owned`` False."""
    _clear_ghost_credential_env(monkeypatch)
    _with_direct_config(monkeypatch, _bare_custom_cfg())

    provider, api_key, base_url, custom_owned = (
        config.apply_custom_provider_connection_authority(
            "custom:ghost",
            "ambient-key-sentinel",
            "https://ambient.example/v1",
            lookup_provider="custom:ghost",
            runtime_provider=copy.deepcopy(_SELF_LABELLED_GHOST_RUNTIME),
        )
    )

    assert base_url is None
    assert api_key is None
    assert provider == "custom:ghost"
    assert custom_owned is False


def test_bare_custom_route_still_uses_the_bare_custom_authority(monkeypatch):
    """The carve-out: closing the generic path for NAMED slugs, not for bare ``custom``.

    ``providers['custom']`` is the bare route's own identity record, so it stays
    eligible there — the tightening above is about a named slug borrowing an
    authority that never named it.
    """
    _clear_ghost_credential_env(monkeypatch)
    _with_direct_config(monkeypatch, _bare_custom_cfg(model_block=False))

    bare = config.resolve_custom_provider_bundle("custom:custom")
    assert bare["status"] == config.CUSTOM_SELECTION_KEYED
    assert bare["source"] == "providers"
    assert bare["base_url"] == _BARE_CUSTOM_PROVIDERS_RECORD["base_url"]
    assert bare["api_key"] == _BARE_CUSTOM_PROVIDERS_RECORD["api_key"]
    assert bare["owned"]["api_mode"] == "anthropic_messages"
    assert bare["owned"]["credential_pool"] == ["bare-pool-sentinel"]
    assert bare["owned"]["acp_command"] == "bare-acp-sentinel"


def test_model_block_naming_the_slug_outright_still_owns_it(monkeypatch):
    """Only the GENERIC ``custom`` spelling is refused, not an explicit slug.

    A ``model:`` block whose provider is ``custom:ghost`` names this identity, so
    it remains the route's authority.
    """
    _clear_ghost_credential_env(monkeypatch)
    cfg_dict = _bare_custom_cfg(model_block=False)
    cfg_dict["model"].update(
        {
            "provider": "custom:ghost",
            "base_url": "https://named-model-sentinel.example/v1",
            "api_key": "named-model-key-sentinel",
        }
    )
    _with_direct_config(monkeypatch, cfg_dict)

    ghost = config.resolve_custom_provider_bundle("custom:ghost")
    assert ghost["status"] == config.CUSTOM_SELECTION_KEYED
    assert ghost["source"] == "model"
    assert ghost["base_url"] == "https://named-model-sentinel.example/v1"
    assert ghost["api_key"] == "named-model-key-sentinel"
    # And it beats the bare-custom record that is still sitting in providers:.
    assert ghost["base_url"] != _BARE_CUSTOM_PROVIDERS_RECORD["base_url"]


# ─────────────────────────────────────────────────────────────────────────────
# An identity-keyed record owning ONLY side fields or a dynamic credential
#
# ``providers['custom:<slug>']`` names this provider outright, so whatever it
# declares, it owns. Judging it by "did a STATIC api_key resolve, or is there a
# base_url?" classified a perfectly valid ``key_cmd``-only / ``api_mode``-only /
# pool-only / ACP-only record as ``missing`` — and the merge then CLEARED the
# very fields that record exists to supply, reverting them to the ambient
# runtime's.
# ─────────────────────────────────────────────────────────────────────────────


def _keyed_side_field_cfg(record):
    """Config whose only authority for ``omni`` is the keyed record ``record``."""
    return {
        "model": {"default": "active/model", "provider": "custom:active"},
        "providers": {"custom:omni": copy.deepcopy(record)},
        "custom_providers": [
            {
                "name": "active",
                "base_url": "https://active.example/v1",
                "api_key": "active-key",
            },
        ],
    }


_KEYED_SIDE_FIELD_ONLY_RECORDS = [
    (
        {"api_mode": "anthropic_messages"},
        {"api_mode": "anthropic_messages"},
        True,
        "api_mode only",
    ),
    (
        {"transport": "anthropic"},
        {"api_mode": "anthropic_messages"},
        True,
        "transport alias only",
    ),
    (
        {"credential_pool": ["keyed-pool-sentinel"]},
        {"credential_pool": ["keyed-pool-sentinel"]},
        # A configured pool is a declared credential source, so NOT keyless.
        False,
        "credential_pool only",
    ),
    (
        {"acp_command": "keyed-acp-sentinel", "acp_args": ["--keyed-arg-sentinel"]},
        {"acp_command": "keyed-acp-sentinel", "acp_args": ["--keyed-arg-sentinel"]},
        True,
        "ACP transport only",
    ),
]


@pytest.mark.parametrize(
    "record,expected_owned,expected_keyless,label",
    _KEYED_SIDE_FIELD_ONLY_RECORDS,
    ids=["api-mode", "transport-alias", "credential-pool", "acp"],
)
def test_keyed_record_owning_only_side_fields_is_not_missing(
    monkeypatch, record, expected_owned, expected_keyless, label
):
    """A keyed record with no static URL/key still OWNS what it declares."""
    _clear_credential_env(monkeypatch)
    _with_direct_config(monkeypatch, _keyed_side_field_cfg(record))

    bundle = config.resolve_custom_provider_bundle("custom:omni")
    assert bundle["status"] == config.CUSTOM_SELECTION_KEYED, (
        f"{label}: a keyed record that owns side fields was classified as missing"
    )
    assert bundle["source"] == "providers", label
    assert bundle["record"] is not None, label
    assert bundle["owned"] == expected_owned, label
    assert bundle["is_exact"] is False, label
    # It declares no endpoint of its own; that is unowned, not invalid.
    assert bundle["base_url"] is None, label
    assert bundle["keyless"] is expected_keyless, label


def test_keyed_record_owning_only_key_cmd_is_not_missing(monkeypatch):
    """``key_cmd`` is a real credential source, so a ``key_cmd``-only record owns the route.

    It resolves no STATIC key by design — the command mints a short-lived bearer
    per request — so a static-key test reported the record as missing and the
    route fell back to the ambient authority.
    """
    _clear_credential_env(monkeypatch)

    def _token_provider():
        return "minted-bearer-sentinel"

    _fake_command_token_source(monkeypatch, lambda _key_cmd, _name: _token_provider)
    _with_direct_config(monkeypatch, _keyed_side_field_cfg({"key_cmd": "print-omni-bearer"}))

    bundle = config.resolve_custom_provider_bundle("custom:omni")
    assert bundle["status"] == config.CUSTOM_SELECTION_KEYED
    assert bundle["source"] == "providers"
    assert bundle["api_key"] is _token_provider, "the keyed record's key_cmd token source was lost"
    assert bundle["keyless"] is False, "a declared key_cmd must never be reported keyless"


def test_keyed_record_owning_only_unresolved_env_credential_is_not_missing(monkeypatch):
    """A declared-but-unset ``key_env`` still names this slug's credential source.

    Treating it as missing sends the route to the ambient endpoint instead of
    surfacing the misconfiguration.
    """
    _clear_credential_env(monkeypatch)
    _with_direct_config(monkeypatch, _keyed_side_field_cfg({"key_env": _MISSING_ENV_VAR}))

    bundle = config.resolve_custom_provider_bundle("custom:omni")
    assert bundle["status"] == config.CUSTOM_SELECTION_KEYED
    assert bundle["api_key"] is None
    assert bundle["keyless"] is False, "a declared-but-unresolved credential is not keyless"


_KEYED_SIDE_FIELDS_ONLY_CFG = {
    "model": {"default": "active/model", "provider": "custom:active"},
    "providers": {
        "custom:omni": {
            # No base_url, no api_key: side fields are ALL this record declares.
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


def test_keyed_record_side_fields_survive_the_merge_without_a_static_pair(monkeypatch):
    """The merge applies the record's owned fields instead of clearing them."""
    _clear_credential_env(monkeypatch)
    _with_direct_config(monkeypatch, copy.deepcopy(_KEYED_SIDE_FIELDS_ONLY_CFG))

    bundle = config.merge_custom_provider_runtime_bundle(
        "custom:omni",
        "ambient-key-sentinel",
        "https://ambient.example/v1",
        _ambient_runtime(
            api_mode="chat_completions",
            command="ambient-acp-sentinel",
            args=["--ambient-arg-sentinel"],
            credential_pool=["ambient-pool-sentinel"],
        ),
        lookup_provider="custom:omni",
    )

    _assert_side_fields(
        bundle,
        {
            "api_mode": "anthropic_messages",
            "credential_pool": ["keyed-pool-sentinel"],
            "acp_command": "keyed-acp-sentinel",
            "acp_args": ["--keyed-arg-sentinel"],
        },
        "keyed record owning only side fields",
    )


def test_keyed_record_side_fields_survive_the_runtime_bundle(monkeypatch):
    """End to end: the production-composed send constructs with the record's fields.

    The record declares no endpoint, so the runtime's stands (the keyed
    fill-only rule) — but every constructor field the record DOES own reaches
    the agent instead of the ambient provider's value.
    """
    _clear_credential_env(monkeypatch)

    init_kwargs = _run_composed_send(
        monkeypatch,
        copy.deepcopy(_KEYED_SIDE_FIELDS_ONLY_CFG),
        _ambient_runtime(
            api_mode="chat_completions",
            command="ambient-acp-sentinel",
            args=["--ambient-arg-sentinel"],
            credential_pool=["ambient-pool-sentinel"],
        ),
        "session-1806-keyed-side-fields-only",
    )

    _assert_side_fields(
        init_kwargs,
        {
            "api_mode": "anthropic_messages",
            "credential_pool": ["keyed-pool-sentinel"],
            "acp_command": "keyed-acp-sentinel",
            "acp_args": ["--keyed-arg-sentinel"],
        },
        "keyed side-field-only record through the runtime bundle",
    )
    assert init_kwargs["provider"] == "custom"


# ─────────────────────────────────────────────────────────────────────────────
# Non-streaming consumers: the verdict is terminal off the streaming path too
#
# ``api/streaming.py`` is not the only thing that builds an AIAgent from a
# merged bundle. POST /api/chat and the four auxiliary consumers all go through
# ``routes._resolve_agent_connection_bundle()``, which is the single chokepoint
# that must RAISE rather than hand back a bundle with a hole in it: an
# incomplete ``(api_key, base_url)`` pair is not a refusal at the constructor,
# it is the signal to re-resolve a provider through ``_routed_client_kwargs()``.
# These pin the raise, the 400 POST /api/chat turns it into, and the negative
# control that a routable bundle still comes back untouched.
# ─────────────────────────────────────────────────────────────────────────────


def _terminal_route_cfg(kind):
    """Config producing each of the three terminal verdicts for one slug."""
    if kind == "unowned":
        # Exactly one row, named something else: nothing owns ``custom:ghost``.
        return copy.deepcopy(_SOLE_UNRELATED_ROW_CFG)
    if kind == "no_endpoint":
        # The exact row owns the slug and blanks the endpoint, so the keyed
        # row's URL must not be reachable by falling through.
        return _exact_list_row_cfg(base_url="")
    # The exact row declares a credential source that resolves to nothing.
    return _exact_list_row_cfg(drop=("api_key",), key_env=_MISSING_ENV_VAR)


_TERMINAL_ROUTE_CASES = [
    ("unowned", "custom:ghost", config.CUSTOM_ROUTE_UNOWNED, "is not configured", "Settings"),
    (
        "no_endpoint",
        "custom:omni",
        config.CUSTOM_ROUTE_NO_ENDPOINT,
        "resolved no endpoint",
        "base_url",
    ),
    (
        "no_credential",
        "custom:omni",
        config.CUSTOM_ROUTE_NO_CREDENTIAL,
        "produced no API key",
        "key_env",
    ),
]


@pytest.mark.parametrize(
    "kind,slug,expected_reason,message_fragment,hint_fragment", _TERMINAL_ROUTE_CASES
)
def test_agent_connection_bundle_raises_on_terminal_route(
    monkeypatch, kind, slug, expected_reason, message_fragment, hint_fragment
):
    """The non-streaming chokepoint refuses instead of returning a holed bundle.

    Returning ``{"base_url": None, ...}`` here would look terminal to the caller
    and be the opposite at the constructor, so all five consumers that share
    this helper would each have had to remember to check. Raising makes the
    refusal structural — and the exception subclasses ``ValueError``, which is
    what the existing handlers at those call sites already catch.
    """
    import api.routes as routes

    _clear_credential_env(monkeypatch)
    _with_direct_config(monkeypatch, _terminal_route_cfg(kind))

    # Mirror POST /api/chat's own sequence: resolve the model's provider, then
    # let the ambient runtime fill the endpoint/credential it did not resolve.
    _model, provider, base_url = config.resolve_model_provider(
        f"@{slug}:antigravity/gemini-3.7-flash-tiered"
    )
    runtime = copy.deepcopy(_ROUTE_KEYED_RUNTIME)
    api_key = runtime["api_key"]
    if not base_url:
        base_url = runtime["base_url"]

    with pytest.raises(config.CustomProviderRouteError) as excinfo:
        routes._resolve_agent_connection_bundle(provider, api_key, base_url, runtime)

    err = excinfo.value
    assert isinstance(err, ValueError), (
        "the existing except-ValueError handlers at the five call sites must keep catching it"
    )
    assert err.reason == expected_reason
    assert err.provider == slug
    assert message_fragment in err.message, err.message
    assert hint_fragment in err.hint, err.hint


def test_agent_connection_bundle_returns_routable_bundle_unchanged(monkeypatch):
    """The negative control: a routable named route still comes back, not raised.

    Without this, the three cases above are equally satisfied by a helper that
    refuses everything — which would take every working custom provider down.
    """
    import api.routes as routes

    _with_direct_config(monkeypatch, copy.deepcopy(_KEYED_VS_LIST_CFG))

    bundle = routes._resolve_agent_connection_bundle(
        "custom:omni",
        "keyed-key-sentinel-abc",
        "https://keyed-url-sentinel.example/v1",
        copy.deepcopy(_ROUTE_KEYED_RUNTIME),
    )

    assert bundle["base_url"] == _LIST_ROW_URL
    assert bundle["api_key"] == _LIST_ROW_KEY
    assert bundle[config.CUSTOM_ROUTE_ERROR_FIELD] is None


def _drive_sync_chat_route_expecting_refusal(monkeypatch, cfg_dict, slug):
    """Drive POST /api/chat to its refusal; return ``(payload, status, captured)``.

    The session carries the explicit ``@custom:<slug>:<model>`` form the picker
    emits, which ``model_with_provider_context()`` passes through untouched.
    That matters for the unowned case: a BARE model plus a stale
    ``model_provider`` never mints an ``@custom:ghost:`` route in the first
    place (the #7356 guard drops the hint), so the explicit form is the shape in
    which an unowned slug actually reaches this route.
    """
    import api.routes as routes

    captured, fake_session = _setup_route_consumer_runtime(
        monkeypatch,
        session_messages=_four_route_messages(),
        cfg_dict=cfg_dict,
        runtime_dict=copy.deepcopy(_ROUTE_KEYED_RUNTIME),
    )
    fake_session.model = f"@{slug}:antigravity/gemini-3.7-flash-tiered"
    fake_session.model_provider = slug

    responses = []
    monkeypatch.setattr(routes, "get_session", lambda _sid: fake_session)
    monkeypatch.setattr(routes, "_agent_runtime_barrier_response", lambda **_k: None)
    monkeypatch.setattr(routes, "resolve_trusted_workspace", lambda ws: "/tmp")
    monkeypatch.setattr(
        routes, "_get_session_agent_lock", lambda _sid: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        routes, "_read_profile_model_config", lambda *_a, **_k: (None, None, None)
    )
    monkeypatch.setattr(
        routes,
        "_resolve_compatible_session_model_state",
        lambda model, provider, **_k: (model, provider),
    )
    # Capture the status alongside the payload -- "which HTTP code" is half the
    # contract here (400 user-fixable, not a 500 traceback).
    def _capture_j(_handler, payload, **kwargs):
        responses.append((payload, kwargs.get("status", 200)))
        return payload

    monkeypatch.setattr(routes, "j", _capture_j)
    monkeypatch.setattr(routes, "bad", lambda _handler, message, *_a, **_k: {"error": message})

    routes._handle_chat_sync(
        object(), {"session_id": "session-1806-route", "message": "hi"}
    )

    assert responses, "POST /api/chat returned no response at all"
    payload, status = responses[-1]
    return payload, status, captured


@pytest.mark.parametrize(
    "kind,slug,expected_reason,message_fragment,hint_fragment", _TERMINAL_ROUTE_CASES
)
def test_sync_chat_route_answers_400_on_unroutable_custom_provider(
    monkeypatch, kind, slug, expected_reason, message_fragment, hint_fragment
):
    """POST /api/chat answers the actionable cause, and never builds the agent.

    400, not 500: an unroutable ``custom:<slug>`` is a user-fixable provider
    misconfiguration, exactly like the ambiguous-slug collision. The constructor
    must not be reached at all — the capturing double raises on ``__init__``, so
    a bundle that got that far would surface as a leaked
    ``_RouteAgentCaptured`` rather than a quiet pass.
    """
    _clear_credential_env(monkeypatch)

    payload, status, captured = _drive_sync_chat_route_expecting_refusal(
        monkeypatch, _terminal_route_cfg(kind), slug
    )

    assert status == 400, payload
    assert payload["type"] == "custom_provider_unroutable", payload
    assert payload["reason"] == expected_reason, payload
    assert message_fragment in payload["error"], payload
    assert slug in payload["error"], payload
    assert hint_fragment in payload["hint"], payload

    assert "init_kwargs" not in captured, (
        "an agent was constructed for an unroutable route on the non-streaming path"
    )
    # The refusal must not hand back the endpoint or credential it declined to
    # route through.
    assert "keyed-url-sentinel" not in str(payload), payload
    assert "keyed-key-sentinel-abc" not in str(payload), payload
