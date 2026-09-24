"""Claude Code sessions must open under a NAMED (non-root) active profile.

``get_claude_code_sessions()`` scans ``~/.claude/projects`` and stamps
``profile: None`` on every row — those JSONL transcripts belong to no Hermes
profile. ``/api/sessions`` lists them regardless of the active profile, but the
``GET /api/session`` detail load ran them through
``_session_visible_to_active_profile``, which coerces ``None`` -> ``'default'``
via ``_profiles_match``. Under a named profile (e.g. ``feng-family``) that gate
404'd before ``_claim_or_synthesize_cli_session`` ever ran, so every Claude Code
row in the sidebar rendered "Session not available in web UI." when clicked.

These pin the exemption: profile-less Claude Code rows bypass the gate, while
profile-tagged foreign rows stay fully scoped (the #5419 409 contract).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest

import api.routes as routes
from api.models import Session


CLAUDE_SID = "claude_code_491fbe3e6ea1248d70a4177f"


def _claude_code_row():
    """A row shaped exactly like get_claude_code_sessions() emits."""
    return {
        "session_id": CLAUDE_SID,
        "title": "Claude Code transcript",
        "workspace": "/home/user/project",
        "model": "claude-code",
        "message_count": 2,
        "created_at": 1.0,
        "updated_at": 2.0,
        "last_message_at": 2.0,
        "pinned": False,
        "archived": False,
        "project_id": None,
        "profile": None,
        "source_tag": "claude_code",
        "raw_source": "claude_code",
        "session_source": "external_agent",
        "source_label": "Claude Code",
        "is_cli_session": True,
        "read_only": True,
    }


def _synth_for(row):
    s = Session(
        session_id=row["session_id"],
        title=row["title"],
        workspace=row["workspace"],
        model=row["model"],
        messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        profile=row["profile"],
        is_cli_session=True,
        source_tag=row["source_tag"],
        raw_source=row["raw_source"],
        session_source=row["session_source"],
        source_label=row["source_label"],
        read_only=True,
    )
    return s


def _capture(monkeypatch):
    cap = {}

    def fake_j(handler, data, status=200, extra_headers=None):
        cap["data"] = data
        cap["status"] = status
        return True

    def fake_bad(handler, msg, status=400, extra_headers=None):
        cap["error"] = msg
        cap["status"] = status
        return True

    monkeypatch.setattr(routes, "j", fake_j)
    monkeypatch.setattr(routes, "bad", fake_bad)
    return cap


def test_claude_code_detail_load_survives_named_active_profile(monkeypatch):
    row = _claude_code_row()
    cap = _capture(monkeypatch)
    synth = _synth_for(row)

    handler = MagicMock()
    parsed = urlparse(
        "/api/session?session_id=%s&messages=0&resolve_model=0" % CLAUDE_SID
    )

    with (
        patch("api.routes.get_session", side_effect=KeyError(CLAUDE_SID)),
        patch("api.routes._get_active_profile_name", return_value="feng-family"),
        patch("api.routes._lookup_cli_session_metadata", return_value=row),
        patch(
            "api.routes._claim_or_synthesize_cli_session",
            return_value=(synth, "not_claimable"),
        ),
    ):
        assert routes.handle_get(handler, parsed) is True

    assert cap.get("error") is None, (
        "profile-less Claude Code row must not be 404'd by the detail-load "
        "profile gate under a named active profile"
    )
    assert cap["status"] == 200
    sess = cap["data"]["session"]
    assert sess["session_id"] == CLAUDE_SID
    assert sess["read_only"] is True
    assert sess["is_cli_session"] is True
    assert sess["source_tag"] == "claude_code"
    assert len(sess["messages"]) == 2


def test_profile_tagged_foreign_session_still_scoped(monkeypatch):
    """Negative control: a row that DOES carry a profile keeps the #5419 409."""
    row = dict(_claude_code_row())
    row.update(
        session_id="20260101_000000_abc123",
        profile="other-profile",
        source_tag="telegram",
        raw_source="telegram",
        session_source="messaging",
        source_label="Telegram",
    )
    cap = _capture(monkeypatch)

    handler = MagicMock()
    parsed = urlparse(
        "/api/session?session_id=%s&messages=0&resolve_model=0" % row["session_id"]
    )

    with (
        patch("api.routes.get_session", side_effect=KeyError(row["session_id"])),
        patch("api.routes._get_active_profile_name", return_value="feng-family"),
        patch("api.routes._lookup_cli_session_metadata", return_value=row),
    ):
        assert routes.handle_get(handler, parsed) is True

    assert cap["status"] == 409
    assert cap["data"]["code"] == "session_profile_mismatch"


def test_profile_agnostic_predicate_is_narrow():
    assert routes._is_profile_agnostic_foreign_session(_claude_code_row()) is True
    # A Claude Code row that somehow carries a profile stays scoped.
    tagged = dict(_claude_code_row(), profile="feng-family")
    assert routes._is_profile_agnostic_foreign_session(tagged) is False
    # A profile-less row from any other source stays scoped.
    other = dict(_claude_code_row(), source_tag="cli", raw_source="cli")
    assert routes._is_profile_agnostic_foreign_session(other) is False
    # A Claude Code row that is not read-only stays scoped.
    writable = dict(_claude_code_row(), read_only=False)
    assert routes._is_profile_agnostic_foreign_session(writable) is False
    # A Claude Code row that is not from external-agent provenance stays scoped.
    non_external = dict(_claude_code_row(), session_source="webui")
    assert routes._is_profile_agnostic_foreign_session(non_external) is False
    # Missing / empty metadata is never exempt.
    assert routes._is_profile_agnostic_foreign_session({}) is False
    assert routes._is_profile_agnostic_foreign_session(None) is False


def test_isolated_profile_mode_blocks_claude_code_detail_load(monkeypatch):
    row = _claude_code_row()
    cap = _capture(monkeypatch)

    handler = MagicMock()
    parsed = urlparse(
        "/api/session?session_id=%s&messages=0&resolve_model=0" % CLAUDE_SID
    )

    with (
        patch("api.routes.get_session", side_effect=KeyError(CLAUDE_SID)),
        patch("api.routes._get_active_profile_name", return_value="feng-family"),
        patch("api.routes._lookup_cli_session_metadata", return_value=row),
        patch("api.routes._is_isolated_profile_mode", return_value=True),
    ):
        assert routes.handle_get(handler, parsed) is True

    # Under isolated profile mode, detail load must fail with 404
    assert cap["status"] == 404
    assert cap.get("error") == "Session not found"


def test_isolated_profile_mode_blocks_claude_code_sharing(monkeypatch):
    row = _claude_code_row()
    handler = MagicMock()

    with (
        patch("api.routes.get_session", side_effect=KeyError(CLAUDE_SID)),
        patch("api.routes._lookup_cli_session_metadata", return_value=row),
        patch("api.routes._is_isolated_profile_mode", return_value=True),
    ):
        import pytest

        with pytest.raises(KeyError):
            routes._resolve_share_session_pair(CLAUDE_SID, handler)


def test_isolated_profile_mode_blocks_stored_claude_code_sharing(monkeypatch):
    row = _claude_code_row()
    handler = MagicMock()
    mock_stored = MagicMock()
    mock_stored.profile = None
    mock_stored.read_only = True
    mock_stored.is_cli_session = True
    mock_stored.session_source = "external_agent"
    mock_stored.source_tag = "claude_code"
    mock_stored.compact.return_value = row

    mock_ensure = MagicMock()

    with (
        patch("api.routes.get_session", return_value=mock_stored),
        patch("api.routes._lookup_cli_session_metadata", return_value=row),
        patch("api.routes._is_isolated_profile_mode", return_value=True),
        patch("api.routes._ensure_full_session_before_mutation", mock_ensure),
    ):
        import pytest

        with pytest.raises(KeyError):
            routes._resolve_share_session_pair(CLAUDE_SID, handler)

    assert mock_ensure.call_count == 0


def test_isolated_profile_mode_blocks_claude_code_import(monkeypatch):
    row = _claude_code_row()
    cap = _capture(monkeypatch)
    body = {"session_id": CLAUDE_SID, "profile": "feng-family"}

    handler = MagicMock()
    mock_get_msgs = MagicMock(
        return_value=[{"role": "user", "content": "secret isolated content"}]
    )
    mock_load = MagicMock(return_value=None)

    with (
        patch("api.routes.Session.load", mock_load),
        patch("api.routes._lookup_cli_session_metadata", return_value=row),
        patch("api.routes.get_cli_session_messages", mock_get_msgs),
        patch("api.routes._is_isolated_profile_mode", return_value=True),
    ):
        assert routes._handle_session_import_cli(handler, body) is True

    # Under isolated profile mode, import must fail with 404
    assert cap["status"] == 404
    assert cap.get("error") == "Session not found in CLI store"
    # Ensure that neither sidecar load nor get_cli_session_messages were ever called
    assert mock_load.call_count == 0
    assert mock_get_msgs.call_count == 0


def test_isolated_profile_mode_blocks_stored_claude_code_detail_load(monkeypatch):
    row = _claude_code_row()
    cap = _capture(monkeypatch)

    mock_stored = MagicMock()
    mock_stored.profile = None
    mock_stored.read_only = True
    mock_stored.is_cli_session = True
    mock_stored.session_source = "external_agent"
    mock_stored.source_tag = "claude_code"
    mock_stored.compact.return_value = row

    handler = MagicMock()
    parsed = urlparse(
        "/api/session?session_id=%s&messages=1&resolve_model=0" % CLAUDE_SID
    )

    with (
        patch("api.routes.get_session", return_value=mock_stored),
        patch("api.routes._get_active_profile_name", return_value="feng-family"),
        patch("api.routes._lookup_cli_session_metadata", return_value=row),
        patch("api.routes._is_isolated_profile_mode", return_value=True),
    ):
        assert routes.handle_get(handler, parsed) is True

    assert cap["status"] == 404
    assert cap.get("error") == "Session not found"


@pytest.mark.parametrize("messages", ["0", "1"])
def test_isolated_profile_mode_stored_foreign_profile_returns_409(
    monkeypatch, messages
):
    sid = "stored_other_profile"
    stored = Session(
        session_id=sid,
        title="Other profile session",
        workspace="/tmp",
        model="test-model",
        messages=[],
        created_at=1.0,
        updated_at=2.0,
        profile="other-profile",
    )
    cap = _capture(monkeypatch)

    with (
        patch("api.routes.get_session", return_value=stored),
        patch("api.routes._get_active_profile_name", return_value="feng-family"),
        patch("api.routes._is_isolated_profile_mode", return_value=True),
    ):
        assert routes.handle_get(
            MagicMock(),
            urlparse(
                f"/api/session?session_id={sid}&messages={messages}&resolve_model=0"
            ),
        ) is True

    assert cap["status"] == 409
    assert cap["data"] == {
        "error": "Session belongs to a different profile",
        "code": "session_profile_mismatch",
        "session_id": sid,
        "profile": "other-profile",
    }


def test_isolated_profile_mode_filters_gateway_sse_snapshot():
    claude_row = _claude_code_row()
    active_row = {"session_id": "active-1", "profile": "feng-family"}
    other_row = {"session_id": "other-1", "profile": "other-profile"}
    rows = [claude_row, active_row, other_row]

    # Non-isolated mode: active row and agnostic Claude row are visible, other-profile row is excluded
    non_isolated_scoped = routes._scope_rows_to_active_profile(
        rows, "feng-family", is_isolated=False
    )
    assert {r["session_id"] for r in non_isolated_scoped} == {
        CLAUDE_SID,
        "active-1",
    }

    # Isolated mode: only the active row is visible; agnostic Claude row and other-profile row are excluded
    isolated_scoped = routes._scope_rows_to_active_profile(
        rows, "feng-family", is_isolated=True
    )
    assert {r["session_id"] for r in isolated_scoped} == {"active-1"}

    # Isolated mode under default profile: agnostic Claude row is still excluded
    default_isolated_scoped = routes._scope_rows_to_active_profile(
        rows, "default", is_isolated=True
    )
    assert {r["session_id"] for r in default_isolated_scoped} == set()


def test_isolated_profile_mode_gateway_sse_stream_handler(monkeypatch):
    claude_row = _claude_code_row()
    active_row = {"session_id": "active-1", "profile": "feng-family"}
    other_row = {"session_id": "other-1", "profile": "other-profile"}
    initial_rows = [claude_row, active_row, other_row]

    sent_events = []

    def mock_sse(handler, event_type, data):
        sent_events.append((event_type, data))
        if len(sent_events) >= 2:
            raise ConnectionResetError("test stop after loop event")

    handler = MagicMock()
    mock_queue = MagicMock()
    shared_event = {"type": "sessions_changed", "sessions": [claude_row, active_row, other_row]}
    mock_queue.get.return_value = shared_event
    mock_watcher = MagicMock()
    mock_watcher.is_alive.return_value = True
    mock_watcher.subscribe.return_value = mock_queue

    with (
        patch("api.routes.load_settings", return_value={"show_cli_sessions": True}),
        patch("api.gateway_watcher.get_watcher", return_value=mock_watcher),
        patch("api.routes._is_isolated_profile_mode", return_value=True),
        patch("api.routes._get_active_profile_name", return_value="feng-family"),
        patch("api.models.get_cli_sessions", return_value=initial_rows),
        patch("api.routes._sse", side_effect=mock_sse),
        patch("api.routes.end_sse_headers"),
        patch("api.routes._sse_set_write_deadline"),
    ):
        routes._handle_gateway_sse_stream(handler, urlparse("/api/sessions/gateway/stream"))

    assert len(sent_events) == 2
    # Snapshot:
    assert sent_events[0][0] == "sessions_changed"
    assert {r["session_id"] for r in sent_events[0][1]["sessions"]} == {"active-1"}
    # Stream event:
    assert sent_events[1][0] == "sessions_changed"
    assert {r["session_id"] for r in sent_events[1][1]["sessions"]} == {"active-1"}
    # Original shared event dictionary was not mutated in place:
    assert len(shared_event["sessions"]) == 3


def test_stored_claude_code_detail_load_survives_named_profile(monkeypatch):
    row = _claude_code_row()
    cap = _capture(monkeypatch)

    mock_stored = MagicMock()
    mock_stored.session_id = CLAUDE_SID
    mock_stored.profile = None
    mock_stored.read_only = True
    mock_stored.is_cli_session = True
    mock_stored.session_source = "external_agent"
    mock_stored.source_tag = "claude_code"
    mock_stored.messages = []
    mock_stored.active_stream_id = None
    mock_stored.compact.return_value = row

    handler = MagicMock()
    parsed = urlparse(
        "/api/session?session_id=%s&messages=0&resolve_model=0" % CLAUDE_SID
    )

    with (
        patch("api.routes.get_session", return_value=mock_stored),
        patch("api.routes._get_active_profile_name", return_value="feng-family"),
        patch("api.routes._lookup_cli_session_metadata", return_value=row),
        patch("api.routes._is_isolated_profile_mode", return_value=False),
    ):
        assert routes.handle_get(handler, parsed) is True

    assert cap.get("error") is None
    assert cap.get("status") == 200


def test_stored_claude_code_sharing_survives_named_profile(monkeypatch):
    row = _claude_code_row()
    handler = MagicMock()
    mock_stored = MagicMock()
    mock_stored.session_id = CLAUDE_SID
    mock_stored.profile = None
    mock_stored.read_only = True
    mock_stored.is_cli_session = True
    mock_stored.session_source = "external_agent"
    mock_stored.source_tag = "claude_code"
    mock_stored.messages = [{"role": "user", "content": "hello"}]
    mock_stored.compact.return_value = row

    with (
        patch("api.routes.get_session", return_value=mock_stored),
        patch("api.routes._get_active_profile_name", return_value="feng-family"),
        patch("api.routes._lookup_cli_session_metadata", return_value=row),
        patch("api.routes._is_isolated_profile_mode", return_value=False),
        patch("api.routes._ensure_full_session_before_mutation", side_effect=lambda sid, s: s),
    ):
        snap, stored, meta = routes._resolve_share_session_pair(CLAUDE_SID, handler)
        assert snap is not None
        assert stored is mock_stored


def test_synthesized_claude_code_sharing_survives_named_profile(monkeypatch):
    row = _claude_code_row()
    handler = MagicMock()
    synth = _synth_for(row)

    with (
        patch("api.routes.get_session", side_effect=KeyError(CLAUDE_SID)),
        patch("api.routes._get_active_profile_name", return_value="feng-family"),
        patch("api.routes._lookup_cli_session_metadata", return_value=row),
        patch("api.routes._is_isolated_profile_mode", return_value=False),
        patch("api.routes._claim_or_synthesize_cli_session", return_value=(synth, "not_claimable")),
    ):
        snap, stored, meta = routes._resolve_share_session_pair(CLAUDE_SID, handler)
        assert snap is synth
        assert stored is None


def test_reimport_existing_claude_code_session_survives_named_profile(monkeypatch):
    row = _claude_code_row()
    cap = _capture(monkeypatch)
    body = {"session_id": CLAUDE_SID, "profile": "feng-family"}

    existing = MagicMock()
    existing.session_id = CLAUDE_SID
    existing.profile = None
    existing.read_only = True
    existing.is_cli_session = True
    existing.session_source = "external_agent"
    existing.source_tag = "claude_code"
    existing.messages = [{"role": "user", "content": "original"}]
    existing.compact.return_value = row

    fresh_messages = [
        {"role": "user", "content": "original"},
        {"role": "assistant", "content": "reply"},
    ]

    handler = MagicMock()
    with (
        patch("api.routes.Session.load", return_value=existing),
        patch("api.routes._get_active_profile_name", return_value="feng-family"),
        patch("api.routes._lookup_cli_session_metadata", return_value=row),
        patch("api.routes._is_isolated_profile_mode", return_value=False),
        patch("api.routes.get_cli_session_messages", return_value=fresh_messages),
    ):
        assert routes._handle_session_import_cli(handler, body) is True

    assert cap.get("error") is None
    assert cap.get("status") == 200
    assert existing.messages == fresh_messages


def test_isolated_profile_mode_blocks_existing_claude_code_import(monkeypatch):
    row = _claude_code_row()
    cap = _capture(monkeypatch)
    body = {"session_id": CLAUDE_SID, "profile": "feng-family"}

    existing = MagicMock()
    existing.session_id = CLAUDE_SID
    existing.profile = None
    existing.read_only = True
    existing.is_cli_session = True
    existing.session_source = "external_agent"
    existing.source_tag = "claude_code"
    existing.raw_source = "claude_code"
    existing.messages = [{"role": "user", "content": "secret isolated content"}]
    existing.compact.return_value = row

    mock_get_msgs = MagicMock()
    handler = MagicMock()
    with (
        patch("api.routes.Session.load", return_value=existing),
        patch("api.routes._get_active_profile_name", return_value="feng-family"),
        patch("api.routes._lookup_cli_session_metadata", return_value=None),
        patch("api.routes._is_isolated_profile_mode", return_value=True),
        patch("api.routes.get_cli_session_messages", mock_get_msgs),
    ):
        assert routes._handle_session_import_cli(handler, body) is True

    assert cap["status"] == 404
    assert cap.get("error") == "Session not found in CLI store"
    assert mock_get_msgs.call_count == 0


def test_import_cli_existing_foreign_profile_returns_409_mismatch(monkeypatch):
    cap = _capture(monkeypatch)
    body = {"session_id": "other_sid"}

    existing = MagicMock()
    existing.session_id = "other_sid"
    existing.profile = "work-profile"
    existing.read_only = False
    existing.is_cli_session = True
    existing.session_source = "hermes"
    existing.source_tag = "cli"
    existing.messages = [{"role": "user", "content": "secret"}]

    mock_get_msgs = MagicMock()
    handler = MagicMock()
    with (
        patch("api.routes.Session.load", return_value=existing),
        patch("api.routes._get_active_profile_name", return_value="default"),
        patch("api.routes._lookup_cli_session_metadata", return_value=None),
        patch("api.routes._is_isolated_profile_mode", return_value=True),
        patch("api.routes.get_cli_session_messages", mock_get_msgs),
    ):
        assert routes._handle_session_import_cli(handler, body) is True

    assert cap["status"] == 409
    assert cap.get("data", {}).get("code") == "session_profile_mismatch"
    assert cap.get("data", {}).get("profile") == "work-profile"
    assert cap.get("data", {}).get("session_id") == "other_sid"
    assert mock_get_msgs.call_count == 0


# ── Cache-stale isolation (fix spec #1) ──────────────────────────────────────
#
# The isolation gates used to decide "profile-agnostic" from the CLI metadata
# row, and a MISSING row reads as "not agnostic". The message readers do not
# share that blind spot: get_cli_session_messages() routes on the
# ``claude_code_`` id prefix and scans ~/.claude/projects directly. So with a
# transcript already on disk but not yet in the metadata cache, an isolated
# deployment could import it (as a writable sidecar), open it, and share it.
# These pin the id-first gate: 404 with no sidecar load, no metadata-driven
# synthesis and no transcript read.


def _stale_cache_disk_messages():
    """What the JSONL scanner would return for a transcript not yet cached."""
    return [
        {"role": "user", "content": "isolated deployment must never see this"},
        {"role": "assistant", "content": "leaked reply"},
    ]


def test_isolated_import_rejects_claude_code_when_metadata_cache_is_stale(monkeypatch):
    cap = _capture(monkeypatch)
    body = {"session_id": CLAUDE_SID, "profile": "ops"}

    # Cold/stale cache: no row for a transcript whose JSONL already exists.
    mock_lookup = MagicMock(return_value=None)
    mock_resolve = MagicMock(return_value={})
    mock_load = MagicMock(return_value=None)
    mock_get_msgs = MagicMock(return_value=_stale_cache_disk_messages())
    mock_import = MagicMock()

    with (
        patch("api.routes.Session.load", mock_load),
        patch("api.routes._get_active_profile_name", return_value="ops"),
        patch("api.routes._lookup_cli_session_metadata", mock_lookup),
        patch("api.routes._resolve_cli_import_metadata", mock_resolve),
        patch("api.routes.get_cli_session_messages", mock_get_msgs),
        patch("api.routes.import_cli_session", mock_import),
        patch("api.routes._is_isolated_profile_mode", return_value=True),
    ):
        assert routes._handle_session_import_cli(handler := MagicMock(), body) is True
        assert handler is not None

    assert cap["status"] == 404
    assert cap.get("error") == "Session not found in CLI store"
    # No sidecar read/mutation and no transcript read on the rejected path.
    assert mock_load.call_count == 0
    assert mock_import.call_count == 0
    assert mock_get_msgs.call_count == 0
    assert mock_lookup.call_count == 0
    assert mock_resolve.call_count == 0


def test_isolated_detail_load_rejects_claude_code_when_metadata_cache_is_stale(monkeypatch):
    cap = _capture(monkeypatch)

    mock_get_session = MagicMock(side_effect=KeyError(CLAUDE_SID))
    mock_lookup = MagicMock(return_value={})
    mock_synth = MagicMock(return_value=(_synth_for(_claude_code_row()), "not_claimable"))
    mock_get_msgs = MagicMock(return_value=_stale_cache_disk_messages())

    parsed = urlparse(
        "/api/session?session_id=%s&messages=1&resolve_model=0" % CLAUDE_SID
    )
    with (
        patch("api.routes.get_session", mock_get_session),
        patch("api.routes._get_active_profile_name", return_value="ops"),
        patch("api.routes._lookup_cli_session_metadata", mock_lookup),
        patch("api.routes._claim_or_synthesize_cli_session", mock_synth),
        patch("api.routes.get_cli_session_messages", mock_get_msgs),
        patch("api.routes._is_isolated_profile_mode", return_value=True),
    ):
        assert routes.handle_get(MagicMock(), parsed) is True

    assert cap["status"] == 404
    assert cap.get("error") == "Session not found"
    # The gate runs before the sidecar load, the metadata lookup and any
    # synthesis, so nothing on disk is touched.
    assert mock_get_session.call_count == 0
    assert mock_lookup.call_count == 0
    assert mock_synth.call_count == 0
    assert mock_get_msgs.call_count == 0


def test_isolated_share_rejects_claude_code_when_metadata_cache_is_stale(monkeypatch):
    mock_get_session = MagicMock(side_effect=KeyError(CLAUDE_SID))
    mock_lookup = MagicMock(return_value={})
    mock_synth = MagicMock(return_value=(_synth_for(_claude_code_row()), "not_claimable"))

    with (
        patch("api.routes.get_session", mock_get_session),
        patch("api.routes._get_active_profile_name", return_value="ops"),
        patch("api.routes._lookup_cli_session_metadata", mock_lookup),
        patch("api.routes._claim_or_synthesize_cli_session", mock_synth),
        patch("api.routes._is_isolated_profile_mode", return_value=True),
    ):
        with pytest.raises(KeyError):
            routes._resolve_share_session_pair(CLAUDE_SID, MagicMock())

    assert mock_get_session.call_count == 0
    assert mock_lookup.call_count == 0
    assert mock_synth.call_count == 0


def test_isolated_sidebar_drops_claude_code_row_that_carries_a_profile():
    """A stale row with a profile misses the metadata shape — the id still wins."""
    stale_row = dict(_claude_code_row(), profile="ops", read_only=False)
    assert routes._is_profile_agnostic_foreign_session(stale_row) is False
    scoped = routes._scope_rows_to_active_profile(
        [stale_row, {"session_id": "active-1", "profile": "ops"}],
        "ops",
        is_isolated=True,
    )
    assert {r["session_id"] for r in scoped} == {"active-1"}


def test_profile_agnostic_session_id_predicate(monkeypatch):
    assert routes._is_profile_agnostic_session_id(CLAUDE_SID) is True
    assert routes._is_profile_agnostic_session_id("  " + CLAUDE_SID + "  ") is True
    assert routes._is_profile_agnostic_session_id("claude_code_") is True
    assert routes._is_profile_agnostic_session_id("20260101_000000_abc123") is False
    assert routes._is_profile_agnostic_session_id("") is False
    assert routes._is_profile_agnostic_session_id(None) is False

    # When codex_sessions module is available, codex_ prefix is also recognized
    import sys
    import types
    fake_codex = types.ModuleType("api.codex_sessions")
    fake_codex.CODEX_SOURCE = "codex"
    monkeypatch.setitem(sys.modules, "api.codex_sessions", fake_codex)
    assert routes._is_profile_agnostic_session_id("codex_session_123") is True


def test_isolated_rejection_with_real_jsonl_file_and_stale_cache(monkeypatch, tmp_path):
    """End-to-end regression: real JSONL exists on disk but CLI metadata cache is stale."""
    import json
    import api.models as models

    projects_dir = tmp_path / "claude" / "projects"
    session_file = projects_dir / "proj" / "session.jsonl"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"summary": "Secret Real Session"},
        {"timestamp": "2026-04-18T12:00:01Z", "message": {"role": "user", "content": [{"type": "text", "text": "secret unread text"}]}},
    ]
    session_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_WEBUI_CLAUDE_PROJECTS_DIR", str(projects_dir))

    sid = models._claude_code_session_id(session_file)
    assert sid.startswith("claude_code_")

    # Real scanner verifies the transcript exists on disk and is readable
    disk_msgs = models.get_claude_code_session_messages(sid, projects_dir=projects_dir)
    assert len(disk_msgs) == 1
    assert disk_msgs[0]["content"] == "secret unread text"

    cap = _capture(monkeypatch)
    with (
        patch("api.routes._get_active_profile_name", return_value="ops"),
        patch("api.routes._lookup_cli_session_metadata", return_value=None),
        patch("api.routes._is_isolated_profile_mode", return_value=True),
    ):
        # 1. Import returns 404 and does not mutate or read messages
        assert routes._handle_session_import_cli(MagicMock(), {"session_id": sid, "profile": "ops"}) is True
        assert cap["status"] == 404
        assert cap.get("error") == "Session not found in CLI store"

        # 2. Detail load returns 404
        parsed = urlparse(f"/api/session?session_id={sid}&messages=1&resolve_model=0")
        assert routes.handle_get(MagicMock(), parsed) is True
        assert cap["status"] == 404
        assert cap.get("error") == "Session not found"

        # 3. Share resolution raises KeyError
        with pytest.raises(KeyError):
            routes._resolve_share_session_pair(sid, MagicMock())


# ── Stored foreign owner outranks a profile-less metadata row (fix spec #2) ──
#
# The profile-less exemption ran BEFORE the stored-profile check, so a
# persisted sidecar owned by `other` matched against a Claude metadata row with
# no profile was treated as belonging to no profile at all: detail load
# returned 200 with the transcript instead of the #5419 409, and sharing was
# allowed.


FOREIGN_OWNED_SECRET = "message owned by the other profile"


def _foreign_owned_claude_sidecar():
    return Session(
        session_id=CLAUDE_SID,
        title="Imported Claude Code transcript",
        workspace="/home/user/project",
        model="claude-code",
        messages=[{"role": "user", "content": FOREIGN_OWNED_SECRET}],
        created_at=1.0,
        updated_at=2.0,
        profile="other",
        is_cli_session=True,
        source_tag="claude_code",
        raw_source="claude_code",
        session_source="external_agent",
        source_label="Claude Code",
        read_only=True,
    )


@pytest.mark.parametrize("messages", ["0", "1"])
def test_stored_foreign_owner_beats_profile_less_claude_metadata_on_detail_load(
    monkeypatch, messages
):
    cap = _capture(monkeypatch)
    stored = _foreign_owned_claude_sidecar()
    # The Claude metadata row carries no profile — the old exemption fired here.
    agnostic_meta = _claude_code_row()

    mock_get_session = MagicMock(return_value=stored)
    parsed = urlparse(
        "/api/session?session_id=%s&messages=%s&resolve_model=0" % (CLAUDE_SID, messages)
    )
    with (
        patch("api.routes.get_session", mock_get_session),
        patch("api.routes._get_active_profile_name", return_value="ops"),
        patch("api.routes._lookup_cli_session_metadata", return_value=agnostic_meta),
        patch("api.routes._is_isolated_profile_mode", return_value=False),
    ):
        assert routes.handle_get(MagicMock(), parsed) is True

    assert cap["status"] == 409
    assert cap["data"] == {
        "error": "Session belongs to a different profile",
        "code": "session_profile_mismatch",
        "session_id": CLAUDE_SID,
        "profile": "other",
    }
    # Never hydrated: the gate rejects on metadata alone.
    assert mock_get_session.call_count >= 1
    assert all(
        call.kwargs.get("metadata_only") is True
        for call in mock_get_session.call_args_list
    )
    assert FOREIGN_OWNED_SECRET not in repr(cap["data"])


def test_stored_foreign_owner_beats_profile_less_claude_metadata_on_share(monkeypatch):
    stored = _foreign_owned_claude_sidecar()
    mock_get_session = MagicMock(return_value=stored)
    mock_snapshot = MagicMock()

    with (
        patch("api.routes.get_session", mock_get_session),
        patch("api.routes._get_active_profile_name", return_value="ops"),
        patch("api.routes._lookup_cli_session_metadata", return_value=_claude_code_row()),
        patch("api.routes._is_isolated_profile_mode", return_value=False),
        patch("api.routes._share_snapshot_messages_for_session", mock_snapshot),
    ):
        with pytest.raises(KeyError):
            routes._resolve_share_session_pair(CLAUDE_SID, MagicMock())

    # Denied before the share snapshot (and its message load) is built.
    assert mock_snapshot.call_count == 0


def test_stored_active_owner_with_profile_less_claude_metadata_still_loads(monkeypatch):
    """Negative control: the owner check only denies a FOREIGN owner."""
    cap = _capture(monkeypatch)
    stored = _foreign_owned_claude_sidecar()
    stored.profile = "ops"

    parsed = urlparse(
        "/api/session?session_id=%s&messages=0&resolve_model=0" % CLAUDE_SID
    )
    with (
        patch("api.routes.get_session", return_value=stored),
        patch("api.routes._get_active_profile_name", return_value="ops"),
        patch("api.routes._lookup_cli_session_metadata", return_value=_claude_code_row()),
        patch("api.routes._is_isolated_profile_mode", return_value=False),
    ):
        assert routes.handle_get(MagicMock(), parsed) is True

    assert cap.get("error") is None
    assert cap["status"] == 200


def test_stored_foreign_owner_beats_active_profile_cli_metadata_on_share(monkeypatch):
    """Stored foreign sidecar profile='other' outranks conflicting CLI metadata profile='ops'."""
    stored = _foreign_owned_claude_sidecar()
    stored.profile = "other"
    conflicting_meta = dict(_claude_code_row(), profile="ops")
    mock_get_session = MagicMock(return_value=stored)
    mock_snapshot = MagicMock()

    with (
        patch("api.routes.get_session", mock_get_session),
        patch("api.routes._get_active_profile_name", return_value="ops"),
        patch("api.routes._lookup_cli_session_metadata", return_value=conflicting_meta),
        patch("api.routes._is_isolated_profile_mode", return_value=False),
        patch("api.routes._share_snapshot_messages_for_session", mock_snapshot),
    ):
        with pytest.raises(KeyError):
            routes._resolve_share_session_pair(CLAUDE_SID, MagicMock())

    assert mock_snapshot.call_count == 0
