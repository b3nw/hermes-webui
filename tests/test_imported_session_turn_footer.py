"""Per-turn footer metadata for state.db-imported sessions.

Imported agent transcripts (delegated subagents, CLI, TUI, cron) are not run by
the WebUI, so ``_run_agent_streaming`` never stamps the per-turn footer fields
and the transcript renders with no model, duration, or token information at all.
These tests pin the server-side backfill that reads those facts off the session
row instead.
"""

import sqlite3
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlparse

import pytest

import api.agent_sessions as agent_sessions


def _make_state_db(path):
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            model TEXT,
            started_at REAL,
            ended_at REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_read_tokens INTEGER,
            cache_write_tokens INTEGER,
            estimated_cost_usd REAL,
            api_call_count INTEGER
        );
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL,
            tool_calls TEXT,
            active INTEGER
        );
        """
    )
    return conn


def _add_session(conn, sid, *, model="x-ai/grok-4.5", api_calls=1, input_tokens=10189,
                 output_tokens=221, cache_read=128, cache_write=0, cost=0.0025):
    conn.execute(
        """
        INSERT INTO sessions (id, source, model, started_at, ended_at, input_tokens,
                              output_tokens, cache_read_tokens, cache_write_tokens,
                              estimated_cost_usd, api_call_count)
        VALUES (?, 'subagent', ?, 1000.0, 1010.0, ?, ?, ?, ?, ?, ?)
        """,
        (sid, model, input_tokens, output_tokens, cache_read, cache_write, cost, api_calls),
    )
    conn.commit()


def _add_turn(conn, sid, n, *, user_ts, assistant_ts):
    conn.executemany(
        "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, 'x', ?)",
        [
            (f"{sid}_u{n}", sid, "user", user_ts),
            (f"{sid}_a{n}", sid, "assistant", assistant_ts),
        ],
    )
    conn.commit()


def _transcript(conn, sid):
    """Message dicts in the shape get_state_db_session_messages returns."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
        (sid,),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in rows]


def test_single_call_subagent_turn_gets_model_duration_and_usage(tmp_path):
    """The common delegate_task child: one call, so totals provably describe it."""
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "sub_single", api_calls=1)
        _add_turn(conn, "sub_single", 1, user_ts=1000.0, assistant_ts=1004.076)
        msgs = _transcript(conn, "sub_single")
    finally:
        conn.close()

    stats = agent_sessions.read_agent_session_turn_footer_stats(db, "sub_single")
    agent_sessions.stamp_imported_turn_footers(msgs, stats)

    assistant = msgs[-1]
    assert assistant["_usedModel"] == "x-ai/grok-4.5"
    assert assistant["_turnDuration"] == 4.076
    assert assistant["_turnUsage"]["input_tokens"] == 10189
    assert assistant["_turnUsage"]["output_tokens"] == 221
    assert assistant["_turnUsage"]["estimated_cost"] == 0.0025
    # cache_read / (input + cache_read + cache_write) — same denominator the
    # in-process path uses via prompt_cache_hit_percent.
    assert assistant["_turnUsage"]["cache_hit_percent"] == 1
    # The user message is never a turn footer carrier.
    assert "_usedModel" not in msgs[0]


def test_multi_call_session_gets_model_and_duration_but_no_token_totals(tmp_path):
    """Session totals must not be presented as one turn's usage.

    ``messages.token_count`` is not populated for imported sessions, so with
    more than one API call there is no way to split the totals. Model and
    per-turn duration are still exact and are still shown.
    """
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "sub_multi", api_calls=2)
        _add_turn(conn, "sub_multi", 1, user_ts=1000.0, assistant_ts=1006.404)
        _add_turn(conn, "sub_multi", 2, user_ts=1010.0, assistant_ts=1020.574)
        msgs = _transcript(conn, "sub_multi")
    finally:
        conn.close()

    stats = agent_sessions.read_agent_session_turn_footer_stats(db, "sub_multi")
    agent_sessions.stamp_imported_turn_footers(msgs, stats)

    assistants = [m for m in msgs if m["role"] == "assistant"]
    assert len(assistants) == 2
    assert [m["_usedModel"] for m in assistants] == ["x-ai/grok-4.5", "x-ai/grok-4.5"]
    assert [m["_turnDuration"] for m in assistants] == [6.404, 10.574]
    assert all("_turnUsage" not in m for m in assistants)


def test_tool_using_turn_stamps_only_the_settled_assistant(tmp_path):
    """An assistant row that requests tools is not the turn's answer.

    `get_state_db_session_messages` preserves `tool_calls`, so an imported
    tool-using turn arrives as user -> assistant(tool_calls) -> tool ->
    assistant. Stamping both assistant rows would put a premature elapsed time
    on the tool-call segment and render two footers for one logical turn; the
    frontend contract puts settled metadata on the last metadata-bearing
    assistant row. The final answer must carry the full user-to-completion
    duration.
    """
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        # A tool-using turn is at least two API calls, so usage is withheld
        # regardless; this test is about model/duration ownership.
        _add_session(conn, "sub_tools", api_calls=2)
        conn.executemany(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            [
                ("t_u1", "sub_tools", "user", "do the thing", 1000.0),
                ("t_a1", "sub_tools", "assistant", "", 1002.0),
                ("t_r1", "sub_tools", "tool", "tool output", 1003.0),
                ("t_a2", "sub_tools", "assistant", "done", 1008.5),
            ],
        )
        conn.commit()
        msgs = _transcript(conn, "sub_tools")
    finally:
        conn.close()

    # The reader parses tool_calls onto the intermediate assistant row.
    msgs[1]["tool_calls"] = [{"id": "call_1", "function": {"name": "read_file"}}]

    stats = agent_sessions.read_agent_session_turn_footer_stats(db, "sub_tools")
    agent_sessions.stamp_imported_turn_footers(msgs, stats)

    intermediate, final = msgs[1], msgs[3]
    assert "_usedModel" not in intermediate
    assert "_turnDuration" not in intermediate
    assert "_turnUsage" not in intermediate
    assert final["_usedModel"] == "x-ai/grok-4.5"
    # Full user -> final answer, not user -> tool-call segment (which would be 2.0).
    assert final["_turnDuration"] == 8.5
    assert "_turnUsage" not in final


def test_single_call_turn_without_tool_calls_is_the_direct_control(tmp_path):
    """Control for the tool-using case: a plain turn still gets full metadata."""
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "sub_plain", api_calls=1)
        _add_turn(conn, "sub_plain", 1, user_ts=1000.0, assistant_ts=1003.0)
        msgs = _transcript(conn, "sub_plain")
    finally:
        conn.close()

    stats = agent_sessions.read_agent_session_turn_footer_stats(db, "sub_plain")
    agent_sessions.stamp_imported_turn_footers(msgs, stats)

    assistant = msgs[-1]
    assert assistant["_usedModel"] == "x-ai/grok-4.5"
    assert assistant["_turnDuration"] == 3.0
    assert assistant["_turnUsage"]["input_tokens"] == 10189


def test_footer_stats_read_state_db_through_the_read_only_opener(tmp_path):
    """A pure-read projection must not take a write-capable handle on state.db.

    Mirrors the module's existing rule (#5455): the live WAL state.db is opened
    via ``open_state_db_readonly`` so a read never adds checkpoint/lock surface.
    """
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "ro_check")
        _add_turn(conn, "ro_check", 1, user_ts=1.0, assistant_ts=2.0)
    finally:
        conn.close()

    calls = []
    real_opener = agent_sessions.open_state_db_readonly

    def _spy(path, *args, **kwargs):
        calls.append(str(path))
        return real_opener(path, *args, **kwargs)

    with patch.object(agent_sessions, "open_state_db_readonly", side_effect=_spy):
        stats = agent_sessions.read_agent_session_turn_footer_stats(db, "ro_check")

    assert calls == [str(db)], "stats read did not go through open_state_db_readonly"
    assert stats["model"] == "x-ai/grok-4.5"


def test_stitched_transcript_is_not_stamped_with_one_segments_stats(tmp_path):
    """A stitched continuation chain spans session rows with their own models.

    ``get_state_db_session_messages`` returns segments stitched together, so a
    transcript longer than the requested session's own message count must be
    left alone rather than attributed to this segment's model and totals.
    """
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "tip", model="openai/gpt-5", api_calls=1)
        _add_turn(conn, "tip", 1, user_ts=2000.0, assistant_ts=2001.0)
        msgs = _transcript(conn, "tip")
        # Simulate the stitched parent segment prepended by the reader.
        msgs = [
            {"role": "user", "content": "older", "timestamp": 1000.0},
            {"role": "assistant", "content": "older reply", "timestamp": 1001.0},
        ] + msgs
    finally:
        conn.close()

    stats = agent_sessions.read_agent_session_turn_footer_stats(db, "tip")
    assert stats["own_message_count"] == 2
    agent_sessions.stamp_imported_turn_footers(msgs, stats)

    assert all("_usedModel" not in m for m in msgs)
    assert all("_turnUsage" not in m for m in msgs)


def test_footer_stats_degrade_quietly(tmp_path):
    """Missing db, unknown session, and empty input must never raise."""
    missing = tmp_path / "nope.db"
    assert agent_sessions.read_agent_session_turn_footer_stats(missing, "whatever") == {}

    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "known")
    finally:
        conn.close()

    assert agent_sessions.read_agent_session_turn_footer_stats(db, "unknown") == {}
    assert agent_sessions.read_agent_session_turn_footer_stats(db, "") == {}
    assert agent_sessions.read_agent_session_turn_footer_stats(db, None) == {}
    # No-op paths on the stamping side.
    assert agent_sessions.stamp_imported_turn_footers([], {"own_message_count": 0}) == []
    assert agent_sessions.stamp_imported_turn_footers(None, {}) is None


def test_transcript_without_an_assistant_reply_is_untouched(tmp_path):
    """A user-only transcript has no turn to attribute anything to."""
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "pending")
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp) VALUES ('p_u1','pending','user','x',5.0)"
        )
        conn.commit()
        msgs = _transcript(conn, "pending")
    finally:
        conn.close()

    stats = agent_sessions.read_agent_session_turn_footer_stats(db, "pending")
    agent_sessions.stamp_imported_turn_footers(msgs, stats)
    assert msgs == [{"role": "user", "content": "x", "timestamp": 5.0}]


def test_missing_model_does_not_stamp_an_empty_model_chip(tmp_path):
    """An unrecorded model must leave the chip absent, not blank."""
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "no_model", model=None, api_calls=1)
        _add_turn(conn, "no_model", 1, user_ts=1.0, assistant_ts=2.5)
        msgs = _transcript(conn, "no_model")
    finally:
        conn.close()

    stats = agent_sessions.read_agent_session_turn_footer_stats(db, "no_model")
    agent_sessions.stamp_imported_turn_footers(msgs, stats)

    assistant = msgs[-1]
    assert "_usedModel" not in assistant
    assert assistant["_turnDuration"] == 1.5


# --------------------------------------------------------------------------
# Ownership count under the reader's visibility predicate
# --------------------------------------------------------------------------


def test_own_message_count_excludes_inactive_rows_like_the_reader(tmp_path):
    """The count must live in the same coordinate space as the transcript.

    ``get_state_db_session_messages`` drops ``active = 0`` rows — compacted
    pre-compression history — by default, so a compressed session returns fewer
    messages than its ``messages`` table holds. Counting every row here reports a
    transcript this session can never produce, and the stitched-lineage guard
    reads that mismatch as a stitched chain and suppresses the footer on an
    ordinary single-segment session.
    """
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "compressed", api_calls=1)
        conn.executemany(
            "INSERT INTO messages (id, session_id, role, content, timestamp, active) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                # Archived by a compression pass: invisible to the reader.
                ("c_u0", "compressed", "user", "ancient", 900.0, 0),
                ("c_u1", "compressed", "user", "current", 1000.0, 1),
                ("c_a1", "compressed", "assistant", "reply", 1003.5, 1),
            ],
        )
        conn.commit()
        # The visible transcript, exactly as the reader would return it.
        msgs = [
            {"role": "user", "content": "current", "timestamp": 1000.0},
            {"role": "assistant", "content": "reply", "timestamp": 1003.5},
        ]
    finally:
        conn.close()

    stats = agent_sessions.read_agent_session_turn_footer_stats(db, "compressed")
    assert stats["own_message_count"] == 2, (
        "own_message_count must exclude active=0 rows so it matches the "
        "transcript get_state_db_session_messages actually returns"
    )

    stamped = agent_sessions.stamp_imported_turn_footers(msgs, stats, detach=True)
    assert stamped[-1]["_usedModel"] == "x-ai/grok-4.5"
    assert stamped[-1]["_turnDuration"] == 3.5


def test_include_inactive_count_mirrors_the_readers_recovery_mode(tmp_path):
    """``include_inactive`` exists so the two stay paired, not as a default."""
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "recovery", api_calls=1)
        conn.executemany(
            "INSERT INTO messages (id, session_id, role, content, timestamp, active) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("r_u0", "recovery", "user", "ancient", 900.0, 0),
                ("r_u1", "recovery", "user", "current", 1000.0, 1),
                ("r_a1", "recovery", "assistant", "reply", 1002.0, 1),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    assert agent_sessions.read_agent_session_turn_footer_stats(
        db, "recovery"
    )["own_message_count"] == 2
    assert agent_sessions.read_agent_session_turn_footer_stats(
        db, "recovery", include_inactive=True
    )["own_message_count"] == 3


def test_detach_leaves_the_caller_transcript_untouched(tmp_path):
    """Response-path stamping must not reach the sidecar or model context.

    The dicts handed in are the same objects the sidecar holds and the same ones
    that feed model-context reconstruction, so display-only keys must land on
    copies.
    """
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "detached", api_calls=1)
        _add_turn(conn, "detached", 1, user_ts=1000.0, assistant_ts=1002.0)
        msgs = _transcript(conn, "detached")
    finally:
        conn.close()

    stats = agent_sessions.read_agent_session_turn_footer_stats(db, "detached")
    stored_assistant = msgs[-1]
    stamped = agent_sessions.stamp_imported_turn_footers(msgs, stats, detach=True)

    assert stamped is not msgs
    assert stamped[-1]["_usedModel"] == "x-ai/grok-4.5"
    assert "_usedModel" not in stored_assistant, "stamped a dict the sidecar still holds"
    assert stamped[-1] is not stored_assistant
    # Untouched rows are shared by reference: only settled turns are copied.
    assert stamped[0] is msgs[0]


# --------------------------------------------------------------------------
# GET /api/session — both branches must reach the same backfill
# --------------------------------------------------------------------------


class _ImportedSidecar:
    """Persisted WebUI sidecar for a session another surface executed.

    Read-only imported sidecars carry the source markers and an empty message
    list: the transcript itself lives in state.db and is merged in on load.
    """

    def __init__(self, sid, *, source_tag="cli", messages=None):
        self.session_id = sid
        self.title = "Imported"
        self.workspace = "/tmp"
        self.model = "openai/gpt-5"
        self.model_provider = None
        self.messages = list(messages or [])
        self.tool_calls = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.estimated_cost = 0
        self.context_length = 1
        self.threshold_tokens = 0
        self.last_prompt_tokens = 0
        self.active_stream_id = None
        self.pending_user_message = None
        self.pending_attachments = []
        self.pending_started_at = None
        self.pending_user_source = None
        self.composer_draft = {}
        self.is_cli_session = True
        self.read_only = True
        self.session_source = source_tag
        self.source_tag = source_tag
        self.raw_source = source_tag
        self.source_label = source_tag
        self.truncation_watermark = None
        self.truncation_boundary = None
        self.anchor_activity_scenes = None
        self.profile = None
        self.created_at = 1000.0
        self.updated_at = 1010.0
        self.last_message_at = 1010.0
        self.pinned = False
        self.archived = False
        self.project_id = None

    def compact(self, **kwargs):
        return {
            "session_id": self.session_id,
            "title": self.title,
            "workspace": self.workspace,
            "model": self.model,
            "model_provider": self.model_provider,
            "message_count": len(self.messages),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_message_at": self.last_message_at,
            "context_length": self.context_length,
            "threshold_tokens": self.threshold_tokens,
            "last_prompt_tokens": self.last_prompt_tokens,
            "active_stream_id": self.active_stream_id,
            "pending_user_message": self.pending_user_message,
            "composer_draft": self.composer_draft,
            "is_cli_session": self.is_cli_session,
            "read_only": self.read_only,
            "session_source": self.session_source,
            "source_tag": self.source_tag,
            "raw_source": self.raw_source,
            "source_label": self.source_label,
            "pinned": self.pinned,
            "archived": self.archived,
            "project_id": self.project_id,
            "profile": self.profile,
        }


class _NativeWebuiSidecar(_ImportedSidecar):
    """A session the WebUI itself ran: no foreign source markers at all."""

    def __init__(self, sid, *, messages=None):
        super().__init__(sid, messages=messages)
        self.is_cli_session = False
        self.read_only = False
        self.session_source = None
        self.source_tag = None
        self.raw_source = None
        self.source_label = None
        self.platform = None


def _get_api_session(db, sid, *, session, cli_meta, query="", synth=None, spy_stats=False):
    """Drive GET /api/session against a purpose-built state.db.

    The real ``get_state_db_session_messages`` reads the transcript so the
    ownership count and the transcript it gates are produced by the same schema
    the production reader sees, rather than by a hand-written fixture pair.
    """
    import api.models as models
    import api.routes as routes

    captured = {}

    def fake_j(_handler, data, status=200, extra_headers=None):
        captured["data"] = data
        captured["status"] = status
        return data

    def fake_get_session(_sid, **kwargs):
        if session is None:
            raise KeyError(_sid)
        return session

    parsed = urlparse(f"/api/session?session_id={sid}&messages=1&resolve_model=0{query}")
    stack = [
        patch.object(models, "_active_state_db_path", return_value=db),
        patch.object(routes, "_active_state_db_path", return_value=db),
        patch.object(routes, "get_session", side_effect=fake_get_session),
        patch.object(routes, "_clear_stale_stream_state", return_value=False),
        patch.object(routes, "_lookup_cli_session_metadata", return_value=cli_meta),
        patch.object(routes, "j", side_effect=fake_j),
    ]
    if synth is not None:
        stack.append(
            patch.object(routes, "_claim_or_synthesize_cli_session", return_value=(synth, "")),
        )
    stats_reads = []
    if spy_stats:
        real_reader = routes.read_agent_session_turn_footer_stats

        def _spy_reader(*args, **kwargs):
            stats_reads.append(args[:2])
            return real_reader(*args, **kwargs)

        stack.append(
            patch.object(
                routes, "read_agent_session_turn_footer_stats", side_effect=_spy_reader
            )
        )
    with ExitStack() as es:
        for cm in stack:
            es.enter_context(cm)
        routes.handle_get(SimpleNamespace(), parsed)
    captured["stats_reads"] = stats_reads
    return captured


def test_persisted_imported_sidecar_gets_the_footer_on_the_normal_branch(tmp_path):
    """The gap the first round left open.

    A CLI/TUI session imported into a WebUI sidecar takes the ordinary
    ``get_session`` branch, which returned the transcript without ever calling
    the backfill — so the footer appeared once, on the synthesized load, and
    vanished on every load after a sidecar existed.
    """
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "cli_persisted", model="openai/gpt-5", api_calls=1)
        _add_turn(conn, "cli_persisted", 1, user_ts=1000.0, assistant_ts=1007.25)
    finally:
        conn.close()

    captured = _get_api_session(
        db,
        "cli_persisted",
        session=_ImportedSidecar("cli_persisted"),
        cli_meta={"session_id": "cli_persisted", "source_tag": "cli", "raw_source": "cli"},
    )

    assert captured["status"] == 200
    msgs = captured["data"]["session"]["messages"]
    assistant = [m for m in msgs if m.get("role") == "assistant"][-1]
    assert assistant["_usedModel"] == "openai/gpt-5"
    assert assistant["_turnDuration"] == 7.25
    assert assistant["_turnUsage"]["output_tokens"] == 221


def test_no_sidecar_subagent_branch_still_gets_the_footer(tmp_path):
    """The synthesized branch must keep working through the shared helper."""
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "sub_synth", model="x-ai/grok-4.5", api_calls=1)
        _add_turn(conn, "sub_synth", 1, user_ts=2000.0, assistant_ts=2004.5)
        conn.row_factory = sqlite3.Row
        transcript = _transcript(conn, "sub_synth")
    finally:
        conn.close()

    synth = _ImportedSidecar("sub_synth", source_tag="subagent", messages=transcript)
    synth.is_cli_session = False
    captured = _get_api_session(
        db,
        "sub_synth",
        session=None,  # no WebUI sidecar -> KeyError -> synthesized branch
        cli_meta={"session_id": "sub_synth", "source_tag": "subagent", "raw_source": "subagent"},
        synth=synth,
    )

    assert captured["status"] == 200
    msgs = captured["data"]["session"]["messages"]
    assistant = [m for m in msgs if m.get("role") == "assistant"][-1]
    assert assistant["_usedModel"] == "x-ai/grok-4.5"
    assert assistant["_turnDuration"] == 4.5
    # The synthesized Session's own message dicts must not have been stamped.
    assert all("_usedModel" not in m for m in transcript)


def test_compressed_imported_session_is_stamped_through_the_route(tmp_path):
    """One inactive row plus one visible turn: the two counts must agree.

    End-to-end proof of the ownership-count fix — the real reader hides the
    ``active = 0`` row, so a whole-table count would suppress the footer here.
    """
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "cli_compressed", model="openai/gpt-5", api_calls=1)
        conn.executemany(
            "INSERT INTO messages (id, session_id, role, content, timestamp, active) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("z_u0", "cli_compressed", "user", "archived by compression", 900.0, 0),
                ("z_u1", "cli_compressed", "user", "current question", 1000.0, 1),
                ("z_a1", "cli_compressed", "assistant", "current answer", 1006.0, 1),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    captured = _get_api_session(
        db,
        "cli_compressed",
        session=_ImportedSidecar("cli_compressed"),
        cli_meta={"session_id": "cli_compressed", "source_tag": "cli", "raw_source": "cli"},
    )

    msgs = captured["data"]["session"]["messages"]
    assert [m.get("content") for m in msgs] == ["current question", "current answer"], (
        "the reader should have hidden the active=0 row"
    )
    assistant = msgs[-1]
    assert assistant["_usedModel"] == "openai/gpt-5", (
        "an inactive row inflated own_message_count and suppressed the footer"
    )
    assert assistant["_turnDuration"] == 6.0


def test_paginated_imported_load_still_stamps_the_visible_assistant(tmp_path):
    """Ownership is judged on the full visible segment, not the returned window.

    A window can never match the session's own message count, so gating on the
    sliced list would leave every paginated load unstamped. Stamping runs before
    the slice, which also lets the newest assistant resolve the user row that
    opened its turn even when that row falls outside the window.
    """
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "cli_paged", model="openai/gpt-5", api_calls=3)
        _add_turn(conn, "cli_paged", 1, user_ts=1000.0, assistant_ts=1002.0)
        _add_turn(conn, "cli_paged", 2, user_ts=1010.0, assistant_ts=1013.0)
        _add_turn(conn, "cli_paged", 3, user_ts=1020.0, assistant_ts=1029.5)
    finally:
        conn.close()

    sidecar = _ImportedSidecar("cli_paged")
    captured = _get_api_session(
        db,
        "cli_paged",
        session=sidecar,
        cli_meta={"session_id": "cli_paged", "source_tag": "cli", "raw_source": "cli"},
        # A one-row window, so the user message that opened this turn is
        # provably outside it.
        query="&msg_limit=1",
    )

    session_payload = captured["data"]["session"]
    msgs = session_payload["messages"]
    assert [m.get("role") for m in msgs] == ["assistant"]
    assert session_payload["_messages_offset"] == 5, "expected the tail window of six rows"
    assistant = msgs[0]
    assert assistant["_usedModel"] == "openai/gpt-5"
    # 1029.5 - 1020.0: resolved from a user row the window does not contain,
    # which is only possible because stamping ran before the slice.
    assert assistant["_turnDuration"] == 9.5
    # Three API calls: session totals cannot be attributed to one turn.
    assert "_turnUsage" not in assistant
    # Nothing display-only may have been written back to the stored session.
    assert all("_usedModel" not in m for m in sidecar.messages)


def test_webui_native_session_is_left_alone(tmp_path):
    """The ownership gate: the WebUI stamps its own turns and must not be double-stamped.

    A native session has no foreign source markers, so the backfill must not run
    even when a state.db row happens to exist for the same id.
    """
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "native_sess", model="should-not-appear", api_calls=1)
        _add_turn(conn, "native_sess", 1, user_ts=1000.0, assistant_ts=1004.0)
    finally:
        conn.close()

    # Mirror the state.db rows exactly so the merge dedupes to a transcript whose
    # length DOES match own_message_count. Otherwise the stitched-lineage length
    # guard would suppress the footer on its own and this test would pass without
    # the ownership gate ever being consulted.
    native = _NativeWebuiSidecar(
        "native_sess",
        messages=[
            {"role": "user", "content": "x", "timestamp": 1000.0},
            {"role": "assistant", "content": "x", "timestamp": 1004.0},
        ],
    )
    captured = _get_api_session(db, "native_sess", session=native, cli_meta={})

    msgs = captured["data"]["session"]["messages"]
    assert len(msgs) == 2, "the length guard must not be what blocks this case"
    assert all("_usedModel" not in m for m in msgs), (
        "WebUI-native sessions must not be backfilled from state.db"
    )
    assert all("_turnUsage" not in m for m in msgs)


def test_webui_origin_session_touched_by_another_surface_is_left_alone(tmp_path):
    """A ``webui`` marker on either source wins over a foreign marker on the other.

    A WebUI-origin session later updated through the Gateway API server does get a
    state.db row, and its sidebar row can carry an ``api`` source marker — but the
    WebUI ran those turns and already stamped their footers. Merging the two
    sources' markers into one dict would let the ``api`` marker mask the ``webui``
    one and backfill session-wide totals over correct per-turn ones.
    """
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "webui_via_api", model="should-not-appear", api_calls=1)
        _add_turn(conn, "webui_via_api", 1, user_ts=1000.0, assistant_ts=1004.0)
    finally:
        conn.close()

    sidecar = _ImportedSidecar("webui_via_api", source_tag="webui")
    sidecar.messages = [
        {"role": "user", "content": "x", "timestamp": 1000.0},
        {"role": "assistant", "content": "x", "timestamp": 1004.0},
    ]
    captured = _get_api_session(
        db,
        "webui_via_api",
        session=sidecar,
        # The sidebar row disagrees: it saw the session through the API server.
        cli_meta={"session_id": "webui_via_api", "source_tag": "api", "raw_source": "api"},
    )

    msgs = captured["data"]["session"]["messages"]
    assert len(msgs) == 2, "the length guard must not be what blocks this case"
    assert all("_usedModel" not in m for m in msgs), (
        "a foreign marker on the sidebar row masked the sidecar's webui marker"
    )




# --------------------------------------------------------------------------
# Turn ownership across schema shapes: absent, migrated-NULL, and populated
# tool_calls metadata
# --------------------------------------------------------------------------


def _make_state_db_with_shape(path, *, tool_calls_column: bool):
    """Build a state.db with or without the ``messages.tool_calls`` column.

    ``tool_calls_column=False`` is a legacy database predating the column.
    ``True`` with NULL values is the *migrated* shape: Agent startup back-fills
    missing message columns, so the column exists on an upgraded database while
    every historical row leaves it NULL. Neither shape can be told apart from
    "this assistant requested no tools" by looking at the key.
    """
    conn = sqlite3.connect(str(path))
    tool_calls_ddl = "            tool_calls TEXT,\n" if tool_calls_column else ""
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            model TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_read_tokens INTEGER,
            cache_write_tokens INTEGER,
            estimated_cost_usd REAL,
            api_call_count INTEGER
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            role TEXT,
            content TEXT,
"""
        + tool_calls_ddl
        + """            timestamp REAL
        );
        """
    )
    return conn


def _read_real_transcript(db, sid):
    """Read through the production reader so the fixture cannot drift from it."""
    import api.models as models

    with patch.object(models, "_active_state_db_path", return_value=db):
        return models.get_state_db_session_messages(sid)


def _seed_shape(db, sid, rows, *, api_calls=2, tool_calls_column):
    conn = _make_state_db_with_shape(db, tool_calls_column=tool_calls_column)
    try:
        conn.execute(
            "INSERT INTO sessions VALUES (?, 'cli', 'openai/gpt-5', 100, 10, 0, 0, 0.001, ?)",
            (sid, api_calls),
        )
        conn.executemany(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            [(sid, role, content, ts) for role, content, ts in rows],
        )
        conn.commit()
    finally:
        conn.close()


# Both production shapes the review named. Neither leaves a usable tool_calls
# value on the rows, so the settled-assistant decision cannot come from the key.
_AMBIGUOUS_SHAPES = [
    pytest.param(False, id="column-physically-absent"),
    pytest.param(True, id="column-present-but-null"),
]


@pytest.mark.parametrize("tool_calls_column", _AMBIGUOUS_SHAPES)
def test_narrated_tool_segment_is_not_mistaken_for_the_answer(tmp_path, tool_calls_column):
    """A tool-call assistant can carry real prose, so content is not a discriminator.

    Real CLI transcripts record narrated tool-call assistants — "Let me search" —
    ahead of the tool row. Treating any nonblank assistant as settled hands the
    intermediate segment a premature 2.0s duration and renders two footers for
    one turn. Only the final answer may be stamped, with the full
    user-to-completion duration.
    """
    db = tmp_path / "state.db"
    _seed_shape(
        db,
        "narrated",
        [
            ("user", "find the config", 1000.0),
            ("assistant", "Let me search", 1002.0),
            ("tool", "3 matches", 1003.0),
            ("assistant", "It lives in config.json", 1008.5),
        ],
        tool_calls_column=tool_calls_column,
    )

    msgs = _read_real_transcript(db, "narrated")
    assert all(not m.get("tool_calls") for m in msgs), (
        "fixture must reproduce a shape where tool_calls cannot discriminate"
    )
    stats = agent_sessions.read_agent_session_turn_footer_stats(db, "narrated")
    stamped = agent_sessions.stamp_imported_turn_footers(msgs, stats, detach=True)

    narrated, final = stamped[1], stamped[3]
    assert "_usedModel" not in narrated, "narrated tool segment was stamped"
    assert "_turnDuration" not in narrated, "narrated tool segment got a premature 2.0s"
    assert "_turnUsage" not in narrated
    assert final["_usedModel"] == "openai/gpt-5"
    assert final["_turnDuration"] == 8.5


@pytest.mark.parametrize("tool_calls_column", _AMBIGUOUS_SHAPES)
@pytest.mark.parametrize(
    "narration", ["Let me search", ""], ids=["narrated", "blank"]
)
def test_aborted_turn_with_no_final_answer_stamps_nothing(
    tmp_path, tool_calls_column, narration
):
    """Tool work outstanding and no answer recorded: fail closed.

    A footer here would assert a duration and a model for a turn that never
    finished. Covers both the narrated and the blank intermediate, since neither
    content shape tells you an answer arrived.
    """
    db = tmp_path / "state.db"
    _seed_shape(
        db,
        "aborted",
        [
            ("user", "find the config", 1000.0),
            ("assistant", narration, 1002.0),
            ("tool", "3 matches", 1003.0),
        ],
        tool_calls_column=tool_calls_column,
    )

    msgs = _read_real_transcript(db, "aborted")
    stats = agent_sessions.read_agent_session_turn_footer_stats(db, "aborted")
    stamped = agent_sessions.stamp_imported_turn_footers(msgs, stats, detach=True)

    assert all("_usedModel" not in m for m in stamped), (
        "an unfinished turn was given a footer"
    )
    assert all("_turnDuration" not in m for m in stamped)
    assert all("_turnUsage" not in m for m in stamped)


@pytest.mark.parametrize("tool_calls_column", _AMBIGUOUS_SHAPES)
def test_plain_answer_control_keeps_its_footer(tmp_path, tool_calls_column):
    """Control: hardening ownership must not cost ordinary turns their footer."""
    db = tmp_path / "state.db"
    _seed_shape(
        db,
        "plain",
        [
            ("user", "hello", 1000.0),
            ("assistant", "hi there", 1003.0),
        ],
        api_calls=1,
        tool_calls_column=tool_calls_column,
    )

    msgs = _read_real_transcript(db, "plain")
    stats = agent_sessions.read_agent_session_turn_footer_stats(db, "plain")
    stamped = agent_sessions.stamp_imported_turn_footers(msgs, stats, detach=True)

    assert stamped[-1]["_usedModel"] == "openai/gpt-5"
    assert stamped[-1]["_turnDuration"] == 3.0
    assert stamped[-1]["_turnUsage"]["output_tokens"] == 10


@pytest.mark.parametrize("tool_calls_column", _AMBIGUOUS_SHAPES)
def test_multi_round_tool_turn_stamps_only_the_last_answer(tmp_path, tool_calls_column):
    """Several tool rounds inside one turn still yield exactly one footer."""
    db = tmp_path / "state.db"
    _seed_shape(
        db,
        "rounds",
        [
            ("user", "do it", 1000.0),
            ("assistant", "Let me look", 1001.0),
            ("tool", "partial", 1002.0),
            ("assistant", "Now let me check the other file", 1003.0),
            ("tool", "more", 1004.0),
            ("assistant", "All set", 1009.0),
        ],
        tool_calls_column=tool_calls_column,
    )

    msgs = _read_real_transcript(db, "rounds")
    stats = agent_sessions.read_agent_session_turn_footer_stats(db, "rounds")
    stamped = agent_sessions.stamp_imported_turn_footers(msgs, stats, detach=True)

    stamped_idxs = [i for i, m in enumerate(stamped) if "_usedModel" in m]
    assert stamped_idxs == [5], "exactly one footer belongs to a multi-round turn"
    assert stamped[5]["_turnDuration"] == 9.0


def test_populated_tool_calls_metadata_is_still_honoured(tmp_path):
    """The explicit signal must keep working where the schema does supply it.

    Structural inference is the fallback, not a replacement: an assistant that
    declares tool_calls is intermediate even if nothing follows it yet.
    """
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "explicit", api_calls=2)
        conn.executemany(
            "INSERT INTO messages (id, session_id, role, content, timestamp, tool_calls) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("e_u1", "explicit", "user", "do it", 1000.0, None),
                ("e_a1", "explicit", "assistant", "Let me search", 1002.0,
                 '[{"id": "c1", "function": {"name": "grep"}}]'),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    msgs = _read_real_transcript(db, "explicit")
    assert msgs[1].get("tool_calls"), "reader should have parsed the tool_calls column"
    stats = agent_sessions.read_agent_session_turn_footer_stats(db, "explicit")
    stamped = agent_sessions.stamp_imported_turn_footers(msgs, stats, detach=True)

    assert all("_usedModel" not in m for m in stamped), (
        "an assistant declaring tool_calls is a request, not an answer"
    )


@pytest.mark.parametrize("native_on", ["sidecar", "state_metadata"])
def test_source_label_webui_alone_denies_backfill(tmp_path, native_on):
    """``source_label="WebUI"`` is a native marker and must veto the backfill.

    ``_session_requires_cli_metadata_lookup`` accepts ``source_label`` as a source
    marker, and the canonical ``is_cli_session_row`` accepts
    ``source_label == "webui"`` as native authority — but the narrower
    ``_session_source_is_webui`` does not read that key at all. A native row whose
    only surviving native marker is ``source_label`` therefore passed the foreign
    gate, and same-ID session totals got stamped onto a WebUI transcript as false
    model/duration/token attribution.

    Parametrized over which authority carries the native marker, with the other
    deliberately carrying a foreign one, so the veto is proven per-authority
    rather than on a merged marker set.
    """
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "label_native", model="should-not-appear", api_calls=1)
        _add_turn(conn, "label_native", 1, user_ts=1000.0, assistant_ts=1004.0)
    finally:
        conn.close()

    mirror = [
        {"role": "user", "content": "x", "timestamp": 1000.0},
        {"role": "assistant", "content": "x", "timestamp": 1004.0},
    ]
    sidecar = _ImportedSidecar("label_native")
    sidecar.messages = mirror
    # Strip the markers the narrow veto would have caught, leaving source_label
    # as the only native signal on whichever authority is under test.
    sidecar.source_tag = None
    sidecar.raw_source = None
    sidecar.session_source = None
    sidecar.source = None

    if native_on == "sidecar":
        sidecar.source_label = "WebUI"
        cli_meta = {"session_id": "label_native", "source_tag": "cli", "raw_source": "cli"}
    else:
        sidecar.source_label = None
        sidecar.is_cli_session = True  # foreign marker on the sidecar authority
        cli_meta = {"session_id": "label_native", "source_label": "WebUI"}

    captured = _get_api_session(
        db, "label_native", session=sidecar, cli_meta=cli_meta, spy_stats=True
    )

    msgs = captured["data"]["session"]["messages"]
    assert len(msgs) == 2, "the length guard must not be what blocks this case"
    assert captured["stats_reads"] == [], (
        "a WebUI marker must deny backfill BEFORE the state.db stats read"
    )
    assert all("_usedModel" not in m for m in msgs), (
        "session totals were stamped onto a WebUI-native transcript"
    )
    assert all("_turnUsage" not in m for m in msgs)


def test_source_label_foreign_value_still_allows_backfill(tmp_path):
    """Control: widening the veto to source_label must not deny genuine imports."""
    db = tmp_path / "state.db"
    conn = _make_state_db(db)
    try:
        _add_session(conn, "label_cli", model="openai/gpt-5", api_calls=1)
        _add_turn(conn, "label_cli", 1, user_ts=1000.0, assistant_ts=1005.5)
    finally:
        conn.close()

    sidecar = _ImportedSidecar("label_cli")
    sidecar.source_tag = None
    sidecar.raw_source = None
    sidecar.session_source = None
    sidecar.source = None
    sidecar.source_label = "CLI"

    captured = _get_api_session(
        db,
        "label_cli",
        session=sidecar,
        cli_meta={"session_id": "label_cli", "source_label": "CLI"},
        spy_stats=True,
    )

    msgs = captured["data"]["session"]["messages"]
    assistant = [m for m in msgs if m.get("role") == "assistant"][-1]
    assert captured["stats_reads"], "a genuine import should still read stats"
    assert assistant["_usedModel"] == "openai/gpt-5"
    assert assistant["_turnDuration"] == 5.5
