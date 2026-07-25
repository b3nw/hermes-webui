"""Per-turn footer metadata for state.db-imported sessions.

Imported agent transcripts (delegated subagents, CLI, TUI, cron) are not run by
the WebUI, so ``_run_agent_streaming`` never stamps the per-turn footer fields
and the transcript renders with no model, duration, or token information at all.
These tests pin the server-side backfill that reads those facts off the session
row instead.
"""

import sqlite3

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
            timestamp REAL
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
