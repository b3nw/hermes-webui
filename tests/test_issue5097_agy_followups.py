"""#5097 follow-up: truncate rejects read-only sessions."""

import pathlib
import re

ROUTES = pathlib.Path("api/routes.py").read_text(encoding="utf-8")


def test_truncate_route_read_only_guard():
    start = ROUTES.index('if parsed.path == "/api/session/truncate":')
    block = ROUTES[start : start + 2500]
    assert 'getattr(s, "read_only", False)' in block
    assert "Session is read-only" in block
    assert "403" in block


def test_ui_regenerate_loads_before_user_search():
    ui = pathlib.Path("static/ui.js").read_text(encoding="utf-8")
    body = ui[ui.index("async function regenerateResponse") :]
    end = body.index("\nfunction postProcessRenderedMessages")
    body = body[:end]
    load_pos = body.find("_ensureAllMessagesLoaded")
    loop_pos = body.find("for(let i = absoluteKeepCount - 1")
    assert load_pos != -1 and loop_pos != -1 and load_pos < loop_pos