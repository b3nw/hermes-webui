"""#5097: any user message may show edit; mid-history uses absolute keep_count."""

import pathlib
import re

UI = pathlib.Path("static/ui.js").read_text(encoding="utf-8")


def test_any_user_message_gets_edit_button():
    assert "const isEditableUser=isUser;" in UI
    assert "const isEditableUser=isUser&&rawIdx===lastUserRawIdx;" not in UI
    assert "const editBtn  = isEditableUser ?" in UI


def test_submit_edit_absolute_keep_count():
    body = UI[UI.index("async function submitEdit") :]
    end = body.index("\nasync function regenerateResponse")
    body = body[:end]
    assert re.search(r"absoluteKeepCount\s*=\s*_oldestIdx\s*\+\s*msgIdx", body)
    assert "keep_count: absoluteKeepCount" in body