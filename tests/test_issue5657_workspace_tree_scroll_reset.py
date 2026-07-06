"""Regression coverage for #5657 — workspace tree re-render preserves scrollTop.

`renderFileTree()` in `static/ui.js` rebuilds the file tree by clearing
`#fileTree.innerHTML` and re-appending every row. Setting `innerHTML = ''` on
the scrollable container collapses its `scrollHeight`, which the browser
treats as "no overflow" and clamps `scrollTop` to `0`. Every folder
expand/collapse, breadcrumb navigation, refresh, and hidden-files toggle
that re-runs the renderer therefore snapped the user's view back to the top
of long directory listings.

The fix captures `box.scrollTop` before the wipe and restores it after the
re-paint. This test pins that invariant as a static guard against the
renderer silently regressing to the wipe-and-rebuild-only shape.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UI_JS = (REPO_ROOT / "static" / "ui.js").read_text(encoding="utf-8")


def _render_file_tree_body() -> str:
    start = UI_JS.index("function renderFileTree()")
    # The renderer ends at the next blank line followed by a top-level
    # `let _wsActiveDragPath` declaration, which is unique to the workspace
    # tree section.
    end = UI_JS.index("\n\nlet _wsActiveDragPath", start)
    return UI_JS[start:end]


def test_render_file_tree_captures_scrolltop_before_wipe():
    body = _render_file_tree_body()
    # Capture the value BEFORE wiping: `prevScrollTop=box?box.scrollTop:0;`
    capture_marker = "prevScrollTop=box?box.scrollTop:0"
    capture_idx = body.index(capture_marker)
    # The wipe statement (code, not the comment that mentions it): `box.innerHTML='';`
    wipe_marker = "box.innerHTML='';"
    wipe_idx = body.index(wipe_marker)
    assert capture_idx != -1, "renderFileTree must read box.scrollTop before the wipe"
    assert wipe_idx != -1, "renderFileTree must still clear the container"
    assert capture_idx < wipe_idx, (
        "scrollTop must be captured BEFORE innerHTML='' so the previous offset "
        "is observable — otherwise the wipe collapses scrollHeight first and "
        "the browser clamps scrollTop to 0 (#5657)."
    )


def test_render_file_tree_restores_scrolltop_after_repaint():
    body = _render_file_tree_body()
    repaint_idx = body.index("_renderTreeItems(box, visibleEntries, 0)")
    restore_marker = "box.scrollTop=prevScrollTop"
    assert repaint_idx != -1
    assert restore_marker in body, (
        "renderFileTree must assign prevScrollTop back to box.scrollTop after "
        "_renderTreeItems repaints the rows (#5657)."
    )
    # Use the LAST occurrence — the capture line also contains "box.scrollTop"
    # but the restore line is the assign-to-prevScrollTop tail.
    restore_idx = body.rindex(restore_marker)
    assert restore_idx > repaint_idx, (
        "scrollTop restore must run AFTER _renderTreeItems repopulates the "
        "container — restoring before the rows are appended would clamp back "
        "to 0 because scrollHeight would still be collapsed."
    )


def test_render_file_tree_keeps_empty_state_early_returns():
    """The two early-return paths (no workspace, empty dir) hide #fileTree or
    show the empty-state placeholder. They must keep returning early without
    touching scrollTop — restoring it there is harmless but pointless, and
    the early return shape must not regress into a stale capture."""
    body = _render_file_tree_body()
    assert "box.style.display='none';" in body
    assert "return;" in body
    # Restore must be at the tail of the function, past both early returns.
    restore_idx = body.rindex("box.scrollTop=prevScrollTop")
    first_early_return = body.index("return;", body.index("box.style.display='none';"))
    assert restore_idx > first_early_return, (
        "scrollTop restore must sit past the empty-state early returns so a "
        "wiped-but-not-repopulated container never reaches the assign."
    )


def test_no_data_path_anchor_branch_added():
    """The original issue suggested a `data-path` row-anchor branch, but
    `_renderTreeItems` rows only carry `dataset.wsType` / `dataset.wsIsDir`.
    That anchor selector would match nothing. The fix is plain scrollTop
    capture/restore; this guard pins that the dead branch is NOT added."""
    body = _render_file_tree_body()
    assert "_lastClickedDirPath" not in body
    assert "_lastClickedRowRect" not in body
    assert "querySelector('.file-item[data-path=" not in body
    assert "getBoundingClientRect" not in body