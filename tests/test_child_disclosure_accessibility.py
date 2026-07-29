"""Browser coverage for the production delegated-child disclosure (#6510)."""

import os
from pathlib import Path

import pytest


STATIC = Path(__file__).resolve().parents[1] / "static"
SESSIONS_JS = (STATIC / "sessions.js").read_text(encoding="utf-8")
STYLE_CSS = (STATIC / "style.css").read_text(encoding="utf-8")

MOBILE = {"width": 390, "height": 844}
DESKTOP = {"width": 1440, "height": 900}
TOUCH_FLOOR = 44
CONTROL_SELECTOR = ".session-child-disclosure, .session-title-row > .session-child-count"

HARNESS = """
window.S = {session: null, activeProfile: 'default'};
window.$ = (id) => document.getElementById(id);
window.li = () => '';
window.t = (key, count) => {
  if (key === 'session_meta_children') return `${count} 子`;
  if (key === 'session_select_mode') return 'Select';
  return key;
};
window.api = async () => ({});
window.showToast = () => {};
window.closeSessionActionMenu = () => {};
"""

SESSIONS = [
    {
        "session_id": "gateway_parent",
        "title": "Call the delegate_task TOOL directly",
        "source": "api_server",
        "raw_source": "api_server",
        "source_tag": "api_server",
        "session_source": "api",
        "source_label": "API",
        "message_count": 4,
        "updated_at": 30,
    },
    {
        "session_id": "subagent_a",
        "title": "Subagent Session A",
        "parent_session_id": "gateway_parent",
        "relationship_type": "child_session",
        "raw_source": "subagent",
        "source_tag": "subagent",
        "session_source": "other",
        "source_label": "Subagent",
        "parent_source": "api_server",
        "_parent_lineage_root_id": "gateway_parent",
        "_cross_surface_child_session": True,
        "message_count": 2,
        "updated_at": 20,
    },
    {
        "session_id": "subagent_b",
        "title": "Subagent Session B",
        "parent_session_id": "gateway_parent",
        "relationship_type": "child_session",
        "raw_source": "subagent",
        "source_tag": "subagent",
        "session_source": "other",
        "source_label": "Subagent",
        "parent_source": "api_server",
        "_parent_lineage_root_id": "gateway_parent",
        "_cross_surface_child_session": True,
        "message_count": 2,
        "updated_at": 10,
    },
]


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:  # pragma: no cover - optional local dependency
        pytest.skip("playwright is unavailable; run the browser test job")
    return sync_playwright


def _page(playwright, viewport):
    executable_path = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH") or None
    browser = playwright.chromium.launch(
        headless=True,
        executable_path=executable_path,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )
    page = browser.new_page(viewport=viewport)
    page.set_content(
        "<!doctype html><html><body>"
        '<input id="sessionSearch"><div id="sessionList"></div>'
        '<div id="batchActionBar"></div></body></html>'
    )
    page.add_style_tag(content=STYLE_CSS)
    page.add_script_tag(content=HARNESS)
    page.add_script_tag(content=SESSIONS_JS)
    page.evaluate(
        """(sessions) => {
          _allSessions = sessions;
          _sidebarReferenceSessions = sessions;
          _allSessionsScope = {
            profile: 'default', allProfiles: false,
            sidebarSource: 'webui', excludeHidden: false
          };
          renderSessionListFromCache();
        }""",
        SESSIONS,
    )
    page.locator(CONTROL_SELECTOR).wait_for()
    return browser, page


def _target_geometry(page):
    return page.locator(CONTROL_SELECTOR).evaluate(
        """(control) => {
          const target = control.getBoundingClientRect();
          const visualEl = control.querySelector('.session-child-count') || control;
          const visual = visualEl.getBoundingClientRect();
          const points = [];
          for (const x of [target.left + .5, target.left + target.width / 2, target.right - .5]) {
            for (const y of [target.top + .5, target.top + target.height / 2, target.bottom - .5]) {
              const hit = document.elementFromPoint(x, y);
              points.push(hit === control || control.contains(hit));
            }
          }
          return {
            width: Math.round(target.width),
            height: Math.round(target.height),
            visualWidth: Math.round(visual.width),
            visualHeight: Math.round(visual.height),
            allPointsHit: points.every(Boolean),
          };
        }"""
    )


def test_production_disclosure_keeps_focus_across_native_keyboard_toggles():
    sync_playwright = _require_playwright()
    with sync_playwright() as playwright:
        browser, page = _page(playwright, MOBILE)
        control = page.locator(CONTROL_SELECTOR)
        control.focus()

        page.keyboard.press("Enter")
        after_enter = page.locator(CONTROL_SELECTOR)
        assert after_enter.get_attribute("aria-expanded") == "true"
        assert page.locator(".session-child-session").count() == 2
        assert after_enter.evaluate("el => document.activeElement === el")

        page.keyboard.press("Space")
        after_space = page.locator(CONTROL_SELECTOR)
        assert after_space.get_attribute("aria-expanded") == "false"
        assert page.locator(".session-child-session").count() == 0
        assert after_space.evaluate("el => document.activeElement === el")
        browser.close()


def test_production_disclosure_and_children_clear_mobile_touch_floor():
    sync_playwright = _require_playwright()
    with sync_playwright() as playwright:
        browser, page = _page(playwright, MOBILE)
        disclosure = _target_geometry(page)
        page.locator(CONTROL_SELECTOR).click()
        children = page.locator(".session-child-session")
        child_heights = children.evaluate_all(
            "rows => rows.map(row => Math.round(row.getBoundingClientRect().height))"
        )
        browser.close()

    assert disclosure["width"] >= TOUCH_FLOOR
    assert disclosure["height"] >= TOUCH_FLOOR
    assert disclosure["allPointsHit"], disclosure
    assert disclosure["visualWidth"] < TOUCH_FLOOR
    assert disclosure["visualHeight"] < TOUCH_FLOOR
    assert len(child_heights) == 2
    assert min(child_heights) >= TOUCH_FLOOR


def test_production_disclosure_keeps_compact_desktop_geometry():
    sync_playwright = _require_playwright()
    with sync_playwright() as playwright:
        browser, page = _page(playwright, DESKTOP)
        disclosure = _target_geometry(page)
        page.locator(CONTROL_SELECTOR).click()
        child_heights = page.locator(".session-child-session").evaluate_all(
            "rows => rows.map(row => Math.round(row.getBoundingClientRect().height))"
        )
        browser.close()

    assert disclosure["width"] < TOUCH_FLOOR
    assert disclosure["height"] < TOUCH_FLOOR
    assert max(child_heights) < TOUCH_FLOOR
