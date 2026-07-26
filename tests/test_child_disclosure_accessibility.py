"""Browser regression coverage for the delegated-child disclosure (#6510 gate).

Nesting delegated subagents made this disclosure the only route to a child row,
which promoted its pre-existing keyboard and touch-target debt to a blocker.

Both assertions here are computed in a real browser against the real
`static/style.css`, and activation goes through real key events — no `.onclick()`
invocation and no CSS source-string matching, per the gate's test-plan
requirement. A third check pins that production `sessions.js` actually emits the
button semantics, so the browser evidence cannot pass while the wiring regresses.
"""
from pathlib import Path

import pytest


STATIC = Path(__file__).resolve().parents[1] / "static"
SESSIONS_JS = (STATIC / "sessions.js").read_text(encoding="utf-8")
STYLE_CSS = (STATIC / "style.css").read_text(encoding="utf-8")

MOBILE = {"width": 390, "height": 844}
DESKTOP = {"width": 1440, "height": 900}
TOUCH_FLOOR = 44

# Mirrors the markup the production disclosure branch builds: a
# .session-child-count pill carrying full button semantics, a .session-actions
# trigger pinned to the row, and the .session-child-session rows it reveals.
def _production_disclosure_branch() -> str:
    """The inline branch in sessions.js that builds the child-count disclosure."""
    start = SESSIONS_JS.find("childCountEl.className='session-child-count'")
    assert start > 0, "child-count disclosure branch not found"
    return SESSIONS_JS[start:start + 1400]


def production_semantics() -> dict:
    """Which button semantics production actually emits.

    The fixture below applies ONLY these, so the browser tests exercise the real
    contract: if production stops setting `tabindex`, focus fails here; if it
    stops binding `onkeydown`, the key presses do nothing. That keeps the
    keyboard test a production regression test rather than a test of its own
    fixture.
    """
    branch = _production_disclosure_branch()
    return {
        "role": "setAttribute('role','button')" in branch,
        "tabindex": "setAttribute('tabindex','0')" in branch,
        "aria": "setAttribute('aria-expanded'" in branch,
        "keydown": "childCountEl.onkeydown" in branch
        and "e.key==='Enter'||e.key===' '" in branch,
    }


FIXTURE = """
window.__buildRow = (semantics) => {
  document.body.innerHTML = '';
  const list = document.createElement('div');
  list.id = 'sessionList';
  const item = document.createElement('div');
  item.className = 'session-item';
  const titleRow = document.createElement('div');
  titleRow.className = 'session-title-row';
  const title = document.createElement('span');
  title.className = 'session-title';
  title.textContent = 'Call the delegate_task TOOL directly';
  titleRow.appendChild(title);

  const pill = document.createElement('span');
  pill.className = 'session-child-count';
  pill.textContent = '2 children';
  if (semantics.role) pill.setAttribute('role', 'button');
  if (semantics.tabindex) pill.setAttribute('tabindex', '0');
  if (semantics.aria) pill.setAttribute('aria-expanded', 'false');
  const kids = document.createElement('div');
  kids.className = 'session-child-sessions';
  kids.style.display = 'none';
  window.__activations = 0;
  const toggle = (e) => {
    e.preventDefault();
    e.stopPropagation();
    window.__activations += 1;
    const next = pill.getAttribute('aria-expanded') !== 'true';
    if (semantics.aria) pill.setAttribute('aria-expanded', next ? 'true' : 'false');
    kids.style.display = next ? 'flex' : 'none';
  };
  pill.onclick = toggle;
  if (semantics.keydown) {
    pill.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') toggle(e); };
  }
  titleRow.appendChild(pill);
  item.appendChild(titleRow);

  const actions = document.createElement('div');
  actions.className = 'session-actions';
  const trigger = document.createElement('button');
  trigger.className = 'session-actions-trigger';
  trigger.textContent = 'x';
  actions.appendChild(trigger);
  item.appendChild(actions);

  for (let i = 0; i < 2; i++) {
    const b = document.createElement('button');
    b.className = 'session-child-session';
    b.textContent = '-> Subagent Session - 5m';
    kids.appendChild(b);
  }
  list.appendChild(item);
  list.appendChild(kids);
  document.body.appendChild(list);
};

window.__disclosureHitHeight = () => {
  const el = document.querySelector('.session-child-count');
  const box = el.getBoundingClientRect();
  const overlay = parseFloat(getComputedStyle(el, '::after').height) || 0;
  return {visual: Math.round(box.height), hit: Math.round(Math.max(box.height, overlay))};
};

window.__childGeometry = () => {
  const el = document.querySelector('.session-child-session');
  const box = el.getBoundingClientRect();
  return {
    height: Math.round(box.height),
    clipped: el.scrollHeight > el.clientHeight + 1,
  };
};

window.__actionsTriggerReachable = () => {
  const t = document.querySelector('.session-actions-trigger');
  const r = t.getBoundingClientRect();
  if (!r.width) return 'zero-size';
  const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
  return (hit === t || t.contains(hit)) ? 'reachable' : 'shadowed';
};
"""


def _page(playwright, viewport):
    browser = playwright.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )
    page = browser.new_page(viewport=viewport)
    page.set_content("<!doctype html><html><body></body></html>")
    page.add_style_tag(content=STYLE_CSS)
    page.add_script_tag(content=FIXTURE)
    page.evaluate("(semantics) => window.__buildRow(semantics)", production_semantics())
    return browser, page


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except Exception:  # pragma: no cover - dependency missing path
        pytest.skip("playwright is unavailable; run the child disclosure browser test")
    return sync_playwright


def test_child_disclosure_is_reachable_by_native_keyboard():
    """Enter and Space must both reveal the nested children.

    Activation is driven by real key events on the focused element, so a handler
    that only responds to pointer input fails here.
    """
    sync_playwright = _require_playwright()
    with sync_playwright() as playwright:
        browser, page = _page(playwright, MOBILE)
        pill = page.locator(".session-child-count")
        pill.focus()
        focused = page.evaluate("() => document.activeElement.className")

        page.keyboard.press("Enter")
        after_enter = {
            "activations": page.evaluate("() => window.__activations"),
            "aria": pill.get_attribute("aria-expanded"),
            "children_visible": page.locator(".session-child-session").first.is_visible(),
        }

        page.keyboard.press(" ")
        after_space = {
            "activations": page.evaluate("() => window.__activations"),
            "aria": pill.get_attribute("aria-expanded"),
        }
        browser.close()

    assert focused == "session-child-count", "disclosure did not accept keyboard focus"
    assert after_enter == {"activations": 1, "aria": "true", "children_visible": True}
    # Space toggles closed again, and aria-expanded tracks state both ways.
    assert after_space == {"activations": 2, "aria": "false"}


def test_disclosure_and_child_targets_clear_the_mobile_touch_floor():
    """Computed geometry at 390x844 against the real stylesheet.

    The disclosure keeps its compact visual while its hit area reaches the floor,
    the revealed rows reach it directly, nothing clips, and the row's own actions
    trigger is not shadowed by the enlarged hit area.
    """
    sync_playwright = _require_playwright()
    with sync_playwright() as playwright:
        browser, page = _page(playwright, MOBILE)
        page.locator(".session-child-count").click()
        disclosure = page.evaluate("() => window.__disclosureHitHeight()")
        child = page.evaluate("() => window.__childGeometry()")
        trigger = page.evaluate("() => window.__actionsTriggerReachable()")
        browser.close()

    assert disclosure["hit"] >= TOUCH_FLOOR, f"disclosure hit area {disclosure} below floor"
    # Compact visual preserved — the fix must not bulk up the sidebar row.
    assert disclosure["visual"] < TOUCH_FLOOR, "disclosure pill grew visually"
    assert child["height"] >= TOUCH_FLOOR, f"child target {child} below floor"
    assert child["clipped"] is False, "child row text is clipped"
    assert trigger == "reachable", f"actions trigger {trigger} by the disclosure hit area"


def test_desktop_layout_keeps_its_compact_rows():
    """The touch floor is scoped to the mobile breakpoint, not applied globally."""
    sync_playwright = _require_playwright()
    with sync_playwright() as playwright:
        browser, page = _page(playwright, DESKTOP)
        page.locator(".session-child-count").click()
        child = page.evaluate("() => window.__childGeometry()")
        disclosure = page.evaluate("() => window.__disclosureHitHeight()")
        browser.close()

    assert child["height"] < TOUCH_FLOOR, "desktop child rows gained mobile padding"
    assert disclosure["hit"] < TOUCH_FLOOR, "desktop disclosure gained the mobile overlay"


def test_production_disclosure_wires_button_semantics():
    """Guard the wiring the browser tests above assume.

    The disclosure is built inline inside the sidebar render path and cannot be
    extracted standalone, so this pins that production still emits the semantics
    rather than the fixture being the only place they exist.
    """
    start = SESSIONS_JS.find("childCountEl.className='session-child-count'")
    assert start > 0, "child-count disclosure branch not found"
    branch = SESSIONS_JS[start:start + 1400]
    for expected in (
        "setAttribute('role','button')",
        "setAttribute('tabindex','0')",
        "setAttribute('aria-expanded'",
        "childCountEl.onkeydown",
        "e.key==='Enter'||e.key===' '",
    ):
        assert expected in branch, f"disclosure lost {expected}"
