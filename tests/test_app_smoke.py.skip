"""Smoke test: the Streamlit dashboard must boot without raising.

This is the tripwire for the app.py refactor — every extraction step must
keep this green. It does not assert on layout, only that the script runs
end-to-end and renders something.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_dashboard_boots_without_exception():
    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    assert not at.exception, f"Dashboard raised on boot: {at.exception}"


def test_dashboard_renders_content():
    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    # The app must produce at least some markdown/title output; an empty
    # render means the script bailed out silently.
    rendered = list(at.markdown) + list(at.title) + list(at.header)
    assert rendered, "Dashboard rendered no content at all"
