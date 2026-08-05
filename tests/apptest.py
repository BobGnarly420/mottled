"""One place that knows where the Streamlit app lives.

`AppTest.from_file` resolves a *relative* path differently across Streamlit
releases — older versions resolved against the working directory, 1.61
resolves against the file that called it, which turns `"ui.py"` into
`tests/ui.py` and fails. The path is absolute here so the tests do not depend
on which behaviour the installed Streamlit has.
"""
from pathlib import Path

APP = str(Path(__file__).resolve().parent.parent / "ui.py")


def app_test(**kwargs):
    """`AppTest` for the explorer, with an unambiguous path."""
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP, **kwargs)
