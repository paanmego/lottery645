from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders_from_local_cache(monkeypatch) -> None:
    import lottery_data

    monkeypatch.setattr(
        lottery_data,
        "sync_missing_draws",
        lambda dataframe, client, today=None: (dataframe, 0),
    )
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"

    app = AppTest.from_file(str(app_path), default_timeout=20).run()

    assert not app.exception
    assert any("Lotto Insight" in markdown.value for markdown in app.markdown)
    assert any("Bryah Cho 제작" in markdown.value for markdown in app.markdown)
    assert any("1등 당첨금" in markdown.value for markdown in app.markdown)
    assert len(app.tabs) == 3
    assert len(app.dataframe) == 1
    assert len(app.button) >= 2
