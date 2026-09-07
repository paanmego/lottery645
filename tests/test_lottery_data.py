from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import requests

from lottery_data import (
    LotteryClient,
    LotteryDataError,
    estimate_latest_draw,
    normalize_prize_info,
    parse_api_draw,
    sync_missing_draws,
)


def api_row(draw_number: int = 1240) -> dict:
    return {
        "ltEpsd": draw_number,
        "tm1WnNo": 11,
        "tm2WnNo": 13,
        "tm3WnNo": 19,
        "tm4WnNo": 20,
        "tm5WnNo": 31,
        "tm6WnNo": 44,
        "bnsWnNo": 7,
        "ltRflYmd": "20260905",
        "rnk1WnNope": 9,
        "rnk1WnAmt": 3_000_000_000,
    }


def test_parse_current_api_draw() -> None:
    parsed = parse_api_draw(api_row())

    assert parsed["draw_number"] == 1240
    assert parsed["numbers"] == [11, 13, 19, 20, 31, 44]
    assert parsed["bonus_number"] == 7
    assert parsed["draw_date"] == pd.Timestamp("2026-09-05")
    assert parsed["prize_info"][0] == {
        "rank": "1등",
        "winners": 9,
        "prize_amount": 3_000_000_000,
    }


def test_legacy_prize_columns_are_repaired() -> None:
    legacy = [
        {
            "rank": "1등",
            "winners": "26,343,470,256원",
            "prize_amount": "12",
            "winners_per_prize": "2195289188.00000000",
        }
    ]

    assert normalize_prize_info(legacy) == [
        {"rank": "1등", "winners": 12, "prize_amount": 2_195_289_188}
    ]


def test_draw_estimate_matches_known_saturday() -> None:
    assert estimate_latest_draw(date(2026, 9, 5)) == 1240


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"data": {"list": [api_row()]}}


class FakeSession:
    def __init__(self) -> None:
        self.call = None

    def get(self, url: str, **kwargs):
        self.call = (url, kwargs)
        return FakeResponse()


def test_client_uses_bounded_connect_and_read_timeouts() -> None:
    session = FakeSession()
    client = LotteryClient(session=session, connect_timeout=2.0, read_timeout=5.0)

    draw = client.fetch_draw(1240)

    assert draw is not None
    assert session.call[1]["timeout"] == (2.0, 5.0)
    assert session.call[1]["params"] == {"srchLtEpsd": 1240}


class TimeoutSession:
    def get(self, *args, **kwargs):
        raise requests.ConnectTimeout("offline")


def test_network_details_are_wrapped_in_safe_domain_error() -> None:
    client = LotteryClient(session=TimeoutSession())

    with pytest.raises(LotteryDataError, match="연결 또는 응답 처리"):
        client.fetch_draw(1240)


class WindowClient:
    def fetch_latest_window(self, today=None):
        return [parse_api_draw(api_row(3))]

    def fetch_draw_window(self, center_draw):
        return [parse_api_draw(api_row(number)) for number in (2, 3)]

    def fetch_draw(self, draw_number):
        return parse_api_draw(api_row(draw_number))


def test_sync_fills_all_missing_draws() -> None:
    first = parse_api_draw(api_row(1))
    existing = pd.DataFrame([first])

    refreshed, count = sync_missing_draws(existing, WindowClient())

    assert count == 2
    assert refreshed["draw_number"].tolist() == [1, 2, 3]
