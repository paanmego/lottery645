"""Official Lotto 6/45 data access and resilient local-cache handling."""

from __future__ import annotations

import ast
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger(__name__)

OFFICIAL_BASE_URL = "https://www.dhlottery.co.kr"
DRAW_ENDPOINT = "/lt645/selectPstLt645Info.do"
DRAW_WINDOW_ENDPOINT = "/lt645/selectPstLt645InfoNew.do"
FIRST_DRAW_DATE = date(2002, 12, 7)
REQUIRED_COLUMNS = {
    "draw_number",
    "numbers",
    "bonus_number",
    "prize_info",
    "draw_date",
}


class LotteryDataError(RuntimeError):
    """Raised when official draw data cannot be fetched or validated."""


@dataclass(frozen=True)
class RefreshStatus:
    online: bool
    updated_count: int
    latest_draw: int
    message: str


def _digits(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(value)
    digits = "".join(char for char in str(value) if char.isdigit())
    return int(digits) if digits else 0


def parse_draw_date(value: Any) -> pd.Timestamp:
    """Parse both the legacy Korean date and the current API date."""
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).normalize()

    text = str(value).strip().replace(" 추첨", "")
    for date_format in ("%Y-%m-%d", "%Y%m%d", "%Y년 %m월 %d일"):
        try:
            return pd.Timestamp(datetime.strptime(text, date_format).date())
        except ValueError:
            continue
    raise LotteryDataError(f"지원하지 않는 추첨일 형식입니다: {text}")


def normalize_prize_info(value: Any) -> list[dict[str, Any]]:
    """Normalize current API data and repair columns swapped by the old app."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise LotteryDataError("당첨금 데이터 형식이 올바르지 않습니다.") from exc

    if not isinstance(value, list):
        raise LotteryDataError("당첨금 데이터는 목록이어야 합니다.")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue

        raw_winners = item.get("winners", 0)
        raw_amount = item.get("prize_amount", 0)
        # The old scraper saved the total pool in `winners` and winner count in
        # `prize_amount`. Its derived field contains the per-person amount.
        legacy_swapped = "원" in str(raw_winners) and "원" not in str(raw_amount)
        if legacy_swapped:
            winners = _digits(raw_amount)
            per_person = item.get("winners_per_prize")
            try:
                prize_amount = int(round(float(per_person)))
            except (TypeError, ValueError):
                total_amount = _digits(raw_winners)
                prize_amount = total_amount // winners if winners else 0
        else:
            winners = _digits(raw_winners)
            prize_amount = _digits(raw_amount)

        normalized.append(
            {
                "rank": str(item.get("rank") or f"{index}등"),
                "winners": winners,
                "prize_amount": prize_amount,
            }
        )
    return normalized


def _parse_numbers(value: Any) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    numbers = sorted(int(number) for number in value)
    if len(numbers) != 6 or len(set(numbers)) != 6 or any(not 1 <= n <= 45 for n in numbers):
        raise LotteryDataError("당첨번호는 중복 없는 1~45의 숫자 6개여야 합니다.")
    return numbers


def load_lottery_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))

    dataframe = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise LotteryDataError(f"데이터 파일에 필수 열이 없습니다: {', '.join(sorted(missing))}")

    dataframe = dataframe.copy()
    dataframe["draw_number"] = dataframe["draw_number"].astype(int)
    dataframe["numbers"] = dataframe["numbers"].apply(_parse_numbers)
    dataframe["bonus_number"] = dataframe["bonus_number"].astype(int)
    dataframe["prize_info"] = dataframe["prize_info"].apply(normalize_prize_info)
    dataframe["draw_date"] = dataframe["draw_date"].apply(parse_draw_date)
    dataframe = (
        dataframe.drop_duplicates("draw_number", keep="last")
        .sort_values("draw_number")
        .reset_index(drop=True)
    )
    return dataframe


def save_lottery_data(dataframe: pd.DataFrame, path: str | Path) -> None:
    """Atomically persist normalized data so restarts never see a partial CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = dataframe.copy().sort_values("draw_number")
    output["numbers"] = output["numbers"].apply(
        lambda value: json.dumps(list(value), ensure_ascii=False)
    )
    output["prize_info"] = output["prize_info"].apply(
        lambda value: json.dumps(normalize_prize_info(value), ensure_ascii=False)
    )
    output["draw_date"] = output["draw_date"].apply(
        lambda value: parse_draw_date(value).strftime("%Y-%m-%d")
    )

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as temp_file:
            temp_name = temp_file.name
            output.to_csv(temp_file, index=False)
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def estimate_latest_draw(today: date | None = None) -> int:
    """Estimate the current draw; the API remains authoritative on publication."""
    today = today or date.today()
    if today < FIRST_DRAW_DATE:
        return 1
    return ((today - FIRST_DRAW_DATE).days // 7) + 1


def parse_api_draw(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        draw_number = int(payload["ltEpsd"])
        numbers = sorted(int(payload[f"tm{rank}WnNo"]) for rank in range(1, 7))
        bonus_number = int(payload["bnsWnNo"])
        draw_date = parse_draw_date(payload["ltRflYmd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LotteryDataError("공식 응답에 필수 추첨 정보가 없습니다.") from exc

    if len(set(numbers)) != 6 or any(not 1 <= number <= 45 for number in numbers):
        raise LotteryDataError(f"{draw_number}회 당첨번호 검증에 실패했습니다.")
    if not 1 <= bonus_number <= 45 or bonus_number in numbers:
        raise LotteryDataError(f"{draw_number}회 보너스번호 검증에 실패했습니다.")

    prize_info = [
        {
            "rank": f"{rank}등",
            "winners": int(payload.get(f"rnk{rank}WnNope") or 0),
            "prize_amount": int(payload.get(f"rnk{rank}WnAmt") or 0),
        }
        for rank in range(1, 6)
    ]
    return {
        "draw_number": draw_number,
        "numbers": numbers,
        "bonus_number": bonus_number,
        "prize_info": prize_info,
        "draw_date": draw_date,
    }


class LotteryClient:
    """Official-site client with bounded latency and automatic retries."""

    def __init__(
        self,
        session: requests.Session | None = None,
        connect_timeout: float = 3.05,
        read_timeout: float = 8.0,
    ) -> None:
        self.session = session or self._build_session()
        self.timeout = (connect_timeout, read_timeout)

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=2,
            connect=2,
            read=2,
            status=2,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "Lottery645Dashboard/2.0 (+https://github.com/paanmego/lottery645)",
            }
        )
        return session

    def _get_list(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            response = self.session.get(
                f"{OFFICIAL_BASE_URL}{endpoint}", params=params, timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("data", {}).get("list", [])
            if not isinstance(rows, list):
                raise LotteryDataError("공식 응답의 목록 형식이 올바르지 않습니다.")
            return rows
        except LotteryDataError:
            raise
        except (requests.RequestException, ValueError, AttributeError) as exc:
            raise LotteryDataError("공식 사이트 연결 또는 응답 처리에 실패했습니다.") from exc

    def fetch_draw(self, draw_number: int) -> dict[str, Any] | None:
        rows = self._get_list(DRAW_ENDPOINT, {"srchLtEpsd": int(draw_number)})
        if not rows:
            return None
        return parse_api_draw(rows[0])

    def fetch_draw_window(self, center_draw: int) -> list[dict[str, Any]]:
        rows = self._get_list(
            DRAW_WINDOW_ENDPOINT,
            {"srchDir": "center", "srchLtEpsd": int(center_draw)},
        )
        return [parse_api_draw(row) for row in rows]

    def fetch_latest_window(self, today: date | None = None) -> list[dict[str, Any]]:
        estimated = estimate_latest_draw(today)
        rows = self.fetch_draw_window(estimated)
        if rows:
            return rows

        # A holiday or publication delay can make the estimate unavailable.
        for draw_number in range(estimated, max(estimated - 8, 0), -1):
            draw = self.fetch_draw(draw_number)
            if draw:
                return [draw]
        raise LotteryDataError("발표된 최신 회차를 확인할 수 없습니다.")


def _rows_to_dataframe(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    dataframe = pd.DataFrame(list(rows))
    if dataframe.empty:
        return dataframe
    dataframe["draw_date"] = dataframe["draw_date"].apply(parse_draw_date)
    return dataframe


def sync_missing_draws(
    dataframe: pd.DataFrame,
    client: LotteryClient,
    today: date | None = None,
) -> tuple[pd.DataFrame, int]:
    """Fetch every missing draw with windows, then verify continuity."""
    latest_window = client.fetch_latest_window(today)
    latest_draw = max(row["draw_number"] for row in latest_window)
    local_latest = int(dataframe["draw_number"].max()) if not dataframe.empty else 0
    if local_latest >= latest_draw:
        return dataframe, 0

    fetched: dict[int, dict[str, Any]] = {
        row["draw_number"]: row for row in latest_window if row["draw_number"] > local_latest
    }

    # The center API returns ten draws. Stepping by ten makes stale-cache
    # recovery around ten times faster than one HTTP request per draw.
    for center in range(local_latest + 5, latest_draw + 6, 10):
        for row in client.fetch_draw_window(min(center, latest_draw)):
            if local_latest < row["draw_number"] <= latest_draw:
                fetched[row["draw_number"]] = row

    # Defensive single requests cover boundary changes in the window API.
    missing = [
        draw_number
        for draw_number in range(local_latest + 1, latest_draw + 1)
        if draw_number not in fetched
    ]
    for draw_number in missing:
        row = client.fetch_draw(draw_number)
        if row:
            fetched[draw_number] = row

    still_missing = [
        draw_number
        for draw_number in range(local_latest + 1, latest_draw + 1)
        if draw_number not in fetched
    ]
    if still_missing:
        raise LotteryDataError(f"누락 회차를 동기화하지 못했습니다: {still_missing[0]}회")

    new_rows = _rows_to_dataframe(fetched.values())
    combined = pd.concat([dataframe, new_rows], ignore_index=True)
    combined = (
        combined.drop_duplicates("draw_number", keep="last")
        .sort_values("draw_number")
        .reset_index(drop=True)
    )
    return combined, len(fetched)


def refresh_lottery_data(
    path: str | Path,
    client: LotteryClient | None = None,
    today: date | None = None,
) -> tuple[pd.DataFrame, RefreshStatus]:
    """Refresh when possible, but serve a valid local cache on outage."""
    dataframe = load_lottery_data(path)
    client = client or LotteryClient()
    local_latest = int(dataframe["draw_number"].max()) if not dataframe.empty else 0

    try:
        refreshed, updated_count = sync_missing_draws(dataframe, client, today)
        latest_draw = int(refreshed["draw_number"].max())
        if updated_count:
            try:
                save_lottery_data(refreshed, path)
            except OSError:
                # Some managed hosts use an ephemeral/read-only checkout.
                LOGGER.warning("Could not persist refreshed lottery data", exc_info=True)
        message = (
            f"공식 데이터 {updated_count}개 회차를 업데이트했습니다."
            if updated_count
            else "공식 데이터와 동기화되었습니다."
        )
        return refreshed, RefreshStatus(True, updated_count, latest_draw, message)
    except LotteryDataError as exc:
        if dataframe.empty:
            raise
        LOGGER.warning("Official lottery refresh failed; using local cache: %s", exc)
        return dataframe, RefreshStatus(
            False,
            0,
            local_latest,
            "공식 사이트 연결이 지연되어 마지막 저장 데이터를 안전하게 표시합니다.",
        )
