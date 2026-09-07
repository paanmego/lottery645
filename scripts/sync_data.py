"""Refresh the committed CSV from the official Lotto 6/45 source."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lottery_data import refresh_lottery_data  # noqa: E402


def main() -> None:
    data_path = PROJECT_ROOT / "lotto_data.csv"
    dataframe, status = refresh_lottery_data(data_path)
    print(status.message)
    print(
        f"저장 범위: 1회 ~ {int(dataframe['draw_number'].max())}회 "
        f"({len(dataframe):,}개 회차)"
    )


if __name__ == "__main__":
    main()
