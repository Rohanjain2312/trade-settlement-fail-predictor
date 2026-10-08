from __future__ import annotations

import pytest

from src.config import load_config
from src.data.reference_data import build_calendar, home_business_days, home_holidays


@pytest.mark.parametrize("mode,months", [("full", 24), ("smoke", 6)])
def test_n_days_covers_whole_months(mode, months):
    cfg = load_config(mode)
    cal = build_calendar(cfg)
    assert cal.month_index.max() == months
    last = cal.bdays[-1]
    following = home_business_days(last, 2)[1]
    assert following.month != last.month, "the last generated month would be partial"
    assert cfg["split"]["test_months"][1] == months


def test_settlement_is_on_or_after_trade_date_and_flags_holidays():
    cal = build_calendar(load_config("smoke"))
    assert (cal.settle_ord >= cal.ords[None, None, :]).all()
    assert (cal.settle_ord[:, 2, :] > cal.settle_ord[:, 1, :]).all()
    assert cal.holiday_in_window.any() and cal.settle_is_period_end.any() and cal.vol_spike.any()


def test_home_holidays_are_weekdays():
    for year in (2024, 2025):
        assert all(d.weekday() < 5 for d in home_holidays(year))
