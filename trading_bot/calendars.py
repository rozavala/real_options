"""Exchange-specific trading calendars. Commodity-agnostic."""

from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday,
    USMemorialDay, USLaborDay, USThanksgivingDay, USFederalHolidayCalendar
)
from pandas.tseries.offsets import DateOffset
from dateutil.easter import easter as Easter
from datetime import date, timedelta
import pandas as pd


class ICEHolidayCalendar(AbstractHolidayCalendar):
    """ICE US trading calendar (includes Good Friday)."""
    rules = [
        Holiday('New Year', month=1, day=1, observance=nearest_workday),
        USMemorialDay,
        Holiday('Independence Day', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday),
        # ICE-specific: Good Friday (2 days before Easter Sunday)
        # Note: pandas GoodFriday holiday object handles this
        Holiday("Good Friday", month=1, day=1, offset=[Easter(), DateOffset(days=-2)]),
    ]


def get_exchange_calendar(exchange: str) -> AbstractHolidayCalendar:
    """Get calendar for exchange. Commodity-agnostic."""
    calendars = {
        'ICE': ICEHolidayCalendar,
        'NYBOT': ICEHolidayCalendar,
        'CME': USFederalHolidayCalendar,  # Approximation
    }
    cal_class = calendars.get(exchange, USFederalHolidayCalendar)
    return cal_class()


def is_trading_day(dt: date, exchange: str = 'ICE') -> bool:
    """Check if a date is a trading day."""
    if dt.weekday() >= 5:  # Weekend
        return False
    cal = get_exchange_calendar(exchange)
    holidays = cal.holidays(start=dt, end=dt)
    return len(holidays) == 0
