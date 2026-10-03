"""
Feature engineering for flight-delay prediction.

Target: did the flight depart 15+ minutes late?
Every feature must be knowable shortly before scheduled departure, so
nothing from the flight's own departure or arrival is used.

Leakage rules applied here:
  * The aircraft's previous leg contributes only its *departure* delay,
    which is known hours before the next departure. Its arrival delay
    is not used.
  * Historical route/carrier delay rates are computed from strictly
    earlier dates (expanding mean, shifted), never from the same day.
  * Weather at the scheduled hour is used as a stand-in for a short-range
    forecast (see README limitations).
"""
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

US_HOLIDAYS_2013 = pd.to_datetime([
    "2013-01-01", "2013-01-21", "2013-02-18", "2013-05-27", "2013-07-04",
    "2013-09-02", "2013-10-14", "2013-11-11", "2013-11-28", "2013-12-25",
])

CATEGORICAL = ["carrier", "origin", "dest", "manufacturer"]
NUMERIC = [
    # "month" is deliberately excluded: test months (Nov-Dec) are outside the
    # training range, so a tree model would extrapolate from August.
    "day_of_week", "sched_dep_hour", "distance",
    "days_to_holiday", "origin_hour_departures", "dest_daily_flights",
    "leg_of_day", "prev_leg_dep_delay", "has_prev_leg", "mins_since_prev_leg",
    "plane_age", "seats",
    "temp", "humid", "wind_speed", "precip", "visib", "pressure",
    "route_hist_delay_rate", "carrier_hist_delay_rate", "origin_prev_day_delay_rate",
]
TARGET = "delayed_15"


def load_raw() -> dict[str, pd.DataFrame]:
    with zipfile.ZipFile(DATA / "flights.csv.zip") as z:
        name = next(n for n in z.namelist() if n.endswith("flights.csv"))
        with z.open(name) as f:
            flights = pd.read_csv(f, na_values="NA")
    read = lambda n: pd.read_csv(DATA / f"{n}.csv", na_values="NA")
    return {"flights": flights, "planes": read("planes"), "weather": read("weather")}


def _hist_rate(df: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    """Delay rate for each key using only days strictly before the flight date."""
    daily = (df.groupby(keys + ["date"])[TARGET]
               .agg(["sum", "count"]).reset_index()
               .sort_values(keys + ["date"]))
    g = daily.groupby(keys)
    daily["cum_sum"] = g["sum"].cumsum() - daily["sum"]
    daily["cum_cnt"] = g["count"].cumsum() - daily["count"]
    # Smooth toward the global prior so sparse routes don't get extreme values
    prior, k = df[TARGET].mean(), 20
    daily[name] = (daily["cum_sum"] + prior * k) / (daily["cum_cnt"] + k)
    return df.merge(daily[keys + ["date", name]], on=keys + ["date"], how="left")


def build_features() -> pd.DataFrame:
    raw = load_raw()
    f = raw["flights"]

    f["date"] = pd.to_datetime(dict(year=f.year, month=f.month, day=f.day))

    # ---- schedule congestion: counted on the full published schedule,
    #      before removing cancellations, so it carries no outcome information ----
    f["origin_hour_departures"] = f.groupby(["origin", "date", "hour"])["flight"].transform("size")
    f["dest_daily_flights"] = f.groupby(["dest", "date"])["flight"].transform("size")

    # Cancelled flights have no departure; this model predicts delay given operation.
    f = f[f["dep_time"].notna()].copy()
    f[TARGET] = (f["dep_delay"] >= 15).astype(int)

    # ---- calendar ----
    f["day_of_week"] = f["date"].dt.dayofweek
    f["sched_dep_hour"] = f["hour"]
    gaps = np.abs(f["date"].values[:, None] - US_HOLIDAYS_2013.values[None, :])
    f["days_to_holiday"] = (gaps.min(axis=1) / np.timedelta64(1, "D")).astype(int)

    # ---- aircraft rotation: previous NYC departure by the same plane that day ----
    f["sched_minutes"] = (f["sched_dep_time"] // 100) * 60 + f["sched_dep_time"] % 100
    f = f.sort_values(["tailnum", "date", "sched_minutes"])
    grp = f.groupby(["tailnum", "date"], dropna=True)
    f["leg_of_day"] = grp.cumcount() + 1
    f["prev_leg_dep_delay"] = grp["dep_delay"].shift(1)
    f["mins_since_prev_leg"] = f["sched_minutes"] - grp["sched_minutes"].shift(1)
    f.loc[f["tailnum"].isna(), ["leg_of_day", "prev_leg_dep_delay", "mins_since_prev_leg"]] = np.nan
    f["has_prev_leg"] = f["prev_leg_dep_delay"].notna().astype(int)

    # ---- aircraft attributes ----
    planes = raw["planes"].rename(columns={"year": "year_built"})
    f = f.merge(planes[["tailnum", "year_built", "manufacturer", "seats"]], on="tailnum", how="left")
    f["plane_age"] = 2013 - f["year_built"]

    # ---- weather at origin, scheduled hour ----
    w = raw["weather"].drop_duplicates(["origin", "time_hour"])
    f = f.merge(w[["origin", "time_hour", "temp", "humid", "wind_speed", "precip", "visib", "pressure"]],
                on=["origin", "time_hour"], how="left")

    # ---- leakage-safe historical rates ----
    f = _hist_rate(f, ["origin", "dest"], "route_hist_delay_rate")
    f = _hist_rate(f, ["carrier"], "carrier_hist_delay_rate")
    prev_day = (f.groupby(["origin", "date"])[TARGET].mean()
                  .groupby(level=0).shift(1).rename("origin_prev_day_delay_rate").reset_index())
    f = f.merge(prev_day, on=["origin", "date"], how="left")

    for c in CATEGORICAL:
        f[c] = f[c].fillna("UNKNOWN").astype("category")
    return f.sort_values(["date", "sched_minutes"]).reset_index(drop=True)


def time_split(df: pd.DataFrame):
    """Train Jan-Aug, validate Sep-Oct, test Nov-Dec (no shuffling across time)."""
    train = df[df["month"] <= 8]
    valid = df[df["month"].between(9, 10)]
    test = df[df["month"] >= 11]
    return train, valid, test


if __name__ == "__main__":
    d = build_features()
    print(d.shape, d[TARGET].mean().round(3))
    print(d[NUMERIC].isna().mean().round(3).sort_values(ascending=False).head(8))
