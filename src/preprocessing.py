import pandas as pd
from statsmodels.tsa.seasonal import MSTL
import numpy as np
import holidays


def model_features(df):
    FEATURES_SARIMAX = ["is_weekend", "is_holiday"]
    FOURIER_COLS = [c for c in df.columns if c.startswith(("sin_", "cos_"))]
    FEATURES_DHR = FOURIER_COLS + ["is_weekend", "is_holiday"]
    FEATURES_ML = [
        "energy",
        "lag_24",
        "lag_168",
        "roll_mean_24",
        "roll_mean_168",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_holiday",
    ]
    return {"sarimax": FEATURES_SARIMAX, "dhr": FEATURES_DHR, "ml": FEATURES_ML}


def split_data(df, val_start="2016-01-01", test_start="2017-01-01"):
    df = df.dropna()
    train = df[df.index < val_start]
    val = df[(df.index >= val_start) & (df.index < test_start)]
    test = df[df.index >= test_start]
    return train, val, test


def calculate_rolling_mean_std(df, periods):
    for period in periods:
        df[f"roll_mean_{period}"] = df["energy"].rolling(period).mean()
        df[f"roll_std_{period}"] = df["energy"].rolling(period).std()
    return df


def calculate_lags(df, lags):
    for lag in lags:
        df[f"lag_{lag}"] = df["energy"].shift(lag)
    return df


def calculate_fourier_terms(df, periods, ks):
    t = np.arange(len(df))
    for p, k in zip(periods, ks):
        df[f"sin_{p}_k{k}"] = np.sin(2 * np.pi * k * t / p)
        df[f"cos_{p}_k{k}"] = np.cos(2 * np.pi * k * t / p)
    return df


def encode_cyclical(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def calculate_holiday_weekend(df):
    holidays_by_years = pd.to_datetime(
        list(holidays.UnitedStates(years=df.index.year.unique()).keys())
    )
    df["is_holiday"] = df.index.normalize().isin(holidays_by_years).astype("int8")
    df["is_weekend"] = (df["dayofweek"] > 4).astype("int8")
    return df


def calculate_hour_dayofweek_month(df):
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    return df


def format_time_series(df):
    df["Datetime"] = pd.to_datetime(
        df["Datetime"], errors="raise", format="%Y-%m-%d %H:%M:%S"
    )
    df.rename(columns={"PJME_MW": "energy"}, inplace=True)
    return df


def reindex_time_series(df):
    start_date = df["Datetime"].min()
    end_date = df["Datetime"].max()
    n_expected_hours = int((end_date - start_date).total_seconds() // 3600) + 1
    full_date_range = pd.date_range(start=start_date, end=end_date, freq="h")

    print("== Date Range ==")
    print(f"Start: {start_date}")
    print(f"End:   {end_date}")
    print(f"Expected hourly observations: {n_expected_hours:,}")

    df.set_index("Datetime", inplace=True)

    n_duplicates = df.index.duplicated().sum()
    print(f"== Duplicate Timestamps: {n_duplicates} ==")
    if n_duplicates:
        print(df[df.index.duplicated()])

    df = df[~df.index.duplicated()]
    df = df.reindex(full_date_range)

    print(f"Rows after reindexing: {len(df):,}")
    return df


def impute_time_series(df):
    df["energy"] = df["energy"].interpolate(method="linear")
    return df


def fit_mstl(series, periods):
    model = MSTL(series, periods=periods)
    return model.fit()
