#!/usr/bin/env python3
"""
Prophet MCP Server — Statistical traffic forecasting for SEO.

Runs Meta's Prophet locally to forecast clicks, impressions, or any time
series metric. Designed to work alongside GSC MCP or BigQuery MCP servers
in Claude Desktop.

Usage:
    python3 server.py

Claude Desktop config:
    {
        "mcpServers": {
            "prophet-forecast": {
                "command": "python3",
                "args": ["/absolute/path/to/server.py"]
            }
        }
    }
"""

import json
import os
import sys
import warnings
import logging

# Suppress noisy libraries
warnings.filterwarnings("ignore")
os.environ["CMDSTAN_VERBOSE"] = "false"
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "Prophet Forecast",
    instructions="Statistical traffic forecasting using Meta's Prophet. Runs locally, no cloud services, no API costs.",
)


@mcp.tool()
def forecast_traffic(
    dates: list[str],
    values: list[float],
    horizon: int = 30,
    metric: str = "clicks",
) -> str:
    """
    Forecast future traffic using Meta's Prophet statistical model.

    Takes historical date/value pairs (e.g. daily clicks from GSC) and
    returns a forecast with trend analysis, confidence intervals, and
    weekly seasonality patterns.

    This is NOT an LLM doing the math. Prophet is a proper statistical
    forecasting library that produces reliable, reproducible results
    with confidence intervals.

    Args:
        dates: List of dates in YYYY-MM-DD format, e.g. ["2026-01-01", "2026-01-02", ...]
        values: List of numeric values corresponding to each date, e.g. [150, 163, ...]
        horizon: Number of days to forecast into the future (default 30, max 365)
        metric: Name of the metric being forecast, e.g. "clicks" or "impressions" (for labelling only)

    Returns:
        JSON string with trend analysis, daily forecast with confidence intervals,
        and weekly seasonality breakdown.
    """
    try:
        import pandas as pd
        from prophet import Prophet
    except ImportError:
        return json.dumps({
            "error": "Prophet is not installed. Run: pip3 install prophet",
            "fix": "Open your terminal and run: pip3 install prophet"
        })

    if len(dates) != len(values):
        return json.dumps({
            "error": f"Mismatched data: {len(dates)} dates but {len(values)} values. These must be the same length."
        })

    if len(dates) < 14:
        return json.dumps({
            "error": f"Need at least 14 data points for a reliable forecast. Got {len(dates)}. Try pulling more historical data."
        })

    # Cap horizon
    horizon = min(max(horizon, 7), 365)

    # Build dataframe
    df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": values})
    df = df.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)

    # Fit Prophet
    m = Prophet(
        yearly_seasonality=len(df) >= 365,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
    )
    m.fit(df)

    # Generate forecast
    future = m.make_future_dataframe(periods=horizon)
    forecast = m.predict(future)

    # Split historical vs future
    historical_end = df["ds"].max()
    future_only = forecast[forecast["ds"] > historical_end].copy()

    # Trend analysis
    first_trend = forecast["trend"].iloc[0]
    last_trend = forecast["trend"].iloc[-1]
    trend_change = last_trend - first_trend
    trend_pct = (trend_change / first_trend) * 100 if first_trend != 0 else 0
    trend_direction = "up" if trend_change > 0 else ("down" if trend_change < 0 else "flat")

    # Current vs forecast
    recent_7d = df.tail(7)["y"].mean()
    forecast_last_7d = (
        future_only.tail(7)["yhat"].mean()
        if len(future_only) >= 7
        else future_only["yhat"].mean()
    )

    # Weekly seasonality
    weekday_effects = {}
    days_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                4: "Friday", 5: "Saturday", 6: "Sunday"}
    if "weekly" in forecast.columns:
        for dow, name in days_map.items():
            day_data = forecast[forecast["ds"].dt.dayofweek == dow]
            if len(day_data) > 0:
                weekday_effects[name] = round(day_data["weekly"].mean(), 2)

    # Sort by effect to find best/worst days
    best_day = max(weekday_effects, key=weekday_effects.get) if weekday_effects else "unknown"
    worst_day = min(weekday_effects, key=weekday_effects.get) if weekday_effects else "unknown"

    # Build forecast table (weekly summary for readability)
    daily_forecast = [
        {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "day": row["ds"].strftime("%A"),
            "predicted": round(row["yhat"], 0),
            "lower_bound": round(row["yhat_lower"], 0),
            "upper_bound": round(row["yhat_upper"], 0),
        }
        for _, row in future_only.iterrows()
    ]

    result = {
        "metric": metric,
        "historical_days": len(df),
        "forecast_days": horizon,
        "data_range": {
            "start": df["ds"].min().strftime("%Y-%m-%d"),
            "end": df["ds"].max().strftime("%Y-%m-%d"),
        },
        "trend": {
            "direction": trend_direction,
            "change_percent": round(trend_pct, 1),
            "current_daily_avg": round(recent_7d, 1),
            "forecast_daily_avg": round(forecast_last_7d, 1),
            "summary": (
                f"Traffic is trending {trend_direction} by {abs(round(trend_pct, 1))}% over the historical period. "
                f"Current 7-day average is {round(recent_7d, 0)} {metric}/day. "
                f"Prophet predicts {round(forecast_last_7d, 0)} {metric}/day by the end of the forecast period."
            ),
        },
        "weekly_seasonality": {
            "effects": weekday_effects,
            "best_day": best_day,
            "worst_day": worst_day,
            "summary": (
                f"Best performing day is {best_day}. Weakest day is {worst_day}."
                if weekday_effects
                else "Not enough data to determine weekly patterns."
            ),
        },
        "forecast": daily_forecast,
        "model_info": {
            "model": "Prophet (Meta)",
            "yearly_seasonality": len(df) >= 365,
            "weekly_seasonality": True,
            "confidence_interval": "95%",
            "note": "This forecast was generated by Meta's Prophet statistical model running locally on your machine. No data was sent to any cloud service.",
        },
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def forecast_from_csv(
    file_path: str,
    horizon: int = 30,
    metric: str = "clicks",
) -> str:
    """
    Forecast traffic from a CSV file using Meta's Prophet.

    The CSV must have at least two columns. The first date-like column will
    be used as the date, and the first numeric column will be used as the
    value. Or you can use columns named 'ds' (date) and 'y' (value).

    Args:
        file_path: Absolute path to a CSV file with date and value columns
        horizon: Number of days to forecast (default 30, max 365)
        metric: Name of the metric, e.g. "clicks" or "impressions" (for labelling)

    Returns:
        JSON string with trend analysis, forecast, and weekly seasonality.
    """
    try:
        import pandas as pd
    except ImportError:
        return json.dumps({"error": "pandas not installed. Run: pip3 install prophet"})

    if not os.path.exists(file_path):
        return json.dumps({"error": f"File not found: {file_path}"})

    df = pd.read_csv(file_path)

    # Try to find date and value columns
    if "ds" in df.columns and "y" in df.columns:
        dates = df["ds"].tolist()
        values = df["y"].tolist()
    else:
        # Auto-detect: first date-like column and first numeric column
        date_col = None
        value_col = None
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                if date_col is None:
                    date_col = col
            except (ValueError, TypeError):
                pass
            if pd.api.types.is_numeric_dtype(df[col]) and value_col is None:
                value_col = col

        if date_col is None or value_col is None:
            return json.dumps({
                "error": f"Could not auto-detect date and value columns. Found columns: {list(df.columns)}. Either name them 'ds' and 'y', or ensure there's one date column and one numeric column."
            })

        dates = df[date_col].tolist()
        values = df[value_col].tolist()

    return forecast_traffic(dates=dates, values=values, horizon=horizon, metric=metric)


if __name__ == "__main__":
    mcp.run()
