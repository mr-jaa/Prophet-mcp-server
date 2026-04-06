#!/usr/bin/env python3
"""
Prophet MCP Server — Statistical traffic forecasting for SEO.

Runs Meta's Prophet locally to forecast clicks, impressions, or any time
series metric. Designed to work alongside GSC MCP or BigQuery MCP servers
in Claude Desktop.

Features:
    - Trend detection with confidence intervals
    - Weekly seasonality breakdown
    - Event annotations (algo updates, migrations, launches) that improve accuracy
    - Interactive Plotly charts with forecast visualisation

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
    instructions="Statistical traffic forecasting using Meta's Prophet. Runs locally, no cloud services, no API costs. Supports event annotations and interactive charts.",
)


def _build_holidays_df(events: list[dict] | None):
    """Convert event annotations into a Prophet holidays DataFrame."""
    if not events:
        return None

    try:
        import pandas as pd
    except ImportError:
        return None

    rows = []
    for event in events:
        date = event.get("date")
        label = event.get("label", "event")
        # window_before and window_after let the event affect surrounding days
        lower = event.get("window_before", 1)
        upper = event.get("window_after", 3)
        rows.append({
            "holiday": label,
            "ds": pd.to_datetime(date),
            "lower_window": -abs(int(lower)),
            "upper_window": abs(int(upper)),
        })

    return pd.DataFrame(rows) if rows else None


def _run_forecast(
    dates: list[str],
    values: list[float],
    horizon: int = 30,
    metric: str = "clicks",
    events: list[dict] | None = None,
) -> dict:
    """Core forecasting logic shared by all tools. Returns a dict."""
    try:
        import pandas as pd
        from prophet import Prophet
    except ImportError:
        return {
            "error": "Prophet is not installed. Run: pip3 install prophet",
            "fix": "Open your terminal and run: pip3 install prophet"
        }

    if len(dates) != len(values):
        return {
            "error": f"Mismatched data: {len(dates)} dates but {len(values)} values. These must be the same length."
        }

    if len(dates) < 14:
        return {
            "error": f"Need at least 14 data points for a reliable forecast. Got {len(dates)}. Try pulling more historical data."
        }

    # Cap horizon
    horizon = min(max(horizon, 7), 365)

    # Build dataframe
    df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": values})
    df = df.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)

    # Build holidays/events dataframe
    holidays_df = _build_holidays_df(events)

    # Fit Prophet
    m = Prophet(
        yearly_seasonality=len(df) >= 365,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
        holidays=holidays_df,
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

    # Event impact analysis
    event_impact = {}
    if holidays_df is not None and not holidays_df.empty:
        for _, event_row in holidays_df.iterrows():
            label = event_row["holiday"]
            event_date = event_row["ds"]
            # Find the holiday effect columns Prophet creates
            col_name = f"{label}"
            # Prophet creates columns like 'event_name' for each holiday
            matching_cols = [c for c in forecast.columns if c == label]
            if matching_cols:
                col = matching_cols[0]
                effect = forecast[col].mean()
                # Also get the effect around the event date
                nearby = forecast[
                    (forecast["ds"] >= event_date + pd.Timedelta(days=event_row["lower_window"])) &
                    (forecast["ds"] <= event_date + pd.Timedelta(days=event_row["upper_window"]))
                ]
                if len(nearby) > 0:
                    peak_effect = nearby[col].abs().max()
                    event_impact[label] = {
                        "date": event_date.strftime("%Y-%m-%d"),
                        "estimated_daily_impact": round(peak_effect, 1),
                        "direction": "positive" if nearby[col].mean() > 0 else "negative",
                    }

    # Build forecast table
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
            "events_included": len(events) if events else 0,
            "note": "This forecast was generated by Meta's Prophet statistical model running locally on your machine. No data was sent to any cloud service.",
        },
        # Store raw data for chart generation
        "_chart_data": {
            "historical": df.to_dict(orient="records"),
            "forecast": forecast.to_dict(orient="records"),
            "events": events,
            "metric": metric,
        },
    }

    if event_impact:
        result["event_impact"] = event_impact

    return result


@mcp.tool()
def forecast_traffic(
    dates: list[str],
    values: list[float],
    horizon: int = 30,
    metric: str = "clicks",
    events: list[dict] | None = None,
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
        events: Optional list of event annotations that may have affected traffic.
                Each event is a dict with:
                    - "date": YYYY-MM-DD (required)
                    - "label": short description like "core update" or "site migration" (required)
                    - "window_before": days before the event it may have had impact (default 1)
                    - "window_after": days after the event it may have had impact (default 3)
                Example: [{"date": "2026-01-15", "label": "core update", "window_after": 7}]
                Prophet uses these as special events to improve forecast accuracy.

    Returns:
        JSON string with trend analysis, daily forecast with confidence intervals,
        weekly seasonality breakdown, and event impact analysis if events were provided.
    """
    result = _run_forecast(dates, values, horizon, metric, events)
    # Remove internal chart data from the JSON output
    result.pop("_chart_data", None)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def forecast_from_csv(
    file_path: str,
    horizon: int = 30,
    metric: str = "clicks",
    events: list[dict] | None = None,
    events_csv: str | None = None,
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
        events: Optional list of event annotations (same format as forecast_traffic)
        events_csv: Optional path to a CSV file with event annotations.
                    Must have columns: date (YYYY-MM-DD), label (text).
                    Optional columns: window_before (int), window_after (int).
                    Example rows:
                        date,label,window_after
                        2026-01-15,core update,7
                        2026-02-01,site migration,14

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

    # Load events from CSV if provided
    if events_csv and os.path.exists(events_csv):
        events_df = pd.read_csv(events_csv)
        loaded_events = []
        for _, row in events_df.iterrows():
            event = {
                "date": str(row.get("date", "")),
                "label": str(row.get("label", "event")),
            }
            if "window_before" in events_df.columns:
                event["window_before"] = int(row.get("window_before", 1))
            if "window_after" in events_df.columns:
                event["window_after"] = int(row.get("window_after", 3))
            loaded_events.append(event)
        # Merge with any directly provided events
        if events:
            events = events + loaded_events
        else:
            events = loaded_events

    result = _run_forecast(dates=dates, values=values, horizon=horizon, metric=metric, events=events)
    result.pop("_chart_data", None)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def forecast_chart(
    dates: list[str],
    values: list[float],
    horizon: int = 30,
    metric: str = "clicks",
    events: list[dict] | None = None,
    output_path: str | None = None,
) -> str:
    """
    Generate an interactive Plotly chart of the Prophet forecast.

    Creates an HTML file with:
    - Historical data as a scatter plot
    - Forecast line with 95% confidence band (shaded)
    - Event annotations as vertical markers with labels
    - Hover tooltips showing exact values and dates
    - Zoomable, pannable, downloadable as PNG

    The chart opens automatically in the default browser.

    Args:
        dates: List of dates in YYYY-MM-DD format
        values: List of numeric values corresponding to each date
        horizon: Number of days to forecast (default 30, max 365)
        metric: Name of the metric, e.g. "clicks" or "impressions"
        events: Optional list of event annotations (same format as forecast_traffic)
        output_path: Optional path to save the HTML chart. Defaults to ~/Desktop/prophet_forecast.html

    Returns:
        JSON string with the chart file path and a summary of the forecast.
    """
    try:
        import plotly.graph_objects as go
        import pandas as pd
    except ImportError:
        return json.dumps({
            "error": "Plotly is not installed. Run: pip3 install plotly",
            "fix": "Open your terminal and run: pip3 install plotly"
        })

    # Run the forecast
    result = _run_forecast(dates, values, horizon, metric, events)

    if "error" in result:
        return json.dumps(result)

    chart_data = result.get("_chart_data", {})
    historical_records = chart_data.get("historical", [])
    forecast_records = chart_data.get("forecast", [])

    hist_df = pd.DataFrame(historical_records)
    hist_df["ds"] = pd.to_datetime(hist_df["ds"])

    fc_df = pd.DataFrame(forecast_records)
    fc_df["ds"] = pd.to_datetime(fc_df["ds"])

    historical_end = hist_df["ds"].max()
    future_df = fc_df[fc_df["ds"] > historical_end]

    fig = go.Figure()

    # Historical data points
    fig.add_trace(go.Scatter(
        x=hist_df["ds"],
        y=hist_df["y"],
        mode="markers",
        name=f"Historical {metric}",
        marker=dict(color="#527ED7", size=3, opacity=0.5),
        hovertemplate="%{x|%a %d %b %Y}<br>" + metric.capitalize() + ": %{y:.0f}<extra></extra>",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_df["ds"],
        y=future_df["yhat"],
        mode="lines",
        name="Forecast",
        line=dict(color="#1a1a1a", width=2),
        hovertemplate="%{x|%a %d %b %Y}<br>Predicted: %{y:.0f}<extra></extra>",
    ))

    # Confidence band (upper)
    fig.add_trace(go.Scatter(
        x=future_df["ds"],
        y=future_df["yhat_upper"],
        mode="lines",
        name="95% upper",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Confidence band (lower, fills to upper)
    fig.add_trace(go.Scatter(
        x=future_df["ds"],
        y=future_df["yhat_lower"],
        mode="lines",
        name="95% confidence",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(82, 126, 215, 0.15)",
        hovertemplate="%{x|%a %d %b %Y}<br>Range: %{y:.0f} to " +
                      future_df["yhat_upper"].apply(lambda x: f"{x:.0f}").tolist().__getitem__(0) +
                      "<extra></extra>" if len(future_df) > 0 else "",
    ))

    # Trend line across full range
    fig.add_trace(go.Scatter(
        x=fc_df["ds"],
        y=fc_df["trend"],
        mode="lines",
        name="Trend",
        line=dict(color="#999", width=1, dash="dot"),
        hovertemplate="%{x|%a %d %b %Y}<br>Trend: %{y:.0f}<extra></extra>",
    ))

    # Event annotations
    if events:
        for event in events:
            event_date = pd.to_datetime(event["date"])
            label = event.get("label", "event")

            # Vertical line
            fig.add_vline(
                x=event_date,
                line_width=1.5,
                line_dash="dash",
                line_color="#e74c3c",
                opacity=0.7,
            )

            # Label
            fig.add_annotation(
                x=event_date,
                y=1.05,
                yref="paper",
                text=label,
                showarrow=False,
                font=dict(size=10, color="#e74c3c"),
                textangle=-30,
            )

    # Divider between historical and forecast
    fig.add_vline(
        x=historical_end,
        line_width=1,
        line_dash="dot",
        line_color="#ccc",
    )
    fig.add_annotation(
        x=historical_end,
        y=0.02,
        yref="paper",
        text="forecast →",
        showarrow=False,
        font=dict(size=9, color="#999"),
        xanchor="left",
    )

    # Layout
    trend_info = result.get("trend", {})
    title_text = (
        f"{metric.capitalize()} Forecast: "
        f"{trend_info.get('current_daily_avg', 0):.0f}/day → "
        f"{trend_info.get('forecast_daily_avg', 0):.0f}/day "
        f"({trend_info.get('direction', 'flat')} {abs(trend_info.get('change_percent', 0)):.1f}%)"
    )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16, color="#1a1a1a")),
        xaxis_title="",
        yaxis_title=metric.capitalize(),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=80, b=60, l=60, r=30),
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
    )

    # Save the chart
    if not output_path:
        output_path = os.path.expanduser("~/Desktop/prophet_forecast.html")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.write_html(output_path, include_plotlyjs=True, full_html=True)

    # Also try to open in browser
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(output_path)}")
    except Exception:
        pass

    # Return summary with file path
    result.pop("_chart_data", None)
    summary = {
        "chart_saved": os.path.abspath(output_path),
        "opened_in_browser": True,
        "trend": result.get("trend", {}),
        "weekly_seasonality": result.get("weekly_seasonality", {}),
        "events_plotted": len(events) if events else 0,
        "forecast_days": horizon,
        "model_info": result.get("model_info", {}),
    }

    if "event_impact" in result:
        summary["event_impact"] = result["event_impact"]

    return json.dumps(summary, indent=2, default=str)


if __name__ == "__main__":
    mcp.run()
