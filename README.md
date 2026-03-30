# Prophet MCP Server

Statistical traffic forecasting for SEO using Meta's Prophet. Connects to Claude Desktop as an MCP server.

No LLM guesswork. Prophet is a proper statistical model that produces reliable forecasts with confidence intervals.

## What it does

Two tools available in Claude Desktop once connected.

**forecast_traffic** — Pass in daily date/value pairs (e.g. clicks from GSC) and get back a full forecast with trend analysis, confidence intervals, and weekly seasonality patterns.

**forecast_from_csv** — Point it at a CSV file and it auto-detects the date and value columns. Works with Google Search Console CSV exports out of the box.

## Setup

### 1. Install dependencies

```bash
pip3 install prophet mcp pandas
```

### 2. Add to Claude Desktop

Open Claude Desktop settings, go to Developer > MCP Servers, and add this config.

**macOS**
```json
{
  "mcpServers": {
    "prophet-forecast": {
      "command": "python3",
      "args": ["/absolute/path/to/prophet-mcp-server/server.py"]
    }
  }
}
```

**Windows**
```json
{
  "mcpServers": {
    "prophet-forecast": {
      "command": "python",
      "args": ["C:\\absolute\\path\\to\\prophet-mcp-server\\server.py"]
    }
  }
}
```

Replace the path with wherever you saved this repo.

### 3. Restart Claude Desktop

The Prophet Forecast tools will appear in your tool list.

## Usage

Just ask Claude naturally.

- "Forecast my traffic for the next 30 days" (works when you also have GSC MCP or BigQuery MCP connected)
- "Read my GSC export at ~/Downloads/search-console.csv and forecast the next 60 days"
- "Forecast impressions for the next 90 days"

## Requirements

- Python 3.8+
- No GPU needed
- Works on Mac, Windows, and Linux
- Uses ~200MB RAM for typical SEO datasets

## How it works

Prophet decomposes your time series data into trend, weekly seasonality, and yearly seasonality (if you have 365+ days of data). It then projects these components forward to produce a forecast with 95% confidence intervals.

The forecast runs entirely on your machine. No data is sent anywhere.

## Blog post

Full walkthrough with screenshots: [How to Forecast SEO Traffic Using Prophet and Claude Code](https://suganthan.com/blog/forecast-seo-traffic-prophet-claude-code/)

## Licence

Apache 2.0
