# OpenBB Single Asset Lite + Quant

This is a lighter custom backend for OpenBB Workspace focused on single-stock analysis with stable price-based quantitative widgets.

## Includes
- Summary and company snapshot
- Price, growth, drawdown
- Rolling volatility and beta
- RSI, MACD, support/resistance
- Return stats, monthly heatmap, risk scorecard
- Relative strength
- Rolling alpha
- Rolling Sharpe and Sortino
- Momentum scorecard
- Peer correlation table

## Run on Mac
```bash
python3 -m pip install -r requirements.txt
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 7779
```

Then add `http://127.0.0.1:7779` in OpenBB Workspace and open **Single Asset Lite + Quant**.
Leave **Validate Widget** off.

This clean-fix build removes duplicate widget instances and uses a table-style monthly heatmap for reliability.
