# Pairs Convergence/Divergence Trading Strategy

A quantitative trading strategy implementation using the Alpaca API for live trading and backtesting, designed to exploit **statistical convergence and divergence between correlated stock pairs**.

## Overview

This project implements a **pairs trading strategy** that:

- Identifies correlated stock pairs using historical data
- Monitors price spreads for **convergence/divergence signals**
- Executes long/short positions on both legs of the pair
- Dynamically sizes positions based on statistical confidence
- Supports **both live trading and backtesting**

Unlike a mean-reversion strategy on individual tickers, this approach trades **relative value** between two correlated stocks rather than absolute price levels.

## Features

- **Real-time market data processing** using yfinance
- **Automated trade execution** through Alpaca API
- **Z-score based entry and exit signals**
- **Rolling beta calculation** for dynamic hedge ratios
- **Risk management** via max capital per pair and total portfolio caps
- **Support for multiple pairs simultaneously**
- **Backtesting module** to evaluate performance before going live

## Automation

The automation is configured in `.github/workflows/main.yml`.

- Add your Alpaca API keys as **GitHub Action secrets**
- The workflow will execute trades automatically on your linked Alpaca account

## Installation

1. Clone the repository:

```bash
git clone https://github.com/alexmueller07/pairsConvergenceStrategy.git
cd pairsConvergenceStrategy
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your Alpaca API credentials in `.env` (do not hardcode keys):

```bash
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

## Usage

### Live Trading

Run the main trading script:

```bash
python main.py
```

## Configuration

You can adjust the strategy behavior by modifying:

- `tickers` list in `main.py` (stock universe)
- `ROLLING_WINDOW`, `ENTRY_Z`, `EXIT_Z` in `config.py` (signal tuning)
- `MAX_OPEN_PAIRS`, `PER_PAIR_MAX_DOLLAR` for risk management
- `EOD_FLATTEN_HHMM_EST` for when to stop trading each day

## Project Structure

- `main.py` – Main trading script
- `config.py` – Strategy parameters and API settings
- `requirements.txt` – Project dependencies
- `strategy.py` - Strategy Logic
- `get_tickers.py` - Scrapes tickers to trade
- `get_related.py` - Determines the pairs to trade based on divergence

## Dependencies

- yfinance
- pandas
- numpy
- alpaca-trade-api
- python-dotenv

## License

MIT License

## Disclaimer

This software is for educational purposes only. Do not risk money you cannot afford to lose. USE THE SOFTWARE AT YOUR OWN RISK. The authors and all affiliates assume no responsibility for your trading results.

## Author

**Alexander Mueller**

- GitHub: [alexmueller07](https://github.com/alexmueller07)
- LinkedIn: [Alexander Mueller](https://www.linkedin.com/in/alexander-mueller-021658307/)
- Email: amueller.code@gmail.com
