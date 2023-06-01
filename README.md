# 📈 Stock-GPT

This script utilizes the GPT-4 model from OpenAI to analyze stock market data from CNBC and make buy, sell, or hold decisions based on the current portfolio. The analysis is executed using OpenAI's API and the Alpaca Trading API for handling stock transactions.

## 🚀 Features

- 📰 Scrape latest financial news articles from CNBC
- 📊 Extract relevant stock data from the articles
- 🧠 Analyze stocks using OpenAI's GPT model
- 💡 Make buy, sell, or hold decisions
- 💸 Execute the decisions using Alpaca Trading API
- 📝 Analyze the current portfolio

## 📋 Requirements

- Python 3
- `requests` library
- `beautifulsoup4` library
- `pandas` library
- `ycnbc` library
- `openai` library
- `alpaca-trading-api` library
- `re` library
- `pytz` library
- `numpy` library

- API keys:
  - OpenAI API key
  - Alpaca Trading API key
  - Alpaca Trading Secret key

## 🎬 Usage

1. Set your OpenAI API key, Alpaca API key, and Alpaca Secret key in the environment variable or provide them directly in the script.
2. Create a `StockAnalyzer` instance with your API keys:

```python
stock_analyzer = StockAnalyzer(OPENAI_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY)
```

3. Run the analysis:

```python
stock_analyzer.run()
```

## 🛠 Methods

### get_account_info()

Retrieve user's current trading positions and buying power.

### scrape_articles()

Scrape the latest financial news articles from CNBC and extract relevant stock data.

### extract_stock_data(stock_links)

Extract stock data from given stock links and concatenate it into a DataFrame.

### analyze_stocks(queries, article_keys)

Use OpenAI's GPT model to analyze the stock data and make buy, sell, or hold decisions.

### process_recommendations(responses, article_keys=None)

Process the GPT model's buy/sell/hold recommendations and return them as a DataFrame.

### execute_decisions(mast_df, article_keys=None)

Execute buy/sell decisions based on the GPT model's recommendations using Alpaca Trading API by placing limit orders.

### analyze_current_portfolio()

Analyze the current portfolio and update the daily transactions.

### run()

Run the complete stock analysis process.