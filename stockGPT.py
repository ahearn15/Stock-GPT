import requests
from bs4 import BeautifulSoup
import pandas as pd
import ycnbc
import openai
import json
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import re
import pytz
from datetime import datetime, time
import numpy as np

import config

class StockAnalyzer:
    def __init__(self, openai_api_key, alpaca_api_key, alpaca_secret_key):
        self.openai_api_key = openai_api_key
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_secret_key = alpaca_secret_key

        # Set OpenAI API key
        openai.api_key = self.openai_api_key

        # Set up Alpaca client instance
        self.client = TradingClient(self.alpaca_api_key, self.alpaca_secret_key, paper=True)
        self.buying_power, self.current_holdings, self.current_portfolio, self.all_positions = self.get_account_info()
        self.current_date = datetime.now().strftime("%Y/%m/%d")
        self.failed_articles = []
        try:
            scraped_df = pd.read_csv('scraped.csv')
            scraped_df = scraped_df[scraped_df['0'].str.contains(self.current_date)]
            self.scraped = list(scraped_df['0'])

        except Exception as e:
            self.scraped = []

    def get_account_info(self):
        """
        Retrieve user's current trading positions and buying power.
        """
        account = dict(self.client.get_account())

        if not self.client.get_all_positions():
            current_portfolio = {"STOCK": "", "MARKET_VALUE": "", "SHARES": ""}
            current_holdings = []
        else:
            current_portfolio = [
                {
                    "STOCK": a.symbol,
                    "MARKET_VALUE": round(float(a.market_value), 2),
                    "SHARES": round(float(a.qty), 2),
                }
                for a in self.client.get_all_positions()
            ]
            current_holdings = [i["STOCK"] for i in current_portfolio]

        buying_power = round(float(account["cash"]), 2)
        all_positions = self.client.get_all_positions()
        return buying_power, current_holdings, current_portfolio, all_positions

    def scrape_articles(self):
        """
        Scrape the latest financial news articles from CNBC and extract relevant stock data.
        """
        data = ycnbc.News()
        latest_ = data.latest()

        queries = []
        article_keys = []
        latest_nonscraped = latest_[~latest_["Link"].isin(self.scraped)]
        for article_link in list(latest_nonscraped["Link"]):
            r = requests.get(article_link)
            article = r.text
            soup = BeautifulSoup(article, "html.parser")
            stock_links = []
            try:
                art = soup.find("div", {"class": "ArticleBody-articleBody"})
                art = art.find("div", {"class": "group"})
                if art is None:  ### pro article
                    print(article_link + ' (PRO)')
                    article_text = soup.find("span", {"class": "xyz-data"}).text
                    script = str(soup.find_all('script', {'charset': 'UTF-8'})[2])
                    js = script[script.find('tickerSymbols') - 1:]
                    js = js[js.find('[{'):js.find('}]') + 2]
                    js = js.replace('\\', '')
                    js = json.loads(js)
                    relevant_stocks = [i['symbol'] for i in js]
                    for stock in relevant_stocks:
                        stock_links.append('https://www.cnbc.com/quotes/' + stock + '/')
                else:
                    print(article_link)
                    article_text = art.text
                    relevant_stocks = soup.find('ul', {'class': 'RelatedQuotes-list'})
                    if relevant_stocks is not None:
                        relevant_stocks = relevant_stocks.find_all('a')
                        for a in relevant_stocks:
                            stock_links.append('https://www.cnbc.com' + a['href'] + '/')
                    for sp in soup.find_all('span', {'data-test': 'QuoteInBody'}):
                        quo = sp.find('a', href=True)['href']
                        stock_links.append("https://www.cnbc.com" + quo)

                stock_links = [*set(stock_links)]
                if not stock_links == []:
                    mast_data = pd.DataFrame()
                    mast_data = self.extract_stock_data(stock_links)
                    query = mast_data.to_csv()
                    query += '\n\n' + article_text
                    queries.append(query)
                    article_keys.append(article_link)
                else:
                    self.scraped.append(article_link) # no stocks to scrape

            except Exception as e:
                print(f"Error occurred in {article_link}: {e}")
                if ((str(e) == "'NoneType' object has no attribute 'find'") |
                    (str(e) == "'symbol'")):
                    self.scraped.append(article_link) # Unscrapable
                else:
                    self.failed_articles.append(article_link)

        return queries, article_keys

    def extract_stock_data(self, stock_links):
        """
        Extract stock data from given stock links and concatenate to a DataFrame.
        """
        mast_data = pd.DataFrame()
        for link in stock_links:
            r_stock = requests.get(link)
            stock_soup = BeautifulSoup(r_stock.text, "html.parser")
            curr_price = stock_soup.find("span", {"class": "QuoteStrip-lastPrice"}).text
            txt = stock_soup.find("div", {"class": "QuoteStrip-lastPriceStripContainer"}).text
            daily_change = re.findall(r"\((.*?)\)", txt)[0]

            stats_ml, vals_ml = ["Current Price", "Daily Change"], [curr_price, daily_change]

            for subsection in stock_soup.find_all("div", {"class": "Summary-subsection"}):
                stats = subsection.find_all("span", {"class": "Summary-label"})
                vals = subsection.find_all("span", {"class": "Summary-value"})
                for i in range(0, len(stats)):
                    stats_ml.append(stats[i].text)
                    vals_ml.append(vals[i].text)

            stock_data = pd.DataFrame(vals_ml).T
            stock_data.columns = stats_ml
            stock_data["stock"] = link.replace("https://www.cnbc.com/quotes/", "").replace('/', '')
            stock_data = stock_data.set_index("stock")
            mast_data = pd.concat([mast_data, stock_data])

        return mast_data

    def analyze_stocks(self, queries, article_keys):
        """
        Use OpenAI's GPT model to analyze the stock data and make buy, sell, or hold decisions.
        """
        responses, article_keys2 = [], []
        i = 0
        for query in queries:
            print(f'Analyzing {article_keys[i]}')
            prompt = (
                    f"I have ${self.buying_power} in buying power. For each stock in the json data below, tell me if "
                    f"I should buy, sell, or hold."
                    "If BUY, list how many shares you would buy (AMOUNT) considering my buying power."
                    "If SELL, list how many shares (AMOUNT) you would sell considering my buying power. "
                    "Respond in json format with zero whitespace, including the following keys: "
                    "STOCK, ACTION, AMOUNT. Use the stock symbol."
                    "Here is the article and accompanying data from which you should base your decision: \n" + query
            )
            ## TODO: add boolean response for reasoning
            ## TODO: associate website link with response

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are both a qualitative and quantitative stock market expert who's only "
                                       "goal in life is to beat the market and make money, channeling Warren Buffet's "
                                       "investment strategy of finding undervalued stocks. You are to provide stock "
                                       "market recommendations based on the data and context provided",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=2000,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                resp = response["choices"][0]["message"]["content"]

                ### ToDo: parse json and print suggestions
                print(resp)
                responses.append(resp)
                article_keys2.append(article_keys[i])
            except Exception as e:
                print(f'Query failed: {e}')
                self.failed_articles.append(article_keys)
            i += 1

        print("Done")
        return responses, article_keys2

    def process_recommendations(self, responses, article_keys=None):
        """
        Process the GPT model's buy/sell/hold recommendations and return as a DataFrame.
        """
        if article_keys is None:
            article_keys = [0]
        mast_df = pd.DataFrame()
        i = 0
        for resp in responses:
            resp = str(resp)
            if not resp.startswith("["):
                resp = "[" + resp
            if not resp.endswith("]"):
                resp = resp + "]"

            ## find first '{'
            first_brack = resp.find('{')
            last_brack = resp.rfind('}') + 1

            resp = resp[first_brack:last_brack]
            resp = '[' + resp + ']'
            resp = resp.replace('\n', ',')

            try:
                stock_df = pd.DataFrame(json.loads(resp))
                stock_df['ARTICLE_SOURCE'] = article_keys[i]
                mast_df = pd.concat([mast_df, stock_df])

            except Exception as e:
                print(f"Unable to parse JSON: {resp}")
                self.failed_articles.append(article_keys[i])
            i += 1

        mast_df['ACTION'] = mast_df['ACTION'].str.upper()
        mast_df.to_csv('mast_df.csv', index=False)

        mast_df = mast_df[mast_df['ACTION'] != 'HOLD']
        mast_df["AMOUNT"] = pd.to_numeric(mast_df["AMOUNT"], errors="coerce") / self.buying_power * 10

        # mast_df = mast_df.drop_duplicates(subset=["STOCK"], keep=False)
        mast_df = mast_df[~mast_df["STOCK"].str.contains("=")]
        mast_df["STOCK"] = mast_df["STOCK"].str.strip()
        return mast_df, article_keys

    def execute_decisions(self, mast_df, article_keys=None):
        """
        Execute buy/sell decisions based on the GPT model's recommendations using Alpaca Trading API by placing limit
        orders.
        """
        if article_keys is None:
            article_keys = []

        def is_extended_hours():
            # Set timezone to Eastern Time (ET)
            timezone_et = pytz.timezone("US/Eastern")
            now = datetime.now(timezone_et)

            # Stock market opening and closing times
            market_open = time(9, 30)
            market_close = time(16, 0)  # 4:00 PM

            # Check if the current time is between opening and closing times and if it's a weekday
            if market_open <= now.time() <= market_close and 0 <= now.weekday() <= 4:
                return False
            else:
                return True

        mast_df.to_csv('mast_df.csv', index = False)
        ext_hours = is_extended_hours()
        if 'AMOUNT' not in mast_df.columns:
            mast_df['AMOUNT'] = mast_df['Qty']
        for index, row in mast_df.iterrows():
            break_loop = False
            if row["ACTION"] == "BUY":
                side_ = OrderSide.BUY
                print(
                    f'PLACING ORDER BUY {round(row["AMOUNT"], 2)} SHARES OF {row["STOCK"]}')
            elif row["ACTION"] == "SELL" and row["STOCK"] in self.current_holdings:
                side_ = OrderSide.SELL
                print(
                    f'PLACING ORDER TO SELL {round(row["AMOUNT"], 2)} SHARES OF {row["STOCK"]}')
            else:
                self.scraped.append(row['ARTICLE_SOURCE'])
                side_ = False
                break_loop = True

            if not break_loop:
                self.scraped.append(row['ARTICLE_SOURCE'])

                try:
                    ## ToDo: make qty/notional >= $1
                    limit_order_data = MarketOrderRequest(
                        symbol=row["STOCK"],
                        qty=row["AMOUNT"],
                        side=side_,
                        extended_hours=ext_hours,
                        time_in_force=TimeInForce.DAY)

                    # Place limit order
                    limit_order = self.client.submit_order(order_data=limit_order_data)

                except Exception as e:
                    print(f"Order failed: {e}")

    def analyze_current_portfolio(self):

        def get_holdings_info(my_stock_data):
            pos_df = pd.DataFrame()
            for pos in self.all_positions:
                pos_df = pd.concat([pos_df, pd.DataFrame.from_dict(pos).set_index(0).T])
            pos_df = pos_df[['symbol', 'avg_entry_price', 'qty', 'market_value']]

            pos_df['qty'] = pd.to_numeric(pos_df['qty'], errors = 'coerce')
            for col in pos_df.drop(columns=['symbol','qty']).columns:
                pos_df[col] = round(pd.to_numeric(pos_df[col], errors='coerce'), 2)

            pos_df.columns = ['STOCK', 'Avg. Entry Price', 'Qty', 'Market Value']
            pos_df = pos_df.set_index('STOCK')

            mast_stock_data = pd.concat([pos_df, my_stock_data], axis=1)

            mast_stock_data['Portfolio Diversity'] = (pd.to_numeric(mast_stock_data['Market Value']) /
                                                      pd.to_numeric(mast_stock_data['Market Value']).sum())
            mast_stock_data['Portfolio Diversity'] = mast_stock_data['Portfolio Diversity'].round(2).astype(str) + '%'

            mast_stock_data['Net P/L (%)'] = (pd.to_numeric(mast_stock_data['Current Price'].str.replace(',', '')) /
                                              pd.to_numeric(mast_stock_data['Avg. Entry Price'])) - 1

            mast_stock_data['Net P/L (%)'] = np.where(mast_stock_data['Net P/L (%)'] > 0,
                                                      '+' + (mast_stock_data['Net P/L (%)'] * 100).round(2).astype(
                                                          str) + '%',
                                                      (mast_stock_data['Net P/L (%)'] * 100).round(2).astype(str) + '%')

            imp_cols = ['Avg. Entry Price', 'Qty', 'Portfolio Diversity', 'Net P/L (%)', 'Current Price',
                        '52 Week High',
                        '52 Week Low', 'Market Cap', 'P/E (TTM)', 'Fwd P/E (NTM)', 'EPS (TTM)', 'Beta', 'YTD % Change',
                        'Debt To Equity (MRQ)', 'ROE (TTM)', 'Gross Margin (TTM)', 'Revenue (TTM)', 'EBITDA (TTM)',
                        'Net Margin (TTM)', 'Dividend Yield']

            mast_stock_data = mast_stock_data[imp_cols].dropna(subset=['Current Price'])

            return mast_stock_data

        def gpt_portfolio(my_stock_info):
            prompt = ('Below is my portfolio. Which of these stocks should be held and which should be sold? '
                      'Respond in json format with zero whitespace and include the keys "STOCK", "ACTION".'
                      'Only include the stocks you would sell.\n' + my_stock_info.to_csv()
                      )
            responses = []
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are both a qualitative and quantitative stock market expert who's only "
                                       "goal in life is to beat the market and make money, channeling Warren Buffet's "
                                       "investment strategy of finding undervalued stocks. You are to provide stock "
                                       "market recommendations based on the data and context provided",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                resp = response["choices"][0]["message"]["content"]

                ### ToDo: parse json and print suggestions
                print(resp)
                responses.append(resp)

            except Exception as e:
                print(f'Query failed: {e}')
            print("Done")
            return responses

        with open('daily.txt') as f:
            file_date = f.read()

        if file_date == self.current_date:
            return
        print('Analyzing current portfolio')
        current_holdings_links = ["https://www.cnbc.com/quotes/" + stock for stock in self.current_holdings]
        my_stock_data = self.extract_stock_data(current_holdings_links)
        my_stock_info = get_holdings_info(my_stock_data)
        responses = gpt_portfolio(my_stock_info)
        recs, article_keys = self.process_recommendations(responses)
        my_stock_info = my_stock_info.reset_index().rename(columns={'index': 'STOCK'})
        recs = recs.merge(my_stock_info, on='STOCK', how='left')
        self.execute_decisions(recs)
        with open('daily.txt', 'w') as f:
            f.write(self.current_date)
        print('Done')

    def run(self):
        self.analyze_current_portfolio()
        queries, article_keys = self.scrape_articles()

        if queries:
            responses, article_keys = self.analyze_stocks(queries, article_keys)
            recs, article_keys = self.process_recommendations(responses, article_keys)
            self.execute_decisions(recs, article_keys)
            print('Failed articles:', self.failed_articles)
        else:
            #print('No new data')
            pass
        # Update dataframe with successfully scraped transactions
        pd.concat([pd.DataFrame(self.scraped)]).drop_duplicates().to_csv('scraped.csv', index=False)

# Load API keys from environment variables or another secure source
OPENAI_API_KEY = config.OPENAI_API_KEY
ALPACA_API_KEY = config.ALPACA_API_KEY
ALPACA_SECRET_KEY = config.ALPACA_SECRET_KEY

# Create a StockAnalyzer instance and run the analysis
stock_analyzer = StockAnalyzer(OPENAI_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY)

loop = True
while loop:
    stock_analyzer.run()
