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
from selenium import webdriver
from fake_useragent import UserAgent
from yahooquery import Ticker
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

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

        with open('daily.txt') as f:
            file_date = f.read()

        if file_date == self.current_date:
            self.new_day = False
            try:
                self.daily_transactions = pd.read_csv('daily_transactions.csv')
            except:
                self.daily_transactions = pd.DataFrame()
        else:
            self.new_day = True
            self.daily_transactions = pd.DataFrame()

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
        latest_ = data.latest().head()

        queries = []
        article_keys = []
        latest_nonscraped = latest_[~latest_["Link"].isin(self.scraped)]
        for article_link in list(latest_nonscraped["Link"]):
            r = requests.get(article_link)
            article = r.text
            soup = BeautifulSoup(article, "html.parser")
            stocks = []
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
                        stocks.append(stock)
                else:
                    print(article_link)
                    article_text = art.text
                    relevant_stocks = soup.find('ul', {'class': 'RelatedQuotes-list'})
                    if relevant_stocks is not None:
                        relevant_stocks = relevant_stocks.find_all('a')
                        for a in relevant_stocks:
                            quo = a['href'].replace('/quotes/', '').replace('/', '')
                            stocks.append(quo)
                    for sp in soup.find_all('span', {'data-test': 'QuoteInBody'}):
                        quo = sp.find('a', href=True)['href']
                        quo = quo.replace('/quotes/', '').replace('/', '')
                        stocks.append(quo)

                stocks = [None if stock in ('ETH.CM=', 'BTC.CM=') else stock for stock in stocks]
                stocks = [stock for stock in stocks if stock is not None]
                if not stocks == []:
                    # replace indices with SPY if it references index
                    stocks = ['SPY' if stock.startswith('.') else stock for stock in stocks]
                    stocks = ['VIX' if stock == '@VX.1' else stock for stock in stocks]
                    stocks = ['WTI' if stock == '@CL.1' else stock for stock in stocks]

                    stocks = [*set(stocks)]
                    print(stocks)
                    mast_data = pd.DataFrame()
                    stock_data = self.extract_stock_data(stocks)
                    stock_data = stock_data.reset_index().drop_duplicates(subset=['stock'])
                    stock_data = stock_data.set_index('stock')
                    for i in range(0, stock_data.shape[0]):
                        query = (stock_data.iloc[i:i + 1].dropna(axis=1).to_csv())
                        query += '\n' + article_text
                        print(query)
                        queries.append(query)
                        article_keys.append(article_link)

                else:
                    self.scraped.append(article_link)  # no stocks to scrape

            except Exception as e:
                print(f"Error occurred in {article_link}: {e}")
                if ((str(e) == "'NoneType' object has no attribute 'find'") |
                        (str(e) == "'symbol'")):
                    self.scraped.append(article_link)  # Unscrapable
                else:
                    self.failed_articles.append(article_link)

        return queries, article_keys

    def extract_stock_data(self, stocks):
        """
        Extract stock data from given stock links and concatenate to a DataFrame.
        """

        def get_cnbc_data(stocks):
            stock_links = ["https://www.cnbc.com/quotes/" + stock for stock in stocks]
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

        def get_tech_data(stocks):
            op = webdriver.ChromeOptions()
            op.add_argument('headless')
            ua = UserAgent()

            driver = webdriver.Chrome('/Users/ahearn/Downloads/chromedriver_mac_arm64/chromedriver', options=op)
            mast_df = pd.DataFrame()

            for ticker in stocks:
                url = f'http://tradingview.com/symbols/{ticker}/technicals/'
                # print(url)
                driver.get(url)
                html = driver.page_source
                soup = BeautifulSoup(html, "html.parser")
                tabs = soup.find_all('table')

                stock_df = pd.DataFrame()
                for table in tabs:
                    headers = [header.text for header in table.find_all("th")]

                    # Get table rows
                    rows = []
                    for row in table.find_all("tr")[1:]:
                        data = [cell.text for cell in row.find_all("td")]
                        rows.append(data)

                    df = pd.DataFrame(rows, columns=headers)

                    # Create dataframe
                    if 'Pivot' in df.columns:
                        df = df.melt(id_vars=['Pivot'], var_name='Method', value_name='Value', col_level=0)
                        df['Name'] = df['Pivot'] + ' ' + df['Method']

                    df = df[['Name', 'Value']].set_index('Name')
                    stock_df = pd.concat([stock_df, df])

                stock_df = stock_df.T
                stock_df['stock'] = ticker
                mast_df = pd.concat([mast_df, stock_df])
            driver.quit()
            mast_df = mast_df.rename_axis(None, axis=1).set_index('stock')
            return mast_df

        def get_yahoo_data(stocks):
            mast_df = pd.DataFrame()
            for stock in stocks:
                stock_df = Ticker(stock)
                try:
                    stock_df = stock_df.all_financial_data().tail(1).reset_index().rename(columns={'symbol': 'stock'})
                    stock_df['stock'] = stock_df['stock'].str.upper()
                    stock_df = stock_df.set_index('stock')
                    stock_df = stock_df.dropna(axis=1)
                    for col in stock_df.columns:
                        if (abs(pd.to_numeric(stock_df[col], errors='coerce').max() > 1000000) and
                                (stock_df[col].dtype == 'float64')):
                            stock_df[col] = stock_df[col] / 1000000
                            stock_df[col] = stock_df[col].round(3).astype(str) + str('M')
                        elif (abs(pd.to_numeric(stock_df[col], errors='coerce').max() > 1000) and
                              (stock_df[col].dtype == 'float64')):
                            stock_df[col] = stock_df[col] / 1000
                            stock_df[col] = stock_df[col].round(3).astype(str) + str('K')
                    mast_df = pd.concat([mast_df, stock_df])
                except Exception as e:
                    pass
            return mast_df

        ## ToDo: Get market indicators
        # def get_market_indicators()
        # GDP Growth Rate
        # Interest Rate
        # Inflation Rate
        # Unemployment Rate
        # Government Debt to GDP
        # Balance of Trade
        # Current Account to GDP
        # Credit Rating

        cnbc = get_cnbc_data(stocks)
        tv = get_tech_data(stocks)
        yahoo = get_yahoo_data(stocks)

        stock_data = pd.concat([tv, cnbc, yahoo], axis=1)

        stock_data['Today Date'] = self.current_date
        imp_cols = [
            'stock',
            'Relative Strength Index (14)',
            'Stochastic %K (14, 3, 3)',
            'Commodity Channel Index (20)',
            'Average Directional Index (14)',
            'Awesome Oscillator',
            'Momentum (10)',
            'MACD Level (12, 26)',
            'Stochastic RSI Fast (3, 3, 14, 14)',
            'Williams Percent Range (14)',
            'Bull Bear Power',
            'Ultimate Oscillator (7, 14, 28)',
            'Exponential Moving Average (10)',
            'Simple Moving Average (10)',
            'Exponential Moving Average (20)',
            'Simple Moving Average (20)',
            'Exponential Moving Average (30)',
            'Simple Moving Average (30)',
            'Exponential Moving Average (50)',
            'Simple Moving Average (50)',
            'Exponential Moving Average (100)',
            'Simple Moving Average (100)',
            'Exponential Moving Average (200)',
            'Simple Moving Average (200)',
            'Ichimoku Base Line (9, 26, 52, 26)',
            'Volume Weighted Moving Average (20)',
            'Hull Moving Average (9)',
            'S3 Classic',
            'S2 Classic',
            'S1 Classic',
            'P Classic',
            'R1 Classic',
            'R2 Classic',
            'R3 Classic',
            'S3 Fibonacci',
            'S2 Fibonacci',
            'S1 Fibonacci',
            'P Fibonacci',
            'R1 Fibonacci',
            'R2 Fibonacci',
            'R3 Fibonacci',
            'S3 Camarilla',
            'S2 Camarilla',
            'S1 Camarilla',
            'P Camarilla',
            'R1 Camarilla',
            'R2 Camarilla',
            'R3 Camarilla',
            'S3 Woodie',
            'S2 Woodie',
            'S1 Woodie',
            'P Woodie',
            'R1 Woodie',
            'R2 Woodie',
            'R3 Woodie',
            'S3 DM',
            'S2 DM',
            'S1 DM',
            'P DM',
            'R1 DM',
            'R2 DM',
            'R3 DM',
            'Current Price',
            'Daily Change',
            'Open',
            'Day High',
            'Day Low',
            'Prev Close',
            '52 Week High',
            '52 Week Low',
            'Market Cap',
            'Shares Out',
            '10 Day Average Volume',
            'Dividend',
            'Dividend Yield',
            'Beta',
            'YTD % Change',
            'EPS (TTM)',
            'P/E (TTM)',
            'Fwd P/E (NTM)',
            'EBITDA (TTM)',
            'ROE (TTM)',
            'Revenue (TTM)',
            'Gross Margin (TTM)',
            'Net Margin (TTM)',
            'Debt To Equity (MRQ)',
            'Today Date',
            'Earnings Date',
            'Ex Div Date',
            'Div Amount',
            'Split Date',
            'Split Factor',
            'CashAndCashEquivalents',
            'CurrentAssets',
            'TotalAssets',
            'CurrentLiabilities',
            'TotalLiabilitiesNetMinorityInterest',
            'TotalEquityGrossMinorityInterest',
            'OperatingIncome',
            'GrossProfit',
            'NetIncome',
            'CurrentDebt']
        cols_to_keep = [col for col in stock_data.columns if col in imp_cols]
        stock_data = stock_data[cols_to_keep]
        return stock_data

    def analyze_stocks(self, queries, article_keys):
        """
        Use OpenAI's GPT model to analyze the stock data and make buy, sell, or hold decisions.
        """
        responses, article_keys2 = [], []
        i = 0
        max_retry_attempts = 3
        for query in queries:
            print(f'Analyzing {article_keys[i]}')
            prompt = (
                    f"I have ${self.buying_power * 100} in buying power. For the stock in the json data below, tell me "
                    f"if I should buy, sell, or hold."
                    "If BUY, list how many shares you would buy (AMOUNT) considering my buying power. "
                    "If SELL, list how many shares (AMOUNT) you would sell considering my buying power. "
                    "Respond in json format with zero whitespace, including the following keys: "
                    "STOCK, ACTION, AMOUNT. Use the stock symbol."
                    "Here is the article and accompanying data from which you should base your decision: \n" + query
            )
            retry_count = 0
            success = False
            while not success and retry_count < max_retry_attempts:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are both a qualitative and quantitative stock market expert who's only "
                                           "goal in life is to beat the market and make money using day trading strategies "
                                           "and maximizing short-term gain. You are to provide stock market recommendations"
                                           " based on the data and context provided. Focus primarily on the data, but"
                                           " incorporate context from the article as needed.",
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

                    print(resp)
                    responses.append(resp)
                    article_keys2.append(article_keys[i])
                    success = True
                except Exception as e:
                    print(f'Query failed: {e}')
                    retry_count += 1
                    if retry_count >= max_retry_attempts:
                        try:
                            self.failed_articles.append(article_keys)
                        except:
                            pass
            i += 1

        print("Done")
        return responses, article_keys2

    def process_recommendations(self, responses, article_keys=None, portfolio=False):
        """
        Process the GPT model's buy/sell/hold recommendations and return as a DataFrame.
        """
        if article_keys is None:
            article_keys = []
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
                if not portfolio:
                    stock_df['ARTICLE_SOURCE'] = article_keys[i]
                mast_df = pd.concat([mast_df, stock_df])

            except Exception as e:
                print(f"Unable to parse JSON: {resp}, {e}")
                try:
                    self.failed_articles.append(article_keys[i])
                except:
                    pass
            i += 1

        mast_df['ACTION'] = mast_df['ACTION'].str.upper()
        mast_df.to_csv('mast_df.csv', index=False)

        # mast_df = mast_df[mast_df['ACTION'] == 'HOLD']
        if 'AMOUNT' in mast_df.columns:
            mast_df["AMOUNT"] = pd.to_numeric(mast_df["AMOUNT"], errors="coerce") / (self.buying_power * 100) * 10

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

        ext_hours = is_extended_hours()
        if 'ACTION' in self.daily_transactions.columns:
            daily_buys = list(self.daily_transactions[self.daily_transactions['ACTION'] == 'BUY']['STOCK'])
        else:
            daily_buys = []

        if 'AMOUNT' not in mast_df.columns:
            mast_df['AMOUNT'] = pd.to_numeric(mast_df['Qty'])
        for index, row in mast_df.iterrows():
            break_loop = False
            if row["ACTION"] == "BUY":
                side_ = OrderSide.BUY
                print(f'PLACING ORDER BUY {row["STOCK"]}')
                market_order_data = MarketOrderRequest(
                    symbol=row["STOCK"],
                    notional=50,
                    side=side_,
                    time_in_force=TimeInForce.DAY)

            elif ((row["ACTION"] == "SELL") and (row["STOCK"] in self.current_holdings) and
                  (row['STOCK'] not in daily_buys)):
                side_ = OrderSide.SELL
                print(f'PLACING ORDER SELL {row["STOCK"]}')
                d = pd.DataFrame([dict(d) for d in self.client.get_all_positions()])
                amount = list(d[d['symbol'] == row['STOCK']]['qty'])[0]

                market_order_data = MarketOrderRequest(
                    symbol=row["STOCK"],
                    qty=amount,
                    side=side_,
                    time_in_force=TimeInForce.DAY)
            elif row["ACTION"] == "HOLD" and row["STOCK"] in self.current_holdings:
                self.scraped.append(row['ARTICLE_SOURCE'])
                break_loop = True

            else:
                self.scraped.append(row['ARTICLE_SOURCE'])
                side_ = False
                break_loop = True

            if not break_loop:
                self.scraped.append(row['ARTICLE_SOURCE'])

                try:
                    ## ToDo: make qty/notional >= $1. Need to get current price

                    # Place market order
                    self.client.submit_order(order_data=market_order_data)
                    self.daily_transactions = pd.concat([self.daily_transactions, pd.DataFrame(row).T])
                except Exception as e:
                    print(f"Order failed: {e}")

        # self.daily_transactions.to_csv('daily_transactions.csv', index=False)

    def analyze_current_portfolio(self):

        def get_holdings_info(my_stock_data):
            pos_df = pd.DataFrame()
            for pos in self.all_positions:
                pos_df = pd.concat([pos_df, pd.DataFrame.from_dict(pos).set_index(0).T])

            pos_df = pos_df[['symbol', 'avg_entry_price', 'qty', 'market_value']]

            pos_df['qty'] = pd.to_numeric(pos_df['qty'], errors='coerce')
            for col in pos_df.drop(columns=['symbol']).columns:
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
            mast_stock_data = mast_stock_data.sample(mast_stock_data.shape[0])
            return mast_stock_data

        def gpt_portfolio(my_stock_info):
            queries = []
            for i in range(0, my_stock_info.shape[0]):
                query = (my_stock_info.iloc[i:i + 1].dropna(axis=1).to_csv())
                queries.append(query)

            responses = []
            max_retry_attempts = 5
            for query in queries:
                prompt = ('Below is a stock I own. Based on the data, should this stock be sold or held?'
                          'Respond in json format with zero whitespace and include the keys "STOCK", "ACTION".'
                          '\n' + query
                          )
                retry_count = 0
                success = False
                while not success and retry_count < max_retry_attempts:
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are both a qualitative and quantitative stock market expert who's only "
                                               "goal in life is to beat the market and make money using day trading strategies "
                                               "and maximizing short-term gain. You are to provide stock market recommendations"
                                               " based on the data provided.",
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
                        print(resp)
                        responses.append(resp)
                        success = True
                    except Exception as e:
                        print(f'Query failed: {e}')
                        retry_count += 1

            print("Done")
            return responses

        if not self.new_day:
            return
        print('Analyzing current portfolio')
        my_stock_data = self.extract_stock_data(self.current_holdings)

        my_stock_info = get_holdings_info(my_stock_data).dropna(subset=['Current Price'])
        responses = gpt_portfolio(my_stock_info)
        recs, article_keys = self.process_recommendations(responses, portfolio=True)
        my_stock_info = my_stock_info.reset_index().rename(columns={'index': 'STOCK'})
        recs = recs.merge(my_stock_info, on='STOCK', how='left')
        recs.to_csv('recs.csv')
        self.execute_decisions(recs)
        with open('daily.txt', 'w') as f:
            f.write(self.current_date)
            self.new_day = False
        print('Done')

    def run(self):
        self.analyze_current_portfolio()
        queries, article_keys = self.scrape_articles()

        if queries:
            responses, article_keys = self.analyze_stocks(queries, article_keys)
            recs, article_keys = self.process_recommendations(responses, article_keys)
            self.execute_decisions(recs, article_keys)
            # print('Failed articles:', [*set(self.failed_articles)])
        else:
            # print('No new data')
            pass
        # Update dataframe with successfully scraped transactions
        pd.concat([pd.DataFrame(self.scraped)]).drop_duplicates().to_csv('scraped.csv', index=False)
        self.daily_transactions.to_csv('daily_transactions.csv')


# Load API keys from environment variables or another secure source
OPENAI_API_KEY = config.OPENAI_API_KEY
ALPACA_API_KEY = config.ALPACA_API_KEY
ALPACA_SECRET_KEY = config.ALPACA_SECRET_KEY

# Create a StockAnalyzer instance and run the analysis
stock_analyzer = StockAnalyzer(OPENAI_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY)

loop = True
while loop:
    try:
        stock_analyzer.run()
    except Exception as e:
        print(f'Error occurred: {e}')
