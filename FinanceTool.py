
import ssl
import certifi
from urllib.request import urlopen
import json
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import os


class MyToolInput(BaseModel):
  """Input schema for FinanceTool."""
  ticker: str = Field(..., description="stock ticker symbol of the company (for example  AAPL for Apple, TSLA for Tesla, AMZN for amazon, MSFT for micrsoft, NVDA for nvidia, GOOGL for google, META for meta, NFLX for netflix, etc)")


class FinanceTool(BaseTool):
  name: str = "Company Finance Data Tool"
  description: str = """
      Fetches financial data for a given company ticker.

      This tool retrieves data from multiple endpoints provided by Financial Modeling Prep:

      1. Ratings Snapshot:
        - Provides a comprehensive snapshot of financial ratings based on key financial ratios.

      2. Company Profile:
        - Offers key financial and operational information such as market capitalization, stock price, and industry details.

      3. Shares Float:
        - Supplies information on the liquidity and volatility of a stock, including the total number of publicly traded shares.

      4. Financial Scores:
        - Presents key metrics such as the Altman Z-Score and Piotroski Score for assessing a company's overall financial health.

      5. TTM Metrics:
        - Delivers trailing twelve-month (TTM) key performance metrics related to profitability, capital efficiency, and liquidity.

      6. Balance Sheet:
        - Provides the balance sheet of the company, showing assets, liabilities, and equity.

      7. Income Statement:
        - Provides the income statement of the company, showing revenue, expenses, and net income.

      Returns:
          dict: A dictionary containing the retrieved data along with descriptions for each data set.
      """
  args_schema: Type[BaseModel] = MyToolInput

  def get_jsonparsed_data(self, url: str):
    """
    Retrieve and parse JSON data from the given URL.
    """
    context = ssl.create_default_context(cafile=certifi.where())

    # Fetch data using urlopen with the custom context
    response = urlopen(url, context=context)
    data = response.read().decode("utf-8")
    return json.loads(data)

  def _run(self,ticker: str,api_key: str = None) -> dict:
    if api_key is None:
      api_key = os.getenv("FMP_API_KEY")
      if api_key is None:
        raise ValueError("FMP_API key is missing. Please provide it as an argument or set it in the environment.")

    try:
      base_url = "https://financialmodelingprep.com/stable/"
      endpoints = {
          "ratings_snapshot": {
              "url": f"{base_url}ratings-snapshot?symbol={ticker}&apikey={api_key}",
              "description": "Provides a comprehensive snapshot of financial ratings based on various key financial ratios."
          },
          "profile": {
              "url": f"{base_url}profile?symbol={ticker}&apikey={api_key}",
              "description": "Contains key financial and operational information, including market capitalization, stock price, and industry details."
          },
          "shares_float": {
              "url": f"{base_url}shares-float?symbol={ticker}&apikey={api_key}",
              "description": "Details on liquidity and volatility of the stock by providing the total number of publicly traded shares."
          },
          "financial_scores": {
              "url": f"{base_url}financial-scores?symbol={ticker}&apikey={api_key}",
              "description": "Includes key metrics like the Altman Z-Score and Piotroski Score, giving insights into overall financial health and stability."
          },
          "ttm_metrics": {
              "url": f"{base_url}key-metrics-ttm?symbol={ticker}&apikey={api_key}",
              "description": "Retrieves trailing twelve-month (TTM) key performance metrics related to profitability, capital efficiency, and liquidity."
          },
          "balance_sheet": {
              "url": f"{base_url}balance-sheet-statement?symbol={ticker}&limit=3&apikey={api_key}",
              "description": "Provides the balance sheet of the company, showing assets, liabilities, and equity."
          },
       

      }

      results = {}
      for key, info in endpoints.items():
          try:
              data = self.get_jsonparsed_data(info["url"])
          except Exception as e:
              data = f"Error retrieving data: {e}"
          results[key] = {
              "description": info["description"],
              "data": data
          }

    except Exception as e:
      print(e)

    return results

