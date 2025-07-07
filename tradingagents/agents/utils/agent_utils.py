from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from typing import List
from typing import Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import RemoveMessage
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage # Changed ToolOutput to ToolMessage
from datetime import date, timedelta, datetime
import functools
import pandas as pd
import os
import json # Added for parsing potential JSON error messages
import re # For the new (now removed by patch) parse_json_object, but good to keep if other regex ops are needed
from datetime import datetime, timedelta # For Finnhub date calculations
# from chromadb.config import Settings # This was in the diff but seems unrelated to the core changes, likely a merge artifact. Keeping commented.
from finnhub import Client as FinnhubClient # For Finnhub API
from tradingagents.dataflows import interface # Import the interface module

# Initialize Finnhub client
# Ensure FINNHUB_API_KEY is set as an environment variable
finnhub_api_key = os.getenv("FINNHUB_API_KEY")
if not finnhub_api_key:
    print("WARNING: FINNHUB_API_KEY environment variable not set. Finnhub-dependent tools may fail.")
finnhub_client = FinnhubClient(api_key=finnhub_api_key)


def ensure_not_plaintext_error(response, tool_name: str):
    """Raise ValueError if response is a simple string error from Finnhub."""
    if isinstance(response, str):
        # Finnhub can return plain-text errors like "No data available..."
        # or if the API key is invalid, it might also return a string error.
        # This check assumes any unexpected string response is an error.
        # More specific keyword checking could be added if certain string responses are valid.
        raise ValueError(f"{tool_name}: Finnhub API returned a plain-text response, suspected error: {response[:200]}")


# The @tool decorator and ToolMessage wrapping will be applied to the functions below
# where they are defined as methods of the Toolkit class.
# The functions provided in the patch are the new core logic for these methods.

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]
        
        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]
        
        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")
        
        return {"messages": removal_operations + [placeholder]}
    
    return delete_messages


class Toolkit:
    _config = None  # Initialize class attribute as None

    @classmethod
    def _ensure_config_loaded(cls):
        """Ensures the class-level config is loaded."""
        if cls._config is None:
            from tradingagents.default_config import DEFAULT_CONFIG as toolkit_default_config # Import locally
            cls._config = toolkit_default_config.copy()

    @classmethod
    def update_config(cls, config_update: dict):
        """Update the class-level configuration."""
        cls._ensure_config_loaded() # Ensure config is loaded before updating
        cls._config.update(config_update)

    @property
    def config(self) -> dict:
        """Access the shared class-level configuration."""
        Toolkit._ensure_config_loaded() # Ensure config is loaded before accessing
        return Toolkit._config

    def __init__(self, config: dict = None):
        """
        Initializes the Toolkit.
        Ensures the shared class configuration is loaded.
        If an initial 'config' is provided, it updates the shared class configuration.
        """
        Toolkit._ensure_config_loaded() # Ensure default config is loaded

        if config is not None: # If an overriding config is passed to constructor
            Toolkit.update_config(config) # Update the shared class config

    @staticmethod
    @tool
    def get_reddit_news( # type: ignore
        curr_date: Annotated[str, "Date you want to get news for in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_reddit_news_signature_fallback_id"
    ) -> ToolMessage:
        """
        Retrieve global news from Reddit within a specified time frame.
        Args:
            curr_date (str): Date you want to get news for in yyyy-mm-dd format
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the news or an error message.
        """
        tool_name = "get_reddit_news"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        try:
            global_news_result = interface.get_reddit_global_news(curr_date, 7, 5)
            if not isinstance(global_news_result, str):
                global_news_result = str(global_news_result)
            return ToolMessage(content=global_news_result, name=tool_name, tool_call_id=effective_tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for date {curr_date}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=effective_tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_finnhub_news( # type: ignore
        ticker: Annotated[
            str,
            "Search query of a company, e.g. 'AAPL, TSM, etc.",
        ],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_finnhub_news_signature_fallback_id"
    ) -> ToolMessage:
        """
        Retrieve the latest news about a given stock from Finnhub within a date range
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the news or an error message.
        """
        tool_name = "get_finnhub_news"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        try:
            end_date_str = end_date
            parsed_end_date = datetime.strptime(end_date, "%Y-%m-%d")
            parsed_start_date = datetime.strptime(start_date, "%Y-%m-%d")
            look_back_days = (parsed_end_date - parsed_start_date).days

            finnhub_news_result = interface.get_finnhub_news(
                ticker, end_date_str, look_back_days
            )
            if not isinstance(finnhub_news_result, str):
                finnhub_news_result = str(finnhub_news_result)
            return ToolMessage(content=finnhub_news_result, name=tool_name, tool_call_id=effective_tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=effective_tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_reddit_stock_info( # type: ignore
        ticker: Annotated[
            str,
            "Ticker of a company. e.g. AAPL, TSM",
        ],
        curr_date: Annotated[str, "Current date you want to get news for"],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_reddit_stock_info_signature_fallback_id"
    ) -> ToolMessage:
        """
        Retrieve the latest news about a given stock from Reddit, given the current date.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): current date in yyyy-mm-dd format to get news for
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the news or an error message.
        """
        tool_name = "get_reddit_stock_info"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        try:
            stock_news_results = interface.get_reddit_company_news(ticker, curr_date, 7, 5)
            if not isinstance(stock_news_results, str):
                stock_news_results = str(stock_news_results)
            return ToolMessage(content=stock_news_results, name=tool_name, tool_call_id=effective_tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=effective_tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_YFin_data( # type: ignore
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_YFin_data_signature_fallback_id"
    ) -> ToolMessage:
        """
        Retrieve the stock price data for a given ticker symbol from Yahoo Finance.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the stock data or an error message.
        """
        tool_name = "get_YFin_data"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        try:
            result_data = interface.get_YFin_data(symbol, start_date, end_date)
            if not isinstance(result_data, str):
                result_data = str(result_data)
            return ToolMessage(content=result_data, name=tool_name, tool_call_id=effective_tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {symbol}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=effective_tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_YFin_data_online( # type: ignore
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_YFin_data_online_signature_fallback_id"
    ) -> ToolMessage:
        """
        Retrieve the stock price data for a given ticker symbol from Yahoo Finance.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the stock data or an error message.
        """
        tool_name = "get_YFin_data_online"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        try:
            result_data = interface.get_YFin_data_online(symbol, start_date, end_date)
            if not isinstance(result_data, str):
                result_data = str(result_data)
            return ToolMessage(content=result_data, name=tool_name, tool_call_id=effective_tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {symbol}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=effective_tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_stockstats_indicators_report( # type: ignore
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[
            str, "The current trading date you are trading on, YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_stockstats_indicators_report_signature_fallback_id"
    ) -> ToolMessage:
        """
        Retrieve stock stats indicators for a given ticker symbol and indicator.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            indicator (str): Technical indicator to get the analysis and report of
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): How many days to look back, default is 30
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the indicators report or an error message.
        """
        tool_name = "get_stockstats_indicators_report"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        try:
            result_stockstats = interface.get_stock_stats_indicators_window(
                symbol, indicator, curr_date, look_back_days, False
            )
            if not isinstance(result_stockstats, str):
                result_stockstats = str(result_stockstats)
            return ToolMessage(content=result_stockstats, name=tool_name, tool_call_id=effective_tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {symbol}, indicator {indicator}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=effective_tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_stockstats_indicators_report_online( # type: ignore
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[
            str, "The current trading date you are trading on, YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_stockstats_indicators_report_online_signature_fallback_id"
    ) -> ToolMessage:
        """
        Retrieve stock stats indicators for a given ticker symbol and indicator.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            indicator (str): Technical indicator to get the analysis and report of
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): How many days to look back, default is 30
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the indicators report or an error message.
        """
        tool_name = "get_stockstats_indicators_report_online"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        try:
            result_stockstats = interface.get_stock_stats_indicators_window(
                symbol, indicator, curr_date, look_back_days, True
            )
            if not isinstance(result_stockstats, str):
                result_stockstats = str(result_stockstats)
            return ToolMessage(content=result_stockstats, name=tool_name, tool_call_id=effective_tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {symbol}, indicator {indicator}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=effective_tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_finnhub_company_insider_sentiment( # type: ignore
        ticker: Annotated[str, "ticker symbol for the company"],
        curr_date: Annotated[
            str,
            "current date of you are trading at, yyyy-mm-dd",
        ],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_finnhub_company_insider_sentiment_signature_fallback_id"
    ) -> ToolMessage:
        """
        Retrieve insider sentiment information about a company for the past 30 days
        Args:
            ticker (str): ticker symbol of the company
            curr_date (str): current date you are trading at, yyyy-mm-dd
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the sentiment data or an error message.
        """
        tool_name = "get_finnhub_company_insider_sentiment"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        try:
            data_sentiment = interface.get_finnhub_company_insider_sentiment(
                ticker, curr_date, 30
            )
            if not isinstance(data_sentiment, str):
                data_sentiment = str(data_sentiment)
            return ToolMessage(content=data_sentiment, name=tool_name, tool_call_id=effective_tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=effective_tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_finnhub_company_insider_transactions( # type: ignore
        ticker: Annotated[str, "ticker symbol"],
        curr_date: Annotated[
            str,
            "current date you are trading at, yyyy-mm-dd",
        ],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_finnhub_company_insider_transactions_signature_fallback_id"
    ) -> ToolMessage:
        """
        Retrieve insider transaction information about a company for the past 30 days
        Args:
            ticker (str): ticker symbol of the company
            curr_date (str): current date you are trading at, yyyy-mm-dd
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the transaction data or an error message.
        """
        tool_name = "get_finnhub_company_insider_transactions"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        try:
            data_trans = interface.get_finnhub_company_insider_transactions(
                ticker, curr_date, 30
            )
            if not isinstance(data_trans, str):
                data_trans = str(data_trans)
            return ToolMessage(content=data_trans, name=tool_name, tool_call_id=effective_tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=effective_tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_simfin_balance_sheet( # type: ignore
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_simfin_balance_sheet_signature_fallback_id"
    ) -> ToolMessage:
        """
        Retrieve the most recent balance sheet of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency: annual / quarterly
            curr_date (str): current date, yyyy-mm-dd
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the balance sheet data or an error message.
        """
        tool_name = "get_simfin_balance_sheet"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        try:
            data_balance_sheet = interface.get_simfin_balance_sheet(ticker, freq, curr_date)
            if not isinstance(data_balance_sheet, str):
                data_balance_sheet = str(data_balance_sheet)
            return ToolMessage(content=data_balance_sheet, name=tool_name, tool_call_id=effective_tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}, freq {freq}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=effective_tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_simfin_cashflow( # type: ignore
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_simfin_cashflow_signature_fallback_id"
    ) -> ToolMessage:
        """
        Retrieve the most recent cash flow statement of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency: annual / quarterly
            curr_date (str): current date, yyyy-mm-dd
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the cash flow data or an error message.
        """
        tool_name = "get_simfin_cashflow"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        try:
            data_cashflow = interface.get_simfin_cashflow(ticker, freq, curr_date)
            if not isinstance(data_cashflow, str):
                data_cashflow = str(data_cashflow)
            return ToolMessage(content=data_cashflow, name=tool_name, tool_call_id=effective_tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}, freq {freq}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=effective_tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_simfin_income_stmt( # type: ignore
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_simfin_income_stmt_signature_fallback_id"
    ) -> ToolMessage:
        """
        Retrieve the most recent income statement of a company
        Args:
            ticker (str): ticker symbol of the company
            freq (str): reporting frequency: annual / quarterly
            curr_date (str): current date, yyyy-mm-dd
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the income statement data or an error message.
        """
        tool_name = "get_simfin_income_stmt"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        try:
            data_income_stmt = interface.get_simfin_income_statements(
                ticker, freq, curr_date
            )
            if not isinstance(data_income_stmt, str):
                data_income_stmt = str(data_income_stmt)
            return ToolMessage(content=data_income_stmt, name=tool_name, tool_call_id=effective_tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}, freq {freq}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=effective_tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_google_news( # type: ignore
        query: Annotated[str, "Query to search with"],
        curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_google_news_signature_fallback_id"
    ) -> ToolMessage:
        """
        Retrieve the latest news from Google News based on a query and date range.
        Args:
            query (str): Query to search with
            curr_date (str): Current date in yyyy-mm-dd format
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the news or an error message.
        """
        tool_name = "get_google_news"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        try:
            google_news_results = interface.get_google_news(query, curr_date, 7)
            if not isinstance(google_news_results, str):
                google_news_results = str(google_news_results)
            return ToolMessage(content=google_news_results, name=tool_name, tool_call_id=effective_tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for query '{query}': {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=effective_tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_stock_news_openai( # type: ignore
        ticker: Annotated[str, "the company's ticker"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_stock_news_openai_signature_fallback_id"
    ) -> ToolMessage:
        """
        Fetch company news from Finnhub; handles Finnhub-specific errors.
        Note: Tool name includes 'openai' for historical/compatibility reasons, but now uses Finnhub.
        """
        tool_name = "get_stock_news_openai"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"

        try:
            # Calculate date range for the last 7 days
            to_date_obj = datetime.fromisoformat(curr_date.split('T')[0])
            from_date_obj = to_date_obj - timedelta(days=7)
            from_date_str = from_date_obj.date().isoformat()
            to_date_str = to_date_obj.date().isoformat()

            # Call the new interface function for FinancialDatasets.ai
            raw_data = interface.fetch_financialdatasets_news(
                ticker=ticker,
                start_date_str=from_date_str,
                end_date_str=to_date_str
            )

            news_list_to_process = None
            is_error_flag = False
            content_to_return = ""

            if isinstance(raw_data, str) and raw_data.startswith("Error:"):
                content_to_return = raw_data
                is_error_flag = True
            elif isinstance(raw_data, dict):
                news_list_to_process = raw_data.get("news")
                if not isinstance(news_list_to_process, list):
                    content_to_return = f"Error: 'news' key in response is not a list, got: {type(news_list_to_process)}"
                    is_error_flag = True
                    news_list_to_process = None # Ensure it's None so formatting is skipped
            elif isinstance(raw_data, list): # Should not happen with current FinancialDatasets.ai /news but good for robustness
                news_list_to_process = raw_data
            else:
                content_to_return = f"Error: Unexpected data type from fetch_financialdatasets_news: {type(raw_data)}"
                is_error_flag = True

            if not is_error_flag and news_list_to_process is not None:
                if not news_list_to_process: # Empty list
                    content_to_return = "No news found for the given period from FinancialDatasets.ai."
                else:
                    formatted_news = []
                    for item in news_list_to_process[:10]: # Limit to 10 news items
                        title = item.get('title', 'N/A')
                        published_at = item.get('published_at', 'N/A')
                        source_name = item.get('source_name', 'N/A')
                        formatted_news.append(f"Title: {title}\nPublished: {published_at}\nSource: {source_name}\n---")
                    content_to_return = "\n\n".join(formatted_news)
            elif not is_error_flag and news_list_to_process is None: # Should be caught by above checks, but as a safeguard
                content_to_return = "Error: News data is None after initial processing."
                is_error_flag = True

            return ToolMessage(
                name=tool_name,
                content=content_to_return,
                tool_call_id=effective_tool_call_id,
                is_error=is_error_flag
            )
        except Exception as e:
            # Catch-all for any other unexpected errors during this tool's execution
            error_content = f"Unexpected error in {tool_name} for {ticker} processing date {curr_date}: {e}"
            return ToolMessage(
                name=tool_name,
                content=error_content,
                tool_call_id=effective_tool_call_id,
                is_error=True
            )

    @staticmethod
    @tool
    def get_global_news_openai( # type: ignore
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format. Used for context if the macro endpoint supports it, otherwise fetches latest snapshot."],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_global_news_openai_signature_fallback_id"
    ) -> ToolMessage:
        """
        Fetches macroeconomic interest-rate snapshot data
        from FinancialDatasets.ai (latest central-bank rates).
        See: https://docs.financialdatasets.ai/api-reference/endpoint/macro/interest-rates/snapshot
        Args:
            curr_date (str): Current date, potentially for context (though snapshot is latest).
            tool_call_id (str): The ID of the tool call.
        Returns:
            ToolMessage: Contains formatted interest rate data or an error message.
        """
        tool_name = "get_global_news_openai"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        is_error_flag = False
        content_to_return = ""

        try:
            raw_data = interface.fetch_financialdatasets_macro_interest_rates_snapshot()

            if isinstance(raw_data, str) and raw_data.startswith("Error:"):
                 content_to_return = raw_data
                 is_error_flag = True
            elif isinstance(raw_data, list):
                if not raw_data:
                    content_to_return = "No interest rate snapshot data available from FinancialDatasets.ai."
                else:
                    formatted_rates = ["Central Bank Interest Rates Snapshot:"]
                    for item in raw_data:
                        bank = item.get('bank', 'N/A')
                        name = item.get('name', 'N/A')
                        rate = item.get('rate', 'N/A')
                        item_date = item.get('date', 'N/A')
                        formatted_rates.append(f"- {name} ({bank}): {rate}% (as of {item_date})")
                    content_to_return = "\n".join(formatted_rates)
            else: # Handles if raw_data is not a list (e.g. dict without 'interest_rates' or other types)
                content_to_return = f"Error: Expected list of rate records from macro snapshot, got {type(raw_data)}. Data: {str(raw_data)[:200]}"
                is_error_flag = True

        except Exception as e:
            content_to_return = f"Unexpected error in {tool_name} (fetching interest rates) for date {curr_date}: {e}"
            is_error_flag = True

        return ToolMessage(
            name=tool_name,
            content=content_to_return,
            tool_call_id=effective_tool_call_id,
            is_error=is_error_flag
        )

    @staticmethod
    @tool
    def get_fundamentals_openai( # type: ignore
        ticker: Annotated[str, "the company's ticker"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format (used for context, not directly by API if fetching latest)"],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_fundamentals_openai_signature_fallback_id"
    ) -> ToolMessage:
        """
        Fetch company fundamentals (Balance Sheet) using FinancialDatasets.ai.
        Note: Tool name includes 'openai' for historical/compatibility reasons.
        """
        tool_name = "get_fundamentals_openai"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        is_error_flag = False
        content_to_return = ""

        try:
            # Fetches latest annual balance sheet data
            raw_data = interface.fetch_financialdatasets_statement(
                ticker=ticker,
                statement_type="balance-sheets",
                period="annual",
                limit=1
            )

            actual_statements_data = None
            # is_error_flag and content_to_return are already defined outside try-except

            if isinstance(raw_data, str) and raw_data.startswith("Error:"):
                content_to_return = raw_data
                is_error_flag = True
            elif isinstance(raw_data, dict):
                if "data" in raw_data and (isinstance(raw_data["data"], list) or isinstance(raw_data["data"], dict)):
                    actual_statements_data = raw_data["data"]
                elif not raw_data: # Empty dict
                     actual_statements_data = {} # or an empty list depending on expected structure by LLM
                else: # A dict, but not the expected wrapper structure, assume it's the data itself
                    actual_statements_data = raw_data
            elif isinstance(raw_data, list): # Direct list of statements
                actual_statements_data = raw_data
            else:
                content_to_return = f"Error: Unexpected data type from fetch_financialdatasets_statement for {ticker}: {type(raw_data)}"
                is_error_flag = True

            if not is_error_flag and actual_statements_data is not None:
                if not actual_statements_data: # Empty list or dict after extraction or if it was initially empty
                    content_to_return = f"No annual balance sheet data found for {ticker} from FinancialDatasets.ai."
                else:
                    try:
                        content_to_return = json.dumps(actual_statements_data, indent=2)
                    except TypeError as te:
                        content_to_return = f"Error: Could not serialize fundamentals data to JSON for {ticker}: {te}"
                        is_error_flag = True
            elif not is_error_flag and actual_statements_data is None:
                 content_to_return = f"Error: Fundamentals data is None after initial processing for {ticker}."
                 is_error_flag = True

        except Exception as e:
            content_to_return = f"Unexpected error in {tool_name} for {ticker}: {e}"
            is_error_flag = True

        return ToolMessage(
            name=tool_name,
            content=content_to_return,
            tool_call_id=effective_tool_call_id,
            is_error=is_error_flag
        )
