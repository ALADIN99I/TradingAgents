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
from dateutil.relativedelta import relativedelta
from langchain_openai import ChatOpenAI
import tradingagents.dataflows.interface as interface
from tradingagents.default_config import DEFAULT_CONFIG
from langchain_core.messages import HumanMessage


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
    _config = DEFAULT_CONFIG.copy()

    @classmethod
    def update_config(cls, config):
        """Update the class-level configuration."""
        cls._config.update(config)

    @property
    def config(self):
        """Access the configuration."""
        return self._config

    def __init__(self, config=None):
        if config:
            self.update_config(config)

    @staticmethod
    @tool
    def get_reddit_news(
        curr_date: Annotated[str, "Date you want to get news for in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"]
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
        try:
            global_news_result = interface.get_reddit_global_news(curr_date, 7, 5)
            if not isinstance(global_news_result, str):
                global_news_result = str(global_news_result)
            return ToolMessage(content=global_news_result, name=tool_name, tool_call_id=tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for date {curr_date}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_finnhub_news(
        ticker: Annotated[
            str,
            "Search query of a company, e.g. 'AAPL, TSM, etc.",
        ],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"]
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
            return ToolMessage(content=finnhub_news_result, name=tool_name, tool_call_id=tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_reddit_stock_info(
        ticker: Annotated[
            str,
            "Ticker of a company. e.g. AAPL, TSM",
        ],
        curr_date: Annotated[str, "Current date you want to get news for"],
        tool_call_id: Annotated[str, "The ID of the tool call"]
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
        try:
            stock_news_results = interface.get_reddit_company_news(ticker, curr_date, 7, 5)
            if not isinstance(stock_news_results, str):
                stock_news_results = str(stock_news_results)
            return ToolMessage(content=stock_news_results, name=tool_name, tool_call_id=tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_YFin_data(
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"]
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
        try:
            result_data = interface.get_YFin_data(symbol, start_date, end_date)
            if not isinstance(result_data, str):
                result_data = str(result_data)
            return ToolMessage(content=result_data, name=tool_name, tool_call_id=tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {symbol}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_YFin_data_online(
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"]
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
        try:
            result_data = interface.get_YFin_data_online(symbol, start_date, end_date)
            if not isinstance(result_data, str):
                result_data = str(result_data)
            return ToolMessage(content=result_data, name=tool_name, tool_call_id=tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {symbol}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_stockstats_indicators_report(
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[
            str, "The current trading date you are trading on, YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
        tool_call_id: Annotated[str, "The ID of the tool call"] = ""
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
        try:
            result_stockstats = interface.get_stock_stats_indicators_window(
                symbol, indicator, curr_date, look_back_days, False
            )
            if not isinstance(result_stockstats, str):
                result_stockstats = str(result_stockstats)
            return ToolMessage(content=result_stockstats, name=tool_name, tool_call_id=tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {symbol}, indicator {indicator}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_stockstats_indicators_report_online(
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[
            str, "technical indicator to get the analysis and report of"
        ],
        curr_date: Annotated[
            str, "The current trading date you are trading on, YYYY-mm-dd"
        ],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
        tool_call_id: Annotated[str, "The ID of the tool call"] = ""
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
        try:
            result_stockstats = interface.get_stock_stats_indicators_window(
                symbol, indicator, curr_date, look_back_days, True
            )
            if not isinstance(result_stockstats, str):
                result_stockstats = str(result_stockstats)
            return ToolMessage(content=result_stockstats, name=tool_name, tool_call_id=tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {symbol}, indicator {indicator}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_finnhub_company_insider_sentiment(
        ticker: Annotated[str, "ticker symbol for the company"],
        curr_date: Annotated[
            str,
            "current date of you are trading at, yyyy-mm-dd",
        ],
        tool_call_id: Annotated[str, "The ID of the tool call"]
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
        try:
            data_sentiment = interface.get_finnhub_company_insider_sentiment(
                ticker, curr_date, 30
            )
            if not isinstance(data_sentiment, str):
                data_sentiment = str(data_sentiment)
            return ToolMessage(content=data_sentiment, name=tool_name, tool_call_id=tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_finnhub_company_insider_transactions(
        ticker: Annotated[str, "ticker symbol"],
        curr_date: Annotated[
            str,
            "current date you are trading at, yyyy-mm-dd",
        ],
        tool_call_id: Annotated[str, "The ID of the tool call"]
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
        try:
            data_trans = interface.get_finnhub_company_insider_transactions(
                ticker, curr_date, 30
            )
            if not isinstance(data_trans, str):
                data_trans = str(data_trans)
            return ToolMessage(content=data_trans, name=tool_name, tool_call_id=tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_simfin_balance_sheet(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
        tool_call_id: Annotated[str, "The ID of the tool call"]
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
        try:
            data_balance_sheet = interface.get_simfin_balance_sheet(ticker, freq, curr_date)
            if not isinstance(data_balance_sheet, str):
                data_balance_sheet = str(data_balance_sheet)
            return ToolMessage(content=data_balance_sheet, name=tool_name, tool_call_id=tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}, freq {freq}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_simfin_cashflow(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
        tool_call_id: Annotated[str, "The ID of the tool call"]
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
        try:
            data_cashflow = interface.get_simfin_cashflow(ticker, freq, curr_date)
            if not isinstance(data_cashflow, str):
                data_cashflow = str(data_cashflow)
            return ToolMessage(content=data_cashflow, name=tool_name, tool_call_id=tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}, freq {freq}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_simfin_income_stmt(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
        tool_call_id: Annotated[str, "The ID of the tool call"]
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
        try:
            data_income_stmt = interface.get_simfin_income_statements(
                ticker, freq, curr_date
            )
            if not isinstance(data_income_stmt, str):
                data_income_stmt = str(data_income_stmt)
            return ToolMessage(content=data_income_stmt, name=tool_name, tool_call_id=tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}, freq {freq}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_google_news(
        query: Annotated[str, "Query to search with"],
        curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"]
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
        try:
            google_news_results = interface.get_google_news(query, curr_date, 7)
            if not isinstance(google_news_results, str):
                google_news_results = str(google_news_results)
            return ToolMessage(content=google_news_results, name=tool_name, tool_call_id=tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for query '{query}': {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_stock_news_openai(
        ticker: Annotated[str, "the company's ticker"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"]
    ):
        """
        Retrieve the latest news about a given stock by using OpenAI's news API.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): Current date in yyyy-mm-dd format
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the news or an error message.
        """
        tool_name = "get_stock_news_openai" # It's good practice to define the tool name
        try:
            openai_news_results = interface.get_stock_news_openai(ticker, curr_date)
            # Ensure openai_news_results is a string, not another ToolMessage or complex object
            if not isinstance(openai_news_results, str):
                # Attempt to serialize or get a string representation if it's not a simple string
                # This depends on what interface.get_stock_news_openai actually returns
                # For now, let's assume it should be a string. If it's complex, this needs adjustment.
                openai_news_results = str(openai_news_results)

            return ToolMessage(
                content=openai_news_results,
                name=tool_name,
                tool_call_id=tool_call_id
            )
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}: {e}"
            return ToolMessage(
                content=error_string,
                name=tool_name,
                tool_call_id=tool_call_id,
                is_error=True # Optional: some frameworks might use this
            )

    @staticmethod
    @tool
    def get_global_news_openai(
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"]
    ) -> ToolMessage:
        """
        Retrieve the latest macroeconomics news on a given date using OpenAI's macroeconomics news API.
        Args:
            curr_date (str): Current date in yyyy-mm-dd format
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the news or an error message.
        """
        tool_name = "get_global_news_openai"
        try:
            openai_news_results = interface.get_global_news_openai(curr_date)
            if not isinstance(openai_news_results, str):
                openai_news_results = str(openai_news_results)
            return ToolMessage(content=openai_news_results, name=tool_name, tool_call_id=tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for date {curr_date}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_fundamentals_openai(
        ticker: Annotated[str, "the company's ticker"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"]
    ) -> ToolMessage:
        """
        Retrieve the latest fundamental information about a given stock on a given date by using OpenAI's news API.
        Args:
            ticker (str): Ticker of a company. e.g. AAPL, TSM
            curr_date (str): Current date in yyyy-mm-dd format
            tool_call_id (str): The ID of the tool call, injected by the framework.
        Returns:
            ToolMessage: A ToolMessage object containing the fundamentals data or an error message.
        """
        tool_name = "get_fundamentals_openai"
        try:
            openai_fundamentals_results = interface.get_fundamentals_openai(
                ticker, curr_date
            )
            if not isinstance(openai_fundamentals_results, str):
                openai_fundamentals_results = str(openai_fundamentals_results)
            return ToolMessage(content=openai_fundamentals_results, name=tool_name, tool_call_id=tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for {ticker}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=tool_call_id, is_error=True)
