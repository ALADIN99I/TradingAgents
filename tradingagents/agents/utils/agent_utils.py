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
        Fetches stock news from an external source and includes debugging prints.
        """
        tool_name = "get_stock_news_openai"
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"

        try:
            raw_interface_output = interface.get_stock_news_openai(ticker, curr_date)

            if isinstance(raw_interface_output, str):
                # Heuristic check for common non-JSON error patterns or HTML
                lower_output = raw_interface_output.lower()
                error_keywords = ["error", "failed", "invalid", "unavailable", "forbidden", "unauthorized", "limit exceeded", "not found", "service unavailable"]
                html_tags = ["<html>", "<body>", "<head>", "<!doctype html"]

                if any(keyword in lower_output for keyword in error_keywords) or \
                   any(tag in lower_output for tag in html_tags):
                    # Limit the length of the raw output in the error message
                    output_snippet = raw_interface_output[:200] + "..." if len(raw_interface_output) > 200 else raw_interface_output
                    raise ValueError(f"Suspected non-JSON error string returned by interface: '{output_snippet}'")

                # Attempt to parse as JSON and check for known error structure (this comes after the heuristic check)
                try:
                    parsed_output = json.loads(raw_interface_output)
                    if isinstance(parsed_output, dict) and "error" in parsed_output:
                        error_detail = parsed_output.get("error")
                        raise ValueError(f"API returned a JSON error: {error_detail}")
                    # If it's JSON but not the error structure, it's likely valid data or an unknown JSON format.
                except json.JSONDecodeError:
                    # It's not JSON. Apply the new/modified heuristic for non-JSON strings.
                    stripped_output = raw_interface_output.strip()
                    if not (stripped_output.startswith('{') or stripped_output.startswith('[')):
                        # It's a string that doesn't start like JSON.
                        # Check if it's a short, potentially affirmative message or a longer, suspicious one.
                        # Threshold can be adjusted. "Success" or "OK" are short.
                        if len(stripped_output) > 20 and "success" not in stripped_output.lower() and "ok" not in stripped_output.lower():
                            output_snippet = stripped_output[:200] + "..." if len(stripped_output) > 200 else stripped_output
                            raise ValueError(f"Suspected generic non-JSON, non-affirmative string returned by interface: '{output_snippet}'")
                    # If it's short or contains "success"/"ok", or starts with {/[, let it pass to be treated as content.
                    pass # Let it be treated as successful content if it's not caught

            successful_data_string = raw_interface_output
            if not isinstance(successful_data_string, str):
                successful_data_string = str(successful_data_string)

            return ToolMessage(
                name=tool_name,
                content=successful_data_string,
                tool_call_id=effective_tool_call_id,
            )
        except Exception as e:
            error_content = f"Error in {tool_name} for {ticker} processing date {curr_date}: {e}"
            # Optionally log this error if a logging framework is in use
            # print(f"ERROR: Caught exception in {tool_name}: {error_content}\n")
            return ToolMessage(
                name=tool_name,
                content=error_content,
                tool_call_id=effective_tool_call_id,
                is_error=True
            )

    @staticmethod
    @tool
    def get_global_news_openai( # type: ignore
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_global_news_openai_signature_fallback_id"
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
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"
        try:
            openai_news_results = interface.get_global_news_openai(curr_date)
            if not isinstance(openai_news_results, str):
                openai_news_results = str(openai_news_results)
            return ToolMessage(content=openai_news_results, name=tool_name, tool_call_id=effective_tool_call_id)
        except Exception as e:
            error_string = f"Error in {tool_name} for date {curr_date}: {e}"
            return ToolMessage(content=error_string, name=tool_name, tool_call_id=effective_tool_call_id, is_error=True)

    @staticmethod
    @tool
    def get_fundamentals_openai( # type: ignore
        ticker: Annotated[str, "the company's ticker"],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        tool_call_id: Annotated[str, "The ID of the tool call"] = "get_fundamentals_openai_signature_fallback_id"
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
        effective_tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else f"{tool_name}_runtime_missing_or_empty_id"

        try:
            raw_interface_output = interface.get_fundamentals_openai(ticker, curr_date)

            if isinstance(raw_interface_output, str):
                # Heuristic check for common non-JSON error patterns or HTML
                lower_output = raw_interface_output.lower()
                error_keywords = ["error", "failed", "invalid", "unavailable", "forbidden", "unauthorized", "limit exceeded", "not found", "service unavailable"]
                html_tags = ["<html>", "<body>", "<head>", "<!doctype html"]

                if any(keyword in lower_output for keyword in error_keywords) or \
                   any(tag in lower_output for tag in html_tags):
                    output_snippet = raw_interface_output[:200] + "..." if len(raw_interface_output) > 200 else raw_interface_output
                    raise ValueError(f"Suspected non-JSON error string returned by interface: '{output_snippet}'")

                # Attempt to parse as JSON and check for known error structure
                try:
                    parsed_output = json.loads(raw_interface_output)
                    if isinstance(parsed_output, dict) and "error" in parsed_output:
                        error_detail = parsed_output.get("error")
                        raise ValueError(f"API returned a JSON error: {error_detail}")
                    # If it's JSON but not the error structure, it's likely valid data or an unknown JSON format.
                except json.JSONDecodeError:
                    # It's not JSON. Apply the new/modified heuristic for non-JSON strings.
                    stripped_output = raw_interface_output.strip()
                    if not (stripped_output.startswith('{') or stripped_output.startswith('[')):
                        # It's a string that doesn't start like JSON.
                        # Check if it's a short, potentially affirmative message or a longer, suspicious one.
                        if len(stripped_output) > 20 and "success" not in stripped_output.lower() and "ok" not in stripped_output.lower():
                            output_snippet = stripped_output[:200] + "..." if len(stripped_output) > 200 else stripped_output
                            raise ValueError(f"Suspected generic non-JSON, non-affirmative string returned by interface: '{output_snippet}'")
                    # If it's short or contains "success"/"ok", or starts with {/[, let it pass to be treated as content.
                    pass # Let it be treated as successful content if it's not caught

            successful_data_string = raw_interface_output
            if not isinstance(successful_data_string, str):
                successful_data_string = str(successful_data_string)

            return ToolMessage(content=successful_data_string, name=tool_name, tool_call_id=effective_tool_call_id)
        except Exception as e:
            error_content = f"Error in {tool_name} for {ticker} processing date {curr_date}: {e}"
            # Optionally log this error
            # print(f"ERROR: Caught exception in {tool_name}: {error_content}\n")
            return ToolMessage(content=error_content, name=tool_name, tool_call_id=effective_tool_call_id, is_error=True)
