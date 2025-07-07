# TradingAgents/graph/trading_graph.py

import os
import time # Added for rate limiting delays
from pathlib import Path
import json
import logging
from datetime import date # Ensure date is imported
from typing import Dict, Any, Tuple, List, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.interface import set_config

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG

        
        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs
        # around line 71 - REPLACEMENT CODE 
        """ it was # Around line 71
if self.config["llm_provider"].lower() == "openai" or self.config["llm_provider"] == "ollama" or self.config["llm_provider"] == "openrouter":
    self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
    self.quick_thinking_llm = ChatOpenAI(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])"""
        
        
        if self.config["llm_provider"].lower() in ["openai", "ollama", "openrouter"]:
            # Create a dictionary for the arguments
            llm_args = {
                "base_url": self.config["backend_url"],
            }
            # If the provider is OpenRouter, add the specific API key
            if self.config["llm_provider"].lower() == "openrouter":
                llm_args["api_key"] = os.environ.get("OPENROUTER_API_KEY")

            # Initialize the LLMs with the correct arguments
            self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"], **llm_args)
            self.quick_thinking_llm = ChatOpenAI(model=self.config["quick_think_llm"], **llm_args)
        elif self.config["llm_provider"].lower() == "anthropic":
            self.deep_thinking_llm = ChatAnthropic(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatAnthropic(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"].lower() == "google":
            self.deep_thinking_llm = ChatGoogleGenerativeAI(model=self.config["deep_think_llm"])
            self.quick_thinking_llm = ChatGoogleGenerativeAI(model=self.config["quick_think_llm"])
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")
        
        self.toolkit = Toolkit(config=self.config)

        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic()
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.toolkit,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources."""
        return {
            "market": ToolNode(
                [
                    # online tools
                    self.toolkit.get_YFin_data_online,
                    self.toolkit.get_stockstats_indicators_report_online,
                    # offline tools
                    self.toolkit.get_YFin_data,
                    self.toolkit.get_stockstats_indicators_report,
                ]
            ),
            "social": ToolNode(
                [
                    # online tools
                    self.toolkit.get_stock_news_openai,
                    # offline tools
                    self.toolkit.get_reddit_stock_info,
                ]
            ),
            "news": ToolNode(
                [
                    # online tools
                    self.toolkit.get_global_news_openai,
                    self.toolkit.get_google_news,
                    # offline tools
                    self.toolkit.get_finnhub_news,
                    self.toolkit.get_reddit_news,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # online tools
                    self.toolkit.get_fundamentals_openai,
                    # offline tools
                    self.toolkit.get_finnhub_company_insider_sentiment,
                    self.toolkit.get_finnhub_company_insider_transactions,
                    self.toolkit.get_simfin_balance_sheet,
                    self.toolkit.get_simfin_cashflow,
                    self.toolkit.get_simfin_income_stmt,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history": final_state["risk_debate_state"]["risky_history"],
                "safe_history": final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)

    def get_portfolio_management_advice(
        self, detailed_open_positions: List[Dict[str, Any]], ticker_to_potentially_trade: str
    ) -> Dict[str, Any]:
        """
        Analyzes open positions and a potential new trade using an LLM, returning structured advice.

        Args:
            detailed_open_positions: A list of dictionaries, each representing an open position.
            ticker_to_potentially_trade: The ticker symbol for a potential new trade.

        Returns:
            A dictionary with position management advice and new trade opportunity analysis.
            Returns placeholder advice on LLM error.
        """
        current_date_str = date.today().isoformat()
        open_positions_json = json.dumps(detailed_open_positions, indent=2)

        prompt_template = """\
You are an expert portfolio manager and trading analyst. Your task is to review a list of currently open positions and a potential new stock to trade. Provide clear, actionable advice in JSON format.

Today's Date: {current_date}

Current Open Positions:
{open_positions_json}

Potential New Trade:
Ticker: {ticker_to_potentially_trade}

Instructions:

1.  For each position in "Current Open Positions":
    *   Analyze its current status (symbol, quantity, entry price, market price, P&L).
    *   Decide on a management action. Valid actions are: "HOLD", "CLOSE", "REDUCE", "ADD".
    *   If action is "REDUCE" or "ADD", include "quantity" (integer).
    *   Provide a brief "reason" for your decision (max 1-2 sentences).
    *   The output for each position should be a JSON object: {{"symbol": "XYZ", "action": "ACTION", "quantity": X (optional), "reason": "Brief reason."}}

2.  For the "Potential New Trade" ({ticker_to_potentially_trade}):
    *   Analyze its potential as a new trade.
    *   Decide on an action. Valid actions are: "BUY", "SELL" (short), or "NONE".
    *   Provide a "conviction_score" (float between 0.0 and 1.0). If "NONE", score can be 0.0.
    *   Provide a brief "reason" (max 1-2 sentences).
    *   The output should be a JSON object: {{"symbol": "XYZ", "decision": "ACTION", "conviction_score": Y.YY, "reason": "Brief reason."}}. If "decision" is "NONE", this entire object can be null or structured with "decision": "NONE".

Output Format:
Return a single JSON object with two keys: "position_management" and "new_trade_opportunity".
"position_management" should be a list of JSON objects (one for each open position).
"new_trade_opportunity" should be a single JSON object or null.

Example:
{{
  "position_management": [
    {{"symbol": "AAPL", "action": "HOLD", "reason": "Monitoring trend."}},
    {{"symbol": "MSFT", "action": "REDUCE", "quantity": 10, "reason": "Profit target."}}
  ],
  "new_trade_opportunity": {{"symbol": "GOOGL", "decision": "BUY", "conviction_score": 0.85, "reason": "Breakout."}}
}}

Provide only the JSON output. Do not include any other explanatory text before or after the JSON.
"""
        prompt = prompt_template.format(
            current_date=current_date_str,
            open_positions_json=open_positions_json,
            ticker_to_potentially_trade=ticker_to_potentially_trade,
        )

        try:
            # Using the deep_thinking_llm for this complex analysis task
            response = self.deep_thinking_llm.invoke(prompt)
            llm_output_content = response.content

            # Basic cleaning: LLMs sometimes wrap JSON in backticks or add "json" prefix
            if llm_output_content.startswith("```json"):
                llm_output_content = llm_output_content[7:]
            if llm_output_content.endswith("```"):
                llm_output_content = llm_output_content[:-3]
            llm_output_content = llm_output_content.strip()

            parsed_advice = json.loads(llm_output_content)

            # Validate structure (basic validation)
            if not isinstance(parsed_advice, dict) or \
               "position_management" not in parsed_advice or \
               "new_trade_opportunity" not in parsed_advice or \
               not isinstance(parsed_advice["position_management"], list):
                raise ValueError("LLM output does not match expected structure.")

            # Further validation can be added here for individual items if needed

            return parsed_advice

        except Exception as e:
            print(f"Error during LLM call or parsing in get_portfolio_management_advice: {e}")
            # Fallback to placeholder/default advice
            placeholder_position_management = []
            for pos in detailed_open_positions:
                placeholder_position_management.append({
                    "symbol": pos["symbol"],
                    "action": "HOLD",
                    "reason": f"Error in LLM processing: {e}. Defaulting to HOLD.",
                })

            placeholder_new_trade = None
            if ticker_to_potentially_trade:
                placeholder_new_trade = {
                    "symbol": ticker_to_potentially_trade,
                    "decision": "NONE",
                    "conviction_score": 0.0,
                    "reason": f"Error in LLM processing: {e}. Defaulting to NONE.",
                }

            return {
                "position_management": placeholder_position_management,
                "new_trade_opportunity": placeholder_new_trade,
            }

    # Helper to get base context (market, news, fundamentals reports)
    def _get_base_research_context(self, symbol: str) -> Dict[str, str]:
        # This method attempts to use existing state or falls back to placeholders.
        # For a truly independent debate, this might need to trigger data fetching.
        # The original full debate likely happens after self.propagate() populates self.curr_state.
        if self.curr_state and self.curr_state.get('company_of_interest') == symbol:
            print(f"Using existing context from self.curr_state for {symbol} for debate.")
            return {
                "market_report": self.curr_state.get("market_report", f"Market report for {symbol} not available in curr_state."),
                "sentiment_report": self.curr_state.get("sentiment_report", f"Sentiment report for {symbol} not available in curr_state."),
                "news_report": self.curr_state.get("news_report", f"News report for {symbol} not available in curr_state."),
                "fundamentals_report": self.curr_state.get("fundamentals_report", f"Fundamentals report for {symbol} not available in curr_state."),
            }
        else:
            # This is a critical fallback. Ideally, if get_new_trade_decision is called for a new symbol,
            # it should first run the data gathering part of the graph (analyst nodes).
            # The user's description "This triggers all analysts and debates sequentially" for the new call
            # implies that get_new_trade_decision should indeed fetch this data.
            # For now, this is a simplified version. A more robust solution would involve
            # invoking the analyst nodes from self.graph for the given symbol if data isn't fresh.
            print(f"Warning: No current state for {symbol}. Analyst reports for debate will be placeholders or fetched ad-hoc if implemented.")
            print("Attempting to run initial graph propagation to gather data for new trade decision...")
            try:
                # Attempt to run the graph to populate self.curr_state for the new symbol
                # This is a simplified way to ensure data is present.
                # The propagate method itself returns final_state, signal
                # We are interested in self.curr_state being updated.
                self.propagate(company_name=symbol, trade_date=date.today().isoformat())
                if self.curr_state and self.curr_state.get('company_of_interest') == symbol:
                    print(f"Data gathered successfully for {symbol} via internal propagate call.")
                    return {
                        "market_report": self.curr_state.get("market_report", f"Market report for {symbol} not available post-propagate."),
                        "sentiment_report": self.curr_state.get("sentiment_report", f"Sentiment report for {symbol} not available post-propagate."),
                        "news_report": self.curr_state.get("news_report", f"News report for {symbol} not available post-propagate."),
                        "fundamentals_report": self.curr_state.get("fundamentals_report", f"Fundamentals report for {symbol} not available post-propagate."),
                    }
                else:
                    print(f"Failed to gather data for {symbol} via internal propagate call. curr_state not updated as expected.")
            except Exception as e:
                print(f"Error during ad-hoc data gathering for {symbol} in _get_base_research_context: {e}")

            # Fallback if propagation failed or didn't update as expected
            return {
                "market_report": f"Market report for {symbol} could not be fetched.",
                "sentiment_report": f"Sentiment report for {symbol} could not be fetched.",
                "news_report": f"News report for {symbol} could not be fetched.",
                "fundamentals_report": f"Fundamentals report for {symbol} could not be fetched.",
            }

    def _get_bull_thesis(self, symbol: str) -> str:
        context = self._get_base_research_context(symbol)
        prompt = f"""You are a Bull Analyst advocating for investing in {symbol}.
        Leverage the provided research:
        Market research: {context['market_report']}
        Sentiment: {context['sentiment_report']}
        News: {context['news_report']}
        Fundamentals: {context['fundamentals_report']}
        Build a strong, evidence-based case for a BUY decision. Emphasize growth, competitive advantages, and positive indicators for {symbol}.
        Output only your bullish thesis.
        """
        response = self.deep_thinking_llm.invoke(prompt)
        return response.content

    def _get_bear_thesis(self, symbol: str) -> str:
        context = self._get_base_research_context(symbol)
        prompt = f"""You are a Bear Analyst making the case against investing in {symbol}.
        Leverage the provided research:
        Market research: {context['market_report']}
        Sentiment: {context['sentiment_report']}
        News: {context['news_report']}
        Fundamentals: {context['fundamentals_report']}
        Build a strong, evidence-based case for a SELL or AVOID decision. Emphasize risks, challenges, and negative indicators for {symbol}.
        Output only your bearish thesis.
        """
        response = self.deep_thinking_llm.invoke(prompt)
        return response.content

    def _get_trader_decision_from_theses(self, symbol: str, bull_thesis: str, bear_thesis: str, portfolio_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        portfolio_json = json.dumps(portfolio_positions, indent=2)
        prompt = f"""You are a Trader deciding on a new trade for {symbol}.
        You have received the following analyses:
        Bullish Thesis: "{bull_thesis}"
        Bearish Thesis: "{bear_thesis}"

        Your Current Portfolio:
        {portfolio_json}

        Based on these theses and your current portfolio, provide a new trade decision for {symbol}.
        Output a JSON object with "symbol", "decision" ("BUY", "SELL", "NONE"), "conviction_score" (0.0-1.0), and "reason".
        Example: {{"symbol": "{symbol}", "decision": "BUY", "conviction_score": 0.75, "reason": "Bull thesis is compelling despite bear points, and it fits portfolio."}}
        If no trade, use "decision": "NONE", "conviction_score": 0.0.
        Provide only the JSON output.
        """
        response = self.deep_thinking_llm.invoke(prompt)
        try:
            llm_output_content = response.content
            if llm_output_content.startswith("```json"):
                llm_output_content = llm_output_content[7:]
            if llm_output_content.endswith("```"):
                llm_output_content = llm_output_content[:-3]
            llm_output_content = llm_output_content.strip()
            decision = json.loads(llm_output_content)
            if not all(k in decision for k in ("symbol", "decision", "conviction_score", "reason")):
                raise ValueError("Missing required keys in trader decision")
            return decision
        except Exception as e:
            print(f"Error parsing trader decision for {symbol}: {e}. LLM Output: {response.content}")
            return {"symbol": symbol, "decision": "NONE", "conviction_score": 0.0, "reason": f"Error in processing trader decision: {e}"}

    def get_new_trade_decision(self, symbol: str, detailed_open_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Orchestrates a multi-agent debate for a robust new-trade decision.
        This method now attempts to gather fresh analyst data by calling self.propagate if needed.
        """
        print(f"\n--- Initiating Multi-Agent Debate for New Trade on {symbol} ---")

        # This will attempt to run the graph if context for the symbol isn't already in self.curr_state
        # _get_base_research_context now calls self.propagate if necessary.

        # 1. Bull Agent’s bullish thesis
        print(f"--- Generating Bullish Thesis for {symbol}... ---")
        bull_research = self._get_bull_thesis(symbol) # _get_base_research_context is called within this
        print(f"Bull research for {symbol} obtained.")
        time.sleep(1) # Delay after LLM call

        # 2. Bear Agent’s bearish thesis
        print(f"--- Generating Bearish Thesis for {symbol}... ---")
        bear_research = self._get_bear_thesis(symbol) # _get_base_research_context is called within this
        print(f"Bear research for {symbol} obtained.")
        time.sleep(1) # Delay after LLM call

        # 3. Trader Agent’s final verdict
        print(f"--- Trader Agent Deliberating on {symbol}... ---")
        final_trade_decision = self._get_trader_decision_from_theses(
            symbol=symbol,
            bull_thesis=bull_research,
            bear_thesis=bear_research,
            portfolio_positions=detailed_open_positions
        )
        # Note: No time.sleep() needed after the last LLM call in this sequence.
        print(f"Trader agent final decision for {symbol} obtained.")

        print("--- Multi-Agent Debate Concluded ---")
        return final_trade_decision
