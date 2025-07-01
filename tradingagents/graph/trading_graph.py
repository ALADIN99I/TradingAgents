# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
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
