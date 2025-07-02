import alpaca_trade_api as tradeapi
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from datetime import datetime
import os
import time # Added import for time.sleep()
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

#os api method for now only
# IMPORTANT: This MUST be a real OpenAI API key for embeddings to work.

# --- Alpaca API Configuration ---
# WARNING: API keys are hardcoded below as per user request.
# This is NOT RECOMMENDED for security reasons. Prefer environment variables.
API_KEY = 'PKHNQQLTBXQFBIRW8T95'
API_SECRET = 'zPd6VAIzKhFFFVNyYFw79gSL4bnXerryeW4kfbMU'
# Use paper trading URL for testing, live trading URL is 'https://api.alpaca.markets'
BASE_URL = 'https://paper-api.alpaca.markets' # Defaulting to paper trading

# Connect to Alpaca
try:
    print(f"Attempting to connect to Alpaca with Key ID: {API_KEY[-4:]} at URL: {BASE_URL}")
    alpaca_api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL)
    data_client = StockHistoricalDataClient(API_KEY, API_SECRET) # Add this line
    # Check if the API connection is successful
    account_info = alpaca_api.get_account()
    print(f"Successfully connected to Alpaca. Account ID: {account_info.id}")
except Exception as e:
    print(f"Error connecting to Alpaca API: {e}")
    alpaca_api = None # Set to None if connection fails
    data_client = None # Also set data_client to None

# --- TradingAgents Configuration ---
# Default configuration uses OpenAI. Below shows how to configure for OpenRouter.
USE_OPENROUTER = True # Set to True to use OpenRouter configuration example

config = DEFAULT_CONFIG.copy()
config["online_tools"] = True  # Ensure live data is used

if USE_OPENROUTER:
    print("Configuring TradingAgentsGraph for OpenRouter...")
    # Ensure you have OPENROUTER_API_KEY set in your environment,
    # or set OPENAI_API_KEY to your OpenRouter key (e.g., "sk-or-v1-...").
    # The ChatOpenAI client might pick up OPENAI_API_KEY by default.
    # Alternatively, you might need to pass the api_key directly to ChatOpenAI if using OpenRouter's specific key format
    # and the library doesn't handle it automatically via OPENAI_API_KEY.
    # Check Langchain documentation for ChatOpenAI with custom providers like OpenRouter.

    config["llm_provider"] = "openrouter" # Ensure TradingAgentsGraph recognizes "openrouter"

    config["deep_think_llm"] = "deepseek/deepseek-chat-v3-0324:free"  # Example model
    config["quick_think_llm"] = "deepseek/deepseek-chat-v3-0324:free" # Example model
    config["embedding_llm"] = "openai/text-embedding-3-small" # <-- UPDATED MODEL NAME
    config["backend_url"] = "https://openrouter.ai/api/v1"
    # Important: You'll also need to have the OPENROUTER_API_KEY environment variable set,
    # or ensure OPENAI_API_KEY is set to your OpenRouter key, for the ChatOpenAI class to authenticate.
    print(f"LLM Provider: {config['llm_provider']}")
    print(f"Deep Think LLM: {config['deep_think_llm']}")
    print(f"Quick Think LLM: {config['quick_think_llm']}")
    print(f"Backend URL: {config['backend_url']}")
else:
    print("Using default LLM provider (ensure OPENAI_API_KEY and potentially FINNHUB_API_KEY are set).")
    # Default config uses OpenAI, ensure OPENAI_API_KEY is set.
    # Also, the TradingAgents framework uses Finnhub, so FINNHUB_API_KEY should be set.

trading_agent = TradingAgentsGraph(debug=True, config=config) # Enabled debug mode
print("TradingAgentsGraph initialized.")

def get_account_state(api): # Removed ticker parameter
    """Gets account balance and a list of all open positions."""
    if api is None:
        print("Alpaca API not initialized. Cannot get account state.")
        return 0.0, [] # Return empty list for positions


    equity = 0.00
    try:
        account = api.get_account()
        # Use portfolio_value instead of equity
        equity = float(account.portfolio_value)
    except Exception as e:
        print(f"Error fetching account information: {e}")
        equity = 0.0 # Default to 0 if account info cannot be fetched

    open_positions = []
    try:
        open_positions = api.list_positions()
    except Exception as e:
        print(f"Error fetching open positions: {e}")
        open_positions = [] # Default to empty list on error

    return equity, open_positions

def calculate_position_size(equity, last_price, conviction_score):
    """
    Calculates the position size based on equity, price, and a conviction score.
    """
    # Determine risk_per_trade based on conviction_score
    if 0.8 <= conviction_score <= 1.0:
        risk_per_trade = 0.02  # High Conviction: 2.0%
    elif 0.6 <= conviction_score < 0.8:
        risk_per_trade = 0.015  # Medium-High Conviction: 1.5%
    elif 0.4 <= conviction_score < 0.6:
        risk_per_trade = 0.01  # Medium Conviction: 1.0%
    elif 0.2 <= conviction_score < 0.4:
        risk_per_trade = 0.005  # Low Conviction (Scout Mode): 0.5%
    else: # conviction_score < 0.2
        risk_per_trade = 0.0   # Very Low Conviction / Disagreement: 0%

    if risk_per_trade == 0.0:
        return 0 # No trade if risk is 0

    dollars_to_risk = equity * risk_per_trade
    if last_price <= 0: # Prevent division by zero or negative price
        return 0
    position_size = dollars_to_risk / last_price
    return int(position_size)

def execute_trade_logic(api, data_api, agent, ticker, decision, conviction_score, equity, position, management_action=None, quantity_for_action=None): # PARAMETER CHANGE
    """
    Handles trade execution for opening new positions or managing existing ones.
    - For new trades: uses `decision` ("BUY"/"SELL") and `conviction_score`.
    - For managing existing positions: uses `management_action` ("CLOSE"/"REDUCE"/"ADD") and `quantity_for_action`.
    """
    #log import
    import logging
    log = logging.getLogger(__name__) #log definition

    # Parameters:
    # decision: "BUY", "SELL" (primarily for new trades)
    # conviction_score: float (for new trades, or if agent provides it for adjustments)
    # equity: float
    # position: Alpaca position object (if managing an existing one), else None
    # management_action: "CLOSE", "REDUCE", "ADD", "NONE" (or None) - specific for managing existing positions
    # quantity_for_action: int/float (for REDUCE/ADD actions on existing positions)

    log_message_parts = [f"Executing trade logic for {ticker}"]
    if management_action and management_action != "NONE":
        log_message_parts.append(f"Management Action: {management_action}")
        if quantity_for_action:
            log_message_parts.append(f"Quantity: {quantity_for_action}")
    else:
        log_message_parts.append(f"Decision for New Trade: {decision}")
        log_message_parts.append(f"Conviction: {conviction_score:.2f}")

    log_message_parts.append(f"Equity: {equity:.2f}")
    log_message_parts.append(f"Existing Position: {position is not None}")
    log.info(". ".join(log_message_parts))

    try:
        # Price fetching is needed for new trades or if logic requires current price for management decisions
        # For simple CLOSE/REDUCE by quantity, last_price might not be strictly needed for the order itself,
        # but calculate_position_size (for new trades) does.
        last_price = None
        # Fetch price if needed for new trades or ADD operations.
        # Also ensure data_api is valid.
        if (not management_action or management_action in ["NONE", "ADD"]) and data_api:
            request_params = StockBarsRequest(
                                symbol_or_symbols=[ticker],
                                timeframe=TimeFrame.Minute,
                                limit=1)
            barset = data_api.get_stock_bars(request_params)
            if barset and barset[ticker]:
                last_price = barset[ticker][0].close
            else:
                log.error(f"Could not get last price for {ticker}. Cannot execute new BUY/SELL or ADD action.")
                return
        elif (not management_action or management_action in ["NONE", "ADD"]) and not data_api:
             log.error(f"data_api not available for price fetching in execute_trade_logic for {ticker}. Cannot proceed.")
             return

        # ---- Management Actions on Existing Positions ----
        if management_action and management_action != "NONE":
            if not position:
                log.warning(f"Management action '{management_action}' requested for {ticker}, but no existing position found. Ignoring.")
                return

            if management_action == "HOLD":
                log.info(f"Management Action: HOLD for {ticker}. No changes made.")
            elif management_action == "CLOSE":
                log.info(f"Management Action: CLOSE for {ticker}. Attempting to close position.")
                try:
                    # 1. Fetch current position
                    position_to_close = api.get_position(ticker)
                    qty_to_sell = position_to_close.qty

                    # 2. Submit a market sell order to close
                    close_order = api.submit_order(
                        symbol=ticker,
                        qty=qty_to_sell,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    log.info(f"Market SELL order to close {qty_to_sell} shares of {ticker} placed. Order ID: {close_order.id}")
                except Exception as e:
                    log.error(f"Error trying to CLOSE position in {ticker}: {e}")
            elif management_action == "REDUCE":
                if quantity_for_action and quantity_for_action > 0:
                    current_qty = float(position.qty)
                    if quantity_for_action >= current_qty:
                        log.info(f"Management Action: REDUCE quantity {quantity_for_action} for {ticker} is >= current quantity {current_qty}. Changing to CLOSE.")
                        try:
                            closed_order = api.close_position(ticker)
                            log.info(f"Submitted order to CLOSE (due to REDUCE full amount) position in {ticker}. Order ID: {closed_order.id}")
                        except Exception as e:
                            log.error(f"Error trying to CLOSE (due to REDUCE full amount) position in {ticker}: {e}")
                    else:
                        log.info(f"Management Action: REDUCE for {ticker}. Attempting to sell {quantity_for_action} shares.")
                        try:
                            reduce_order = api.submit_order(
                                symbol=ticker,
                                qty=quantity_for_action,
                                side='sell',
                                type='market',
                                time_in_force='day'
                            )
                            log.info(f"Submitted order to REDUCE position in {ticker} by {quantity_for_action} shares. Order ID: {reduce_order.id}")
                        except Exception as e:
                            log.error(f"Error trying to REDUCE position in {ticker} by {quantity_for_action} shares: {e}")
                else:
                    log.warning(f"Management Action: REDUCE for {ticker} but quantity_for_action is invalid ({quantity_for_action}). No action taken.")
            elif management_action == "ADD":
                if quantity_for_action and quantity_for_action > 0:
                    log.info(f"Management Action: ADD for {ticker}. Attempting to buy {quantity_for_action} additional shares.")
                    try:
                        # Note: For ADD, last_price might be useful if conviction changes sizing,
                        # but here we assume quantity_for_action is directly provided.
                        # If last_price was fetched earlier (e.g., because management_action was initially "ADD"
                        # in the conditional price fetching), it could be used for logging or other checks.
                        add_order = api.submit_order(
                            symbol=ticker,
                            qty=quantity_for_action,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        log.info(f"Submitted order to ADD {quantity_for_action} shares to position in {ticker}. Order ID: {add_order.id}")
                    except Exception as e:
                        log.error(f"Error trying to ADD {quantity_for_action} shares to position in {ticker}: {e}")
                else:
                    log.warning(f"Management Action: ADD for {ticker} but quantity_for_action is invalid ({quantity_for_action}). No action taken.")
            else:
                log.info(f"Management Action: {management_action} for {ticker} - (Execution logic pending for this action type).")

        # ---- Opening New Positions ----
        # (Only if no specific management_action is given, or it's "NONE", and no existing position)
        elif (not management_action or management_action == "NONE") and position is None:
            if decision == "BUY":
                if last_price is None: log.error(f"Cannot BUY {ticker} (new), last_price not fetched."); return
                log.info(f"Attempting to BUY new position in {ticker} with conviction {conviction_score}")
                qty = calculate_position_size(equity, last_price, conviction_score)
                log.info(f"Calculated position size for new BUY: {qty} shares of {ticker}")
                if qty > 0:
                    api.submit_order(symbol=ticker, qty=qty, side='buy', type='market', time_in_force='day')
                    log.info(f"Market BUY order for {qty} shares of {ticker} placed.")
                else:
                    log.info(f"Position size is 0 for new BUY on {ticker}. No trade.")
            elif decision == "SELL": # For short selling a new position
                if last_price is None: log.error(f"Cannot SELL {ticker} (new short), last_price not fetched."); return
                log.info(f"Attempting to SELL (short) new position in {ticker} with conviction {conviction_score}")
                qty = calculate_position_size(equity, last_price, conviction_score)
                log.info(f"Calculated position size for new SELL (short): {qty} shares of {ticker}")
                if qty > 0:
                    api.submit_order(symbol=ticker, qty=qty, side='sell', type='market', time_in_force='day')
                    log.info(f"Market SELL (short) order for {qty} shares of {ticker} placed.")
                else:
                    log.info(f"Position size is 0 for new SELL (short) on {ticker}. No trade.")
            elif decision == "HOLD":
                log.info(f"Decision is HOLD for new trade on {ticker}. No action taken.")
            else: # Should not happen if decision is always BUY/SELL/HOLD
                log.warning(f"Unknown decision '{decision}' for new trade on {ticker}.")

        # ---- Fallback/Default Behavior for Existing Positions if no specific management_action ----
        elif position is not None and (not management_action or management_action == "NONE"):
            log.info(f"No specific management action for existing position in {ticker}. Decision was '{decision}'. Defaulting to HOLD.")
            # Previously, a 'SELL' decision here would close. Now, explicit 'CLOSE' management_action is preferred.

    except Exception as e:
        log.error(f"Error in trade logic for {ticker}: {e}")

def test_buy_logic_simulation(data_api, ticker, equity, conviction_score): # PARAMETER CHANGE
    #log import
    import logging # Ensure log is available if not globally configured for this scope
    log = logging.getLogger(__name__)
    log.info(f"[SIMULATION] Testing BUY logic for {ticker} with equity {equity} and conviction {conviction_score}")

    try:
        # 1. Get Price (as in execute_trade_logic)
        request_params = StockBarsRequest(
                            symbol_or_symbols=[ticker],
                            timeframe=TimeFrame.Minute,
                            limit=1
                         )
        barset = data_api.get_stock_bars(request_params)
        if barset and barset[ticker]:
            last_price = barset[ticker][0].close # Use .close instead of .c
            log.info(f"[SIMULATION] Fetched last price: {last_price} for {ticker}")
        else:
            log.error(f"[SIMULATION] Could not get last price for {ticker}. No bar data found.")
            return

        # 2. Calculate Position Size (as in execute_trade_logic)
        qty = calculate_position_size(equity, last_price, conviction_score) # PASS conviction_score
        log.info(f"[SIMULATION] Calculated position size: {qty} shares of {ticker} (Conviction: {conviction_score})")

        if qty > 0:
            log.info(f"[SIMULATION] Would attempt to submit BUY order for {qty} shares of {ticker}.")
            # IMPORTANT: Actual api.submit_order(...) is NOT called here
        else:
            log.info(f"[SIMULATION] Position size is 0 for {ticker}. No trade would be executed.")

    except Exception as e:
        log.error(f"[SIMULATION] Error during BUY logic simulation for {ticker}: {e}")

def run_trading_cycle(ticker_to_potentially_trade): # RENAMED and parameter clarified
    """Runs a trading cycle: reviews open positions and considers new trades."""
    if alpaca_api is None or data_client is None: # check for data_client
        print(f"Cannot run trading cycle for {ticker_to_potentially_trade}: Alpaca API not connected.")
        return

    print(f"\n--- Starting trading cycle on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    # 1. Get Overall Account State and Open Positions
    current_equity, open_positions = get_account_state(alpaca_api)
    print(f"Current Equity: ${current_equity:.2f}. Open Positions: {len(open_positions)}")

    if current_equity <= 0 and not open_positions:
        print(f"Equity is ${current_equity:.2f} and no open positions. Cannot make new trades.")
        print("--- Trading cycle complete ---")
        return

    # 2. Gather Detailed Data for Existing Open Positions
    print("\n--- Gathering Details for Open Positions ---")
    detailed_open_positions = []
    if not open_positions:
        print("No open positions.")
    else:
        for position in open_positions:
            try:
                request_params = StockBarsRequest(
                                    symbol_or_symbols=[position.symbol],
                                    timeframe=TimeFrame.Minute,
                                    limit=1
                                 )
                barset = data_client.get_stock_bars(request_params)
                current_price = float(position.current_price) # Fallback
                if barset and barset[position.symbol]:
                    current_price = barset[position.symbol][0].close
                else:
                    print(f"Warning: Could not fetch latest bar for {position.symbol}. Using position.current_price.")

                position_details = {
                    "symbol": position.symbol, "qty": float(position.qty),
                    "avg_entry_price": float(position.avg_entry_price), "market_price": current_price,
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_pl_pct": float(position.unrealized_plpc) * 100,
                }
                detailed_open_positions.append(position_details)
                print(f"  {position.symbol}: Qty {position_details['qty']}, Entry ${position_details['avg_entry_price']:.2f}, "
                      f"Mkt ${position_details['market_price']:.2f}, UPL ${position_details['unrealized_pl']:.2f} ({position_details['unrealized_pl_pct']:.2f}%)")
            except Exception as e:
                print(f"Error processing position {position.symbol} for detailed view: {e}")
                # Fallback to basic info if API call fails
                detailed_open_positions.append({"symbol": position.symbol, "qty": float(position.qty), "avg_entry_price": float(position.avg_entry_price), "market_price": float(position.current_price), "unrealized_pl": float(position.unrealized_pl), "unrealized_pl_pct": float(position.unrealized_plpc) * 100})

    # 3. Get Live Agent Portfolio Advice (New Trade Decision + Position Management)
    print("\n--- Running Full Agent Debate Pipeline for New Trade Decision ---")
    new_trade_decision = None
    position_management_advice = []
    agent_advice = {} # Initialize agent_advice

    try:
        # This triggers all analysts and debates sequentially (can take time)
        # The get_new_trade_decision method now takes detailed_open_positions for context
        new_trade_decision = trading_agent.get_new_trade_decision(ticker_to_potentially_trade, detailed_open_positions)
    except Exception as e:
        print(f"Error getting new trade decision from TradingAgentsGraph: {e}")
        print("Falling back to no new trade action for this cycle.")
        new_trade_decision = {
            "symbol": ticker_to_potentially_trade,
            "decision": "NONE",
            "conviction_score": 0.0,
            "reason": f"Error in get_new_trade_decision: {e}"
        }

    print("\n--- Getting Position Management Advice ---")
    try:
        # Get position management advice separately
        management_advice_full = trading_agent.get_portfolio_management_advice(
            detailed_open_positions,
            ticker_to_potentially_trade # Pass ticker for context, though it's mainly for open positions
        )
        if management_advice_full and "position_management" in management_advice_full:
            position_management_advice = management_advice_full["position_management"]
        else:
            print("Warning: No 'position_management' key in advice from get_portfolio_management_advice.")
            position_management_advice = []

    except AttributeError as ae:
        print(f"CRITICAL ERROR: `trading_agent` does not have the method `get_portfolio_management_advice`. This method needs to be implemented in TradingAgentsGraph. {ae}")
        position_management_advice = []
    except Exception as e:
        print(f"Error getting position management advice from TradingAgentsGraph: {e}")
        print("Falling back to no management actions for this cycle.")
        position_management_advice = []

    # Combine into the agent_advice structure
    agent_advice = {
        "position_management": position_management_advice,
        "new_trade_opportunity": new_trade_decision
    }

    # 4. Process Management Actions for Open Positions
    print("\n--- Processing Management Actions ---")
    # Use the locally scoped position_management_advice which has been error handled
    if position_management_advice:
        for advice in agent_advice["position_management"]:
            # Add this check to validate the advice object
            if not all(k in advice for k in ("symbol", "action")):
                print(f"Warning: Skipping malformed advice object from LLM: {advice}")
                continue

            log_msg_parts = [
                f"Agent advises for {advice['symbol']}: Action: {advice['action']}"
            ]
            if 'quantity' in advice:
                log_msg_parts.append(f"Quantity: {advice['quantity']}")
            if 'reason' in advice:
                log_msg_parts.append(f"Reason: {advice['reason']}")
            print(". ".join(log_msg_parts))

            advised_action = advice['action'].upper() # Normalize to uppercase
            advised_symbol = advice['symbol']

            # Find the corresponding detailed position object and original Alpaca position object
            current_detailed_position = next((p for p in detailed_open_positions if p['symbol'] == advised_symbol), None)
            current_alpaca_position_obj = next((p_obj for p_obj in open_positions if p_obj.symbol == advised_symbol), None)

            if not current_alpaca_position_obj:
                print(f"Warning: Agent advised action for {advised_symbol}, but no open position object found. Skipping.")
                continue

            if advised_action in ["CLOSE", "REDUCE"]:
                quantity_for_action = advice.get('quantity') if advised_action == "REDUCE" else None
                if advised_action == "REDUCE" and (not isinstance(quantity_for_action, (int, float)) or quantity_for_action <= 0):
                    print(f"Warning: Invalid or missing quantity for REDUCE action on {advised_symbol}. Skipping.")
                    continue

                # For management actions, 'decision' and 'conviction_score' are less relevant unless agent provides new ones for adjustment.
                # Passing None or default values for them.
                execute_trade_logic(
                    api=alpaca_api,
                    data_api=data_client,
                    agent=trading_agent, # `agent` object might be used by execute_trade_logic if it evolves
                    ticker=advised_symbol,
                    decision=None, # Not a new trade decision
                    conviction_score=0, # Not directly applicable for pre-decided management action qty
                    equity=current_equity,
                    position=current_alpaca_position_obj, # Crucial: pass the existing position object
                    management_action=advised_action,
                    quantity_for_action=quantity_for_action
                )
            elif advised_action == "ADD":
                quantity_to_add = advice.get('quantity')
                if isinstance(quantity_to_add, (int, float)) and quantity_to_add > 0:
                    execute_trade_logic(
                        api=alpaca_api,
                        data_api=data_client,
                        agent=trading_agent,
                        ticker=advised_symbol,
                        decision=None, # Not a new trade decision from this path
                        conviction_score=0, # Agent might provide a new one for ADD, but mock doesn't yet
                        equity=current_equity,
                        position=current_alpaca_position_obj, # Pass existing position
                        management_action="ADD",
                        quantity_for_action=quantity_to_add
                    )
                else:
                    print(f"Warning: Invalid or missing quantity for ADD action on {advised_symbol}. Skipping.")
            elif advised_action == "HOLD":
                print(f"Agent advises HOLD for {advised_symbol}. No execution needed.")
            else:
                print(f"Unknown management action '{advised_action}' for {advised_symbol}. Skipping.")
    else:
        print("No management actions advised for open positions.")

    # 5. Process New Trade Opportunity (if any)
    # --- Considering New Trade Opportunity ---
    print(f"\n--- Considering New Trade Opportunity for {ticker_to_potentially_trade} ---")
    # The new_trade_decision should already be fetched from agent_advice or directly if agent_advice structure is different
    new_trade_decision = agent_advice.get("new_trade_opportunity") # Assuming it's here from earlier logic

    if not new_trade_decision or not new_trade_decision.get("symbol"): # Ensure new_trade_decision and symbol exist
        print(f"No valid new trade decision available for {ticker_to_potentially_trade}.")
    else:
        decision = new_trade_decision.get("decision")
        symbol = new_trade_decision.get("symbol") # Use symbol from the decision
        conviction = new_trade_decision.get("conviction_score", 0) # Use conviction_score, default to 0
        reason = new_trade_decision.get("reason", "No reason provided.")

        # Ensure the symbol from the decision matches the ticker_to_potentially_trade if that's intended,
        # or clarify if the agent can suggest trades for symbols other than ticker_to_potentially_trade.
        # For now, proceeding with `symbol` from the agent's decision.
        if symbol != ticker_to_potentially_trade:
            print(f"Note: Agent provided new trade decision for {symbol}, while current cycle focuses on {ticker_to_potentially_trade}. Proceeding with {symbol}.")

        if decision == "BUY" and symbol: # Check if symbol is not None or empty
            print(f"Agent suggests NEW trade: {decision} {symbol} with Conviction {conviction:.2f}. Reason: {reason}")

            existing_position_obj = None
            try:
                # Use alpaca_api (the tradeapi.REST instance) to get position
                existing_position_obj = alpaca_api.get_position(symbol)
            except tradeapi.rest.APIError as e: # Catch specific Alpaca APIError for "position does not exist"
                if e.status_code == 404: # Not found
                    pass # Position does not exist, existing_position_obj remains None
                else:
                    print(f"APIError when checking for existing position for {symbol}: {e}") # Log other API errors
                    existing_position_obj = None # Treat as no position on other errors
            except Exception as e:
                print(f"Generic error when checking for existing position for {symbol}: {e}")
                existing_position_obj = None # Treat as no position on other errors


            if existing_position_obj:
                # If position exists, treat the BUY as an ADD
                print(f"Position for {symbol} already exists. Treating BUY signal as an ADD action.")
                # We need the last price to calculate quantity for ADD.
                last_price_for_add = None
                try:
                    request_params = StockBarsRequest(
                                        symbol_or_symbols=[symbol],
                                        timeframe=TimeFrame.Minute, # Or appropriate timeframe
                                        limit=1)
                    barset = data_client.get_stock_bars(request_params) # Use data_client
                    if barset and barset[symbol]:
                        last_price_for_add = barset[symbol][0].close
                    else:
                        print(f"Could not get last price for {symbol} to calculate ADD quantity. No action taken.")
                except Exception as e:
                    print(f"Error fetching price for {symbol} to calculate ADD quantity: {e}. No action taken.")

                if last_price_for_add:
                    # Use the calculate_position_size function for consistency in sizing,
                    # even though it's an ADD. It takes conviction into account.
                    # Or, if the agent provides a specific quantity for ADD, use that.
                    # Assuming new BUY signal's conviction applies to the ADD.
                    add_quantity = calculate_position_size(current_equity, last_price_for_add, conviction) # Corrected parameter order
                    if add_quantity > 0:
                        print(f"Calculated quantity to ADD for {symbol}: {add_quantity}")
                        execute_trade_logic( # Call execute_trade_logic for ADD
                            api=alpaca_api,
                            data_api=data_client,
                            agent=trading_agent, # Pass the agent
                            ticker=symbol,
                            decision=None, # Not a new BUY, it's a management action
                            conviction_score=conviction, # Pass conviction for potential use in logic or logging
                            equity=current_equity,
                            position=existing_position_obj, # Pass the existing position object
                            management_action="ADD",
                            quantity_for_action=add_quantity
                        )
                    else:
                        print(f"Calculated quantity to add for {symbol} is zero (Conviction: {conviction:.2f}, Price: {last_price_for_add}). No action taken.")
                else:
                    print(f"Cannot ADD to {symbol} as last price could not be fetched.")
            else:
                # If no position exists, execute it as a new BUY
                print(f"Executing new BUY trade for {symbol}.")
                # For a new BUY, pass None for 'position' and 'management_action'.
                # 'decision' will be "BUY".
                execute_trade_logic(
                    api=alpaca_api,
                    data_api=data_client,
                    agent=trading_agent, # Pass the agent
                    ticker=symbol,
                    decision="BUY",
                    conviction_score=conviction,
                    equity=current_equity,
                    position=None, # No existing position
                    management_action=None # Not a management action on an existing position
                    # quantity_for_action is not needed for a new BUY handled by calculate_position_size internally
                )
        # Handling for "SELL" (new short) can remain similar if that's intended
        elif decision == "SELL" and symbol: # Check if symbol is not None or empty
            print(f"Agent suggests NEW trade: {decision} {symbol} with Conviction {conviction:.2f}. Reason: {reason}")
            # Check if a position in this symbol already exists (could be long or short)
            existing_position_obj = None
            try:
                existing_position_obj = alpaca_api.get_position(symbol)
            except tradeapi.rest.APIError as e:
                if e.status_code == 404: pass
                else: print(f"APIError checking position for new SELL {symbol}: {e}")
            except Exception as e:
                print(f"Error checking position for new SELL {symbol}: {e}")

            if existing_position_obj:
                # If a position already exists (e.g., it's long), a new "SELL" signal might mean
                # "close the long and go short" or "reduce the long".
                # This part of the logic might need more sophistication based on strategy.
                # For now, let's assume if a position exists, management advice should have handled it.
                # Or, if the agent explicitly says "SELL" for a new trade despite an existing one,
                # it might imply a desire to flip or short against an existing long.
                # This is complex. For simplicity, we'll print a message.
                print(f"Agent suggests new SELL for {symbol}, but a position already exists. Current logic defers to position management for existing assets.")
                # Potentially, one could trigger a CLOSE here, then a new SELL (short) if conviction is high.
                # execute_trade_logic(..., management_action="CLOSE", ...)
                # execute_trade_logic(..., decision="SELL", ...)
            else:
                # If no position exists, execute it as a new SELL (short)
                print(f"Executing new SELL (short) trade for {symbol}.")
                execute_trade_logic(
                    api=alpaca_api,
                    data_api=data_client,
                    agent=trading_agent,
                    ticker=symbol,
                    decision="SELL",
                    conviction_score=conviction,
                    equity=current_equity,
                    position=None, # No existing position
                    management_action=None
                )
        else:
            print(f"Agent advises no new high-conviction trade for {symbol if symbol else ticker_to_potentially_trade} (Decision: {decision}, Conviction: {conviction:.2f}).")


    print(f"\n--- Trading cycle complete for {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")


if __name__ == "__main__":
    # --- To run the script ---
    # Ensure API keys are set as environment variables or replace placeholders above.
    # Example: export ALPACA_API_KEY='YOUR_KEY'
    #          export ALPACA_API_SECRET='YOUR_SECRET'
    # You might also need to set OPENAI_API_KEY and FINNHUB_API_KEY for the TradingAgents framework.


    ticker_to_trade = "GOOGL"  # Example: Trade SPDR S&P 500 ETF Trust
    # You can add more tickers to trade in a loop or manage a portfolio
    # For example:
    # portfolio_tickers = ["AAPL", "MSFT", "GOOGL"]
    # for ticker in portfolio_tickers:
    #     run_daily_trading_session(ticker)

    if API_KEY == 'YOUR_API_KEY_HERE' or API_SECRET == 'YOUR_API_SECRET_HERE':
        print("\nWARNING: Alpaca API keys are placeholders. Please set them environment variables (ALPACA_API_KEY, ALPACA_API_SECRET) or directly in the script for testing.")
        print("Using placeholders will likely result in authentication errors with the Alpaca API.")

    if alpaca_api is not None and data_client is not None: # Only run if API connection was successful
        # print("\n--- Running Simulated Buy Logic Test ---")
        # test_equity = 50000 # Example equity, adjust as needed
        # test_conviction = 0.85 # Example: High conviction for testing
        # # Pass the data_client and conviction_score to the simulation function
        # test_buy_logic_simulation(data_client, ticker_to_trade, test_equity, test_conviction)
        # print("--- Simulated Buy Logic Test Complete ---\n")

        # When you run the actual session, you'll need to pass both clients
        # Continuous trading loop
        while True:
            print(f"\n{'='*20} Starting New Trading Cycle {'='*20}")
            run_trading_cycle(ticker_to_trade) # RENAMED
            sleep_duration = 15 * 60  # 15 minutes
            print(f"Cycle complete. Sleeping for {sleep_duration / 60} minutes...")
            time.sleep(sleep_duration)
    else:
        print("\nCannot start trading session: Alpaca API connection failed at initialization.")

    print("\nScript execution finished.")
    print("Remember to manage your API keys securely and test thoroughly with a paper trading account first.")
    print("This script is for educational purposes and should be reviewed carefully before use with real funds.")
