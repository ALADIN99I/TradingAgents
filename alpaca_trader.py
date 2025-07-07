import os

import alpaca_trade_api as tradeapi
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from datetime import datetime

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
    trade_executed_successfully = False # Initialize a flag

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
                return False # Return False on failure
        elif (not management_action or management_action in ["NONE", "ADD"]) and not data_api:
             log.error(f"data_api not available for price fetching in execute_trade_logic for {ticker}. Cannot proceed.")
             return False # Return False on failure

        # ---- Management Actions on Existing Positions ----
        if management_action and management_action != "NONE":
            if not position:
                log.warning(f"Management action '{management_action}' requested for {ticker}, but no existing position found. Ignoring.")
                return False # Return False

            if management_action == "HOLD":
                log.info(f"Management Action: HOLD for {ticker}. No changes made.")
                # No trade executed, but not an error in execution path itself.
                # Depending on definition, this could be True (logic completed) or False (no order placed).
                # For "acted_in_cycle", False is more appropriate as no order was submitted.
                trade_executed_successfully = False
            elif management_action == "CLOSE":
                log.info(f"Management Action: CLOSE for {ticker}. Attempting to close position.")
                try:
                    position_to_close = api.get_position(ticker)
                    qty_to_sell = position_to_close.qty
                    close_order = api.submit_order(
                        symbol=ticker,
                        qty=qty_to_sell,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    log.info(f"Market SELL order to close {qty_to_sell} shares of {ticker} placed. Order ID: {close_order.id}")
                    trade_executed_successfully = True
                except Exception as e:
                    log.error(f"Error trying to CLOSE position in {ticker}: {e}")
                    trade_executed_successfully = False
            elif management_action == "REDUCE":
                if quantity_for_action and quantity_for_action > 0:
                    current_qty = float(position.qty)
                    if quantity_for_action >= current_qty:
                        log.info(f"Management Action: REDUCE quantity {quantity_for_action} for {ticker} is >= current quantity {current_qty}. Changing to CLOSE.")
                        try:
                            closed_order = api.close_position(ticker)
                            log.info(f"Submitted order to CLOSE (due to REDUCE full amount) position in {ticker}. Order ID: {closed_order.id}")
                            trade_executed_successfully = True
                        except Exception as e:
                            log.error(f"Error trying to CLOSE (due to REDUCE full amount) position in {ticker}: {e}")
                            trade_executed_successfully = False
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
                            trade_executed_successfully = True
                        except Exception as e:
                            log.error(f"Error trying to REDUCE position in {ticker} by {quantity_for_action} shares: {e}")
                            trade_executed_successfully = False
                else:
                    log.warning(f"Management Action: REDUCE for {ticker} but quantity_for_action is invalid ({quantity_for_action}). No action taken.")
                    trade_executed_successfully = False
            elif management_action == "ADD":
                if quantity_for_action and quantity_for_action > 0:
                    log.info(f"Management Action: ADD for {ticker}. Attempting to buy {quantity_for_action} additional shares.")
                    try:
                        add_order = api.submit_order(
                            symbol=ticker,
                            qty=quantity_for_action,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        log.info(f"Submitted order to ADD {quantity_for_action} shares to position in {ticker}. Order ID: {add_order.id}")
                        trade_executed_successfully = True
                    except Exception as e:
                        log.error(f"Error trying to ADD {quantity_for_action} shares to position in {ticker}: {e}")
                        trade_executed_successfully = False
                else:
                    log.warning(f"Management Action: ADD for {ticker} but quantity_for_action is invalid ({quantity_for_action}). No action taken.")
                    trade_executed_successfully = False
            else:
                log.info(f"Management Action: {management_action} for {ticker} - (Execution logic pending for this action type).")
                trade_executed_successfully = False

        # ---- Opening New Positions ----
        elif (not management_action or management_action == "NONE") and position is None:
            if decision == "BUY":
                if last_price is None: log.error(f"Cannot BUY {ticker} (new), last_price not fetched."); return False
                log.info(f"Attempting to BUY new position in {ticker} with conviction {conviction_score}")
                qty = calculate_position_size(equity, last_price, conviction_score)
                log.info(f"Calculated position size for new BUY: {qty} shares of {ticker}")
                if qty > 0:
                    try:
                        api.submit_order(symbol=ticker, qty=qty, side='buy', type='market', time_in_force='day')
                        log.info(f"Market BUY order for {qty} shares of {ticker} placed.")
                        trade_executed_successfully = True
                    except Exception as e:
                        log.error(f"Error submitting BUY order for {ticker}: {e}")
                        trade_executed_successfully = False
                else:
                    log.info(f"Position size is 0 for new BUY on {ticker}. No trade.")
                    trade_executed_successfully = False # No trade attempted
            elif decision == "SELL": # For short selling a new position
                if last_price is None: log.error(f"Cannot SELL {ticker} (new short), last_price not fetched."); return False
                log.info(f"Attempting to SELL (short) new position in {ticker} with conviction {conviction_score}")
                qty = calculate_position_size(equity, last_price, conviction_score)
                log.info(f"Calculated position size for new SELL (short): {qty} shares of {ticker}")
                if qty > 0:
                    try:
                        api.submit_order(symbol=ticker, qty=qty, side='sell', type='market', time_in_force='day')
                        log.info(f"Market SELL (short) order for {qty} shares of {ticker} placed.")
                        trade_executed_successfully = True
                    except Exception as e:
                        log.error(f"Error submitting SELL (short) order for {ticker}: {e}")
                        trade_executed_successfully = False
                else:
                    log.info(f"Position size is 0 for new SELL (short) on {ticker}. No trade.")
                    trade_executed_successfully = False # No trade attempted
            elif decision == "HOLD":
                log.info(f"Decision is HOLD for new trade on {ticker}. No action taken.")
                trade_executed_successfully = False # No trade attempted
            else: # Should not happen if decision is always BUY/SELL/HOLD
                log.warning(f"Unknown decision '{decision}' for new trade on {ticker}.")
                trade_executed_successfully = False

        # ---- Fallback/Default Behavior for Existing Positions if no specific management_action ----
        elif position is not None and (not management_action or management_action == "NONE"):
            log.info(f"No specific management action for existing position in {ticker}. Decision was '{decision}'. Defaulting to HOLD.")
            trade_executed_successfully = False # No trade attempted

    except Exception as e:
        log.error(f"Error in trade logic for {ticker}: {e}")
        trade_executed_successfully = False

    return trade_executed_successfully

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

    # Keep track of any ticker acted upon in this cycle
    acted_in_cycle = set()

    # 3. Get Live Agent Portfolio Advice (New Trade Decision + Position Management)
    #    Order of operations:
    #    a. Get "New Trade Opportunity" decision first.
    #    b. Execute it if applicable and record the symbol.
    #    c. Then, get "Position Management Advice".
    #    d. When processing management advice, skip symbols already acted upon.

    # --- Considering New Trade Opportunity ---
    print(f"\n--- Considering New Trade Opportunity for {ticker_to_potentially_trade} ---")
    new_trade_decision_result = None
    try:
        new_trade_decision_result = trading_agent.get_new_trade_decision(ticker_to_potentially_trade, detailed_open_positions)
    except Exception as e:
        print(f"Error getting new trade decision from TradingAgentsGraph: {e}")
        new_trade_decision_result = {
            "symbol": ticker_to_potentially_trade, "decision": "NONE", "conviction_score": 0.0, "reason": f"Error: {e}"
        }

    if new_trade_decision_result and new_trade_decision_result.get("symbol"):
        decision = new_trade_decision_result.get("decision", "NONE").upper()
        symbol_of_new_trade = new_trade_decision_result.get("symbol")
        conviction = new_trade_decision_result.get("conviction_score", 0.0)
        reason = new_trade_decision_result.get("reason", "No reason provided.")

        print(f"Agent's New Trade Suggestion: {decision} {symbol_of_new_trade} (Conviction: {conviction:.2f}). Reason: {reason}")

        if decision in ["BUY", "SELL"]: # Only proceed if BUY or SELL
            # Check if this symbol is already in open_positions to decide if it's truly "new" or an "ADD/REDUCE"
            existing_position_obj = next((p for p in open_positions if p.symbol == symbol_of_new_trade), None)
            trade_executed = False

            if decision == "BUY":
                if existing_position_obj:
                    print(f"New trade BUY signal for existing position {symbol_of_new_trade}. Treating as ADD.")
                    # Calculate ADD quantity based on conviction (similar to new BUY)
                    last_price_for_add = None
                    try:
                        request_params = StockBarsRequest(symbol_or_symbols=[symbol_of_new_trade], timeframe=TimeFrame.Minute, limit=1)
                        barset = data_client.get_stock_bars(request_params)
                        if barset and barset[symbol_of_new_trade]: last_price_for_add = barset[symbol_of_new_trade][0].close
                    except Exception as e: print(f"Error fetching price for ADD on {symbol_of_new_trade}: {e}")

                    if last_price_for_add:
                        add_quantity = calculate_position_size(current_equity, last_price_for_add, conviction)
                        if add_quantity > 0:
                            print(f"Calculated ADD quantity for {symbol_of_new_trade}: {add_quantity}")
                            trade_executed = execute_trade_logic(
                                api=alpaca_api, data_api=data_client, agent=trading_agent,
                                ticker=symbol_of_new_trade, decision=None, conviction_score=conviction,
                                equity=current_equity, position=existing_position_obj,
                                management_action="ADD", quantity_for_action=add_quantity
                            )
                        else: print(f"ADD quantity for {symbol_of_new_trade} is zero. No action.")
                    else: print(f"Cannot ADD to {symbol_of_new_trade}, last price not available.")
                else: # Truly a new BUY
                    print(f"Executing new BUY for {symbol_of_new_trade}.")
                    trade_executed = execute_trade_logic(
                        api=alpaca_api, data_api=data_client, agent=trading_agent,
                        ticker=symbol_of_new_trade, decision="BUY", conviction_score=conviction,
                        equity=current_equity, position=None, management_action=None
                    )
            elif decision == "SELL": # New SELL (short) or close existing long
                if existing_position_obj:
                     # If agent says SELL for a stock we OWN (long), it means CLOSE.
                     # If agent says SELL for a stock we are SHORT, it means ADD to short (less common for this signal).
                     # If agent says SELL for a stock we DON'T OWN, it means new SHORT.
                     # Assuming 'SELL' on an existing LONG position means 'CLOSE'
                    if float(existing_position_obj.qty) > 0: # It's a long position
                        print(f"New trade SELL signal for existing LONG position {symbol_of_new_trade}. Treating as CLOSE.")
                        trade_executed = execute_trade_logic(
                            api=alpaca_api, data_api=data_client, agent=trading_agent,
                            ticker=symbol_of_new_trade, decision=None, conviction_score=conviction, # Conviction might inform urgency/partial close in future
                            equity=current_equity, position=existing_position_obj,
                            management_action="CLOSE", quantity_for_action=None # Close full
                        )
                    # If it's an existing SHORT position, a "SELL" signal might mean "ADD TO SHORT"
                    # This part needs careful thought based on strategy. For now, we'll assume SELL on existing short is less likely from "new trade"
                    # and more likely from "management". If it occurs, we can log and skip or implement "ADD TO SHORT".
                    elif float(existing_position_obj.qty) < 0: # It's a short position
                         print(f"New trade SELL signal for existing SHORT position {symbol_of_new_trade}. Logic for 'ADD TO SHORT' from this path is TBD. Skipping action.")
                         # trade_executed = execute_trade_logic(...) for ADD TO SHORT if desired
                else: # Truly a new SELL (short)
                    print(f"Executing new SELL (short) for {symbol_of_new_trade}.")
                    trade_executed = execute_trade_logic(
                        api=alpaca_api, data_api=data_client, agent=trading_agent,
                        ticker=symbol_of_new_trade, decision="SELL", conviction_score=conviction,
                        equity=current_equity, position=None, management_action=None
                    )

            if _was_trade_executed(trade_executed): # Use the helper
                acted_in_cycle.add(symbol_of_new_trade)
                print(f"Action taken for {symbol_of_new_trade} in 'New Trade' step. Added to acted_in_cycle.")
        else:
            print(f"Agent advises no new high-conviction BUY/SELL for {symbol_of_new_trade} (Decision: {decision}).")
    else:
        print(f"No valid new trade opportunity decision provided by agent for {ticker_to_potentially_trade}.")


    # --- Getting Position Management Advice ---
    print("\n--- Getting Position Management Advice ---")
    position_management_advice = []
    try:
        management_advice_full = trading_agent.get_portfolio_management_advice(detailed_open_positions, ticker_to_potentially_trade)
        if management_advice_full and "position_management" in management_advice_full:
            position_management_advice = management_advice_full["position_management"]
        else:
            print("Warning: No 'position_management' key in advice from get_portfolio_management_advice.")
    except AttributeError as ae:
        print(f"CRITICAL ERROR: `trading_agent` does not have `get_portfolio_management_advice`. {ae}")
    except Exception as e:
        print(f"Error getting position management advice: {e}")

    # --- Processing Management Actions ---
    print("\n--- Processing Management Actions ---")
    if position_management_advice:
        for advice in position_management_advice:
            if not all(k in advice for k in ("symbol", "action")):
                print(f"Warning: Skipping malformed management advice: {advice}")
                continue

            advised_symbol = advice['symbol']
            advised_action = advice['action'].upper()

            # *** THE CRITICAL FIX ***
            if advised_symbol in acted_in_cycle:
                print(f"Skipping management action for {advised_symbol}, as it was already handled in 'New Trade' step.")
                continue

            log_msg_parts = [f"Agent advises for {advised_symbol}: Action: {advised_action}"]
            if 'quantity' in advice: log_msg_parts.append(f"Quantity: {advice['quantity']}")
            if 'reason' in advice: log_msg_parts.append(f"Reason: {advice['reason']}")
            print(". ".join(log_msg_parts))

            current_alpaca_position_obj = next((p_obj for p_obj in open_positions if p_obj.symbol == advised_symbol), None)
            if not current_alpaca_position_obj:
                print(f"Warning: Agent advised management for {advised_symbol}, but no open position found. Skipping.")
                continue

            management_trade_executed = False
            if advised_action in ["CLOSE", "REDUCE", "ADD"]:
                quantity_for_action = advice.get('quantity')
                # Validate quantity for REDUCE/ADD
                if advised_action in ["REDUCE", "ADD"] and (not isinstance(quantity_for_action, (int, float)) or quantity_for_action <= 0):
                    print(f"Warning: Invalid or missing quantity for {advised_action} on {advised_symbol}. Skipping.")
                    continue

                management_trade_executed = execute_trade_logic(
                    api=alpaca_api, data_api=data_client, agent=trading_agent,
                    ticker=advised_symbol, decision=None, conviction_score=0, # Not a new trade decision
                    equity=current_equity, position=current_alpaca_position_obj,
                    management_action=advised_action, quantity_for_action=quantity_for_action if advised_action != "CLOSE" else None
                )
            elif advised_action == "HOLD":
                print(f"Agent advises HOLD for {advised_symbol}. No execution needed.")
            else:
                print(f"Unknown management action '{advised_action}' for {advised_symbol}. Skipping.")

            if _was_trade_executed(management_trade_executed):
                 acted_in_cycle.add(advised_symbol) # Should not be strictly necessary if logic is correct, but good for consistency
                 print(f"Action taken for {advised_symbol} in 'Management' step.")
    else:
        print("No management actions advised for open positions.")

    # Note: The original step "5. Process New Trade Opportunity (if any)" is now handled *before* management.
    # The agent_advice structure might need adjustment if it was central to passing data between these;
    # however, new_trade_decision_result and position_management_advice are now handled more directly.

    # --- Reflection Step ---
    # Placeholder for determining actual returns/losses from the actions taken in this cycle.
    # This is a complex task and would require tracking trades, their outcomes,
    # and attributing them to the agent's decisions within this cycle.
    # For now, we'll simulate a pseudo outcome.
    # `acted_in_cycle` contains symbols for which new trades were made or existing positions were managed (excluding simple HOLDs from management advice).

    actions_taken_summary = {
        "new_trade_actions": new_trade_decision_result if new_trade_decision_result and new_trade_decision_result.get("decision") not in ["NONE", "HOLD"] else None,
        "managed_positions_actions": [adv for adv in position_management_advice if adv.get("action") not in ["HOLD"]]
    }

    if actions_taken_summary["new_trade_actions"] or actions_taken_summary["managed_positions_actions"]:
        print(f"\n--- Reflecting on Cycle's Actions for {ticker_to_potentially_trade} ---")
        # In a real system, you'd calculate actual P&L or use a more sophisticated evaluation.
        # For this placeholder, we'll just pass a simple string indicating an outcome.
        # The structure of returns_losses should be what `reflector.py` expects.
        # Let's assume it's a string for now, but it could be a dict with P&L, etc.
        pseudo_returns_losses = f"Simulated outcome for cycle involving {ticker_to_potentially_trade}: Positive (Placeholder)"
        if not acted_in_cycle and not any(adv.get('action') not in ["HOLD"] for adv in position_management_advice):
            pseudo_returns_losses = f"Simulated outcome for cycle involving {ticker_to_potentially_trade}: No significant actions taken, market observed."

        # Ensure trading_agent (which is an instance of TradingAgentsGraph) is in scope. It's global in this script.
        try:
            # The trading_agent.curr_state should have been updated by the .propagate() call
            # which is implicitly part of get_new_trade_decision and get_portfolio_management_advice
            # if they internally call propagate for the full analysis.
            # If trading_agent.curr_state is not set correctly by these calls, reflection might be on stale or no data.
            # This depends on TradingAgentsGraph.get_new_trade_decision & get_portfolio_management_advice setting self.curr_state
            if trading_agent.curr_state: # Check if curr_state is populated
                 print(f"Calling reflect_and_remember for {ticker_to_potentially_trade}. Current agent state for reflection pertains to: {trading_agent.curr_state.get('company_of_interest')}")
                 trading_agent.reflect_and_remember(pseudo_returns_losses)
                 print("Reflection complete.")
            else:
                 print(f"Warning: trading_agent.curr_state not set. Skipping reflection for {ticker_to_potentially_trade}.")
        except Exception as e:
            print(f"Error during reflection step for {ticker_to_potentially_trade}: {e}")
    else:
        print(f"\n--- No significant actions taken in this cycle for {ticker_to_potentially_trade}. Skipping reflection. ---")


    print(f"\n--- Trading cycle complete for {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

# Helper function to check if a trade was executed (placeholder for actual return value from execute_trade_logic)
# You'll need to adjust execute_trade_logic to return a boolean indicating success/failure or if a trade was made.
# For now, let's assume it always returns True if it attempts a trade, False otherwise.
# This is a simplification. A more robust solution would be to have execute_trade_logic
# return a more informative status.
def _was_trade_executed(trade_execution_status: bool) -> bool:
    """
    Checks if a trade was successfully executed.
    Relies on execute_trade_logic returning True for success, False otherwise.
    """
    return trade_execution_status

if __name__ == "__main__":
    # --- To run the script ---
    # Ensure API keys are set as environment variables or replace placeholders above.
    # Example: export ALPACA_API_KEY='YOUR_KEY'
    #          export ALPACA_API_SECRET='YOUR_SECRET'
    # You might also need to set OPENAI_API_KEY and FINNHUB_API_KEY for the TradingAgents framework.


    ticker_to_trade = "MSFT"  # Example: Trade SPDR S&P 500 ETF Trust
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
            run_trading_cycle(ticker_to_trade) # Ensure ticker_to_trade is defined, e.g., "GOOGL"
            sleep_duration = 15 * 60  # 15 minutes
            current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"--- Trading cycle complete for {current_time_str} ---") # Log completion time
            print(f"Cycle complete. Sleeping for {sleep_duration / 60:.1f} minutes...")
            time.sleep(sleep_duration)
    else:
        print("\nCannot start trading session: Alpaca API connection failed at initialization.")

    print("\nScript execution finished.")
    print("Remember to manage your API keys securely and test thoroughly with a paper trading account first.")
    print("This script is for educational purposes and should be reviewed carefully before use with real funds.")
