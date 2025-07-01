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
os.environ["_API_KEY"] = "nan"
os.environ["FINNHUB_API_KEY"] = "d0u99jhr01qn5fk3v8rgd0u99jhr01qn5fk3v8s0" # <--- You can also add your Finnhub key here
# This is for OpenRouter chat models.
os.environ["O_API_KEY"] = ""
# --- Alpaca API Configuration ---
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

trading_agent = TradingAgentsGraph(debug=False, config=config)
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

def execute_trade_logic(api, data_api, agent, ticker, decision, conviction_score, equity, position): # PARAMETER CHANGE
    """Contains the logic to BUY, SELL, or HOLD."""
    #log import
    import logging
    log = logging.getLogger(__name__) #log definition

    action_type = "Managing existing" if position else "Opening new"
    log.info(f"Executing trade logic for {ticker}. Action Type: {action_type}. Decision: {decision}. Conviction: {conviction_score:.2f}. Equity: {equity:.2f}. Position: {position is not None}")
    try:
        # Get the most recent 1-minute bar for the stock using the new data_api client
        request_params = StockBarsRequest(
                            symbol_or_symbols=[ticker],
                            timeframe=TimeFrame.Minute,
                            limit=1
                         )
        barset = data_api.get_stock_bars(request_params)

        if barset and barset[ticker]:
            last_price = barset[ticker][0].close  # Use .close instead of .c
        else:
            log.error(f"Could not get last price for {ticker}. No bar data found.")
            return # Exit if we can't get the price

        if position is not None:
            # We have a position, so we can either hold or sell to close
            if decision == "SELL":
                log.info(f"Action: Closing position of {position.qty} shares of {ticker}")
                api.close_position(ticker)
            else:  # "BUY" or "HOLD"
                log.info(f"Action: Holding position in {ticker}")
        else:
            # No position, so we can either open a new position or do nothing
            if decision == "BUY":
                log.info(f"Action: Attempting to BUY {ticker} with conviction {conviction_score}")
                qty = calculate_position_size(equity, last_price, conviction_score)
                log.info(f"Calculated position size: {qty} shares of {ticker} based on conviction {conviction_score}")
                if qty > 0:
                    api.submit_order(
                        symbol=ticker,
                        qty=qty,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    log.info(f"Market buy order for {qty} shares of {ticker} placed.")
                else:
                    log.info(f"Position size is 0 for {ticker}. No trade executed.")

            elif decision == "SELL":
                log.info(f"Action: Attempting to SELL (short) {ticker} with conviction {conviction_score}")
                qty = calculate_position_size(equity, last_price, conviction_score)
                log.info(f"Calculated position size: {qty} shares of {ticker} based on conviction {conviction_score}")
                if qty > 0:
                    api.submit_order(
                        symbol=ticker,
                        qty=qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    log.info(f"Market sell (short) order for {qty} shares of {ticker} placed.")
                else:
                    log.info(f"Position size is 0 for {ticker}. No trade executed.")
            else:  # decision == "HOLD"
                log.info(f"Action: HOLD for {ticker}. No trade executed. (Decision: {decision}, Position: {position})")

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

    # 3. Get Mocked Agent Portfolio Advice
    # In a real system, trading_agent.propagate (or a new method) would be called here with detailed_open_positions.
    print("\n--- Getting (Mocked) Agent Portfolio Advice ---")
    agent_advice = get_mocked_agent_portfolio_advice(detailed_open_positions, ticker_to_potentially_trade)

    # 4. Process Management Actions for Open Positions (Logging only for now)
    print("\n--- Processing (Mocked) Management Actions ---")
    if agent_advice.get("position_management"):
        for advice in agent_advice["position_management"]:
            print(f"Agent advises: Symbol: {advice['symbol']}, Action: {advice['action']}, "
                  f"Details: {advice.get('quantity', 'N/A')}, Reason: {advice.get('reason', 'N/A')}")
            # TODO: Future: Call execute_trade_logic here with new parameters/logic to handle these actions.
            # Example: if advice['action'] == 'REDUCE':
            # execute_trade_logic_for_management(api, data_api, advice['symbol'], 'sell', advice['quantity'], current_equity, find_position_obj(open_positions, advice['symbol']))
            # Note: execute_trade_logic would need significant changes to handle partial sales, adds, etc.
    else:
        print("No management actions advised for open positions.")

    # 5. Process New Trade Opportunity (if any)
    print(f"\n--- Considering (Mocked) New Trade Opportunity ---")
    new_trade_opportunity = agent_advice.get("new_trade_opportunity")
    if new_trade_opportunity:
        nt_symbol = new_trade_opportunity['symbol']
        nt_decision = new_trade_opportunity['decision']
        nt_conviction = new_trade_opportunity['conviction_score']
        nt_reason = new_trade_opportunity.get('reason', 'N/A')

        print(f"Agent suggests NEW trade: {nt_decision} {nt_symbol} with Conviction {nt_conviction:.2f}. Reason: {nt_reason}")

        is_already_open = any(p['symbol'] == nt_symbol for p in detailed_open_positions)
        current_pos_for_new_trade_obj = next((p_obj for p_obj in open_positions if p_obj.symbol == nt_symbol), None)


        if is_already_open:
            print(f"Already have an open position in {nt_symbol}. Agent might be advising to ADD or this is a conflicting signal.")
            # TODO: Future: If agent explicitly says "ADD", then execute_trade_logic should handle adding to position.
            # For now, we are cautious and don't open a new overlapping trade from this "new_trade_opportunity" path
            # if the "position_management" path didn't already handle an "ADD".
        elif nt_decision in ["BUY", "SELL"]:
            print(f"Executing new {nt_decision} trade for {nt_symbol}.")
            execute_trade_logic(alpaca_api, data_client, trading_agent, nt_symbol, nt_decision, nt_conviction, current_equity, None) # Pass None as position for new trade
        else:
            print(f"Decision for new opportunity on {nt_symbol} is {nt_decision}. No new trade action taken.")
    else:
        print("No new high-conviction trade opportunities advised by agent in this cycle.")

    print(f"\n--- Trading cycle complete for {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

# --- Mock Agent Response Function ---
def get_mocked_agent_portfolio_advice(detailed_open_positions, potential_new_trade_ticker):
    """
    Mocks the response from TradingAgentsGraph for portfolio management advice.
    In a real system, this would involve a call to trading_agent.propagate() or a similar method
    with detailed_open_positions and potentially potential_new_trade_ticker.
    """
    print(f"DEBUG: Mocking agent advice for {len(detailed_open_positions)} open positions and new look at {potential_new_trade_ticker}.")
    management_actions = []
    new_trade_decision = None

    # Example logic for managing open positions (mocked)
    for pos in detailed_open_positions:
        # Simple mock: if P/L > 5%, suggest REDUCE, if P/L < -2%, suggest CLOSE, else HOLD
        if pos['unrealized_pl_pct'] > 5.0:
            management_actions.append({
                'symbol': pos['symbol'],
                'action': 'REDUCE',
                'quantity': int(pos['qty'] * 0.1), # Suggest selling 10%
                'reason': f"Mock: Profitable ({pos['unrealized_pl_pct']:.2f}%), suggesting partial profit taking."
            })
        elif pos['unrealized_pl_pct'] < -2.0:
            management_actions.append({
                'symbol': pos['symbol'],
                'action': 'CLOSE',
                'reason': f"Mock: Losing ({pos['unrealized_pl_pct']:.2f}%), suggesting cut loss."
            })
        else:
            management_actions.append({
                'symbol': pos['symbol'],
                'action': 'HOLD',
                'reason': f"Mock: Holding position ({pos['unrealized_pl_pct']:.2f}%)."
            })

    # Example logic for new trade opportunity (mocked)
    # Let's say we only suggest a new trade if there are few open positions or on a specific ticker
    if len(detailed_open_positions) < 3 and potential_new_trade_ticker == "GOOGL": # Arbitrary condition
        new_trade_decision = {
            'symbol': potential_new_trade_ticker,
            'decision': 'BUY', # Could be BUY or SELL
            'conviction_score': 0.75, # Example conviction
            'reason': f'Mock: Favorable conditions for {potential_new_trade_ticker} and portfolio has capacity.'
        }
    elif potential_new_trade_ticker == "MSFT": # Another arbitrary condition for variety
         new_trade_decision = {
            'symbol': potential_new_trade_ticker,
            'decision': 'SELL', # Example short sell
            'conviction_score': 0.65,
            'reason': f'Mock: Bearish signal on {potential_new_trade_ticker}.'
        }


    return {
        "position_management": management_actions,
        "new_trade_opportunity": new_trade_decision
    }


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
        print("\n--- Running Simulated Buy Logic Test ---")
        test_equity = 50000 # Example equity, adjust as needed
        test_conviction = 0.85 # Example: High conviction for testing
        # Pass the data_client and conviction_score to the simulation function
        test_buy_logic_simulation(data_client, ticker_to_trade, test_equity, test_conviction)
        print("--- Simulated Buy Logic Test Complete ---\n")

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
