import alpaca_trade_api as tradeapi
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from datetime import datetime
import os
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

def get_account_state(api, ticker):
    """Gets account balance and any open position for a specific ticker."""
    if api is None:
        print("Alpaca API not initialized. Cannot get account state.")
        return 0.0, None


    equity = 0.00
    try:
        account = api.get_account()
        # Use portfolio_value instead of equity
        equity = float(account.portfolio_value)
    except Exception as e:
        print(f"Error fetching account information: {e}")
        equity = 0.0 # Default to 0 if account info cannot be fetched

    position = None
    try:
        position = api.get_position(ticker)
    except tradeapi.rest.APIError as e:
        if e.status_code == 404: # Position not found
            position = None
        else:
            print(f"APIError getting position for {ticker}: {e}")
            position = None # Treat as no position on other API errors
    except Exception as e:
        print(f"Error getting position for {ticker}: {e}")
        position = None

    return equity, position

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

    log.info(f"Executing trade logic for {ticker}. Decision: {decision}. Equity: {equity:.2f}. Position: {position}")
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

def run_daily_trading_session(ticker_symbol):
    """Runs the full trading session for a given ticker."""
    if alpaca_api is None or data_client is None: # check for data_client
        print(f"Cannot run trading session for {ticker_symbol}: Alpaca API not connected.")
        return

    print(f"\n--- Starting trading session for {ticker_symbol} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    # 1. Get State
    current_equity, current_position = get_account_state(alpaca_api, ticker_symbol)
    print(f"Current Equity: ${current_equity:.2f} | Position in {ticker_symbol}: {'Yes, qty: ' + str(current_position.qty) if current_position else 'No'}")

    if current_equity <= 0 and current_position is None:
        print(f"Equity is ${current_equity:.2f} and no position in {ticker_symbol}. Cannot make new trades.")
        print("--- Trading session complete ---")
        return

    # 2. Get Decision
    analysis_date_str = datetime.now().strftime("%Y-%m-%d")
    print(f"Requesting trading decision for {ticker_symbol} for date: {analysis_date_str}...")
    try:
        # The propagate method in the example returns _, decision.
        # UPDATED ASSUMPTION: now returns (decision, conviction_score)
        # We are discarding the first element (state) if it's still returned, or adjusting if the signature changed.
        # For now, assuming propagate returns: (state_or_similar, (decision, conviction_score)) or just (decision, conviction_score)
        # Let's assume it returns (decision, conviction_score) directly for simplicity here.
        # If it's nested, like (_, (decision, conviction_score)), this will need adjustment.
        raw_output = trading_agent.propagate(ticker_symbol, analysis_date_str)
        if isinstance(raw_output, tuple) and len(raw_output) == 2 and isinstance(raw_output[1], tuple) and len(raw_output[1]) == 2): # Check for (_, (decision, conviction_score))
            _, (agent_decision, conviction_score) = raw_output
        elif isinstance(raw_output, tuple) and len(raw_output) == 2: # Check for (decision, conviction_score)
             agent_decision, conviction_score = raw_output
        else:
            # Fallback or error if the structure isn't as expected.
            # For now, let's assume a default high conviction if structure is old, to prevent crashes, and log a warning.
            # This part would need to be robustly handled based on actual TradingAgentsGraph output.
            print(f"Warning: Unexpected output structure from trading_agent.propagate(): {raw_output}. Defaulting conviction.")
            agent_decision = raw_output[1] if isinstance(raw_output, tuple) and len(raw_output) > 1 else "HOLD" # Default based on old structure
            conviction_score = 0.0 # Default to no trade if structure is unknown or decision is HOLD
            if agent_decision == "BUY" or agent_decision == "SELL":
                conviction_score = 0.8 # Default to high conviction for BUY/SELL if score is missing

        print(f"Agent Decision for {ticker_symbol}: {agent_decision} with Conviction: {conviction_score:.2f}")
    except Exception as e:
        print(f"Error getting decision from TradingAgentsGraph for {ticker_symbol}: {e}")
        print("--- Trading session complete ---")
        return

    # 3. Execute
    execute_trade_logic(alpaca_api, data_client, trading_agent, ticker_symbol, agent_decision, conviction_score, current_equity, current_position)

    print(f"--- Trading session for {ticker_symbol} complete ---")

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
        # run_daily_trading_session(ticker_to_trade)
    else:
        print("\nCannot start trading session: Alpaca API connection failed at initialization.")

    print("\nScript execution finished.")
    print("Remember to manage your API keys securely and test thoroughly with a paper trading account first.")
    print("This script is for educational purposes and should be reviewed carefully before use with real funds.")
