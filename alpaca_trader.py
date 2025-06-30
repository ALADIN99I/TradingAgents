import alpaca_trade_api as tradeapi
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from datetime import datetime
import os

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
    # Check if the API connection is successful
    account_info = alpaca_api.get_account()
    print(f"Successfully connected to Alpaca. Account ID: {account_info.id}")
except Exception as e:
    print(f"Error connecting to Alpaca API: {e}")
    alpaca_api = None # Set to None if connection fails

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

def calculate_position_size(equity, last_price):
    """
    Calculates the position size based on a predefined risk percentage of equity.
    """
    risk_per_trade = 0.02  # 2% of equity
    dollars_to_risk = equity * risk_per_trade
    position_size = dollars_to_risk / last_price
    return int(position_size)

def execute_trade_logic(api, agent, ticker, decision, equity, position): #PARAMETER CHANGE
    """Contains the logic to BUY, SELL, or HOLD."""
    #log import
    import logging
    log = logging.getLogger(__name__) #log definition

    log.info(f"Executing trade logic for {ticker}. Decision: {decision}. Equity: {equity:.2f}. Position: {position}")

    if position is not None:
        # We have a position, so we can either hold or sell to close
        if decision == "SELL":
            log.info(f"Action: Closing position of {position.qty} shares of {ticker}")
            api.close_position(ticker)
        else: # "BUY" or "HOLD"
            log.info(f"Action: Holding position in {ticker}")
    else:
        # No position, so we can either open a new position or do nothing
        if decision == "BUY":
            log.info(f"Action: Attempting to BUY {ticker}")
            try:
                last_price = api.get_last_quote(ticker).askprice #get_last_quote not get_latest_quote
                qty = calculate_position_size(equity, last_price)
                log.info(f"Calculated position size: {qty} shares of {ticker}")
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
            except Exception as e:
                log.error(f"Error fetching latest price for {ticker}: {e}. Cannot calculate position size.")

        elif decision == "SELL":
            log.info(f"Action: Attempting to SELL (short) {ticker}")
            try:
                last_price = api.get_last_quote(ticker).askprice #get_last_quote not get_latest_quote
                qty = calculate_position_size(equity, last_price)
                log.info(f"Calculated position size: {qty} shares of {ticker}")
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
            except Exception as e:
                log.error(f"Error fetching latest price for {ticker}: {e}. Cannot calculate position size.")
        else: # decision == "HOLD"
            log.info(f"Action: HOLD for {ticker}. No trade executed. (Decision: {decision}, Position: {position})")


def run_daily_trading_session(ticker_symbol):
    """Runs the full trading session for a given ticker."""
    if alpaca_api is None:
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
        # Assuming the first returned value is state or similar, and second is the decision string.
        _, agent_decision = trading_agent.propagate(ticker_symbol, analysis_date_str)
        print(f"Agent Decision for {ticker_symbol}: {agent_decision}")
    except Exception as e:
        print(f"Error getting decision from TradingAgentsGraph for {ticker_symbol}: {e}")
        print("--- Trading session complete ---")
        return

    # 3. Execute
    execute_trade_logic(alpaca_api, trading_agent, ticker_symbol, agent_decision, current_equity, current_position)

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

    if alpaca_api is not None: # Only run if API connection was successful
        run_daily_trading_session(ticker_to_trade)
    else:
        print("\nCannot start trading session: Alpaca API connection failed at initialization.")

    print("\nScript execution finished.")
    print("Remember to manage your API keys securely and test thoroughly with a paper trading account first.")
    print("This script is for educational purposes and should be reviewed carefully before use with real funds.")
