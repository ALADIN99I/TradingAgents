import alpaca_trade_api as tradeapi
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from datetime import datetime
import os

#os api method for now only
# IMPORTANT: This MUST be a real OpenAI API key for embeddings to work.
os.environ["OPENAI_API_KEY"] = ""
os.environ["FINNHUB_API_KEY"] = "d0u99jhr01qn5fk3v8rgd0u99jhr01qn5fk3v8s0" # <--- You can also add your Finnhub key here
# This is for OpenRouter chat models.
os.environ["OPENROUTER_API_KEY"] = ""
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

def execute_trade_logic(api, agent, ticker, decision, equity, position):
    """Contains the logic to BUY, SELL, or HOLD."""
    if api is None:
        print("Alpaca API not initialized. Cannot execute trade logic.")
        return

    risk_percent = 2.0  # Risk 2% of equity per trade
    print(f"Executing trade logic for {ticker}. Decision: {decision}. Equity: {equity:.2f}. Position: {'Exists' if position else 'None'}")

    # --- BUY Logic ---
    if decision == "BUY" and position is None:
        if equity <= 0:
            print(f"Action: Attempting to BUY {ticker}, but equity is ${equity:.2f}. Cannot place buy order.")
            return
        print(f"Action: Attempting to BUY {ticker}.")

        # 1. Calculate Position Size
        investment_amount = equity * (risk_percent / 100)
        try:
            # NEW, CORRECTED CODE
            last_price = api.get_latest_quote(ticker).ap
        except Exception as e:
            print(f"Error fetching latest price for {ticker}: {e}. Cannot calculate position size.")
            return

        if last_price <= 0:
            print(f"Latest price for {ticker} is ${last_price:.2f}. Cannot calculate position size.")
            return

        qty_to_buy = round(investment_amount / last_price, 5)  # Use fractional shares

        if qty_to_buy <= 0:
            print(f"Calculated quantity to buy for {ticker} is {qty_to_buy}. Cannot place buy order.")
            return

        # 2. Send Order
        try:
            api.submit_order(
                symbol=ticker,
                qty=str(qty_to_buy), # API expects qty as string
                side='buy',
                type='market',
                time_in_force='day'
            )
            print(f"Market buy order for {qty_to_buy} shares of {ticker} submitted.")
        except Exception as e:
            print(f"Error submitting buy order for {ticker}: {e}")

    # --- SELL Logic ---
    elif decision == "SELL" and position is not None:
        print(f"Action: Attempting to SELL {ticker}.")

        try:
            # Ensure position.qty is a float for calculations if needed, though close_position takes symbol
            # qty_to_sell = float(position.qty) # Not directly used by close_position

            # 1. Send Order to close position
            # The close_position method returns an order object upon successful submission.
            # We need to handle the case where the position might be already closed or an error occurs.
            closed_order_response = api.close_position(ticker)
            print(f"Market sell order to close position in {ticker} submitted. Order ID: {closed_order_response.id}")

            # 2. Feed outcome back to agent for learning
            # The Alpaca API's close_position doesn't immediately return realized P/L.
            # Realized P/L is typically calculated after the order fills.
            # For simplicity here, we'll assume the trade was successful and log a placeholder.
            # A more robust solution would involve monitoring the order status and then fetching P/L.
            # For now, we don't have immediate P/L, so we can't directly use it.
            # We will call reflect_and_remember.
            # The existing method in TradingAgentsGraph is `reflect_and_remember(self, returns_losses)`.
            # `returns_losses` is expected to be a numerical P/L value.
            # However, immediate realized P/L is not available from `close_position()`.
            # A robust solution would monitor the order, wait for it to fill, then calculate P/L.
            # For this integration, we pass a placeholder value (e.g., 0.0) for `returns_losses`.
            # The agent's reflection mechanism will use its `self.curr_state` (set during `propagate`)
            # for contextual information.
            if hasattr(agent, 'reflect_and_remember') and callable(getattr(agent, 'reflect_and_remember')):
                print(f"Position in {ticker} closed (Order ID: {closed_order_response.id}). Attempting to update agent memory.")
                # Pass a placeholder for returns_losses.
                # A real system would calculate this after the sell order is confirmed filled.
                placeholder_profit_loss = 0.0
                agent.reflect_and_remember(returns_losses=placeholder_profit_loss)
                print(f"Agent memory updated for {ticker} using placeholder P/L: {placeholder_profit_loss}.")
            else:
                print(f"Warning: 'reflect_and_remember' method not found in trading_agent. Skipping memory update.")

        except tradeapi.rest.APIError as e:
            if "position not found" in str(e).lower(): # Check if error message indicates no position
                 print(f"Attempted to sell {ticker}, but no open position found in Alpaca.")
            else:
                 print(f"APIError submitting sell order for {ticker}: {e}")
        except Exception as e:
            print(f"Error submitting sell order for {ticker}: {e}")

    # --- HOLD Logic ---
    else:
        print(f"Action: HOLD for {ticker}. No trade executed. (Decision: {decision}, Position: {'Exists' if position else 'None'})")


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
