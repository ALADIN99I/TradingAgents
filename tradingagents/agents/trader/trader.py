import functools
import time
import json
import re # Added for regex operations


def parse_json_decision(raw: str, company_name_for_fallback: str) -> dict:
    """
    Extracts and parses JSON decision from raw LLM output.
    Provides a fallback if parsing or validation fails.
    """
    # Attempt to extract JSON object using regex; this is good for nested structures
    # and handles cases where JSON might be embedded in other text.
    # This regex finds the first complete JSON object.
    match = re.search(r"\{(?:[^{}]|(?R))*\}", raw, re.DOTALL)
    json_str_to_parse = ""

    if match:
        json_str_to_parse = match.group(0)
    else:
        # If regex fails, try a simpler approach: strip common markdown and find first/last brace
        temp_str = raw.strip()
        if temp_str.startswith("```json"): # Handle ```json
            temp_str = temp_str[7:]
        elif temp_str.startswith("```"): # Handle ```
            temp_str = temp_str[3:]

        if temp_str.endswith("```"):
            temp_str = temp_str[:-3]
        temp_str = temp_str.strip()

        start_brace = temp_str.find('{')
        end_brace = temp_str.rfind('}')
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            json_str_to_parse = temp_str[start_brace : end_brace+1]
        else:
            # Fallback to the stripped string if no clear JSON structure is found by braces
            # This might be an already clean JSON or will fail parsing, caught by try-except below.
            json_str_to_parse = temp_str

    try:
        if not json_str_to_parse.strip().startswith("{") or not json_str_to_parse.strip().endswith("}"):
             # If after all attempts, it doesn't look like a JSON object string
            raise json.JSONDecodeError("No valid JSON object found in LLM output", raw, 0)

        data = json.loads(json_str_to_parse)

        # Validate essential keys
        if not isinstance(data, dict) or \
           "action" not in data or \
           "symbol" not in data:
            raise json.JSONDecodeError(
                f"Missing essential keys ('action', 'symbol') in parsed JSON. Parsed: {str(data)[:200]}",
                json_str_to_parse, 0
            )

        # Ensure symbol is a string, using fallback if it's structured unexpectedly or missing
        if not isinstance(data.get("symbol"), str) or not data.get("symbol"):
            data["symbol"] = company_name_for_fallback

        # Default conviction score for HOLD if missing
        if data.get("action") == "HOLD" and "conviction_score" not in data:
            data["conviction_score"] = 0.0

        # Default conviction score for BUY/SELL if missing, and annotate reason
        if data.get("action") in ["BUY", "SELL"] and "conviction_score" not in data:
            data["conviction_score"] = 0.5  # Default to neutral
            current_reason = data.get("reason", "")
            default_reason_note = "(Conviction score defaulted as LLM did not provide it)"
            data["reason"] = f"{current_reason} {default_reason_note}".strip() if current_reason else default_reason_note

        # Ensure reason is present
        if "reason" not in data:
            data["reason"] = "Reason not provided by LLM."

        return data

    except json.JSONDecodeError as e:
        # Using print for now as logging isn't set up in this scope
        print(f"TRADER_NODE (parse_json_decision): Error parsing LLM JSON output: {e}. Attempted to parse: '{json_str_to_parse[:500]}...' Raw input: '{raw[:500]}...'")
        return {
            "action": "HOLD",
            "symbol": company_name_for_fallback,
            "conviction_score": 0.0,
            "reason": f"Failed to parse LLM decision. Error: {e}. Raw LLM output snippet: {raw[:200]}..."
        }


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        # Assuming 'potential_trades' is now available in the state, populated by an upstream node (e.g., Research Manager)
        # Each trade in potential_trades should be a dict with 'action', 'symbol', 'reason', and 'conviction_score'
        potential_trades = state.get("potential_trades", [])

        formatted_trades = "No specific pre-assessed trades provided."
        if potential_trades:
            formatted_trades = "Potential trades under consideration (with conviction scores):\n"
            for trade in potential_trades:
                formatted_trades += f"- {trade.get('action','N/A_ACTION')} {trade.get('symbol','N/A_SYMBOL')}: {trade.get('reason','N/A_REASON')} (Conviction: {trade.get('conviction_score', 'N/A')})\n"
        else:
            # This case means potential_trades was empty or not in state.
            # The LLM will have to rely more on the investment_plan.
            pass

        context_content = (
            f"Company: {company_name}\n\n"
            f"Previously Proposed Investment Plan: {investment_plan}\n\n"
            f"{formatted_trades}\n\n"
            f"Your task is to synthesize all this information, including the market research, sentiment, news, fundamentals, "
            f"the proposed investment plan, and the list of potential trades with their conviction scores (if available), "
            f"and decide on a final, single trade action (BUY, SELL, or HOLD for {company_name}). "
            f"If specific trades were listed, your decision should ideally align with the one that has the best rationale AND highest conviction, "
            f"unless other overriding factors from the broader reports suggest a different course. If you decide on a BUY or SELL action, "
            f"also provide a conviction_score (0.0-1.0) for your chosen action. For HOLD, conviction_score can be 0.0 or omitted. "
            f"Explain your reasoning clearly."
        )

        messages = [
            {
                "role": "system",
                "content": f"""You are an expert Trader. Your goal is to make a final, actionable trading decision (BUY, SELL, or HOLD) for a specific company.
You will be given:
1.  A general investment plan or summary from a Research Manager.
2.  A list of potential trades, each with an action (BUY/SELL), symbol, reason, and a conviction_score (0.0-1.0) assigned by a Risk Manager.
3.  Various market, news, sentiment, and fundamentals reports.
4.  Reflections on past trading decisions in similar situations.

Your task:
- Synthesize ALL available information.
- If a list of potential trades with conviction scores is provided, prioritize the trade with the highest conviction unless other reports strongly contradict it. Your final decision should reflect this.
- If deciding to BUY or SELL, you MUST determine a final `conviction_score` for THIS specific decision.
- Output your decision as a JSON object with the keys: "action" (string: "BUY", "SELL", or "HOLD"), "symbol" (string: the company ticker), "conviction_score" (float: 0.0-1.0, required if action is BUY/SELL), and "reason" (string: your brief justification).
- Example for BUY: {{"action": "BUY", "symbol": "{company_name}", "conviction_score": 0.85, "reason": "Strong bullish signals and high conviction from risk assessment."}}
- Example for HOLD: {{"action": "HOLD", "symbol": "{company_name}", "reason": "Market conditions are too uncertain despite some positive indicators."}}
- IMPORTANT: You MUST respond with JSON only. Do NOT include any additional text, commentary, or markdown formatting such as ```json before or after the JSON object. Your entire response must be the raw JSON object itself.
Utilize lessons from past decisions: {past_memory_str}"""
            },
            {
                "role": "user",
                "content": context_content,
            }
        ]

        response = llm.invoke(messages)

        # Use the new helper function to parse the decision
        # Pass company_name for fallback purposes within the parser
        trade_decision_data = parse_json_decision(response.content, company_name)

        return {
            "messages": [response],
            "trader_investment_plan": trade_decision_data,
            "final_trade_decision": trade_decision_data,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
