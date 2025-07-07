import functools
import time
import json
# import re # No longer needed for the new parser

# New robust JSON parser based on brace counting
def parse_json_decision(text: str): # Removed company_name_for_fallback from signature
    """
    Extract JSON objects from text using robust JSON parsing.
    Returns a list of parsed JSON objects.
    """
    results = []
    i = 0

    while i < len(text):
        if text[i] == '{':
            brace_count = 0
            start = i

            # Count braces to find complete JSON object
            while i < len(text):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1

                    if brace_count == 0:
                        json_str = text[start:i+1]
                        try:
                            # Basic cleaning of common non-JSON prefixes/suffixes before parsing
                            # This is a simplified version of what might be needed.
                            # More sophisticated stripping might be required if LLM output is very messy.
                            if json_str.strip().startswith("```json"):
                                json_str = json_str.strip()[7:]
                            elif json_str.strip().startswith("```"):
                                json_str = json_str.strip()[3:]
                            if json_str.strip().endswith("```"):
                                json_str = json_str.strip()[:-3]

                            parsed = json.loads(json_str.strip())
                            results.append(parsed)
                        except json.JSONDecodeError:
                            # Using print for now as logging isn't set up in this scope
                            # print(f"TRADER_NODE (new parse_json_decision): Skipping invalid JSON segment: '{json_str[:100]}...'")
                            pass  # Skip invalid JSON
                        break # Found a balanced object (valid or not), move outer loop cursor
                i += 1
        i += 1 # Move to next character in the outer loop

    return results


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

        potential_trades = state.get("potential_trades", [])
        formatted_trades = "No specific pre-assessed trades provided."
        if potential_trades:
            formatted_trades = "Potential trades under consideration (with conviction scores):\n"
            for trade in potential_trades:
                formatted_trades += f"- {trade.get('action','N/A_ACTION')} {trade.get('symbol','N/A_SYMBOL')}: {trade.get('reason','N/A_REASON')} (Conviction: {trade.get('conviction_score', 'N/A')})\n"

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

        # Call the new parse_json_decision function
        parsed_json_list = parse_json_decision(response.content)

        trade_decision_data = None
        raw_llm_output_for_fallback = response.content # Store raw output for fallback message

        if parsed_json_list:
            # Assuming the first valid JSON object is the one we want
            potential_data = parsed_json_list[0]

            # --- Apply validation and defaulting logic (moved from old parser) ---
            if isinstance(potential_data, dict) and \
               "action" in potential_data and \
               "symbol" in potential_data:

                trade_decision_data = potential_data.copy() # Work with a copy

                # Ensure symbol is a string, using company_name as fallback
                if not isinstance(trade_decision_data.get("symbol"), str) or not trade_decision_data.get("symbol"):
                    trade_decision_data["symbol"] = company_name

                # Default conviction score for HOLD if missing
                if trade_decision_data.get("action") == "HOLD" and "conviction_score" not in trade_decision_data:
                    trade_decision_data["conviction_score"] = 0.0

                # Default conviction score for BUY/SELL if missing, and annotate reason
                if trade_decision_data.get("action") in ["BUY", "SELL"] and "conviction_score" not in trade_decision_data:
                    trade_decision_data["conviction_score"] = 0.5  # Default to neutral
                    current_reason = trade_decision_data.get("reason", "")
                    default_reason_note = "(Conviction score defaulted as LLM did not provide it)"
                    trade_decision_data["reason"] = f"{current_reason} {default_reason_note}".strip() if current_reason else default_reason_note

                # Ensure reason is present
                if "reason" not in trade_decision_data:
                    trade_decision_data["reason"] = "Reason not provided by LLM."
            else:
                # The extracted object was not a valid decision structure
                print(f"TRADER_NODE (new parser): Extracted JSON missing essential keys. Parsed: {str(potential_data)[:200]} from LLM output: {raw_llm_output_for_fallback[:500]}")
                # trade_decision_data remains None, will be handled by the fallback below

        # Fallback if no JSON found or if the first found JSON was invalid
        if not trade_decision_data:
            if not parsed_json_list: # Specifically if no JSON was found at all
                 print(f"TRADER_NODE (new parser): No JSON objects found in LLM output. Raw: '{raw_llm_output_for_fallback[:500]}...'")
            # If parsed_json_list was not empty but trade_decision_data is still None, it means the first JSON was invalid (message printed above)

            trade_decision_data = {
                "action": "HOLD",
                "symbol": company_name, # Use company_name directly
                "conviction_score": 0.0,
                "reason": f"Failed to parse valid JSON decision from LLM. Raw output snippet: {raw_llm_output_for_fallback[:200]}..."
            }

        return {
            "messages": [response], # Keep raw LLM response for debugging
            "trader_investment_plan": trade_decision_data,
            "final_trade_decision": trade_decision_data,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
