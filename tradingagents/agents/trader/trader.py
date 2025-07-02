import functools
import time
import json


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
- IMPORTANT: Respond ONLY with the JSON object. Do not include any other explanatory text, pleasantries, or markdown formatting like ```json before or after the JSON object. Your entire response should be the raw JSON.
Utilize lessons from past decisions: {past_memory_str}"""
            },
            {
                "role": "user",
                "content": context_content,
            }
        ]

        response = llm.invoke(messages)
        llm_output_str = response.content.strip()

        # Attempt to parse the LLM output as JSON
        try:
            # Robustly extract JSON part from the LLM response
            # Find the first '{' and the last '}'
            start_brace = llm_output_str.find('{')
            end_brace = llm_output_str.rfind('}')

            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                json_str_to_parse = llm_output_str[start_brace : end_brace+1]
            else:
                # If no braces found, or in wrong order, attempt to parse the whole thing
                # (it might be a plain JSON string already, or it will fail)
                json_str_to_parse = llm_output_str

            trade_decision_data = json.loads(json_str_to_parse)

            # Ensure it's a dictionary and has the essential keys
            if not isinstance(trade_decision_data, dict) or \
               "action" not in trade_decision_data or \
               "symbol" not in trade_decision_data:
                raise json.JSONDecodeError("Missing essential keys in parsed JSON output", json_str_to_parse, 0)

            # Add a default conviction score if action is HOLD and score is missing
            if trade_decision_data.get("action") == "HOLD" and "conviction_score" not in trade_decision_data:
                trade_decision_data["conviction_score"] = 0.0
            # Ensure conviction score is present for BUY/SELL
            if trade_decision_data.get("action") in ["BUY", "SELL"] and "conviction_score" not in trade_decision_data:
                 # Fallback if LLM forgets conviction score for BUY/SELL
                trade_decision_data["conviction_score"] = 0.5 # Default to neutral if missing
                if "reason" in trade_decision_data:
                    trade_decision_data["reason"] += " (Conviction score defaulted as it was missing from LLM output)"
                else:
                    trade_decision_data["reason"] = "(Conviction score defaulted as it was missing from LLM output)"


        except json.JSONDecodeError as e:
            print(f"TRADER_NODE: Error parsing LLM JSON output: {e}. Raw output: '{llm_output_str}'")
            # Fallback to a HOLD decision if JSON parsing fails
            trade_decision_data = {
                "action": "HOLD",
                "symbol": company_name,
                "conviction_score": 0.0,
                "reason": f"Failed to parse LLM decision. LLM Raw Output: {llm_output_str}"
            }

        return {
            "messages": [response], # Keep the original response for logging/history
            "trader_investment_plan": trade_decision_data, # This will be the structured decision
            "final_trade_decision": trade_decision_data, # Ensuring this key also holds the final decision
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
