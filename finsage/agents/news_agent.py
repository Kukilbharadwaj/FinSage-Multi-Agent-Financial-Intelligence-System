# agents/news_agent.py
# Fetches news headlines and scores sentiment.
# Model: GROQ_FAST (llama-3.1-8b-instant) — fast sentiment scoring on many headlines
#
# Stage 1 agent — no upstream dependencies. Does NOT use RAG.
# Writes: state["news_analysis"] (read by market_agent)

import json
from groq import Groq
from config.settings import settings
from config.models import GROQ_FAST
from tools.news_tool import get_news


def run(state: dict) -> dict:
    """
    Fetch news for the detected stock/index and score overall sentiment.

    Writes state["news_analysis"] with structured output:
        - headlines: list of article dicts
        - sentiment_score: -1.0 to 1.0
        - key_events: one-line summary of market mood
        - market_mood: "positive" | "negative" | "neutral"
    """
    try:
        entities = state.get("entities", {})

        # Determine search query
        query = (
            entities.get("stock")
            or entities.get("index")
            or "Indian stock market"
        )

        # Fetch news articles
        articles = get_news(query, limit=10)

        if not articles:
            state["news_analysis"] = {
                "headlines": [],
                "sentiment_score": 0.0,
                "key_events": "No recent news found",
                "market_mood": "neutral",
            }
            state["trace"].append("news_agent → no articles found, sentiment = 0.0")
            return state

        # Format top headlines for sentiment analysis
        top_headlines = articles[:8]
        headlines_text = "\n".join(
            f"{i+1}. {article['title']}" for i, article in enumerate(top_headlines)
        )

        # Call Groq for sentiment scoring
        try:
            client = Groq(api_key=settings.GROQ_API_KEY)

            sentiment_prompt = f"""Analyze the overall sentiment of these Indian financial news headlines.

Headlines:
{headlines_text}

Score the overall sentiment from -1.0 (very negative/bearish) to 1.0 (very positive/bullish).
Consider: market crash fears = negative, rally/gains = positive, mixed news = near 0.

Respond ONLY with valid JSON in this exact format, nothing else:
{{"score": 0.0, "summary": "one line summary of overall market mood", "mood": "positive|negative|neutral", "key_events": "most important event from headlines"}}"""

            response = client.chat.completions.create(
                model=GROQ_FAST,
                messages=[
                    {"role": "system", "content": "You are a financial news sentiment analyzer. Respond only with JSON."},
                    {"role": "user", "content": sentiment_prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )

            raw_response = response.choices[0].message.content.strip()

            # Parse JSON response
            json_str = raw_response
            if "```" in json_str:
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()

            result = json.loads(json_str)
            score = float(result.get("score", 0.0))
            score = max(-1.0, min(1.0, score))  # clamp

            mood = result.get("mood", "neutral")
            if mood not in ("positive", "negative", "neutral"):
                mood = "positive" if score > 0.2 else ("negative" if score < -0.2 else "neutral")

            state["news_analysis"] = {
                "headlines": articles,
                "sentiment_score": round(score, 2),
                "key_events": result.get("key_events", result.get("summary", "")),
                "market_mood": mood,
            }

        except (json.JSONDecodeError, Exception):
            state["news_analysis"] = {
                "headlines": articles,
                "sentiment_score": 0.0,
                "key_events": "Sentiment analysis unavailable",
                "market_mood": "neutral",
            }

        score = state["news_analysis"]["sentiment_score"]
        mood = state["news_analysis"]["market_mood"]
        state["trace"].append(f"news_agent → sentiment {score} ({mood})")

    except Exception as e:
        state["news_analysis"] = {
            "headlines": [],
            "sentiment_score": 0.0,
            "key_events": f"Error: {str(e)[:80]}",
            "market_mood": "neutral",
        }
        state["trace"].append(f"news_agent → ERROR: {str(e)[:100]}")

    return state
