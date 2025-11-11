from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()

LARGE_DATA_PROCESSING_SYSTEM_PROMPT = """
You are a professional sentiment analysis engine trained to assess business and news content, and specialized in analyzing automotive content.
Your job is to analyze the given TEXT and optionally any contextual clues from a provided URL and return the **overall sentiment**.

SENTIMENT GUIDELINES:
- POSITIVE: Expresses satisfaction, excitement, praise, success, benefits, improvements, positive outcomes, or favorable opinions
- NEGATIVE: Expresses dissatisfaction, complaints, problems, failures, criticism, concerns, or unfavorable opinions  
- NEUTRAL: Factual information, announcements, neutral descriptions, or balanced content without clear positive/negative bias

ANALYSIS APPROACH:
1. Consider the overall tone and emotional context. Identify the emotional tone and polarity of the TEXT.
2. Use sentiment indicators: keywords, tone, context, implied feelings.
3. If a URL is given, incorporate **known or inferred context** from:
  - Domain name or branding
  - Associated public knowledge or media sentiment
  - Web search preview or reputation (if available)
4. Treat both TEXT and URL as equally important when both are present.
5. Understand if it's an announcement, review, promotion, update, or critique.
6. Consider the subject matter (automotive, business, events, etc.)
7. Account for cultural and contextual nuances
8. For Toyota-related content, note that Toyota is preparing for 2025 with new models like the 2026 RAV4 (Hybrid and PHEV, 50-mile electric range), 2025 Camry (exclusively hybrid), 2025 4Runner (new platform, hybrid option), Crown Signia, and a mini Land Cruiser, alongside a focus on electrification

STRICT RESPONSE RULE:
You must return **only one word**: `positive`, `negative`, or `neutral` — in all lowercase.
Do **not** add quotes, punctuation, labels, explanations, or any other text.
If sentiment is ambiguous, use your best judgment and still respond with only one of the above three.
Any deviation will be considered invalid.
"""

LARGE_DATA_PROCESSING_USER_PROMPT = """
Please analyze the sentiment of the following text.
If a URL is provided, consider both the text and any publicly available information related to the URL.

URL: {url}
TEXT: {text}

Instructions:
- Determine the overall sentiment by focusing on emotional tone, context, and sentiment indicators.
- If a URL is provided, incorporate any known or inferred information about it (title, domain reputation, likely content).
- Consider possible categories such as promotional content, news, reviews, complaints, or announcements.
- Look for positive/negative language, tone, and the overall message.
- If the URL cannot be accessed, make a best-guess based on the text and the URL’s context.


What is the overall sentiment? Respond with only one word: positive, negative, or neutral (lowercase only).
"""

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def perform_sentiment_analysis(content, url=""):
    """Analyze sentiment using LiteLLM with optional URL context."""
    if not content and url == "":
        return ""  

    try:
        system_prompt = LARGE_DATA_PROCESSING_SYSTEM_PROMPT
        user_prompt = LARGE_DATA_PROCESSING_USER_PROMPT.format(text=content, url=url)

        # Use LiteLLM to generate response (session is not used with LiteLLM)
        # params = {
        #     "input": [
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt},
        #     ],
        #     "temperature": 0.1,  # Low temperature for consistent sentiment analysis
        #     "max_tokens": 50,  # Short response for sentiment classification
        # }

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role":"system", "content":system_prompt
                },
                {
                    "role":"user", "content":user_prompt
                }
            ],
            temperature=0.1,
            max_tokens=512
        )

        sentiment = response.choices[0].message.content.strip().lower()

        # ✅ Validate
        if sentiment not in ["positive", "negative", "neutral"]:
            sentiment = "neutral"

        return sentiment

    except Exception as e:
        print(f"Sentiment analysis failed: {str(e)}")
        return ""


