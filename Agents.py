from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

# Tavily
from tavily import TavilyClient

# Agent
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call


# =====================================================
# API KEYS
# =====================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY is missing from .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")


print("API keys loaded successfully!")


# =====================================================
# 🌦️ WEATHER TOOL
# =====================================================

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city in India."""

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city},IN"
        f"&appid={OPENWEATHER_API_KEY}"
        "&units=metric"
    )

    response = requests.get(url, timeout=10)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


# =====================================================
# 📰 NEWS TOOL
# =====================================================

tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)


@tool
def get_news(city: str) -> str:
    """Get latest news about a city."""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []

    for r in results:

        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(
            f"- {title}\n"
            f"  URL: {url}\n"
            f"  Summary: {snippet[:150]}..."
        )

    return (
        f"Latest news in {city}:\n\n"
        + "\n\n".join(news_list)
    )


# =====================================================
# 🧠 GEMINI LLM
# =====================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=GEMINI_API_KEY
)


# =====================================================
# 👤 HUMAN APPROVAL
# =====================================================

@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""

    tool_name = request.tool_call["name"]

    confirm = input(
        f"\nAgent wants to call '{tool_name}'. "
        "Approve? (yes/no): "
    )

    if confirm.lower() != "yes":

        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)


# =====================================================
# 🤖 CREATE AGENT
# =====================================================

agent = create_agent(
    llm,
    tools=[
        get_weather,
        get_news
    ],
    system_prompt=(
        "You are a helpful city assistant. "
        "Use the weather tool when the user asks about weather. "
        "Use the news tool when the user asks about latest news."
    ),
    middleware=[
        human_approval
    ]
)


# =====================================================
# 💬 CHAT LOOP
# =====================================================

print("\n================================")
print("        CITY AI AGENT")
print("================================")
print("Type 'exit' to quit.\n")


while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    try:

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            }
        )

        print("\nBot:")
        print(result["messages"][-1].content)
        print()

    except Exception as e:

        error_message = str(e)

        # =================================================
        # Gemini 429 Quota Error
        # =================================================

        if (
            "429" in error_message
            or "RESOURCE_EXHAUSTED" in error_message
            or "quota" in error_message.lower()
        ):

            print("\n⚠️ Gemini API quota exceeded.")
            print("Your Gemini free-tier request limit has been reached.")
            print("Please wait for the quota to reset and try again.")
            print()

        # =================================================
        # Other Errors
        # =================================================

        else:

            print("\n❌ Something went wrong.")
            print(f"Error: {error_message}")
            print()