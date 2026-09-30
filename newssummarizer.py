from dotenv import load_dotenv
import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -----------------------------------
# 1. Load environment variables
# -----------------------------------

load_dotenv()


# -----------------------------------
# 2. Check API keys
# -----------------------------------

gemini_api_key = os.getenv("GEMINI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is missing from your .env file")

if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY is missing from your .env file")

print("Gemini API key loaded successfully!")
print("Tavily API key loaded successfully!")


# -----------------------------------
# 3. Create Tavily Search Tool
# -----------------------------------

search_tool = TavilySearchResults(
    max_results=5
)


# -----------------------------------
# 4. Create Gemini Model
# -----------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=gemini_api_key
)


# -----------------------------------
# 5. Create Prompt
# -----------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Summarize the following news into clear bullet points.

News:
{news}
"""
)


# -----------------------------------
# 6. Create LCEL Chain
# -----------------------------------

chain = prompt | llm | StrOutputParser()


# -----------------------------------
# 7. Search for Latest AI News
# -----------------------------------

news_result = search_tool.invoke(
    "Latest AI news of 2026"
)


# -----------------------------------
# 8. Send Search Results to Gemini
# -----------------------------------

result = chain.invoke({
    "news": news_result
})


# -----------------------------------
# 9. Print Summary
# -----------------------------------

print("\n===== AI NEWS SUMMARY =====\n")
print(result)


# -----------------------------------
# 10. Display Tool Information
# -----------------------------------

print("\n===== TOOL INFORMATION =====")

print("\nTool Name:")
print(search_tool.name)

print("\nTool Description:")
print(search_tool.description)

print("\nTool Arguments:")
print(search_tool.args)
