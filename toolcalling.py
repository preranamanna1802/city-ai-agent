from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from rich import print


# -----------------------------------
# 1. Load environment variables
# -----------------------------------

load_dotenv()


# -----------------------------------
# 2. Check Gemini API key
# -----------------------------------

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is missing from your .env file")


print("Gemini API key loaded successfully!")


# -----------------------------------
# 3. Create a Tool
# -----------------------------------

@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in a given text."""
    return len(text)


# Store tools in a dictionary
tools = {
    "get_text_length": get_text_length
}


# -----------------------------------
# 4. Create Gemini Model
# -----------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=gemini_api_key,
    temperature=0.3
)


# -----------------------------------
# 5. Bind Tool to Gemini
# -----------------------------------

llm_with_tool = llm.bind_tools([
    get_text_length
])


# -----------------------------------
# 6. Get User Input
# -----------------------------------

messages = []

prompt = input("You: ")

query = HumanMessage(content=prompt)

messages.append(query)


# -----------------------------------
# 7. First Gemini Call
# -----------------------------------

result = llm_with_tool.invoke(messages)

messages.append(result)


# -----------------------------------
# 8. Check Whether Gemini Requested
#    a Tool
# -----------------------------------

if result.tool_calls:

    for tool_call in result.tool_calls:

        tool_name = tool_call["name"]

        tool_result = tools[tool_name].invoke(tool_call)

        messages.append(tool_result)


    # -----------------------------------
    # 9. Send Tool Result Back to Gemini
    # -----------------------------------

    final_result = llm_with_tool.invoke(messages)

    print("\nAssistant:")
    print(final_result.content)

else:

    print("\nAssistant:")
    print(result.content)
