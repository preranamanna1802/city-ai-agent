from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Load .env file
load_dotenv()


# Check whether Gemini API key is loaded
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is missing from your .env file")

print("Gemini API key loaded successfully!")


# 1. Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words."
)


# 2. Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=gemini_api_key,
    temperature=0.3
)


# 3. Output Parser
parser = StrOutputParser()


# 4. LCEL Chain
chain = prompt | model | parser


# 5. Invoke
result = chain.invoke({
    "topic": "Machine Learning"
})


# 6. Print Result
print("\nResult:")
print(result)
