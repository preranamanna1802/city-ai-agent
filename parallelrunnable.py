from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser


# Load .env file
load_dotenv()


# Check API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from your .env file")

print("Gemini API key loaded successfully!")


# Create Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key
)


# Prompt 1
summary_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 2 simple sentences."
)


# Prompt 2
importance_prompt = ChatPromptTemplate.from_template(
    "Explain why {topic} is important in simple words."
)


# Output parser
parser = StrOutputParser()


# Chain 1
summary_chain = summary_prompt | model | parser


# Chain 2
importance_chain = importance_prompt | model | parser


# Run both chains in parallel
chain = RunnableParallel(
    summary=summary_chain,
    importance=importance_chain
)


# Input
result = chain.invoke({
    "topic": "Machine Learning"
})


# Display results
print("\n===== RESULT =====")

print("\nSummary:")
print(result["summary"])

print("\nImportance:")
print(result["importance"])
