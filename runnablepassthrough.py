from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


# Load environment variables
load_dotenv()


# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from your .env file")


# Create Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0.3
)


# Output parser
parser = StrOutputParser()


# Prompt for generating code
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator."),
    ("human", "{topic}")
])


# Prompt for explaining code
explain_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant who explains code in simple terms."
    ),
    (
        "human",
        "Explain the following code in simple words:\n{code}"
    )
])


# Generate the code
seq = code_prompt | model | parser


# Pass the generated code to two parallel branches
seq2 = RunnableParallel(
    {
        "code": RunnablePassthrough(),
        "explanation": explain_prompt | model | parser
    }
)


# Combine both sequences
chain = seq | seq2


# Run the chain
result = chain.invoke({
    "topic": "Please write a code of palindrome in Python."
})


# Print generated code
print("\n===== GENERATED CODE =====")
print(result["code"])


# Print explanation
print("\n===== EXPLANATION =====")
print(result["explanation"])
