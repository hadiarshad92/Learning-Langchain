from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableBranch,  # Used for conditional routing/branching logic based on runtime data
    RunnableLambda,  # (Imported, though not explicitly used here)
    RunnableParallel,  # (Imported, though not explicitly used here)
    RunnablePassthrough,  # Used to pass data through unchanged (fallback option in the branch)
    RunnableSequence,  # Explicitly chains multiple runnables sequentially
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables (like OPENAI_API_KEY) from a .env file
load_dotenv()

# Define Prompt 1: Instructs the model to write a detailed report on a given topic
prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)

# Define Prompt 2: Instructs the model to summarize a provided block of text
prompt2 = PromptTemplate(
    template="Summarize the following text \n {text}", input_variables=["text"]
)

# Initialize the OpenAI chat model
model = ChatOpenAI()

# Initialize the standard string output parser to extract raw text from model outputs
parser = StrOutputParser()

# Stage 1: Create the report generation chain (Topic -> Prompt 1 -> Model -> Parser -> Raw Text)
report_gen_chain = prompt1 | model | parser

# Stage 2: Create a conditional branch chain based on the text length
branch_chain = RunnableBranch(
    # Condition: If word count of the input text is greater than 300 words, summarize it
    (lambda x: len(x.split()) > 300, prompt2 | model | parser),
    # Default Fallback: If 300 words or fewer, return the text completely unchanged
    RunnablePassthrough(),
)

# Combine Stage 1 (report generation) and Stage 2 (conditional summarization) into a final sequence
final_chain = RunnableSequence(report_gen_chain, branch_chain)

# Invoke the final chain with a topic dictionary and print the final output
print(final_chain.invoke({"topic": "Russia vs Ukraine"}))

# ==========================================
# WHAT THIS CODE IS IMPLEMENTING:
# This script implements a "Conditional Report Summarization Pipeline".
# 1. It takes a topic (e.g., 'Russia vs Ukraine') and generates a detailed report using Prompt 1.
# 2. It then passes the generated report into a conditional branch (`RunnableBranch`):
#    - Condition 1: If the generated report is longer than 300 words, it routes the text into Prompt 2 to summarize it.
#    - Fallback: If it is 300 words or fewer, it bypasses summarization and passes the text through unchanged (`RunnablePassthrough`).
# 3. Finally, it outputs either the summarized report or the original report depending on the length condition.
# ==========================================
