from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,  # Explicitly using RunnableSequence (the programmatic equivalent of the '|' pipe operator)
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables (like OPENAI_API_KEY) from a .env file
load_dotenv()

# Define Prompt 1: Instructs the model to write a joke about a given topic
prompt1 = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)

# Initialize the OpenAI chat model
model = ChatOpenAI()

# Initialize the standard string output parser to extract raw text from model outputs
parser = StrOutputParser()

# Define Prompt 2: Instructs the model to explain a provided joke text
prompt2 = PromptTemplate(
    template="Explain the following joke - {text}", input_variables=["text"]
)

# Create the first sequence (equivalent to prompt1 | model | parser)
# This chain takes a topic dictionary and outputs the generated joke as a raw string
joke_gen_chain = RunnableSequence(prompt1, model, parser)

# Create a parallel chain with a dictionary mapping:
# 1. 'joke': Uses RunnablePassthrough to carry the original joke string forward unchanged.
# 2. 'explanation': Passes the joke string into prompt2, model, and parser to generate an explanation.
parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explanation": RunnableSequence(prompt2, model, parser),
    }
)

# Combine the joke generator sequence and the parallel chain sequentially into a final chain
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Invoke the final chain with a topic dictionary and print the resulting dictionary (containing both 'joke' and 'explanation')
print(final_chain.invoke({"topic": "cricket"}))


# ==========================================
# WHAT THIS CODE IS IMPLEMENTING:
# This script implements a "Joke Generator & Explainer Pipeline". 
# 1. It takes a topic (e.g., 'cricket') and generates a joke using Prompt 1.
# 2. It then takes that generated joke and runs it through a parallel branch:
#    - Branch A (`RunnablePassthrough`) keeps the original joke intact.
#    - Branch B (`RunnableSequence`) passes the joke into Prompt 2 to generate an explanation.
# 3. Finally, it combines both into a single dictionary containing both the joke and its explanation.
# ==========================================