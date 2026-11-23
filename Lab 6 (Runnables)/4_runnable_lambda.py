from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,  # Used to wrap custom Python functions into LCEL-compatible runnables
    RunnableParallel,  # Used to run multiple operations concurrently on the same data
    RunnablePassthrough,  # Used to pass the input data through unchanged
    RunnableSequence,  # Explicitly chains multiple runnables sequentially
)
from langchain_openai import ChatOpenAI

# Load environment variables (like OPENAI_API_KEY) from a .env file
load_dotenv()


# Custom helper function to count the number of words in a given text string
def word_count(text):
    return len(text.split())


# Initialize the OpenAI chat model
model = ChatOpenAI()

# Define the Prompt Template to instruct the model to write a joke about a given topic
prompt = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)

# Initialize the standard string output parser to extract raw text from the model output
parser = StrOutputParser()

# Create the first sequential chain (equivalent to prompt | model | parser)
# This takes the topic dictionary and outputs the generated joke as a plain string.
joke_generator_chain = RunnableSequence(prompt, model, parser)

# (Note: The first parallel_chain definition is overwritten by the second one below,
# but it demonstrates using a predefined Python function via RunnableLambda).
parallel_chain = RunnableParallel(
    {"joke": RunnablePassthrough(), "word_count": RunnableLambda(word_count)}
)

# Alternate way: Defining the parallel chain using an inline lambda function
# to directly count words on the input text stream.
parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word_count": RunnableLambda(lambda x: len(x.split())),
    }
)

# Combine the joke generator sequence and the parallel word-counter chain sequentially
final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

# Invoke the final chain with a topic dictionary
result = final_chain.invoke({"topic": "AI"})

# Format the dictionary results into a custom presentation string
final_result = "{} \n word count - {}".format(
    result["joke"], result["word_count"]
)

# Print the final formatted output (Fixed missing closing parenthesis)
print(final_result)

# ==========================================
# WHAT THIS CODE IS IMPLEMENTING:
# This script implements a "Joke Generator & Word Counter Pipeline".
# 1. It takes a topic (e.g., 'AI') and generates a joke using a prompt, model, and string parser.
# 2. It then passes the generated joke into a parallel execution block:
#    - Branch A (`RunnablePassthrough`) keeps the raw joke string intact.
#    - Branch B (`RunnableLambda`) runs a custom Python function to count the words in that joke.
# 3. Finally, it combines both outputs into a clean custom text string displaying both the joke and its word count.
# ==========================================