from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,  # Used for conditional routing based on runtime data (like an if-elif-else statement)
    RunnableLambda,  # Used to wrap custom Python functions into LCEL-compatible runnables
    )
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

parser1 = StrOutputParser()

# Defining a Pydantic schema for feedback sentiment so the output is strictly structured
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# Pydantic parser that instructs the model on schema formatting and validates the output
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Define Prompt 1: Instructs the model to classify feedback and injects schema format instructions
prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

# Define the classifier chain which outputs a validated Pydantic 'Feedback' object
classifier_chain = prompt1 | model | parser2

# Define the classifier chain which outputs a validated Pydantic 'Feedback' object
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback'],
)

# Define Prompt 3: Response template specifically for negative feedback
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback'],
)

# Define Prompt 3: Response template specifically for negative feedback
branch_chain = RunnableBranch(
    # Route 1: If sentiment is positive, run prompt2 through the model and string parser
    (lambda x: x.sentiment == "positive", prompt2 | model | parser1),
    # Route 2: If sentiment is negative, run prompt3 through the model and string parser
    (lambda x: x.sentiment == "negative", prompt3 | model | parser1),
    # Default fallback: Executed if none of the above conditions match
    RunnableLambda(lambda x: "Could not find sentiment"),
)

# Combine the classification chain sequentially with the conditional branch chain
chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a beautiful phone'}))

chain.get_graph().print_ascii()