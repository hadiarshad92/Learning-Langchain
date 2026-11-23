# used when we need dynamic prompt for chat messages

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', "You're a helpful {domain} expert"),
    ('human', "Explain is simple term what is {topic}")
])

prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'dusra'})

print(prompt)