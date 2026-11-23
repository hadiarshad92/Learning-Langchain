from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

model = ChatOpenAI()

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=[],
    # partial_variables is used specify formatting in JsonOutputParser
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# create promt, call model, parse the output
# prompt = template.format(topic='black hole')
# response = model.invoke(prompt)
# final_result = parser.parse(response.content)  


# or use chain
chain = template | model | parser
final_result = chain.invoke({'topic': 'black hole'})

print(final_result)
print(type(final_result))