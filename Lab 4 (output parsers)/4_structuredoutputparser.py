from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

response_schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about topic'),
]

parser = StructuredOutputParser.from_response_schemas(response_schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic': 'black hole'})
print(result) 