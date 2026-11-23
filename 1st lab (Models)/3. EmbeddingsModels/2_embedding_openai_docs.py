from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

document = [
    "Islamabad is the capital of pakistan",
    "Paris is the capital of France",
    "Washington is the capital of US"
]

result = embedding.embed_documents(document)

print(str(result))