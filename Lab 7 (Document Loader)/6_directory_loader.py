from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='5_books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(len(docs))
print(docs[325].page_content)
print(docs[325].metadata)

# use lazy_load in documents are huge
docs = loader.lazy_load()

for document in docs:
    print(document.metadata) 