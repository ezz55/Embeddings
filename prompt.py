from dotenv import load_dotenv

load_dotenv() 

from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from redundant_filter_retriever import RedundantFilterRetriver

import langchain
langchain.debug=True

embeddings = OpenAIEmbeddings()

chat= ChatOpenAI()

db = Chroma(persist_directory="emb",
            embedding_function=embeddings,
            )

retriever = RedundantFilterRetriver(
  embeddings= embeddings,
  chroma= db
)

chain = RetrievalQA.from_chain_type(
  llm= chat,
  retriever=retriever,
  chain_type="stuff"
)

result= chain.invoke("What is an intresting fact about The English Language?")

print(result)