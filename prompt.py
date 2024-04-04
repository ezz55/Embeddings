from dotenv import load_dotenv

load_dotenv() 

from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA


embeddings = OpenAIEmbeddings()

chat= ChatOpenAI()

db = Chroma(persist_directory="emb",
            embedding_function=embeddings,
            )

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
  llm= chat,
  retriever=retriever,
  chain_type="stuff"
)

result= chain.invoke("What is an intresting fact about The English Language?")

print(result)