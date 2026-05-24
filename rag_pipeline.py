from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

def load_and_embed_docs():
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def generate_answer(vectorstore, query):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)
    return qa_chain.run(query)
