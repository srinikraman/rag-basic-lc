# Import necessary libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from openai import OpenAI
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load the PDF document
# Ensure 'input/rag_input_1.pdf' exists in your directory
loader = PyPDFLoader("input/rag_input_1.pdf")
documents = loader.load()
print("Number of documents loaded: ", len(documents))

# Data Preprocessing

# Split Documents into smaller chunks
# chunk_size: maximum characters per chunk
# chunk_overlap: characters of overlap between chunks to maintain context
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print("Number of text chunks created: ", len(docs))

# Create Embeddings + FAISS Vector Store
# OpenAIEmbeddings converts text to vector representations
# FAISS stores these vectors for efficient similarity search
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstorage = FAISS.from_documents(docs, embeddings)
print("FAISS index built successfully")

# Create Retriever & QA chain
# ChatOpenAI: The LLM that will generate answers
# RetrievalQA: Chains the LLM and the retriever to answer questions based on retrieved docs
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                    retriever=vectorstorage.as_retriever(),
                                    return_source_documents=True)

print("\n--- RAG System Ready ---")
print("Type 'exit' to quit the program.")

while True:
    query = input("\nEnter your query: ")
    if query.lower() == "exit":
        print("Exiting...")
        break
    
    result = qa_chain.invoke({"query": query})
    print("Answer: ", result["result"])