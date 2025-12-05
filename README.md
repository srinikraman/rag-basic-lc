# RAG Basic System

This project implements a basic Retrieval-Augmented Generation (RAG) system using Python, LangChain, and OpenAI. It allows users to query information from a PDF document.

## Features

-   **PDF Loading**: Loads a PDF document from the `input` directory.
-   **Text Splitting**: Splits the document into manageable chunks for processing.
-   **Vector Store**: Uses FAISS and OpenAI Embeddings to create a searchable vector index.
-   **Interactive QA**: Provides a command-line interface for users to ask questions about the PDF content.
-   **LangChain Integration**: Utilizes LangChain for efficient chaining of components.

## Prerequisites

-   Python 3.8 or higher
-   An OpenAI API Key

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install dependencies**:
    ```bash
    pip3 install -r requirements.txt
    ```

3.  **Set up Environment Variables**:
    -   Create a `.env` file in the root directory.
    -   Add your OpenAI API key:
        ```env
        OPENAI_API_KEY=your_openai_api_key_here
        ```

## Usage

1.  **Prepare Input**:
    -   Place your PDF file in the `input` directory.
    -   Ensure the filename matches the one in `main.py` (default: `rag_input_1.pdf`) or update the code to match your filename.

2.  **Run the Application**:
    ```bash
    python3 main.py
    ```

3.  **Interact**:
    -   The system will initialize and index the document.
    -   Type your question when prompted.
    -   Type `exit` to quit the application.

## Project Structure

-   `main.py`: The main script containing the RAG logic and interactive loop.
-   `requirements.txt`: List of Python dependencies.
-   `input/`: Directory to store input PDF files.
-   `.env`: Configuration file for API keys (not committed to version control).

## Architecture
```mermaid
flowchart TD
    subgraph Ingestion [Data Ingestion]
        A[PDF Document] -->|1-PyPDFLoader| B[Documents]
        B -->|2-CharacterTextSplitter| C[Text Chunks]
        C -->|3-OpenAIEmbeddings| D[Vector Embeddings]
        D -->|4-FAISS| E[Vector Store]
    end

    subgraph Query [Query Processing]
		    F[User Query] -->|OpenAIEmbeddings| G[Query Vector]
        E -->|Retriever| H[Relevant Context]
        H -->|RetrievalQA| I[LLM]
        G -->|Similarity Search| E
        I --> J[Final Answer]
    end
```

## How it Works

### Data Ingestion 
**This stage is run once to prepare the custom data (the PDF) so the LLM can use it.**
1. **PDF Document $\rightarrow$ Documents:** The process begins with your PDF Document. The ***PyPDFLoader*** tool reads the PDF and converts its contents into a list of digital Documents, which are essentially strings of text.
2. **Documents $\rightarrow$ Text Chunks:** The long text documents are too large for the LLM to process all at once, hence the ***CharacterTextSplitter*** breaks the text into smaller, manageable, fixed-size Text Chunks. This is a crucial step for efficient retrieval.
3. **Text Chunks $\rightarrow$ Vector Embeddings:** Each text chunk is passed to ***OpenAIEmbeddings***. This service converts the meaning and context of the text chunk into a list of numbers called a Vector Embedding. Texts with similar meanings will have vectors that are numerically "close" to each other.
4. **Vector Embeddings $\rightarrow$ Vector Store:** Finally, all the vector embeddings are stored in a ***FAISS index***. FAISS is a library designed for efficient similarity search, acting as the Vector Store. This store is the specialized, searchable knowledge base.

### Query Processing
**This stage runs every time a user asks a question.**
1. **User Query $\rightarrow$ Query Vector:** When the User Query comes in, it is also converted into a Query Vector using the exact same ***OpenAIEmbeddings*** process used during ingestion. This ensures the query and the documents are represented in the same mathematical space.
2. **Query Vector $\rightarrow$ Vector Store $\rightarrow$ Relevant Context:** The query vector is used to perform a ***Similarity Search*** against the Vector Store. This search identifies the stored vector embeddings that are mathematically closest (most similar in meaning) to the user's question. A ***Retriever*** then fetches the original text chunks corresponding to those vectors, resulting in the Relevant Context.
3. **Relevant Context $\rightarrow$ LLM:** The retrieved Relevant Context is combined with the original User Query and passed to the LLM (Large Language Model). This entire process is managed by the ***RetrievalQA*** chain, which effectively creates a prompt telling the LLM to answer the user's question only using the following context.
4. **LLM $\rightarrow$ Final Answer:** The LLM processes the query and the provided context to generate a concise, accurate Final Answer.