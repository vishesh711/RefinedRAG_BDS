# Refined Rag - Private Chat with Your Documents

> Completely local RAG with chat UI

<a href="https://www.mlexpert.io/bootcamp" target="_blank">
  <img src="[[https://raw.githubusercontent.com/curiousily/ragbase/master/.github/ui.png](https://github.com/vishesh711/RefinedRAG_BDS/blob/main/images/Demo.jpeg?raw=true)](https://github.com/vishesh711/RefinedRAG_BDS/blob/main/images/Demo.jpeg?raw=true)">
</a>

## Demo

## Installation

Clone the repo:

```sh
git clone git@github.com:vishes711/RefinedRAG_BDS.git
cd RefinedRAG_BDS
```

Install the dependencies (requires Poetry):

```sh
python -m poetry install
```

Fetch your LLM (gemma2:9b by default) (Optional):

```sh
ollama pull gemma2:9b
```

Run the Ollama server (Optional)

```sh
ollama serve
```

## Add Groq API Key (Optional)

You can also use the Groq API to replace the local LLM, for that you'll need a `.env` file with Groq API key:

```sh
GROQ_API_KEY=YOUR API KEY
```

Start Refined Rag:

```sh
python -m streamlit run app.py
```

## Architecture

<a href="https://www.mlexpert.io/bootcamp" target="_blank">
  <img src="[https://raw.githubusercontent.com/curiousily/ragbase/master/.github/architecture.png](https://github.com/vishesh711/RefinedRAG_BDS/blob/main/images/architecture.png?raw=true)">
</a>

### Ingestor

Extracts text from PDF documents and creates chunks (using semantic and character splitter) that are stored in a vector databse

### Retriever

Given a query, searches for similar documents, reranks the result and applies LLM chain filter before returning the response.

### QA Chain

Combines the LLM with the retriever to answer a given user question

## Tech Stack

- [Ollama](https://ollama.com/) - run local LLM
- [Groq API](https://groq.com/) - fast inference for mutliple LLMs
- [LangChain](https://www.langchain.com/) - build LLM-powered apps
- [Qdrant](https://qdrant.tech/) - vector search/database
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) - fast reranking
- [FastEmbed](https://qdrant.github.io/fastembed/) - lightweight and fast embedding generation
- [Streamlit](https://streamlit.io/) - build UI for data apps
- [PDFium](https://pdfium.googlesource.com/pdfium/) - PDF processing and text extraction
