# AI Assistant: English Tutor & Google Terms Consultant

A versatile Streamlit-based web application that combines an interactive English language learning assistant with a Retrieval-Augmented Generation (RAG) system to consult users on Google's Terms of Service. 

Powered by the **LangGraph** React Agent framework, the assistant intelligently switches modes: it handles grammar and translation queries natively, and queries a **Pinecone vector database** using specialized tools when asked about Google's policies.

## Features

- **English Language Tutor:** - Translates words/phrases and provides real-world sentence examples.
  - Translates full sentences and breaks down complex grammar structures (e.g., passive voice, *there is/are* syntax) for beginners.
- **Google Terms Consultant (RAG):**
  - Seamlessly splits, embeds, and stores large policy documents.
  - Automatically search queries for optimized semantic matching in Pinecone, while returning the final answer in the user's preferred language.
- **Robust Architecture:** Built using object-oriented principles (OOP) encapsulating LangChain, LangGraph, and Streamlit state management.

## Tech Stack

- **Framework:** [Streamlit](https://streamlit.io/) (Web UI)
- **Orchestration:** [LangGraph](https://github.com/langchain-ai/langgraph) (React Agent State Machine) & [LangChain](https://github.com/langchain-ai/langchain)
- **LLM & Embeddings:** Google Gemini (`gemini-2.5-flash` & `text-embedding-004`) via `langchain-google-genai`
- **Vector Database:** [Pinecone](https://www.pinecone.io/) (Serverless AWS index)
