import os
import json
from uuid import uuid4
import streamlit as st
import dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document


class AIAssistantApp:
    def __init__(self):
        """
        Initializes credentials, language models, vector database, and the LangGraph agent.
        """
        dotenv.load_dotenv()

        # Load API Keys
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")

        # Initialize Models
        self.llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', api_key=self.gemini_api_key)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",
                                                       google_api_key=self.gemini_api_key)

        # Initialize Pinecone Vector DB
        self.index_name = "soup"
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.vector_store = self._setup_vector_store()

        # Initialize LangGraph Agent with the bound method as a tool
        self.agent = create_react_agent(model=self.llm, tools=[self.get_conditions_info])

    def _setup_vector_store(self) -> PineconeVectorStore:
        """
        Internal method to ensure Pinecone index exists and returns the VectorStore instance.
        """
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        index = self.pc.Index(self.index_name)
        return PineconeVectorStore(index=index, embedding=self.embeddings)

    def get_conditions_info(self, query: str) -> list:
        """
        Tool function for searching information about Google conditions inside the Vector DB.
        Returns a list of similar documents based on the user's query.
        """
        return self.vector_store.similarity_search(query, k=5)

    def populate_database(self, file_path: str = "data/lesson_rag/huge_file.txt"):
        """
        Optional method to process raw text file, chunk it, and save to Pinecone and JSON.
        """
        try:
            with open(file_path, 'r', encoding="utf-8") as file:
                huge_file_text = file.read()

            file_split_blocks = huge_file_text.split("\n\n\n")
            docs = []
            id_map = {}

            for block in file_split_blocks:
                block_parts = block.split("\n")
                block_title = block_parts[0] if block_parts else "Untitled Block"

                doc = Document(
                    page_content=block,
                    metadata={
                        "file name": "Google Conditions",
                        "block title": block_title,
                        "author": "Google Programmer",
                    }
                )
                docs.append(doc)

            ids = [str(uuid4()) for _ in range(len(docs))]

            # self.vector_store.add_documents(documents=docs, ids=ids)

            for doc, doc_id in zip(docs, ids):
                id_map[doc.metadata["block title"]] = doc_id

            with open('ids.json', 'w', encoding="utf-8") as f:
                json.dump(id_map, f, indent=2, ensure_ascii=False)

            st.success("Database successfully populated!")
        except Exception as e:
            st.error(f"Error populating database: {e}")

    def _init_session_state(self):
        """
        Internal method to inject the initial system prompt into Streamlit's session state.
        """
        if 'history' not in st.session_state:
            st.session_state['history'] = [
                SystemMessage(
                    content=(
                        "You are a versatile and polite AI assistant with two main responsibilities:\n\n"
                        "1. English Language Tutor: If the user asks to translate a word or phrase, provide the translation "
                        "and an example sentence. If they ask to translate a sentence, provide the translation and explain "
                        "the grammar (e.g., 'there is/are' structures, passive voice, etc.). Provide clear explanations for beginners.\n\n"
                        "2. Google Terms Consultant: Users may ask about Google's terms of service. You must strictly use "
                        "the `get_conditions_info` tool to fetch this data."
                        "searching via the tool for better matching, but reply to the user in their original language. "
                        "If the requested data is not found, politely state that you cannot help with that specific inquiry."
                    )
                )
            ]

    def run(self):
        """
        Main method to render the Streamlit Web UI and handle user chat interactions.
        """
        st.title("AI Assistant: English Tutor & Google Consultant")

        # Initialize history
        self._init_session_state()

        # Handle user input
        user_query = st.chat_input("Enter your message here...")

        if user_query:
            # Append user message to state
            human_message = HumanMessage(content=user_query)
            st.session_state['history'].append(human_message)

            # Invoke LangGraph agent
            input_data = {"messages": st.session_state['history']}
            with st.spinner("Thinking..."):
                response = self.agent.invoke(input_data)

            # Update history with agent's response logs
            st.session_state['history'] = response['messages']

        # Render chat logs
        for message in st.session_state['history']:
            if isinstance(message, SystemMessage) or message.type == 'tool':
                continue

            role = "user" if isinstance(message, HumanMessage) else "assistant"

            if message.content:
                with st.chat_message(role):
                    st.markdown(message.content)


if __name__ == "__main__":
    # Create an instance of application class
    app = AIAssistantApp()

    # app.populate_database()
    app.run()
