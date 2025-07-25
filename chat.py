import streamlit as st
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv()

hugging_api_key = os.getenv("HuggingFace")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit app
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

# Input the Groq API Key
api_key = st.text_input("Enter your Groq API KEY:", type="password")

if api_key:
    llm = ChatGroq(
        groq_api_key=api_key,
        model="Gemma2-9b-It",
    )

    session_id = st.text_input("Session ID", value="default_session")

    # Statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose a pdf file", type="pdf", accept_multiple_files=True)

    # Process uploaded PDFs
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()  # ✅ Fixed typo
            documents.extend(docs)

        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500
        )
        splits = text_splitter.split_documents(documents)

        # Create vector store using Chroma
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        # Set up retriever
        retriever = vectorstore.as_retriever()

        context_system_prompt = (
            "given a chat history and the latest user question "
            "which might reference context in the chat history "
            "formulate a standalone question which can be understood "
            "just reformulate it if needed and otherwise return it as is"
        )

        context_q_prompt = ChatPromptTemplate.from_messages([
            ("system", context_system_prompt),
            MessagesPlaceholder("chatHistory"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

        # Answer the question
        system_prompt = (
            "You are an assistant for question answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chatHistory"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chatHistory",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your Question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                }
            )
            st.write("Assistant:", response["answer"])
            st.write("Chat History:", session_history.messages)

else:
    st.warning("Please enter your Groq API key.")
