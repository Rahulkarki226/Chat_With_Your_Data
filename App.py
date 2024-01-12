import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css,bot_template, user_template
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store= FAISS.from_texts(text_chunks,embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm=GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversational_chain=ConversationalRetrievalChain.from_llm(llm=llm,retriever=vector_store.as_retriever(), memory=memory)
    return conversational_chain

def user_input(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question':user_question})
        st.session_state.chatHistory = response['chat_history']
        for i, message in enumerate(st.session_state.chatHistory):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.error("Please process the PDF files first.")

def main():
    load_dotenv()
    st.set_page_config("Chat with Multiple PDFs",page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    st.header("Chat with Mutliple PDFs")
    user_question =st.text_input("Ask a Question from the PDF Files")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "ChatHistory" not in st.session_state:
        st.session_state.chatHistory =None
    if user_question:
         user_input(user_question)
    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your Documents")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store= get_vector_store(text_chunks)
                st.session_state.conversation=get_conversational_chain(vector_store)
                st.success("Done")


if __name__=="__main__":
    main()
