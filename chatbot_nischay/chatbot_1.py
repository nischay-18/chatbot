import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    st.write('Made with ❤️ by Team-10')
    st.markdown("***")
    st.write("***Team-10*** ")
    st.write("Nischay Sai Cherukuri")
    st.write("Pranav Sai Putta")
    st.write("Rajeev Koneru")


os.environ["OPENAI_API_KEY"] = "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"


def main():
    st.header("Chat with PDF 💬")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        pdfReader = PdfReader(pdf)
        raw_text = ''
        for page in pdfReader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Load or create FAISS index
        embeddings = OpenAIEmbeddings()
        # st.write(texts)
        docsearch = FAISS.from_texts(texts, embeddings)

    # Chat input
    query = st.chat_input("Ask questions about your PDF file:")
    if query:
        # Process the query
        docs = docsearch.similarity_search(query=query, k=3)
        llm = OpenAI(model="text-davinci-003")
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append(
            {"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            with st.chat_message("User", avatar="👩‍💻"):
                st.write(content)
        else:
            with st.chat_message("Assistant"):
                st.write(content)


if __name__ == '__main__':
    main()
