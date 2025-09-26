import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 不再需要 Gemini 的 API key
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

# 初始化 embedding 模型（本地 sentence-transformers）
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 初始化本地向量数据库
db = Chroma(collection_name="pharma_database",
            embedding_function=embedding_model,
            persist_directory='./pharma_db')


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def add_to_db(uploaded_files):
    if not uploaded_files:
        st.error("No files uploaded!")
        return

    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join("./temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_file_path)
        data = loader.load()

        doc_metadata = [d.metadata for d in data]
        doc_content = [d.page_content for d in data]

        st_text_splitter = SentenceTransformersTokenTextSplitter(
            model_name="sentence-transformers/all-mpnet-base-v2",
            chunk_size=100,
            chunk_overlap=50
        )
        st_chunks = st_text_splitter.create_documents(doc_content, doc_metadata)

        db.add_documents(st_chunks)
        os.remove(temp_file_path)


def run_rag_chain(query):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 5})

    PROMPT_TEMPLATE = """
    You are a highly knowledgeable assistant specializing in pharmaceutical sciences. 
    Answer the question based only on the following context:
    {context}

    Question: {question}

    Provide a concise and accurate answer. 
    Do not say "according to the context".
    """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # 本地 Chat 模型（Ollama）
    chat_model = ChatOllama(model="llama3")  # 你需要先运行 `ollama pull llama3`

    output_parser = StrOutputParser()

    rag_chain = {"context": retriever | format_docs,
                 "question": RunnablePassthrough()} | prompt_template | chat_model | output_parser

    response = rag_chain.invoke(query)
    return response


def main():
    st.set_page_config(page_title="PharmaQuery", page_icon=":microscope:")
    st.header("Pharmaceutical Insight Retrieval System (Local RAG)")

    query = st.text_area(
        ":bulb: Enter your query about the Pharmaceutical Industry:",
        placeholder="e.g., What are the AI applications in drug discovery?"
    )

    if st.button("Submit"):
        if not query:
            st.warning("Please ask a question")
        else:
            with st.spinner("Thinking..."):
                result = run_rag_chain(query=query)
                st.write(result)

    with st.sidebar:
        st.markdown("---")
        pdf_docs = st.file_uploader(
            "Upload your research documents related to Pharmaceutical Sciences (Optional) :memo:",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload the file")
            else:
                with st.spinner("Processing your documents..."):
                    add_to_db(pdf_docs)
                    st.success(":file_folder: Documents successfully added to the database!")

    st.sidebar.write("Built with ❤️ (local Ollama + HuggingFace)")


if __name__ == "__main__":
    main()
