import streamlit as st
import os
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.agents import Tool, initialize_agent
from langgraph.graph import StateGraph

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=api_key,
    model="llama3-8b-8192"
)

st.title("AI PDF Assistant")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
query = st.text_input("Ask something")

@st.cache_resource
def process_pdf(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever()

if uploaded_file:
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing..."):
        retriever = process_pdf(file_path)

    def summarize_tool(text):
        return llm.invoke(f"Summarize this:\n{text}").content

    def mcq_tool(text):
        return llm.invoke(f"Generate 5 MCQs from this:\n{text}").content

    def json_tool(text):
        return llm.invoke(f"""
        Extract info in JSON:
        {{
          "title": "",
          "summary": "",
          "keywords": []
        }}
        Text:
        {text}
        """).content

    tools = [
        Tool(name="Summarizer", func=summarize_tool, description="Summarizes content"),
        Tool(name="MCQ Generator", func=mcq_tool, description="Creates questions"),
        Tool(name="JSON Extractor", func=json_tool, description="Returns structured JSON")
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description"
    )

    class State(dict):
        pass

    def process_query(state):
        return {"query": state["query"]}

    def retrieve_docs(state):
        docs = retriever.get_relevant_documents(state["query"])
        context = "\n".join([d.page_content for d in docs])
        return {"context": context, "query": state["query"]}

    def generate_answer(state):
        response = agent.run(f"{state['context']}\n{state['query']}")
        return {"answer": response}

    graph = StateGraph(State)
    graph.add_node("process", process_query)
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("generate", generate_answer)

    graph.set_entry_point("process")
    graph.add_edge("process", "retrieve")
    graph.add_edge("retrieve", "generate")

    app_graph = graph.compile()

    if query:
        result = app_graph.invoke({"query": query})
        st.subheader("Answer")
        st.write(result["answer"])
