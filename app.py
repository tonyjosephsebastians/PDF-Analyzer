import streamlit as st
import pymupdf  # PyMuPDF for PDF text extraction
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Sidebar for API key input and model selection
with st.sidebar:
    st.title("📄 PDF Document Analyzer")
    api_key = st.text_input("🔑 Enter your API key:", type="password")
    model_name = st.selectbox("🤖 Select Model:", ["gemini-2.0-flash", "gemini-1.0-pro"])

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.session_state["api_key_entered"] = True
    else:
        st.warning("⚠ Please enter your API key to continue.")
        st.stop()

if "api_key_entered" in st.session_state:
    uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        try:
            # Extract text from PDF
            doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")  # Fixed import issue
            text = "\n".join([page.get_text("text") for page in doc])

            if not text.strip():
                st.error("⚠ No text found in the PDF.")
                st.stop()

            st.success("✅ PDF loaded successfully.")
            st.text_area("📜 Extracted Text Preview:", text[:1000] + "...", height=200)

            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator=" ",
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            # Ask a question
            question = st.text_input("🔍 Ask a question about the document:", placeholder="Enter your question here...")
            if question:
                # Perform vector similarity search
                docs = knowledge_base.similarity_search(question)

                if not docs:
                    st.warning("⚠ No relevant content found for the given question.")
                else:
                    # Load LLM and run retrieval-augmented generation
                    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
                    chain = load_qa_chain(llm=model, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=question)

                    st.write("### 💡 Answer:")
                    st.success(response)

        except Exception as e:
            st.error(f"❌ Error processing PDF: {e}")
