import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_community.vectorstores import FAISS # FAISS to strore the vectors
from langchain.text_splitter import CharacterTextSplitter # Chunking
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings # Needs sentence transformer installed
from pypdf import PdfReader

genai.configure(api_key=os.getenv('google_api_key'))
model = genai.GenerativeModel(model_name='gemini-2.0-flash')

# Configure Embedding Model: sentence-transformers/all-MiniLM-L6-v2
@st.cache_resource(show_spinner="Loading the model...")

def myembedding_model():
    return (HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

embedding_model = myembedding_model()

# Creating Frontend
st.header(':blue[RAG using HuggingFace] Embeddings + FAISS db')
uploaded_file = st.file_uploader('Upload the Document', type=['pdf'])

if uploaded_file:
    raw_text = ""
    pdf = PdfReader(uploaded_file)

    for index, page in enumerate(pdf.pages):
        context = page.extract_text()
        if context:
            raw_text += context

    # Chunking using Schema(Orders the text page by page)
    if raw_text.strip():
        document = Document(page_content=raw_text)
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Overlap: 
        chunks = splitter.split_documents([document])

        # HF Embedding
        texts = [chunk.page_content for chunk in chunks]

        # Vector DB
        vector_db = FAISS.from_texts(texts,embedding_model)

        # Retriever to retrieve the information stored in the form of Vector embddings in the Vector Database
        retriever = vector_db.as_retriever()

        st.markdown("Document Processed Successfully ✅.\nAsk Questions Below")

        user_input = st.chat_input('Enter your query')

        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.spinner("Analysisng the Document..."):
                retrieved_doc = retriever.get_relevant_documents(user_input)
                context = '\n\n'.join(doc.page_content for doc in retrieved_doc)

                prompt=f"""You are an expert assistant and use the context below to answer the query. If unsure, kindly deny that you don't know.
                    Context:{context}, 
                    User Query:{user_input}, 
                    Answer:"""
                response=model.generate_content(prompt) 
                st.markdown('Answer: ')
                st.write(response.text)
        else:
            st.warning("Please upload the PDf for review and analysis")
                