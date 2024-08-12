#"""--------------------------------------------- IMPORT LIBRARIES ---------------------------------------------"""

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


#"""------------------------------------------- LOAD API KEY ---------------------------------------------"""

load_dotenv()

os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#"""------------------------------------------- EXTRACT TEXT FROM PDFS ---------------------------------------------"""

def get_pdf_text(pdf_docs):
    text = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_number, page in enumerate(pdf_reader.pages):
            extracted_text = page.extract_text()
            if extracted_text:
                text.append({
                    'content': extracted_text,
                    'pdf_name': os.path.basename(pdf.name),
                    'page_number': page_number 
                })
    return text


#"""------------------------------------------- MAKE CHUNKS OF TEXT ---------------------------------------------"""

def get_text_chunks(pdf_texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for pdf_text in pdf_texts:
        split_chunks = text_splitter.split_text(pdf_text['content'])
        for chunk in split_chunks:
            chunks.append({
                'chunk': chunk,
                'pdf_name': pdf_text['pdf_name'],
                'page_number': pdf_text['page_number']
            })
    return chunks


#"""------------------------------------------- MAKE TEXT EMBEDDINGS ---------------------------------------------"""

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    documents = []

    for chunk in text_chunks:
        documents.append(Document(
            page_content=chunk['chunk'],
            metadata={
                'pdf_name': chunk['pdf_name'],
                'page_number': chunk['page_number']
            }
        ))

    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")


#"""------------------------------------------- GET CONVERSATIONAL CHAIN ---------------------------------------------"""

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in 
    provided context just say, "I don't have this information. For more information, contact +123456789." Don't provide the wrong answer.
    Also, with every response you provide, make sure to mention the source of the information in next line like "This information is taken from {pdf_name} on page {page_number}."\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "pdf_name", "page_number"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


#"""------------------------------------------- USER INPUT ---------------------------------------------"""

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    # Prepare context for the response
    context = "\n".join([doc.page_content for doc in docs])  # Accessing page_content
    pdf_name = docs[0].metadata['pdf_name'] if docs and 'pdf_name' in docs[0].metadata else "Unknown"
    page_number = docs[0].metadata['page_number'] if docs and 'page_number' in docs[0].metadata else 0

    response = chain(
        {"input_documents": docs, "context": context, "question": user_question, "pdf_name": pdf_name, "page_number": page_number + 1},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])


#"""------------------------------------------- MAIN FUNCTION ---------------------------------------------"""

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDFs using Gemini📚")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()