import boto3
import json
import os
import sys
import streamlit as st
# Titan Embeddings Model: Generating embeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
# Vector Embedding and Vector Store
from langchain.vectorstores import FAISS
# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA




# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


# Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                   chunk_overlap=1000)
    
    docs = text_splitter.split_documents(documents)

    return docs

# Vector Embeddings and Vector Store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs,
                                             bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")


def get_claude_LLM():
    # Create the Anthropic Model
    LLM = Bedrock(model_id="anthropic.claude-v2:1", client=bedrock,
                  model_kwargs={"max_tokens_to_sample": 512})

    return LLM

def get_Llama3_LLM():
     # Create the Meta Model
    LLM = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock,
                  model_kwargs={"max_gen_len": 512})

    return LLM

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end and answer 
comprehensively within 250 words. If you don't know the answer,
just say you don't know instead of trying to make up an answer.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=["context", "question"])

def get_LLM_response(LLM, vectorstore_faiss, query):
    qna = RetrievalQA.from_chain_type(llm=LLM,
                                      chain_type="stuff",
                                      retriever=vectorstore_faiss.as_retriever(
                                          search_type="similarity", search_kwargs={"k": 3}
                                      ),
                                      return_source_documents=True,
                                      chain_type_kwargs={"prompt": prompt}
                                      )
    
    answer = qna({"query": query})

    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF files using AWS Bedrock")

    user_question = st.text_input("Ask a Question about the PDF content")

    with st.sidebar:
        st.title("Update or Create Vector Store")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            LLM = get_Llama3_LLM()

            st.write(get_LLM_response(LLM, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama 3 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            LLM = get_claude_LLM()

            st.write(get_LLM_response(LLM, faiss_index, user_question))
            st.success("Done")


if __name__ == "__main__":
    main()