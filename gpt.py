import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

def main():
    st.header("Hello! Let's chat with your PDF")

    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:

        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        docs = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        embeddings = OpenAIEmbeddings()
        vector_store = Milvus.from_texts(
            docs,
            embeddings,
            connection_args={
                "uri": "https://in03-a7fad683bb75fe5.api.gcp-us-west1.zillizcloud.com",
                "token": "6718e8768b06b1ea0095fe9a31c24658d36f99de3b6a6ea7f7fef6ab613fcbff6940e95c25e0660d0af785f88095072359300937"
            }
        )

        query = st.text_input("Ask questions about your PDF file here:")

        if query:
            docs = vector_store.similarity_search(query=query, k=1)

            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)

            st.write(response)

if __name__ == '__main__':
    main()