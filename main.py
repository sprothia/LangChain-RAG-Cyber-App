from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import os
import asyncio
import streamlit as st
from langchain import chains
from langchain_community.tools import YouTubeSearchTool
from langchain_community.document_loaders import PyPDFLoader

VECTOR_DB_DIR = "chroma_db"

async def step_one_load_files():
    all_data = []
    pdf_paths = [
        "nist_documents/NIST.pdf",
        "nist_documents/JSIG.pdf",
        "nist_documents/NISPOM.pdf",
        "nist_documents/DCSA.pdf"
    ]

    for path in pdf_paths:
        try:
            print(f"Loading PDF File: {path}")
            loader = PyPDFLoader(file_path=path)
            data = loader.load()

            length_of_file = len(data)
            print(f"Loaded these many documents {length_of_file}")

            all_data.extend(data)

        except Exception as e:
            print(f"An error occured while loading {path}: {e}")

    print(f"Total PDFs processed: {len(pdf_paths)}")
    print(f"Total documents loaded: {len(all_data)}")

    return all_data

async def step_two_chunks_embeddings_vectordb(data, openai_key):

    if os.path.exists(VECTOR_DB_DIR):
        st.write("...")
        vector_db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=OpenAIEmbeddings(model='text-embedding-ada-002', api_key=openai_key))
    else:
        st.write("Vectorizing documents and saving the vector database...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)

        model_name = 'text-embedding-ada-002'
        embeddings = OpenAIEmbeddings(
            model=model_name,
            api_key=openai_key
        )

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="local-rag",
            persist_directory=VECTOR_DB_DIR
        ) 

    return vector_db


async def step_three_model(vector_database, openai_key):
    
    os.environ["OPENAI_API_KEY"] = openai_key

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")  

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )   

    retriever = MultiQueryRetriever.from_llm(
        vector_database.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )

    # RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    Your goal is to help small companies utilize this information so they can implement these security practices in their own company. You also need to give them an action plan,
    a "what do I need to do" so the user can get right to action. You should act like an expert giving expert advice. In another paragraph, outline a good comprehensive plan of what they can do. End each answer with a "Legal disclaimer...."
    Question: {question}
    If a question is asked that has nothing to do with the document, reply I dont know.
    """

    prompt = ChatPromptTemplate.from_template(template)

    return retriever, prompt, llm

async def step_four_chains(retriever, prompt, llm):
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    topic_prompt = PromptTemplate(
        input_variables=["summary"],
        template="Based on this full explanation/summary about this topic, give me a couple words of what this topic is about: {summary}"
    )

    topic_chain = topic_prompt | llm | StrOutputParser()

    return chain, topic_chain

def summarize_and_extract_topic(text, chain, topic_chain):
  youtube_tool = YouTubeSearchTool()
  summary = chain.invoke(text)
  topic = topic_chain.invoke(summary)
  topic_to_search_youtube = topic.replace(',', '')
  youtube_links = youtube_tool.run(topic_to_search_youtube)

  return {
      "summary": summary,
      "topic": topic_to_search_youtube,
      "links": youtube_links
  }

async def main(openai_key):
    if not os.path.exists(VECTOR_DB_DIR):
        all_data = await step_one_load_files()
        vector_db = await step_two_chunks_embeddings_vectordb(all_data, openai_key)
    else:
        vector_db = await step_two_chunks_embeddings_vectordb(None, openai_key)


    retriever, prompt, llm = await step_three_model(vector_db, openai_key)
    chain, topic_chain = await step_four_chains(retriever, prompt, llm)
    # text = "Your input text here"
    # result = summarize_and_extract_topic(text, chain, topic_chain)
    # print(result)
    return chain, topic_chain

st.title("NIST-800 AI Advisor")
input_text = st.text_area("Enter your NIST question: ")
open_key = st.text_input("Enter your OpenAI API key:")

if st.button("Run"):
    if input_text and open_key:
        with st.spinner("Processing request"):
            chain, topic_chain = asyncio.run(main(open_key))
            result = summarize_and_extract_topic(input_text, chain, topic_chain)

        # Display the result
        st.subheader("Summary")
        st.write(result['summary'])

        st.subheader("Topic")
        st.write(result['topic'])

        st.subheader("YouTube Links")
        st.write(result['links'])

    else:
        st.warning("Please enter some text or apikey to proceed.")

if __name__ == "__main__":
    if(open_key):
        asyncio.run(main(openai_key=open_key))
    else:
        print("Add openaikey and query")

