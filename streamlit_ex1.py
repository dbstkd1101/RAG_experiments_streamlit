import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["OPENAI_API_KEY"] = "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"


#PDF 파일 로드 및 분할
@st.cache_resource
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load_and_split()

@st.cache_resource
def create_vector_store(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(
        split_docs,
        OpenAIEmbeddings(model = 'text-embedding-3-small'),
        persist_directory=persist_directory
    )
    return vectorstore

#만약 기존에 저장해둔 CrhomaDB가 잇는 경우, 이를 로드
def get_vector_store(_docs):
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings(model = 'text-embedding-3-small'))
    else:
        return create_vector_store(_docs)

#Document 객체의 page_content를 Join
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

def chaining():
    file_path = r"/Users/LeeYunSang/vscode/대한민국헌법(헌법)(제00010호)(19880225).pdf"
    pages = load_and_split_pdf(file_path)
    vector_store = get_vector_store(pages)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    qa_system_prompt = """
    You are an asisttant for question-answering tasks.\
    Use the following pieces of retrieved context to answer the question.\
    If you don't know the answer, just say that you don't know.\
    Keep the answer perfect. please use imogi with the answer.
    Please answer in Korean and use respectful language.\
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# Streamlit UI
st.header("헌법 Q&A 챗봇")
rag_chain = chaining()

st.title("💬 Chatbot")

#session_state에 messages Key값 지정 및 Streamlit 화면 진입 시, AI의 인사말을 기록하기
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "헌법에 대해 무엇이든 물어보세요!"}]

#사용자나 AI가 질문/답변을 주고받을 시, 이를 기록하는 session_state
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


#chat_input()에 입력값이 있는 경우,
if prompt := st.chat_input():
    #messages라는 session_state에 역할은 사용자, 컨텐츠는 프롬프트를 각각 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    #chat_message()함수로 사용자 채팅 버블에 prompt 메시지를 기록
    st.chat_message("user").write(prompt)

    
    response = chat.invoke(prompt)
    msg = response.content

    #messages라는 session_state에 역할은 AI, 컨텐츠는 API답변을 각각 저장
    st.session_state.messages.append({"role": "assistant", "content": msg})
    #chat_message()함수로 AI 채팅 버블에 API 답변을 기록
    st.chat_message("assistant").write(msg)