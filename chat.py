import streamlit as st
import os
import base64
#from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

#load_dotenv()

def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_base64 = get_base64("bg-8.png")
user_icon = get_base64("Meta.png")      
ai_icon = get_base64("Chatbot.png")      

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: white;
        font-family: Inter, sans-serif;
    }}

    .user-msg {{
        background-color: rgba(0,0,0,0.5);
        color: white;
        padding: 10px 15px;
        border-radius: 20px;
        text-align: right;
        margin: 5px;
        max-width: 70%;
    }}

    .assistant-msg {{
        background-color: rgba(255,255,255,0.85);
        color: black;
        padding: 10px 15px;
        border-radius: 20px;
        text-align: left;
        margin: 5px;
        max-width: 70%;
    }}

    .avatar {{
        width: 36px;
        height: 36px;
        border-radius: 50%;
        object-fit: cover;
        margin: 0 8px;
    }}

    .msg-wrapper {{
        display: flex;
        align-items: center;
        margin-bottom: 6px;
    }}

    .user-wrapper {{
        justify-content: flex-end;
    }}

    .assistant-wrapper {{
        justify-content: flex-start;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index(os.environ.get("PINECONE_INDEX_NAME"))

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ.get("OPENAI_API_KEY")
)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        SystemMessage("You are an assistant for question-answering tasks.")
    )

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        st.markdown(
            f"""
            <div class="msg-wrapper user-wrapper">
                <span class="user-msg">{message.content}</span>
                <img class="avatar" src="data:image/png;base64,{user_icon}">
            </div>
            """,
            unsafe_allow_html=True
        )

    elif isinstance(message, AIMessage):
        st.markdown(
            f"""
            <div class="msg-wrapper assistant-wrapper">
                <img class="avatar" src="data:image/png;base64,{ai_icon}">
                <span class="assistant-msg">{message.content}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append(HumanMessage(prompt))
    st.markdown(
        f"""
        <div class="msg-wrapper user-wrapper">
            <span class="user-msg">{prompt}</span>
            <img class="avatar" src="data:image/png;base64,{user_icon}">
        </div>
        """,
        unsafe_allow_html=True
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.3}
    )

    docs = retriever.invoke(prompt)
    context = "".join(d.page_content for d in docs)

    system_prompt = f"""
    You are a helpful assistant that answers questions using the provided context.
    Use ONLY the retrieved information.
    Answer in Mongolian.
    If the answer is not in the context, respond with: "Уучлаарай, би энэ талаар мэдэхгүй байна 🤷‍♂️"

    Context:
    {context}
    """

    result = llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(prompt)
    ]).content

    st.session_state.messages.append(AIMessage(result))
    st.markdown(
        f"""
        <div class="msg-wrapper assistant-wrapper">
            <img class="avatar" src="data:image/png;base64,{ai_icon}">
            <span class="assistant-msg">{result}</span>
        </div>
        """,
        unsafe_allow_html=True
    )