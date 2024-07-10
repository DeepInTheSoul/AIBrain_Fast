#UI框架
import os
import streamlit as st
from rag import rag_chain
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langchain_community.llms import QianfanLLMEndpoint
#加载.env文件中的环境变量
config = dotenv_values(".env")

os.environ["QIANFAN_AK"] = "rYndCW8UyNrh7ZIxAmxG0w1X"
os.environ["QIANFAN_SK"] = "KovKWoaJeKYeIQwLgOUxFof5KI1ggTRq"

ERNIE4_llm = QianfanLLMEndpoint(model="ERNIE-4.0-8K", streaming=True)
Yi_llm = QianfanLLMEndpoint(model="Yi-34B-Chat", streaming=True)
Llama_llm = QianfanLLMEndpoint(model="Meta-Llama-3-8B", streaming=True)
ChatGLM_llm = QianfanLLMEndpoint(model="ChatGLM2-6B-32K", streaming=True)



#UI的标题
def chat_page():
    #UI界面的边栏，通过下拉列表切换llm模型
    with st.sidebar:
        pattern_list = [
            "大模型问答",
            "医疗知识库问答",
        ]

        model_list = [
            "文心一言4.0",
            "ChatGLM-6B",
            "Llama3-8B",
            "Yi-34B",
        ]
        pattern = st.selectbox("对话模式选择：",
                             pattern_list,
                             index=0
                             )
        st.write('You selected:', pattern)

        model = st.selectbox("对话模型选择：",
                             model_list,
                             index=0
                             )
        st.write('You selected:', model)

    st.title("🤖 ClinicaBrain：智能医疗大脑")
    #聊天历史记录初始化
    if "messages" not in st.session_state:
        st.session_state.messages = []

     #聊天界面展示历史聊天
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
             st.markdown(message["content"])

    if pattern == "大模型问答":
        if model == "文心一言4.0":
            llm = ERNIE4_llm
        elif model == "ChatGLM-6B":
            llm = ChatGLM_llm
        elif model == "Llama3-8B":
            llm = Llama_llm
        elif model == "Yi-34B":
            llm = Yi_llm
    elif  pattern == "医疗知识库问答":
        if model == "文心一言4.0":
            llm = rag_chain(ERNIE4_llm)
        elif model == "ChatGLM-6B":
            llm = rag_chain(ChatGLM_llm)
        elif model == "Llama3-8B":
            llm = rag_chain(Llama_llm)
        elif model == "Yi-34B":
            llm = rag_chain(Yi_llm)

    if chat_input := st.chat_input("What is up?"):
        st.chat_message("user").markdown(chat_input)
        # 将用户消息添加到聊天记录
        st.session_state.messages.append({"role": "user", "content": chat_input})
        with st.chat_message("assistant"):
            #以流式输出的方式调用llm的api，并在UI界面显示
            response = st.write_stream(
                llm.stream(chat_input))
        # 将大模型的消息添加到聊天记录
        st.session_state.messages.append({"role": "assistant", "content": response})



