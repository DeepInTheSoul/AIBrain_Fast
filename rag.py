import os
import streamlit as st
from dotenv import dotenv_values
from langchain_community.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
from langchain_community.llms.baidu_qianfan_endpoint import QianfanLLMEndpoint
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyPDFLoader, UnstructuredFileLoader
import codecs
import csv
config = dotenv_values(".env")


os.environ["QIANFAN_AK"] = "rYndCW8UyNrh7ZIxAmxG0w1X"
os.environ["QIANFAN_SK"] = "KovKWoaJeKYeIQwLgOUxFof5KI1ggTRq"
embeddings=QianfanEmbeddingsEndpoint(model='bge-large-zh')

def rag_page():
    st.title("📚知识库管理")
    files = st.file_uploader("上传知识文件(目前支持txt、pdf、md文件，建议将word转换为pdf文件)：")
    if files is not None:
        filepath = os.path.join('file', files.name)
        if filepath.lower().endswith(".csv"):
            num = 0
            with codecs.open(filepath) as f:
                new_json = []
                for row in csv.DictReader(f, skipinitialspace=True):
                    data = {}
                    data['question'] = row['ask']
                    data['answer'] = row['answer']
                    data_str = str(data)
                    num = num + 1
                    new_json.append(data_str)
                    if num > 1000:
                        break
                st.markdown("正在把数据传至向量模型编码, 请耐心等待, 暂时先别切换界面")
                vector_db = Chroma.from_texts(new_json, embedding=embeddings, persist_directory="./chroma_db")
                vector_db.persist()
                st.markdown("---------------已将文件装载进知识库中--------------------")
        else:
                docs = file_loader(files)
                konwlwdge_vec_store(docs)
                st.markdown("---------------已将文件装载进知识库中--------------------")

def file_loader(file):
    filepath = os.path.join('file', file.name)
    with open(filepath, 'wb') as f:
        f.write(file.getbuffer())
    if filepath.lower().endswith(".md"):
        loader = UnstructuredMarkdownLoader(filepath)
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = PyPDFLoader(filepath)
        docs = loader.load()
    elif filepath.lower().endswith(".txt"):
        loader = UnstructuredFileLoader(filepath,encoding='utf8')
        docs = loader.load()
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    return docs

def konwlwdge_vec_store(docs):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    splits = text_splitter.split_documents(docs)

    vector_db = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    vector_db.persist()



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def rag_chain(llm):

    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_db.as_retriever()

    # 加载一个预先定义的提示生成器，用于生成检索问题。
    template = """基于以下已知信息，请专业地回答用户的问题。
                不要乱回答，如果无法从已知信息中找到答案，请诚实地告诉用户。
                已知内容:   
                {context}
                问题:
                {question}"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain

