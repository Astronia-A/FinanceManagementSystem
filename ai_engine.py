import os
import time
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. 设置模型 (核心修改：增加 timeout 时间)
# num_ctx=4096 增加上下文窗口，防止数据多了记不住
# timeout=300 设置超时为 300秒 (5分钟)，给 AI 足够的思考时间
llm = OllamaLLM(
    model="llama3.2",
    base_url="http://127.0.0.1:11434",
    num_ctx=4096,
    timeout=300
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434",
)

# 全局变量
vector_store = None
DB_PATH = "faiss_index"


def init_knowledge_base(file_path):
    global vector_store
    print(f"📂 加载知识库: {file_path}")
    docs = []
    try:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif file_path.endswith('.txt'):
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
            except:
                loader = TextLoader(file_path, encoding='gbk')
                docs = loader.load()
        else:
            return
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    if not docs: return

    # 切片逻辑
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)

    print(f"🧩 共 {len(splits)} 个片段，正在建立索引...")

    try:
        batch_size = 10
        vector_store = None
        for i in range(0, len(splits), batch_size):
            batch = splits[i: i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)
            time.sleep(0.1)  # 稍微快一点点

        vector_store.save_local(DB_PATH)
        print(f"✅ 知识库加载完毕！")
    except Exception as e:
        print(f"❌ 建立索引失败: {e}")


def load_existing_db():
    global vector_store
    if os.path.exists(DB_PATH):
        try:
            vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
            print("✅ 已加载旧知识库")
        except:
            pass


def get_financial_analysis(data_summary):
    global vector_store
    if not vector_store: load_existing_db()
    if not vector_store: return "⚠️ 错误：请先在左侧上传并加载知识库文件！"

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
    你是一位专业的财务顾问。请基于【背景知识】分析【财务数据】。

    【背景知识】:
    {context}

    【财务数据】:
    {input}

    请简明扼要地给出分析意见（中文）：
    """)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    try:
        # 这里 invoke 可能会比较慢，已在上面设置了 timeout
        response = rag_chain.invoke({"input": data_summary})
        return response["answer"]
    except Exception as e:
        return f"分析中断: {e} (请检查 Ollama 是否运行或显存是否足够)"