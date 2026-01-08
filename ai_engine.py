import os
import time
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. 模型配置 ---
llm = OllamaLLM(
    model="llama3.2",
    base_url="http://127.0.0.1:11434",
    num_ctx=4096,
    # 【修改1】超时时间设为无限长或非常长，防止加载模型时报错
    timeout=600,
    # 【修改2】告诉 Ollama：加载进内存后，至少保持 1小时(60m) 不退场
    keep_alive="60m"
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434",
)

vector_store = None
DB_PATH = os.path.join(os.getcwd(), "faiss_index")


def init_knowledge_base(file_path):
    global vector_store
    print(f"📂 正在加载知识库文件: {file_path}")
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
            print("❌ 不支持的文件格式")
            return
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    if not docs: return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)

    print(f"🧩 切分完成，共 {len(splits)} 个片段。正在建立索引...")

    try:
        batch_size = 10
        vector_store = None
        for i in range(0, len(splits), batch_size):
            batch = splits[i: i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)
            time.sleep(0.1)
        vector_store.save_local(DB_PATH)
        print(f"✅ 知识库加载完毕并已保存到 '{DB_PATH}'！")
    except Exception as e:
        print(f"❌ 建立向量索引失败: {e}")


def load_existing_db():
    global vector_store
    if os.path.exists(DB_PATH):
        try:
            vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        except:
            pass


# --- 2. 核心修改点：强化的 Prompt ---
STRONG_PROMPT = """
你是一位专业的财务审计师。请务必遵守以下指令：

1. 【核心任务】：你的唯一任务是分析下面的【财务数据摘要】。
2. 【辅助参考】：【参考知识库】仅作为判断标准（例如，如果知识库说亏损不好，你就依据这个来批评数据）。
3. 【禁止项】：绝对不要总结或评价知识库本身！不要说“这段文字介绍了...”之类的话。

【参考知识库】(理论依据):
{context}

【财务数据摘要】(请重点分析这里的数据):
{input}

请直接输出针对数据的分析结论（用中文）：
"""


def get_financial_analysis(data_summary):
    global vector_store

    # 内存没有就读硬盘
    if vector_store is None:
        load_existing_db()
    if vector_store is None:
        return "⚠️ 错误：请先在左侧上传并加载知识库文件！"

    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_template(STRONG_PROMPT)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # === 【修改3】 增加“自动重试”机制 ===
    # 如果第一次连不上（因为模型在加载），就等2秒再试一次，最多试3次
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"🔄 正在尝试第 {attempt + 1} 次请求 AI...")
            response = rag_chain.invoke({"input": data_summary})
            return response["answer"]
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ 第 {attempt + 1} 次请求失败: {error_msg}")

            # 如果是连接错误，等待模型加载
            if "Connection" in error_msg or "disconnected" in error_msg:
                time.sleep(2)  # 等2秒让 Ollama 喘口气
            else:
                return f"分析过程发生严重错误: {e}"

    return "❌ 连接 Ollama 失败，请检查后台服务是否开启，或者电脑是否卡顿。"


def calculate_similarity_score(text1, text2):
    if not text1 or not text2: return 0.0
    try:
        vec1 = np.array(embeddings.embed_query(text1))
        vec2 = np.array(embeddings.embed_query(text2))
        dot = np.dot(vec1, vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norm == 0: return 0.0
        return round(float(dot / norm), 4)
    except:
        return 0.0


def get_financial_analysis_with_model(data_summary, model_name):
    """竞技场使用的函数"""
    global vector_store
    if not vector_store: load_existing_db()

    temp_llm = OllamaLLM(
        model=model_name,
        base_url="http://127.0.0.1:11434",
        num_ctx=4096,
        timeout=300
    )

    retriever = vector_store.as_retriever() if vector_store else None

    # 竞技场也使用强化后的 Prompt，保证公平
    prompt = ChatPromptTemplate.from_template(STRONG_PROMPT)

    if retriever:
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(temp_llm, prompt))
        input_data = {"input": data_summary}
    else:
        return "请先加载知识库", 0

    start_time = time.time()
    try:
        res = chain.invoke(input_data)
        duration = time.time() - start_time
        return res["answer"], round(duration, 2)
    except Exception as e:
        return f"Error: {str(e)}", 0