import os
import time
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434",
)

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

def get_financial_analysis(data_summary, model_name="llama3.2"):
    """支持传入 model_name"""
    global vector_store
    if not vector_store: load_existing_db()
    if not vector_store: return "⚠️ 错误：请先加载知识库！"

    # 动态创建 LLM 对象
    current_llm = OllamaLLM(
        model=model_name,
        base_url="http://127.0.0.1:11434",
        num_ctx=4096,
        timeout=300
    )

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
    你是一位专业的财务顾问。请基于【背景知识】分析【财务数据】。

    【背景知识】:
    {context}

    【财务数据】:
    {input}

    请简明扼要地给出分析意见（必须用中文回答）：
    """)

    question_answer_chain = create_stuff_documents_chain(current_llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    start_time = time.time()  # 开始计时
    try:
        response = rag_chain.invoke({"input": data_summary})
        end_time = time.time()  # 结束计时
        duration = round(end_time - start_time, 2)
        return response["answer"], duration
    except Exception as e:
        return f"分析错误: {e}", 0

def calculate_similarity_score(text1, text2):
    """
    计算两段文本的语义相似度 (余弦相似度)
    返回 0.0 ~ 1.0 的分值，越高越好
    """
    if not text1 or not text2:
        return 0.0

    # 1. 把文字变成向量 (使用已加载的 embeddings 模型)
    # 这就是 RAG 的核心技术，现在拿来做评测
    vec1 = embeddings.embed_query(text1)
    vec2 = embeddings.embed_query(text2)

    # 2. 转换为 numpy 数组
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    # 3. 计算余弦相似度公式: (A . B) / (|A| * |B|)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    similarity = dot_product / (norm_v1 * norm_v2)
    return round(float(similarity), 4)  # 保留4位小数

# 修改 get_financial_analysis 支持动态换模型
def get_financial_analysis_with_model(data_summary, model_name):
    """支持指定模型的分析函数"""
    global vector_store
    if not vector_store: load_existing_db()
    if not vector_store: return "知识库未加载", 0

    # 动态创建指定模型
    temp_llm = OllamaLLM(
        model=model_name,
        base_url="http://127.0.0.1:11434",
        num_ctx=4096,
        timeout=300
    )

    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_template("""
    你是一位专业的财务顾问。基于以下信息分析财务状况：
    【背景知识】:{context}
    【财务数据】:{input}
    请用中文简要分析：
    """)
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(temp_llm, prompt))

    start_time = time.time()
    try:
        res = chain.invoke({"input": data_summary})
        duration = time.time() - start_time
        return res["answer"], round(duration, 2)
    except Exception as e:
        return f"Error: {str(e)}", 0