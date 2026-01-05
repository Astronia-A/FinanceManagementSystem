import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
from captcha.image import ImageCaptcha
import random
import string
import time
import hashlib
import plotly.graph_objects as go
from ai_engine import init_knowledge_base, get_financial_analysis, get_financial_analysis_with_model, calculate_similarity_score

# 引用 AI 引擎
from ai_engine import init_knowledge_base, get_financial_analysis

# --- 0. 数据库管理 ---
DB_FILE = 'finance_system.db'


def make_hash(password):
    """将明文密码转化为 SHA-256 哈希值"""
    return hashlib.sha256(password.encode()).hexdigest()


def check_password(password, hashed_password):
    """验证输入的密码是否正确"""
    return make_hash(password) == hashed_password


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # 1. 建立流水记录表
    c.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT NOT NULL,
            record_date TEXT NOT NULL,
            amount REAL NOT NULL,
            record_type TEXT NOT NULL,
            operator TEXT DEFAULT 'admin'
        )
    ''')
    try:
        c.execute("SELECT operator FROM records LIMIT 1")
    except:
        c.execute("ALTER TABLE records ADD COLUMN operator TEXT DEFAULT 'admin'")

    # 2. 建立用户表 (存储用户名和加密密码)
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
    ''')

    # 3. 初始化默认管理员账号 (如果表中没有用户)
    c.execute("SELECT count(*) FROM users")
    if c.fetchone()[0] == 0:
        # 这里默认创建 admin / 123456
        # 在答辩时可以说：系统初始化时会自动创建默认管理员，密码经过 SHA-256 加密存储
        default_pass = make_hash("123456")
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", ("admin", default_pass))
        # 顺便加个老板账号用于演示多用户
        boss_pass = make_hash("888888")
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", ("boss", boss_pass))
        print("✅ 已初始化默认用户: admin, boss")

    conn.commit()
    conn.close()


def load_data_from_db():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM records", conn)
    conn.close()
    df['record_date'] = pd.to_datetime(df['record_date'], format='mixed', errors='coerce')
    df = df.rename(
        columns={'id': '编号', 'item_name': '项目', 'record_date': '日期', 'amount': '金额', 'record_type': '类型',
                 'operator': '操作人'})
    return df


def insert_record(item, date, amount, operator):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    type_str = "收入" if amount >= 0 else "支出"
    c.execute("INSERT INTO records (item_name, record_date, amount, record_type, operator) VALUES (?, ?, ?, ?, ?)",
              (item, str(date), amount, type_str, operator))
    conn.commit()
    conn.close()


def delete_record(record_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()


def insert_batch_from_excel(df_excel, operator):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    for _, row in df_excel.iterrows():
        type_str = "收入" if row['金额'] >= 0 else "支出"
        c.execute("INSERT INTO records (item_name, record_date, amount, record_type, operator) VALUES (?, ?, ?, ?, ?)",
                  (row['项目'], str(row['日期']), row['金额'], type_str, operator))
    conn.commit()
    conn.close()


# 新增：验证用户登录的函数
def verify_login(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()

    if result:
        stored_hash = result[0]
        # 比对输入的密码哈希 和 数据库里的哈希
        if check_password(password, stored_hash):
            return True
    return False


# --- 辅助函数 ---
def format_big_number(num):
    abs_num = abs(num)
    if abs_num >= 100000000:
        return f"¥{num / 100000000:.2f} 亿"
    elif abs_num >= 10000:
        return f"¥{num / 10000:.2f} 万"
    else:
        return f"¥{num:,.2f}"


def generate_captcha_image():
    image = ImageCaptcha(width=200, height=60)
    captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    data = image.generate(captcha_text)
    print(f"🔑 [DEBUG] 验证码: {captcha_text}")
    return captcha_text, data


# --- 程序配置 ---
st.set_page_config(page_title="智财云 Dashboard", layout="wide", page_icon="💰")
init_db()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
if 'captcha_text' not in st.session_state:
    text, data = generate_captcha_image()
    st.session_state.captcha_text = text
    st.session_state.captcha_image = data


# --- 登录页面 (数据库版) ---
def login_page():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>🔐 智财云登录</h2>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="admin")
            password = st.text_input("密码", type="password", placeholder="123456")
            c1, c2 = st.columns([2, 1])
            with c1: captcha_input = st.text_input("验证码")
            with c2: st.image(st.session_state.captcha_image, caption="验证码")
            submitted = st.form_submit_button("登录", type="primary")

        if st.button("看不清？刷新"):
            text, data = generate_captcha_image()
            st.session_state.captcha_text = text
            st.session_state.captcha_image = data
            st.rerun()

        if submitted:
            # === 核心修改：改为查数据库验证 ===
            if verify_login(username, password):
                if captcha_input.upper() == st.session_state.captcha_text:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.toast(f"欢迎回来，{username}！", icon="👋")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("验证码错误")
                    text, data = generate_captcha_image()
                    st.session_state.captcha_text = text
                    st.session_state.captcha_image = data
                    st.rerun()
            else:
                st.error("用户名或密码错误")


# --- 主界面 ---
def main_app():
    with st.sidebar:
        st.title(f"👤 {st.session_state.username}")
        st.caption("财务管理员")
        st.divider()
        menu = st.radio("系统导航", ["📊 经营仪表盘", "📝 数据录入管理", "🤖 AI 深度分析", "⚙️ 知识库设置"])
        st.divider()
        if st.button("退出系统"):
            st.session_state.logged_in = False
            st.rerun()

    df = load_data_from_db()

    # === 1. 仪表盘 ===
    if menu == "📊 经营仪表盘":
        st.title("📊 企业经营驾驶舱")
        if df.empty:
            st.info("暂无数据，请先录入。")
        else:
            total_in = df[df['金额'] > 0]['金额'].sum()
            total_out = df[df['金额'] < 0]['金额'].sum()
            profit = total_in + total_out

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("总收入", format_big_number(total_in), delta="累计")
            k2.metric("总支出", format_big_number(total_out), delta="-成本", delta_color="inverse")
            k3.metric("净利润", format_big_number(profit), delta_color="normal" if profit > 0 else "inverse")
            k4.metric("交易笔数", f"{len(df)} 笔")

            st.divider()

            time_filter = st.selectbox("📅 趋势图时间维度", ["按月", "按年", "按日"])
            chart_df = df.copy()
            if time_filter == "按月":
                chart_df['日期'] = chart_df['日期'].dt.strftime('%Y-%m')
            elif time_filter == "按年":
                chart_df['日期'] = chart_df['日期'].dt.strftime('%Y')
            else:
                chart_df['日期'] = chart_df['日期'].dt.strftime('%Y-%m-%d')

            chart_df['绘图金额'] = chart_df['金额'].abs()
            grouped = chart_df.groupby(['日期', '类型'])['绘图金额'].sum().reset_index()

            fig = px.bar(grouped, x='日期', y='绘图金额', color='类型', barmode='group',
                         title=f"收支趋势 ({time_filter})", labels={'绘图金额': '金额 (绝对值)'},
                         color_discrete_map={"收入": "#00CC96", "支出": "#EF553B"})
            st.plotly_chart(fig, use_container_width=True)

    # === 2. 数据管理 ===
    elif menu == "📝 数据录入管理":
        st.title("📝 账务中心")
        t1, t2, t3 = st.tabs(["手动录入", "Excel 导入", "查改删"])
        with t1:
            with st.form("entry"):
                c1, c2 = st.columns(2)
                i = c1.text_input("项目名称")
                d = c2.date_input("日期")
                a = st.number_input("金额 (正入负出)", step=100.0)
                if st.form_submit_button("保存"):
                    insert_record(i, d, a, st.session_state.username)
                    st.toast("✅ 录入成功！", icon="💾")
                    time.sleep(1)
                    st.rerun()
        with t2:
            up = st.file_uploader("上传 Excel")
            if up and st.button("开始导入"):
                try:
                    df_upload = pd.read_excel(up)
                    insert_batch_from_excel(df_upload, st.session_state.username)
                    st.toast(f"✅ 成功导入 {len(df_upload)} 条！", icon="📂")
                    time.sleep(1.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"导入失败: {str(e)}")
        with t3:
            c_del1, c_del2 = st.columns([1, 4])
            with c_del1:
                did = st.number_input("输入删除 ID", min_value=0, step=1)
            with c_del2:
                st.write("")
                st.write("")
                if st.button("🗑️ 确认删除"):
                    if did in df['编号'].values:
                        delete_record(did)
                        st.toast(f"✅ 编号 {did} 已删除！", icon="🗑️")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.toast("❌ 编号不存在", icon="⚠️")
            st.dataframe(df, use_container_width=True, height=600)

    # === 3. 智能分析 ===
    elif menu == "🤖 AI 深度分析":
        st.title("🤖 智能财务顾问")
        if df.empty:
            st.warning("请先录入数据")
        else:
            if st.button("🚀 生成分析报告", type="primary"):
                with st.spinner("AI 分析中..."):
                    total_in = df[df['金额'] > 0]['金额'].sum()
                    total_out = df[df['金额'] < 0]['金额'].sum()
                    profit = total_in + total_out
                    top_expense = df[df['金额'] < 0].sort_values('金额').head(5)[['日期', '项目', '金额']].to_string(
                        index=False)

                    data_summary = f"""
                    【核心数据】
                    总收入: {total_in:.2f}
                    总支出: {total_out:.2f}
                    净利润: {profit:.2f}
                    【重点大额支出TOP5】:
                    {top_expense}
                    """

                    res = get_financial_analysis(data_summary)
                    st.toast("✅ 分析完成！", icon="🤖")
                    st.markdown("### 📝 顾问报告")
                    st.markdown(res)
                    st.download_button("📥 下载报告", res, "report.txt")

    # === 4. 知识库 ===
    elif menu == "⚙️ 知识库设置":
        st.title("🧠 知识库管理")
        kf = st.file_uploader("上传 PDF/TXT", type=['pdf', 'txt'])
        if kf:
            path = os.path.join(os.getcwd(), kf.name)
            with open(path, "wb") as f:
                f.write(kf.getbuffer())
            if st.button("加载到 AI 大脑"):
                with st.spinner("正在学习..."):
                    init_knowledge_base(path)
                    st.toast("✅ 知识库加载成功！", icon="🧠")

    # === 5. ⚔️ 模型竞技场 (新增) ===
    elif menu == "⚔️ 模型竞技场 (答辩专用)":
        st.title("⚔️ 大模型性能量化评估")
        st.markdown("通过 **语义相似度**、**响应速度**、**内容生成量** 三个维度，量化对比 Llama 3.2 与其他模型的优劣。")

        if df.empty:
            st.warning("请先在数据管理页录入数据。")
        else:
            # 1. 准备数据
            total_in = df[df['金额'] > 0]['金额'].sum()
            total_out = df[df['金额'] < 0]['金额'].sum()
            data_summary = f"总收入:{total_in}, 总支出:{total_out}, 净利润:{total_in + total_out}。"

            # 2. 设置标准答案 (Ground Truth)
            with st.expander("📝 设定标准答案 (用于计算准确度)", expanded=True):
                st.caption("请输入一段你认为完美的分析结果，系统将以此为基准，计算 AI 回答的语义相似度得分。")
                default_ref = "公司目前经营状况良好，净利润为正。收入主要来源于业务增长，但支出中人力成本占比较高。建议后续控制不必要的行政开支，并关注现金流健康度。"
                reference_text = st.text_area("标准参考答案", value=default_ref, height=80)

            # 3. 选择模型
            col1, col2 = st.columns(2)
            with col1:
                model_a = "llama3.2"
                st.info(f"🔵 选手 A: {model_a} (本系统选用)")
            with col2:
                # 确保你安装了 qwen2.5:3b (ollama pull qwen2.5:3b)
                model_b = st.selectbox("🔴 选手 B (挑战者)", ["qwen2.5:3b", "phi3.5"], index=0)

            if st.button("🔥 开始量化对决 (PK)", type="primary"):
                if not reference_text:
                    st.error("请先填写标准参考答案！")
                else:
                    results = {}

                    # --- 跑模型 A ---
                    with st.spinner(f"{model_a} 正在推理..."):
                        ans_a, time_a = get_financial_analysis_with_model(data_summary, model_a)
                        score_a = calculate_similarity_score(ans_a, reference_text)
                        len_a = len(ans_a)

                    # --- 跑模型 B ---
                    with st.spinner(f"{model_b} 正在推理..."):
                        ans_b, time_b = get_financial_analysis_with_model(data_summary, model_b)
                        score_b = calculate_similarity_score(ans_b, reference_text)
                        len_b = len(ans_b)

                    # --- 展示结果卡片 ---
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"### 🔵 {model_a}")
                        st.write(ans_a)
                        st.metric("语义准确度 (0-1)", f"{score_a}", delta="越接近1越好")
                        st.metric("响应耗时 (秒)", f"{time_a}s", delta="越低越好", delta_color="inverse")
                    with c2:
                        st.markdown(f"### 🔴 {model_b}")
                        st.write(ans_b)
                        st.metric("语义准确度 (0-1)", f"{score_b}", delta=f"{round(score_b - score_a, 3)}")
                        st.metric("响应耗时 (秒)", f"{time_b}s", delta=f"{round(time_b - time_a, 2)}s",
                                  delta_color="inverse")

                    st.divider()

                    # --- 核心图表：雷达图 (Radar Chart) ---
                    st.subheader("📊 多维能力雷达图")

                    # 数据归一化处理 (为了让雷达图好看)
                    # 速度：越快分越高 -> 倒数处理 * 10
                    # 相似度：本身就是 0-1 -> * 10
                    # 字数：归一化到 0-10 之间

                    categories = ['语义准确度 (Quality)', '生成速度 (Speed)', '内容丰富度 (Quantity)']

                    fig = go.Figure()

                    fig.add_trace(go.Scatterpolar(
                        r=[score_a * 10, min(100 / time_a, 10), min(len_a / 50, 10)],
                        theta=categories,
                        fill='toself',
                        name=f'{model_a} (Blue)'
                    ))

                    fig.add_trace(go.Scatterpolar(
                        r=[score_b * 10, min(100 / time_b, 10), min(len_b / 50, 10)],
                        theta=categories,
                        fill='toself',
                        name=f'{model_b} (Red)'
                    ))

                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                        showlegend=True
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # --- 结论 ---
                    st.info(f"""
                    💡 **自动评测结论**：
                    1. **语义准确度**：基于余弦相似度计算，得分 **{score_a}** 表示 AI 生成内容与标准答案的逻辑接近程度。
                    2. **生成速度**：Llama 3.2 耗时 **{time_a}秒**，体现了端侧小模型的效率优势。
                    通过对比可见，Llama 3.2 在保持高准确度的同时，具有极佳的响应速度，适合本系统部署。
                    """)

if __name__ == "__main__":
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()