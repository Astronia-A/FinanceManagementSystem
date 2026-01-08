import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
import plotly.graph_objects as go
from captcha.image import ImageCaptcha
import random
import string
import time
import hashlib

# 引用后端引擎
from ai_engine import init_knowledge_base, get_financial_analysis, get_financial_analysis_with_model, \
    calculate_similarity_score

# --- 0. 数据库与安全配置 ---
DB_FILE = 'finance_system.db'


def make_hash(password):
    """SHA-256 密码加密"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_login(username, password):
    """验证登录凭证"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0] == make_hash(password):
        return True
    return False


def init_db():
    """数据库初始化：建表、添加默认管理员"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # 流水表
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

    # 用户表
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
    ''')

    # 插入默认管理员 (admin/123456)
    c.execute("SELECT count(*) FROM users")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO users VALUES (?, ?)", ("admin", make_hash("123456")))
        c.execute("INSERT INTO users VALUES (?, ?)", ("boss", make_hash("888888")))

    conn.commit()
    conn.close()


def load_data_from_db():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM records", conn)
    conn.close()
    # 修复日期格式报错
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


# --- 辅助函数 ---
def format_big_number(num):
    """UI优化：大数字转万/亿单位"""
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
    print(f"🔑 [DEBUG] 验证码: {captcha_text}")  # 控制台后门
    return captcha_text, data


# --- 程序入口设置 ---
st.set_page_config(page_title="智财云 Dashboard", layout="wide", page_icon="💰")
init_db()

# Session 初始化
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
if 'captcha_text' not in st.session_state:
    text, data = generate_captcha_image()
    st.session_state.captcha_text = text
    st.session_state.captcha_image = data


# --- 1. 登录页面 (UI 最终优化版) ---
# --- 登录页面 (背景图版) ---
def login_page():
    # === 1. 核心修改：通过 CSS 注入全屏背景图 ===
    # 你可以将 url 里的链接替换为你本地图片的 base64 编码，或者直接使用网络图片 URL
    background_css = """
    <style>
    /* 设置整个应用的背景 */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1554224155-8d04cb21cd6c?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /*为了让登录框在背景上更清晰，给表单添加半透明白色背景和阴影 */
    [data-testid="stForm"] {
        background-color: rgba(255, 255, 255, 0.95); /* 95%不透明度的白色 */
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)

    # === 2. 布局调整：改为居中布局 ===
    # 之前是 [1, 1.2, 1]，现在左右对称 [1, 1.5, 1] 让登录框居中
    col1, col2, col3 = st.columns([1, 1.5, 1], vertical_alignment="center")

    # col1 和 col3 留白，只在 col2 显示登录框
    with col2:
        # 增加一些顶部的空白，让登录框不要贴着浏览器顶端
        st.write("")
        st.write("")

        st.markdown("<h2 style='text-align: center; color: white; text-shadow: 2px 2px 4px #000000;'>🔐 智财云登录</h2>",
                    unsafe_allow_html=True)

        # 纯 Python 方式：通过调整列比例来限制大小
        # [3, 1, 3] 的比例会把中间的列挤得很窄，强迫图片变小
        _, c_logo, _ = st.columns([3, 1, 3])
        with c_logo:
            # 关键：去掉 use_container_width=True，并指定 width 参数
            st.image("icon/icon.png", width=100)

        # 登录表单
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="请输入用户名")
            password = st.text_input("密码", type="password", placeholder="请输入密码")

            # 验证码对齐 (保持之前的逻辑)
            c_input, c_image = st.columns([2.5, 1], vertical_alignment="bottom")
            with c_input:
                captcha_input = st.text_input("验证码", placeholder="不区分大小写")
            with c_image:
                st.image(st.session_state.captcha_image, use_container_width=True)

            st.write("")  # 间距

            # 按钮行 (保持之前的逻辑)
            c_login_btn, c_refresh_btn = st.columns([2.5, 1], vertical_alignment="bottom")
            with c_login_btn:
                submitted = st.form_submit_button("🚀 立即登录", type="primary", use_container_width=True)
            with c_refresh_btn:
                refresh = st.form_submit_button("🔄 刷新", use_container_width=True)

        # 逻辑处理 (保持不变)
        if refresh:
            text, data = generate_captcha_image()
            st.session_state.captcha_text = text
            st.session_state.captcha_image = data
            st.rerun()

        if submitted:
            if verify_login(username, password):
                if captcha_input.upper() == st.session_state.captcha_text:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.toast(f"欢迎回来，{username}！", icon="👋")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("验证码错误")
                    text, data = generate_captcha_image()
                    st.session_state.captcha_text = text
                    st.session_state.captcha_image = data
                    st.rerun()
            else:
                st.error("用户名或密码错误")

# --- 2. 主界面 ---
def main_app():
    with st.sidebar:
        st.title(f"👤 {st.session_state.username}")
        st.caption("财务管理员")
        st.divider()
        menu = st.radio("系统导航",
                        ["📊 经营状况", "📝 数据录入管理", "🤖 AI 深度分析", "⚙️ 知识库设置", "⚔️ 模型比较"])
        st.divider()
        if st.button("退出系统"):
            st.session_state.logged_in = False
            st.rerun()

    df = load_data_from_db()

    # === 功能1: 仪表盘 ===
    if menu == "📊 经营状况":
        st.title("📊 收支总览")
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

            # 趋势图
            time_filter = st.selectbox("📅 趋势图时间维度", ["按月", "按年", "按日"])
            chart_df = df.copy()
            if time_filter == "按月":
                chart_df['日期'] = chart_df['日期'].dt.strftime('%Y-%m')
            elif time_filter == "按年":
                chart_df['日期'] = chart_df['日期'].dt.strftime('%Y')
            else:
                chart_df['日期'] = chart_df['日期'].dt.strftime('%Y-%m-%d')

            # 修复：先取绝对值再分组，解决 Pandas 报错
            chart_df['绘图金额'] = chart_df['金额'].abs()
            grouped = chart_df.groupby(['日期', '类型'])['绘图金额'].sum().reset_index()

            fig = px.bar(grouped, x='日期', y='绘图金额', color='类型', barmode='group',
                         title=f"收支趋势 ({time_filter})", labels={'绘图金额': '金额 (绝对值)'},
                         color_discrete_map={"收入": "#00CC96", "支出": "#EF553B"})
            st.plotly_chart(fig, use_container_width=True)

    # === 功能2: 数据管理 ===
    elif menu == "📝 数据录入管理":
        st.title("📝 账务中心")
        t1, t2, t3 = st.tabs(["手动录入", "Excel 导入", "明细总览"])
        with t1:
            with st.form("entry"):
                c1, c2 = st.columns(2)
                i = c1.text_input("项目名称")
                d = c2.date_input("日期")
                a = st.number_input("金额 (正：收入；负：支出)", step=100.0)
                if st.form_submit_button("保存"):
                    insert_record(i, d, a, st.session_state.username)
                    st.toast("✅ 录入成功！", icon="💾")
                    time.sleep(1)
                    st.rerun()
        with t2:
            st.info("支持 xlsx/xls，列名：项目, 日期, 金额")
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
            # 顶部删除区
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
            st.dataframe(df, use_container_width=True, height=500)

    # === 功能3: 智能分析 ===
    elif menu == "🤖 AI 深度分析":
        st.title("🤖 智能财务顾问")
        if df.empty:
            st.warning("请先录入数据")
        else:
            if st.button("🚀 生成分析报告", type="primary"):
                with st.spinner("AI 正在阅读报表并生成分析..."):
                    # 数据摘要优化：只发统计值和TOP5，防止 Token 溢出
                    total_in = df[df['金额'] > 0]['金额'].sum()
                    total_out = df[df['金额'] < 0]['金额'].sum()
                    profit = total_in + total_out
                    top_expense = df[df['金额'] < 0].sort_values('金额').head(5)[['日期', '项目', '金额']].to_string(
                        index=False)

                    data_summary = f"总收入:{total_in:.2f} 总支出:{total_out:.2f} 净利润:{profit:.2f} 大额支出TOP5:\n{top_expense}"

                    res = get_financial_analysis(data_summary)
                    st.toast("✅ 分析完成！", icon="🤖")
                    st.markdown("### 📝 顾问报告")
                    st.markdown(res)
                    st.download_button("📥 下载报告", res, "report.txt")

    # === 功能4: 知识库 ===
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

    # === 功能5: 模型竞技场 (答辩加分项) ===
    elif menu == "⚔️ 模型比较":
        st.title("⚔️ 多模型性能量化评估")
        if df.empty:
            st.warning("请先录入数据")
        else:
            total_in = df[df['金额'] > 0]['金额'].sum()
            total_out = df[df['金额'] < 0]['金额'].sum()
            data_summary = f"收入:{total_in} 支出:{total_out}。"

            with st.expander("📝 设定标准答案 (Ground Truth)", expanded=True):
                default_ref = "经营状况良好，净利润为正。建议控制人力成本开支。"
                reference_text = st.text_area("标准参考答案", value=default_ref)

            c_m1, c_m2 = st.columns(2)
            with c_m1:
                model_a = "llama3.2"
            with c_m2:
                model_b = st.selectbox("挑战者", ["qwen2.5:3b", "phi3.5"], index=0)

            if st.button("🔥 开始对决"):
                with st.spinner("正在对比推理..."):
                    # 跑模型 A
                    ans_a, time_a = get_financial_analysis_with_model(data_summary, model_a)
                    score_a = calculate_similarity_score(ans_a, reference_text)
                    # 跑模型 B
                    ans_b, time_b = get_financial_analysis_with_model(data_summary, model_b)
                    score_b = calculate_similarity_score(ans_b, reference_text)

                    # 结果展示
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.subheader(f"🔵 {model_a}")
                        st.write(ans_a)
                        st.metric("准确度", score_a)
                        st.metric("耗时", f"{time_a}s")
                    with cc2:
                        st.subheader(f"🔴 {model_b}")
                        st.write(ans_b)
                        st.metric("准确度", score_b, delta=f"{round(score_b - score_a, 3)}")
                        st.metric("耗时", f"{time_b}s", delta=f"{round(time_b - time_a, 2)}s", delta_color="inverse")

                    # 雷达图
                    categories = ['语义准确度', '生成速度', '内容量']
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(r=[score_a * 10, min(100 / time_a, 10), min(len(ans_a) / 50, 10)],
                                                  theta=categories, fill='toself', name=model_a))
                    fig.add_trace(go.Scatterpolar(r=[score_b * 10, min(100 / time_b, 10), min(len(ans_b) / 50, 10)],
                                                  theta=categories, fill='toself', name=model_b))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()