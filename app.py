import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
from captcha.image import ImageCaptcha
import random
import string
import time  # 引入 time 库，用于稍微停顿一下展示成功信息

# 引用 AI 引擎
from ai_engine import init_knowledge_base, get_financial_analysis

# --- 0. 数据库管理 ---
DB_FILE = 'finance_system.db'


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
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


# --- 辅助函数：优化版大数字格式化 ---
def format_big_number(num):
    """
    强制缩写逻辑：
    只要绝对值超过 1万，就缩写，确保 UI 不会炸。
    """
    abs_num = abs(num)
    if abs_num >= 100000000:  # 亿
        return f"¥{num / 100000000:.2f} 亿"
    elif abs_num >= 10000:  # 万
        return f"¥{num / 10000:.2f} 万"
    else:
        # 小于1万，正常显示，保留2位小数
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


# --- 登录页面 ---
def login_page():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>🔐 智财云登录</h2>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="UserName")
            password = st.text_input("密码", type="password", placeholder="password")
            c1, c2 = st.columns([2, 1])
            with c1: captcha_input = st.text_input("验证码")
            with c2: st.image(st.session_state.captcha_image, caption="")
            submitted = st.form_submit_button("登录", type="primary")

        if st.button("看不清？刷新"):
            text, data = generate_captcha_image()
            st.session_state.captcha_text = text
            st.session_state.captcha_image = data
            st.rerun()

        if submitted:
            valid_users = {"admin": "123456", "boss": "888888"}
            if username in valid_users and password == valid_users[username]:
                if captcha_input.upper() == st.session_state.captcha_text:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.toast(f"欢迎回来，{username}！", icon="👋")
                    time.sleep(1)  # 稍等一下让用户看到提示
                    st.rerun()
                else:
                    st.error("验证码错误")
                    text, data = generate_captcha_image()
                    st.session_state.captcha_text = text
                    st.session_state.captcha_image = data
                    st.rerun()
            else:
                st.error("账号密码错误")


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
            # 核心指标卡
            total_in = df[df['金额'] > 0]['金额'].sum()
            total_out = df[df['金额'] < 0]['金额'].sum()
            profit = total_in + total_out

            # 使用 format_big_number 确保不折行
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("总收入", format_big_number(total_in), delta="累计")
            k2.metric("总支出", format_big_number(total_out), delta="-成本", delta_color="inverse")
            k3.metric("净利润", format_big_number(profit), delta_color="normal" if profit > 0 else "inverse")
            k4.metric("交易笔数", f"{len(df)} 笔")

            st.divider()

            # 图表区
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
                         title=f"收支趋势 ({time_filter})",
                         labels={'绘图金额': '金额 (绝对值)'},
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
                    st.toast("✅ 录入成功！已保存到数据库。", icon="💾")
                    time.sleep(1)  # 停顿1秒让用户看到提示
                    st.rerun()

        with t2:
            st.info("支持 xlsx/xls 格式，需包含列：项目, 日期, 金额")
            up = st.file_uploader("上传 Excel")
            if up and st.button("开始导入"):
                try:
                    df_upload = pd.read_excel(up)
                    insert_batch_from_excel(df_upload, st.session_state.username)
                    st.toast(f"✅ 批量导入成功！共导入 {len(df_upload)} 条数据。", icon="📂")
                    time.sleep(1.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"导入失败: {str(e)}")

        with t3:
            # 顶部增加删除区，不用拉到最底下
            c_del1, c_del2 = st.columns([1, 4])
            with c_del1:
                did = st.number_input("输入要删除的编号 ID", min_value=0, step=1)
            with c_del2:
                st.write("")  # 占位
                st.write("")
                if st.button("🗑️ 确认删除该记录", type="primary"):
                    if did in df['编号'].values:
                        delete_record(did)
                        st.toast(f"✅ 编号 {did} 已彻底删除！", icon="🗑️")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.toast("❌ 编号不存在，请检查后输入", icon="⚠️")

            st.markdown("### 📊 数据明细表")
            st.dataframe(df, use_container_width=True, height=600)

    # === 3. 智能分析 ===
    elif menu == "🤖 AI 深度分析":
        st.title("🤖 智能财务顾问")
        if df.empty:
            st.warning("请先录入数据")
        else:
            if st.button("🚀 生成深度分析报告", type="primary"):
                with st.spinner("AI 正在阅读报表并生成分析..."):
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

                    st.toast("✅ 分析报告已生成！", icon="🤖")
                    st.success("分析完成！")
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
                    st.toast("✅ 知识库加载成功！AI 变强了。", icon="🧠")


if __name__ == "__main__":
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()