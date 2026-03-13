import streamlit as st # 【核心】网页界面框架。st 是它的简称，后面所有画按钮、写文字都用它。
import os # 操作系统接口。用来设置环境变量（如下载镜像地址）。
import sys # 系统相关参数。通常用于退出程序或获取路径。
import shutil # 【文件管家】高级文件操作。用来复制、删除整个文件夹。
import re # 【正则表达式】用来从文件名里精准提取数字（如从 "2024 真题" 中提取 "2024"）。
import time # 时间工具。用来让程序“暂停”一下（比如做进度条动画时）。
from pathlib import Path # 【路径处理】比 os 更现代的方式，用来处理文件和文件夹路径。
from datetime import datetime # 日期时间工具。（本代码暂未深度使用，但常备用）。

# --- 引入原有的 LlamaIndex 逻辑 ---
# 注意：为了在 Streamlit 中稳定运行，我们需要缓存模型加载
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage, Document, PromptTemplate
# SimpleDirectoryReader 读取文件夹里的文档
# VectorStoreIndex 创建向量索引
# Settings 全局设置
# StorageContext 存储上下文
# load_index_from_storage 从磁盘加载已建好的索引
# Document 文档对象
# PromptTemplate 提示词模板

from llama_index.core.vector_stores import MetadataFilter, MetadataFilters # 过滤器（比如只查 2024 年的题）。

from llama_index.llms.ollama import Ollama # 连接本地 Ollama 大模型。
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # 连接 HuggingFace 的嵌入模型（把文字转数字）。

import pandas as pd # 【数据表格】用来把文件列表变成漂亮的表格展示。
from collections import Counter, defaultdict # 计数器。用来统计单词出现了多少次。
from tqdm import tqdm # 进度条。在后台运行时显示绿色进度条（Web 版用得少，主要用在命令行版）。

# http://localhost:8501/

# --- Streamlit 设置 (必须放在第一行) ---
# Streamlit 特有配置：页面配置：标题、图标、宽屏模式
st.set_page_config(
    page_title="一建智能备考助手", # 浏览器标签页上显示的名字
    page_icon="🏗️", # 浏览器标签页上的小图标
    layout="wide", # 让页面沾满全屏
    initial_sidebar_state="expanded" # 刚打开时，左侧菜单栏是展开的
)


# ================= 配置区域 =================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 设置国内镜像，防止下载模型时因为网络问题失败
PERSIST_DIR = "./storage" # 定义变量：向量索引存放到哪个文件夹
DATA_DIR = "./data" # 定义变量：原始文档（PDF/TXT）放在哪个文件夹

# ================= 辅助函数 =================
# 模型加载缓存
@st.cache_resource # @st.cache_resource 【魔法装饰器】 告诉 Streamlit：“这个函数跑一次就够了，把结果存起来，下次直接用，别重跑！”
def get_models():
    """缓存模型加载，避免每次交互都重新加载"""
    """
    作用：加载 AI 模型。
    为什么加 @st.cache_resource？
    Streamlit 的特性是：你每点一个按钮，整个代码会从第一行重跑一遍。
    如果没有这个装饰器，每次点按钮都会重新加载几十亿参数的模型，电脑会卡死。
    加上它，Streamlit 会记住：“这个函数跑过了，结果存好了，下次直接用，别重跑！”
    """
    try:
        # 加载嵌入模型 (负责把文字转成向量数字)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
        # 加载大语言模型 (负责思考和说话)
        # model="qwen2.5:3b": 使用通义千问 2.5 的 30 亿参数版本
        # request_timeout=120.0: 如果 AI 思考超过 120 秒，就强制停止，防止死等
        # context_window=4096: AI 一次性能读进去的最大字数
        llm = Ollama(model="qwen2.5:3b", request_timeout=120.0, context_window=4096)
        return embed_model, llm
    except Exception as e:
        st.error(f"模型加载失败: {e}") # 在网页上显示红色报错信息
        st.stop() # 停止运行后面的代码

# 辅助函数：从文件名提取年份
def extract_year_from_filename(file_path: str) -> str:
    match = re.search(r'\d{4}', file_path) # 在字符串里找连续的 4 个数字
    # 如果找到了返回数字，没找到返回 "未知"
    return match.group(0) if match else "未知"

# 构建或加载索引
@st.cache_resource # 同样需要缓存，因为建索引很慢
def build_or_load_index():
    """构建或加载索引 (带缓存)"""
    # 1. 先获取模型
    embed_model, llm = get_models()

    # 2. 告诉 LlamaIndex 全局使用这两个模型
    Settings.embed_model = embed_model
    Settings.llm = llm

    # 3. 检查硬盘上有没有现成的索引 (./storage 文件夹)
    if Path(PERSIST_DIR).exists():
        try:
            # 如果有，直接加载，速度极快 (秒开)
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            return index
        except Exception:
            # 如果加载出错（比如文件损坏），就跳过，执行下面的重建逻辑
            pass
    
    # 4. 重建逻辑 (如果没有索引，或者第一次运行)
    # # 检查 data 文件夹是否存在，不存在就创建一个
    if not Path(DATA_DIR).exists():
        Path(DATA_DIR).mkdir(parents=True)
        st.warning("data 文件夹不存在，已自动创建。请放入文件后刷新页面。")
        return None

    # 5. 开始干活：显示一个旋转的 loading 图标
    with st.spinner("📚 正在扫描并构建向量索引 (首次较慢)..."):
        try:
            # 读取 data 目录下所有文件
            raw_documents = SimpleDirectoryReader(DATA_DIR).load_data()
            documents = []

             # 遍历每一个读到的文档，进行“增强”
            for doc in raw_documents:
                file_path = doc.metadata.get("file_path", "")
                year = extract_year_from_filename(file_path) # 提取年份

                # 创建一个新的 Document 对象，注入额外的元数据 (metadata)
                new_doc = Document(
                    text=doc.text, # 原文内容
                    metadata={
                        **doc.metadata, # 保留原来的信息（如文件名）
                        "year": year, # 新增：年份 (用于以后按年份筛选)
                        "source_type": "真题" if "真题" in file_path else "教材" # 新增：类型
                    }
                )
                documents.append(new_doc)
            
            if not documents:
                st.warning("data 文件夹为空。")
                return None

            # 6. 真正建立向量索引 (最耗时的一步)
            # 过程：切分文本 -> 调用 embed_model 转向量 -> 存入内存
            index = VectorStoreIndex.from_documents(documents)
            # 7. 把建好的索引保存到硬盘 (./storage)，下次就不用重算了
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            st.success("✨ 索引构建完成！")
            return index
        except Exception as e:
            st.error(f"构建索引失败: {e}")
            return None


# ================= 功能模块 =================

def ask_question_logic(index, query, target_year=None):
    """问答逻辑"""
    filters = None
    # 如果用户选了年份，就创建过滤器
    if target_year:
        filters = MetadataFilters(filters=[
            MetadataFilter(key="year", value=target_year, operator="==")
        ])
    
    # 创建查询引擎
    # similarity_top_k=5: 去数据库里找最相似的 5 段文字
    # streaming=True: 开启流式模式 (字一个个蹦出来)
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        filters=filters,
        streaming=True
    )
    
    # 执行查询，返回响应对象
    response = query_engine.query(query)
    return response

def analyze_trends_logic(index):
    """趋势分析逻辑 (简化版用于展示)"""
    if not index:
        return "未找到索引"
    
    # 1. 设置过滤器：只要 "真题"，不要 "教材"
    filters = MetadataFilters(filters=[
        MetadataFilter(key="source_type", value="真题", operator="==")
    ])
    try:
        # 2. 尝试获取所有真题片段 (top_k=500 表示最多取 500 段)
        query_engine = index.as_query_engine(similarity_top_k=500, filters=filters)
        response = query_engine.query("一建真题全部内容") # 随便问个啥，目的是拿到 source_nodes
        nodes = response.source_nodes
    except:
        # 如果过滤失败，就不带过滤查
        query_engine = index.as_query_engine(similarity_top_k=500)
        response = query_engine.query("一建真题全部内容")
        nodes = response.source_nodes

    if not nodes:
        return "未找到真题数据"

    # 3. 拼接文本 (为了省时间，这里只取每段的前 500 字)
    # 简单的关键词统计 (为了演示速度，这里简化了 LLM 逐段提取，改用高频词统计 + 少量 LLM 总结)
    # 在实际生产环境中，你可以保留之前的逐段提取逻辑，但加上 st.progress 进度条
    all_text = " ".join([n.get_content()[:500] for n in nodes])
    
    # 4. 构造提示词，让 AI 总结
    # 调用 LLM 进行整体分析
    prompt = f"""
    你是一建专家。基于以下真题片段内容，请总结：
    1. 出现频率最高的 5 个核心考点。
    2. 预测 2026 年最可能考的 3 个方向。
    
    内容片段：{all_text[:15000]} (截取前 15000 字以防超时)
    
    请以 Markdown 格式输出，包含标题和列表。
    """
    
    # 5. 调用 AI 生成回答
    llm = Settings.llm
    response = llm.complete(prompt)
    return response.text

def manage_files_action(action, file_path=None):
    """文件管理逻辑"""
    if action == "list": # 列出文件
        if not Path(DATA_DIR).exists():
            return []
        # 遍历文件夹，排除隐藏文件 (以 . 开头的)
        files = [f for f in Path(DATA_DIR).iterdir() if not f.name.startswith('.')]
        return sorted(files, key=lambda x: x.name) # 按名字排序
    
    elif action == "add": # 添加文件
        if not file_path: return False
        src = Path(file_path)
        if not src.exists():
            return "文件不存在"
        dest = Path(DATA_DIR) / src.name # 目标路径
        shutil.copy2(src, dest) # 复制文件
        return True
    
    elif action == "delete": # 删除文件
        if not file_path: return False
        target = Path(DATA_DIR) / file_path
        if target.exists():
            target.unlink() # 删除文件
            return True
        return "文件未找到"
    
    elif action == "rebuild": # 重建索引
        if Path(PERSIST_DIR).exists():
            shutil.rmtree(PERSIST_DIR) # 暴力删除整个 storage 文件夹
        return "索引已清除，请刷新页面重建"


# ================= 界面布局 =================

# 网页大标题
st.title("🏗️ 一建智能备考助手 (移动版)")
# 副标题
st.markdown("基于本地大模型的一级建造师备考神器 | 支持问答、趋势分析、知识库管理")

# 侧边栏
# with st.sidebar: 以下内容全部放在左侧边栏
with st.sidebar:
    st.header("⚙️ 控制中心")
    
    # st.session_state: 浏览器的“记忆”。因为脚本会重跑，我们需要把加载好的 index 存在这里，防止丢失
    # 如果 'index' 还没在记忆里，说明第一次加载
    if 'index' not in st.session_state:
        with st.spinner("正在加载模型和索引..."):
            # 调用前面定义的函数，把结果存进记忆
            st.session_state.index = build_or_load_index()
    
    # 检查加载成功没
    if st.session_state.index:
        st.success("✅ 系统就绪")
    else:
        st.error("❌ 系统未就绪 (请检查 data 文件夹)")
        st.stop() # 停止运行，不让用户操作
    
    # 菜单选择： st.radio: 单选按钮，用来切换不同的功能模块。
    menu = st.radio("选择功能", ["💬 智能问答", "📈 趋势与押题", "📂 知识库管理"])
    
    # 画一条分割线
    st.markdown("---")
    st.info("💡 提示：在 Safari 中点击分享按钮 -> '添加到主屏幕'，即可像 App 一样使用！")


# 主内容区
if menu == "💬 智能问答":
    st.header("💬 智能问答")
    
    # 创建两列布局，左边宽 (3)，右边窄 (1)
    col1, col2 = st.columns([3, 1]) 
    with col1:
        # 文本输入框
        query = st.text_input("请输入问题:", placeholder="例如：深基坑支护有哪些类型？")
    with col2:
        # 下拉选择框
        year_filter = st.selectbox("限定年份 (可选)", ["全部", "2024", "2023", "2022", "2021"])
    
    # 当用户点击按钮时
    if st.button("🚀 开始分析"):
        if not query:
            st.warning("请输入问题")
        else:
            # 处理年份逻辑
            target_year = None if year_filter == "全部" else year_filter
            # 模拟聊天机器人的气泡
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    # 调用逻辑函数获取答案
                    response = ask_question_logic(st.session_state.index, query, target_year)
                    
                    # 流式输出
                    placeholder = st.empty() # 创建一个空白容器，准备随时更新内容
                    full_response = "" # 累积完整的回答
                    # 遍历 AI 吐出来的每一个字 (token)
                    for token in response.response_gen:
                        full_response += token
                        # 实时更新容器内容，加上 "▌" 模拟光标
                        placeholder.markdown(full_response + "▌")
                    # 最后去掉光标，显示完整内容
                    placeholder.markdown(full_response)
                    
                    # 来源引用(折叠框)
                    if response.source_nodes:
                        with st.expander("📖 查看依据来源"):
                            for i, node in enumerate(response.source_nodes):
                                fname = node.metadata.get("file_name", "未知")
                                year = node.metadata.get("year", "?")
                                snippet = node.get_content()[:200] + "..."
                                st.markdown(f"**[{i+1}] {fname} ({year})**:\n> {snippet}")

elif menu == "📈 趋势与押题":
    st.header("📈 真题趋势分析与押题")
    st.write("系统将分析所有真题文件，提取高频考点并生成预测报告。")
    
    if st.button("🔍 启动深度分析引擎"):
        with st.spinner("AI 正在阅读大量真题，这可能需要 1-2 分钟..."):
            # 创建进度条 (初始 0%)
            progress_bar = st.progress(0)
            # 模拟进度动画 (因为真正的分析在后台，很难实时知道百分比，这里做个假动画安抚用户)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            
            # 调用真正的分析函数
            result = analyze_trends_logic(st.session_state.index)
            st.markdown(result)

elif menu == "📂 知识库管理":
    st.header("📂 知识库管理")
    
    tab1, tab2, tab3 = st.tabs(["📄 文件列表", "➕ 上传文件", "🗑️ 删除/重建"])
    
    with tab1:
        files = manage_files_action("list")
        if files:
            df = pd.DataFrame([
                {"文件名": f.name, "大小(MB)": round(f.stat().st_size/1024/1024, 2), "年份": extract_year_from_filename(f.name)}
                for f in files
            ])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("暂无文件")
            
    with tab2:
        uploaded_file = st.file_uploader("选择文件上传", type=["pdf", "txt", "md", "docx"])
        if uploaded_file:
            save_path = Path(DATA_DIR) / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ {uploaded_file.name} 上传成功！\n\n⚠️ **重要**: 请点击 '删除/重建' 标签页重建索引以生效。")
            
    with tab3:
        st.warning("⚠️ 修改文件后必须重建索引！")
        
        # 删除文件
        files = manage_files_action("list")
        file_names = [f.name for f in files]
        if file_names:
            to_delete = st.selectbox("选择要删除的文件", file_names)
            if st.button("删除选中文件"):
                manage_files_action("delete", to_delete)
                st.success(f"已删除 {to_delete}，请重建索引。")
                st.rerun()
        
        st.divider()
        
        if st.button("🔄 重建索引 (清除旧缓存并重新扫描)", type="primary"):
            msg = manage_files_action("rebuild")
            st.success(msg)
            st.info("索引已清除。请点击左侧边栏刷新页面或点击任意按钮触发重新构建。")
            # 强制刷新
            time.sleep(2)
            st.rerun()