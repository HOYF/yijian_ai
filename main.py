import os

# 设置 HuggingFace 国内镜像地址
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path

from llama_index.core import Document
import re

# 构建过滤器
from llama_index.core.vector_stores import FilterCondition, MetadataFilter

# =================配置区域=================

# 1. 设置嵌入模型 (将文字转为向量)
# 使用 BGE 小模型，速度快且中文效果好
print("⚙️ 正在加载嵌入模型...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 2. 设置大语言模型 (负责生成回答)
# 【重要修正】改用 qwen2.5:3b，0.5b 太笨了，无法处理复杂逻辑，后续可以根据配置更换更强大的模型
# 确保你已经运行过: ollama pull qwen2.5:3b
print("⚙️ 正在连接本地大模型 (Qwen2.5-3B)...")
try:
    Settings.llm = Ollama(model="qwen2.5:3b", request_timeout=120.0, context_window=4096)
except Exception as e:
    print(f"❌ 连接 Ollama 失败: {e}")
    print("💡 提示: 请确保 Ollama 软件已运行，且已执行 'ollama pull qwen2.5:3b'")
    sys.exit(1)

# =================主程序=================

def extract_year_from_filename(file_path: str) -> str:
    """从文件名中提取年份，如 '2024 环球网校...' -> '2024'"""
    match = re.search(r'\d{4}', file_path)
    return match.group(0) if match else "未知"


PERSIST_DIR = "./storage" # 索引保存目录

def build_or_load_knowledge_base():
    """构建知识库，如果已存在则直接加载，节省时间"""
    
    # 检查是否已经存在建好的索引
    if Path(PERSIST_DIR).exists():
        print("📂 发现已存在的索引，正在快速加载...")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            print("✨ 索引加载完成！")
            return index
        except Exception as e:
            print(f"⚠️ 加载索引失败 ({e})，将重新构建...")

    # 如果没有索引，则从头构建
    print("📚 正在扫描 './data' 文件夹加载教材和真题...")
    
    if not Path("./data").exists():
        print("❌ 错误: 找不到 './data' 文件夹！请先创建该文件夹并放入一些 txt 或 pdf 文件。")
        sys.exit(1)

    try:
        # 读取所有文件
        raw_documents = SimpleDirectoryReader("./data").load_data()

        # 为每个文档添加 metadata（包括年份、文件名等）
        documents = []
        for doc in raw_documents:
            # 获取原始文件路径
            file_path = doc.metadata.get("file_path", "")
            year = extract_year_from_filename(file_path)

            # 创建新文档，附加metadata
            new_doc = Document(
                text=doc.text,
                metadata={
                    **doc.metadata,  # 保留原有 metadata（如 file_name）
                    "year": year,    # 新增字段：年份
                    "source_type": "真题" if "真题" in file_path else "教材"  # 可选：标记类型
                }
            )
            documents.append(new_doc)

        print(f"✅ 成功加载 {len(documents)} 个文档片段（已自动标注年份）。")

    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        sys.exit(1)
    
    if len(documents) == 0:
        print("❌ 错误: './data' 文件夹是空的，请放入文件后再试。")
        sys.exit(1)

    print(f"✅ 成功加载 {len(documents)} 个文档片段。")
    print("🔄 正在建立向量索引 (首次运行可能需要几分钟，请耐心等待)...")
    
    # 创建向量索引
    index = VectorStoreIndex.from_documents(documents)
    
    # 保存到磁盘，下次不用重新算
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"💾 索引已保存至 '{PERSIST_DIR}' 文件夹。")
    print("✨ 知识库构建完成！")
    
    return index

def ask_question(index, query):
    """执行查询"""
    # 创建查询引擎
    # similarity_top_k=5 表示参考最相关的 5 段内容
    # query_engine = index.as_query_engine(similarity_top_k=5)
    """执行查询 (带流式输出) - 支持按年份过滤"""
    year_match = re.search(r'\b(20\d{2})\b', query)
    target_year = year_match.group(1) if year_match else None

    print(f"\n🤖 思考中: {query} ...")
    if target_year:
        print(f"🔍 正在筛选 {target_year} 年的相关资料...")

    filters = None
    if target_year:
        filters = MetadataFilter.from_dict(
            key="year",
            value=target_year,
            operator="=="
        )
    
    # 创建查询引擎，传入过滤器，开启 streaming
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        filters=filters,  # 👈 关键：只检索匹配年份的文档
        streaming=True
    )


    # query_engine = index.as_query_engine(similarity_top_k=5, streaming=True) # 开启 streaming
    
    # print(f"\n🤖 思考中: {query} ...")
    try:
        response = query_engine.query(query)
        
        print(f"\n💡 AI 回答:")
        # 流式打印
        for token in response.response_gen:
            print(token, end="", flush=True)
        print() # 换行


        # 显示引用来源
        if response.source_nodes:
            print("\n📖 依据来源:")
            for i, node in enumerate(response.source_nodes):
                # 清理文本格式，只展示前 80 个字
                content = node.get_content().replace("\n", " ").strip()
                snippet = content[:80] + "..." if len(content) > 80 else content
                # 尝试获取文件名
                file_name = node.metadata.get("file_name", "未知文件")
                year = node.metadata.get("year", "未知年份")
                print(f"   [{i+1}] 📄 {file_name} ({year}): {snippet}")
        else:
            print("\n⚠️ 未找到相关参考资料，回答可能不准确。")
            
    except Exception as e:
        print(f"❌ 生成回答时出错: {e}")
        print("💡 可能是模型响应超时或内容过长，请尝试简化问题。")

if __name__ == "__main__":
    print("🏗️  一建智能备考助手 (本地版) 启动中...")
    print("-" * 40)
    
    # 1. 构建或加载知识库
    index = build_or_load_knowledge_base()
    
    print("-" * 40)
    print("✅ 准备就绪！你可以开始提问了。")
    print("💡 示例问题: '深基坑支护有哪些类型？' 或 '索赔成立的三个条件是什么？'")
    
    # 2. 循环提问
    while True:
        try:
            user_input = input("\n👉 请输入问题 (输入 'q' 退出): ").strip()
            if user_input.lower() == 'q':
                print("👋 再见，祝备考顺利！")
                break
            if not user_input:
                continue
                
            ask_question(index, user_input)
        except KeyboardInterrupt:
            print("\n👋 强制退出，再见！")
            break