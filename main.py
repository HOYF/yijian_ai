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


import pandas as pd
from collections import Counter
from tqdm import tqdm

class TrendAnalyzer:
    def __init__(self, index):
        self.index = index

    def extract_keywords_from_nodes(self, texts):
        """利用 LLM 从文本列表中提取核心考点关键词"""
        if not texts:
            return []
        
        # 截取部分内容防止超时，取前 15 段
        sample_text = " ".join(texts[:15])
        
        prompt_text = f"""
        你是一级建造师考试分析专家。
        以下是从历年真题中提取的若干文本片段：
        ---
        {sample_text}
        ---
        请从中提取出 5-10 个核心考点关键词（例如：'索赔条件', '深基坑支护', '网络计划优化'）。
        只返回关键词列表，用逗号分隔，不要其他废话。
        """
        
        from llama_index.core import PromptTemplate
        prompt = PromptTemplate(prompt_text)
        
        try:
            response = Settings.llm.complete(prompt.format())
            keywords_str = response.text.strip()
            # 清洗数据：处理中文逗号和换行
            keywords = [k.strip() for k in keywords_str.replace('，', ',').replace('\n', ',').split(',') if len(k.strip()) > 1]
            return keywords
        except Exception as e:
            print(f"⚠️ 考点提取失败: {e}")
            return []

    def analyze_trends(self):
        print("\n📊 正在启动真题趋势分析引擎...")
        print("🔄 第一步：扫描所有年份真题，提取高频考点...")
        
        # 【核心修复】正确构建过滤条件
        filters = None
        try:
            from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
            
            # 1. 创建单个过滤条件
            single_filter = MetadataFilter(
                key="source_type",
                value="真题",
                operator="=="
            )
            
            # 2. 包装成 Filters 对象 (注意是复数，且需要放在 filters 列表中)
            filters = MetadataFilters(filters=[single_filter])
            
        except ImportError:
            print("⚠️ 未找到最新的过滤类，将尝试不使用过滤进行全量扫描...")
        except Exception as e:
            print(f"⚠️ 构建过滤器时出错 ({e})，将尝试不使用过滤进行全量扫描...")

        # 建立查询引擎
        query_kwargs = {"similarity_top_k": 200}
        if filters:
            query_kwargs["filters"] = filters
            
        try:
            query_engine_all = self.index.as_query_engine(**query_kwargs)
            
            # 执行查询获取大量真题片段
            print("🔍 正在检索真题库...")
            response = query_engine_all.query("一级建造师历年真题考点内容")
            nodes = response.source_nodes
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            print("💡 尝试不带过滤器重新检索...")
            # 降级方案：不带过滤器重试
            query_engine_fallback = self.index.as_query_engine(similarity_top_k=200)
            response = query_engine_fallback.query("一级建造师历年真题考点内容")
            nodes = response.source_nodes
        
        if not nodes:
            print("❌ 未找到任何相关资料。")
            print("💡 提示：请检查 data 文件夹中是否有文件。")
            return

        all_keywords = []
        total_batches = (len(nodes) + 19) // 20
        
        print(f"🤖 AI 正在分析 {len(nodes)} 个文本片段并提取考点标签...")
        
        # 分批次处理
        for i in range(0, len(nodes), 20):
            batch_nodes = nodes[i:i+20]
            batch_texts = [node.get_content() for node in batch_nodes]
            
            kws = self.extract_keywords_from_nodes(batch_texts)
            all_keywords.extend(kws)
            
            # 显示进度
            current_batch = (i // 20) + 1
            print(f"   进度: [{current_batch}/{total_batches}] 已提取 {len(all_keywords)} 个候选考点...", end='\r')
        
        print("\n✅ 考点提取完成。正在统计频率...")
        
        if not all_keywords:
            print("⚠️ 未能提取到有效考点，可能是文本内容过少或模型响应异常。")
            return

        # 统计词频
        counter = Counter(all_keywords)
        most_common = counter.most_common(10)
        
        print("\n" + "="*50)
        print("🔥 【模块二】真题高频考点 TOP 10")
        print("="*50)
        for i, (keyword, count) in enumerate(most_common, 1):
            bar = "█" * min(int(count / 2), 20) 
            print(f"{i}. {keyword:<15} 出现频次: {count} {bar}")
            
        # 押题逻辑
        print("\n🔮 正在生成 2026 年押题预测...")
        self.generate_prediction(most_common)

    def generate_prediction(self, common_topics):
        topics_str = ", ".join([f"{k}({v}次)" for k, v in common_topics[:5]])
        
        prompt_text = f"""
        你是一级建造师考试命题组专家。
        基于对过去几年真题的分析，以下是最常考的高频考点：
        [{topics_str}]
        
        请结合考试规律（如：重者恒重、轮流考查、结合新规范），预测 2025 年最可能考查的 3 个【案例分析题】方向。
        要求：
        1. 给出预测的考点名称。
        2. 简述预测理由（为什么明年会考这个？）。
        3. 给出一道模拟的简答题题干。
        
        请以清晰的格式输出。
        """
        
        from llama_index.core import PromptTemplate
        prompt = PromptTemplate(prompt_text)
        
        try:
            response = Settings.llm.complete(prompt.format())
            print("\n" + "="*50)
            print("💡 【2025 年押题预测】")
            print("="*50)
            print(response.text)
            print("="*50)
        except Exception as e:
            print(f"❌ 预测生成失败: {e}")




if __name__ == "__main__":
    print("🏗️  一建智能备考助手 (本地版) 启动中...")
    print("-" * 40)
    
    # 1. 构建或加载知识库
    index = build_or_load_knowledge_base()
    
    # 初始化分析器
    analyzer = TrendAnalyzer(index)

    print("-" * 40)
    print("✅ 准备就绪！请选择功能模式：")
    print("   [1] 🗣️  智能问答 (查教材/查真题)")
    print("   [2] 📈  趋势分析与押题 (模块二)")
    print("   [q]  退出")

    # print("💡 示例问题: '深基坑支护有哪些类型？' 或 '索赔成立的三个条件是什么？'")
    
    # 2. 循环提问
    while True:
        try:
            mode = input("\n👉 请输入模式编号 (1/2/q): ").strip()

            if mode.lower() == 'q':
                print("👋 再见，祝备考顺利！")
                break

            if mode == '1':
                print("--- 进入智能问答模式 ---")
                user_input = input("\n👉 请输入问题 (输入 'back' 返回主菜单): ").strip()
                if user_input.lower() == 'back':
                    break
                if not user_input:
                    continue
                ask_question(index, user_input)
            elif mode == '2':
                print("--- 进入趋势分析与押题模式 ---")
                print("⚠️ 注意：此过程需要 AI 阅读大量真题，可能需要 1-2 分钟，请耐心等待...")
                analyzer.analyze_trends()
                # 分析完后自动返回主菜单，或者询问是否继续
                input("\n按回车键返回主菜单...")
            else:
                print("❌ 无效输入，请输入 1, 2 或 q")
        except KeyboardInterrupt:
            print("\n👋 强制退出，再见！")
            break