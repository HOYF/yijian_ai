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
from collections import Counter, defaultdict
from tqdm import tqdm
import time

class TrendAnalyzer:
    def __init__(self, index):
        self.index = index

    def extract_keywords_from_single_text(self, text):
        """
        【核心升级】对单段文本进行标准化考点提取
        强制要求使用官方术语，避免口语化
        """
        if not text or len(text.strip()) < 20:
            return []
        
        prompt_text = f"""
        你是一级建造师考试命题专家。
        请阅读以下一段真题内容，提取其中包含的【核心考点】。
        
        【要求】：
        1. 必须使用《一级建造师官方教材》的标准专业术语（如：'索赔管理' 而不是 '要钱'，'深基坑支护' 而不是 '挖坑'）。
        2. 如果没有明显考点，返回空列表。
        3. 最多提取 2 个最核心的考点。
        4. 只返回考点名称，用逗号分隔，不要编号，不要其他解释。
        
        【真题片段】：
        {text[:800]}  (限制800字以防超长，但保证核心信息)
        """
        # 上面要求最多提取2个考点，而且限制了文字长度，所以还是不够精准；
        
        from llama_index.core import PromptTemplate
        prompt = PromptTemplate(prompt_text)
        
        try:
            response = Settings.llm.complete(prompt.format())
            keywords_str = response.text.strip()
            # 清洗数据：处理中文逗号和换行
            keywords = [k.strip() for k in keywords_str.replace('，', ',').replace('\n', ',').split(',') if len(k.strip()) > 2]
            return keywords
        except Exception as e:
            return []

    def analyze_trends(self):
        print("\n📊 正在启动【专业版】真题趋势分析引擎...")
        print("⚠️ 注意：为了保证全面性，系统将逐段分析所有真题，这可能需要 2-3 分钟，请耐心等待...\n")
        
        # ================= 步骤 1: 获取所有真题节点 =================
        print("🔄 第一步：加载全量真题数据...")
        
        filters = None
        try:
            from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
            single_filter = MetadataFilter(key="source_type", value="真题", operator="==")
            filters = MetadataFilters(filters=[single_filter])
        except Exception:
            pass

        query_kwargs = {"similarity_top_k": 1000} # 增大数量，确保覆盖更多
        if filters:
            query_kwargs["filters"] = filters
            
        try:
            query_engine_all = self.index.as_query_engine(**query_kwargs)
            response = query_engine_all.query("一建真题全部内容")
            nodes = response.source_nodes
        except Exception:
            print("⚠️ 过滤检索失败，尝试全量检索...")
            query_engine_fallback = self.index.as_query_engine(similarity_top_k=1000)
            response = query_engine_fallback.query("一建真题全部内容")
            nodes = response.source_nodes

        if not nodes:
            print("❌ 未找到任何真题资料。")
            return

        # ================= 步骤 2: 展示数据源 =================
        source_files = {}
        nodes_by_year = defaultdict(list) # 按年份分组节点
        
        for node in nodes:
            fname = node.metadata.get("file_name", "未知文件")
            year = node.metadata.get("year", "未知年份")
            key = f"{fname}"
            source_files[key] = source_files.get(key, 0) + 1
            nodes_by_year[year].append(node)
        
        print(f"\n✅ 共加载 {len(nodes)} 个真题片段，来源于以下 {len(source_files)} 个文件：")
        for fname, count in sorted(source_files.items()):
            print(f"   📄 {fname} ({count}片段)")
        print("-" * 60)

        # ================= 步骤 3: 逐段深度分析 (Map 阶段) =================
        print("\n🔄 第二步：AI 正在逐段提取标准化考点 (100% 全覆盖)...")
        
        all_keywords = []
        keywords_by_year = defaultdict(list) # 记录每年的考点
        
        # 使用 tqdm 显示进度条
        for i, node in enumerate(tqdm(nodes, desc="分析进度")):
            text = node.get_content()
            year = node.metadata.get("year", "未知年份")
            
            # 提取关键词
            kws = self.extract_keywords_from_single_text(text)
            
            if kws:
                all_keywords.extend(kws)
                keywords_by_year[year].extend(kws)
            
            # 可选：每处理50个稍微停顿一下，防止API限流 (如果是本地模型可去掉)
            # if i % 50 == 0: time.sleep(0.1) 

        print("\n✅ 全量分析完成！正在生成统计报告...")

        if not all_keywords:
            print("⚠️ 未能提取到有效考点。")
            return

        # ================= 步骤 4: 多维度统计 =================
        
        # 4.1 全局高频 TOP 10
        counter = Counter(all_keywords)
        most_common = counter.most_common(10)
        
        print("\n" + "="*60)
        print("🔥【维度一】历年真题高频考点 TOP 10 (绝对频率)")
        print("="*60)
        for i, (keyword, count) in enumerate(most_common, 1):
            bar = "█" * min(int(count / 2), 25) 
            print(f"{i}. {keyword:<15} | 频次: {count:3d} | {bar}")

        # 4.2 年度趋势分析 (新增专业功能)
        print("\n" + "="*60)
        print("📈【维度二】核心考点年度趋势变化 (判断冷热)")
        print("="*60)
        
        # 找出全局前5的考点，看它们在每年的分布
        top_5_keywords = [k for k, _ in most_common[:5]]
        
        if not top_5_keywords:
            return

        print(f"{'考点名称':<15}", end="")
        years = sorted([y for y in keywords_by_year.keys() if y != "未知年份"])
        if not years:
            years = ["未知年份"]
        
        for y in years:
            print(f"{y:>8}", end="")
        print(f"{'趋势判断':>10}")
        print("-" * 60)

        for kw in top_5_keywords:
            print(f"{kw:<15}", end="")
            counts_per_year = []
            for y in years:
                c = keywords_by_year[y].count(kw)
                counts_per_year.append(c)
                print(f"{c:>8}", end="")
            
            # 简单趋势逻辑
            if len(counts_per_year) >= 2:
                if counts_per_year[-1] > counts_per_year[-2] * 1.5:
                    trend = "🔥 升温"
                elif counts_per_year[-1] < counts_per_year[-2] * 0.5:
                    trend = "❄️ 降温"
                else:
                    trend = "➖ 平稳"
            else:
                trend = "?"
            print(f"{trend:>10}")

        # ================= 步骤 5: 原文依据展示 =================
        print("\n📖【维度三】高频考点原文依据 (随机抽样)")
        print("-" * 60)
        displayed = 0
        for keyword, _ in most_common[:3]:
            # 找一个包含该关键词的节点
            for node in nodes:
                if keyword in node.get_content():
                    fname = node.metadata.get("file_name", "?")
                    year = node.metadata.get("year", "?")
                    snippet = node.get_content().replace("\n", " ")[:80] + "..."
                    print(f"🔹 [{keyword}] 出自《{fname}》({year}): \"{snippet}\"")
                    displayed += 1
                    break
        if displayed == 0:
            print("   (未匹配到具体原文)")

        # ================= 步骤 6: 智能押题 =================
        print("\n🔮【维度四】2026 年押题预测引擎启动...")
        self.generate_smart_prediction(most_common, keywords_by_year, years)

    def generate_smart_prediction(self, common_topics, keywords_by_year, years):
        # 构造更丰富的数据上下文
        trend_data = []
        top_5 = [k for k, _ in common_topics[:5]]
        
        for kw in top_5:
            history = []
            for y in years:
                count = keywords_by_year[y].count(kw)
                history.append(f"{y}年({count}次)")
            trend_data.append(f"- {kw}: {', '.join(history)}")
        
        trend_str = "\n".join(trend_data)
        
        prompt_text = f"""
        你是一级建造师资深命题专家。
        
        【数据分析结果】
        基于对全量真题的逐段分析，核心考点的历史分布如下：
        {trend_str}
        
        【命题规律逻辑】
        1. **重者恒重**：连续多年高频的考点，明年大概率继续考（尤其是案例题）。
        2. **冷热交替**：如果某考点去年考得特别多，今年可能略减；如果某重要考点连续2年未出现，今年极大概率“回补”。
        3. **新纲新点**：结合2025-2026教材变动（虽然此处无具体教材数据，请依据常识推断新技术、新规范）。
        
        【任务】
        请预测 2026 年最可能考的 3 个【案例分析大题】方向。
        
        【输出格式】
        1. **预测考点**：[名称]
           - **数据支撑**：引用上面的历史数据说明理由（例如：已连续3年高分，或沉寂2年需回补）。
           - **模拟题型**：给出一句具体的案例背景描述。
        
        请语气专业、笃定。
        """
        
        from llama_index.core import PromptTemplate
        prompt = PromptTemplate(prompt_text)
        
        try:
            response = Settings.llm.complete(prompt.format())
            print("\n" + "="*60)
            print("💡【2026 年一建·终极押题报告】")
            print("="*60)
            print(response.text)
            print("="*60)
            print(f"\n📝 注：本报告基于 {sum(len(v) for v in keywords_by_year.values())} 个标准化考点数据统计生成。")
            
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