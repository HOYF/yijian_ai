import os
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
import time
import sys
import shutil  # <--- 新增：用于文件复制和删除
import re
from pathlib import Path
from llama_index.llms.ollama import Ollama # 连接本地 Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # 使用 HF 的嵌入模型
# 构建过滤器
from llama_index.core.vector_stores import FilterCondition, MetadataFilter

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage, Document, PromptTemplate
# SimpleDirectoryReader 读取文件夹里的文档
# VectorStoreIndex 创建向量索引
# Settings 全局设置
# StorageContext 存储上下文
# load_index_from_storage 从磁盘加载已建好的索引
# Document 文档对象
# PromptTemplate 提示词模板


# 1. 激活虚拟环境 source venv/bin/activate
# 2. ollama serve 启动本地大模型服务
# 3. streamlit run app.py 启动智能系统Web应用

# 设置 HuggingFace 国内镜像地址
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


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
    # 设置大语言模型：负责思考和生成，context_window=4096 限制了一次性能读多少字，防止爆显存
    Settings.llm = Ollama(model="qwen2.5:3b", request_timeout=120.0, context_window=4096)
except Exception as e:
    print(f"❌ 连接 Ollama 失败: {e}")
    print("💡 提示: 请确保 Ollama 软件已运行，且已执行 'ollama pull qwen2.5:3b'")
    sys.exit(1)

# =================主程序=================
# 提取年份函数
def extract_year_from_filename(file_path: str) -> str:
    """从文件名中提取年份，如 '2024 环球网校...' -> '2024'"""
    match = re.search(r'\d{4}', file_path) # 正则表达式：找连续的4个数字
    return match.group(0) if match else "未知"


PERSIST_DIR = "./storage" # 索引保存目录

# 构建和加载知识库
def build_or_load_knowledge_base():
    """构建知识库，如果已存在则直接加载，节省时间"""
    
    # 检查是否已经存在建好的索引(加速启动)
    if Path(PERSIST_DIR).exists():
        print("📂 发现已存在的索引，正在快速加载...")
        try:
            # 从磁盘加载现成的向量数据库，秒开
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

        # 数据清洗与增强 (添加 metadata)，为每个文档添加 metadata（包括年份、文件名等）
        documents = []
        for doc in raw_documents:
            # 获取原始文件路径
            file_path = doc.metadata.get("file_path", "")
            year = extract_year_from_filename(file_path)

            # 创建新文档对象，附加metadata（注入自定义元数据）
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
    
    # 建立向量索引 (最耗时的步骤)
    # 过程：文本切分 -> 调用 embed_model 转向量 -> 存入内存/磁盘
    # 关键点：VectorStoreIndex.from_documents 内部会自动把长文档切成小块 (Chunks)，然后每一块都调用嵌入模型生成向量
    index = VectorStoreIndex.from_documents(documents)
    
    # 保存到磁盘，下次不用重新算(存到 ./storage 文件夹)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"💾 索引已保存至 '{PERSIST_DIR}' 文件夹。")
    print("✨ 知识库构建完成！")
    
    return index

# 模块一问答功能
def ask_question(index, query):
    """执行查询"""
    # 创建查询引擎
    # similarity_top_k=5 表示参考最相关的 5 段内容
    # query_engine = index.as_query_engine(similarity_top_k=5)
    """执行查询 (带流式输出) - 支持按年份过滤"""
    # 1. 解析用户是否想要特定年份 (正则匹配)
    year_match = re.search(r'\b(20\d{2})\b', query)
    target_year = year_match.group(1) if year_match else None

    print(f"\n🤖 思考中: {query} ...")
    if target_year:
        print(f"🔍 正在筛选 {target_year} 年的相关资料...")

    # 2. 构建过滤器 
    filters = None
    if target_year:
        filters = MetadataFilter.from_dict(
            key="year",
            value=target_year,
            operator="=="
        )
    
    # 3. 创建查询引擎，传入过滤器
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        filters=filters,  # 👈 关键：只检索匹配年份的文档
        streaming=True # 开启流输出（字一个个蹦出来）
    )


    # query_engine = index.as_query_engine(similarity_top_k=5, streaming=True) # 开启 streaming
    
    # print(f"\n🤖 思考中: {query} ...")
    try:
        # 执行查询
        response = query_engine.query(query)
        
        print(f"\n💡 AI 回答:")
        # 流式打印
        for token in response.response_gen:
            print(token, end="", flush=True)
        print() # 换行


        # 显示引用来源（RAG 的核心：可追溯）
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


# 模块二：考题预测
class TrendAnalyzer:
    def __init__(self, index):
        self.index = index
    
    # 构造一个 Prompt，让 LLM 充当“命题专家”，从一段真题文本中提取标准术语（如“网络计划”）
    # 注意：这里是对每一个文本片段单独调用 LLM，所以比较慢，但非常精准
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
        
        
        prompt = PromptTemplate(prompt_text)
        
        try:
            response = Settings.llm.complete(prompt.format())
            keywords_str = response.text.strip()
            # 清洗数据：处理中文逗号和换行
            keywords = [k.strip() for k in keywords_str.replace('，', ',').replace('\n', ',').split(',') if len(k.strip()) > 2]
            return keywords
        except Exception as e:
            return []

    # Step 1: 用过滤器 source_type=="真题" 捞出所有真题片段。
    # Step 2: 统计文件来源，展示数据透明度
    # Step 3 (Map): 循环遍历所有节点，逐个提取关键词。使用 tqdm 显示进度条。
    # Step 4 (Reduce): 使用 Counter 统计词频，找出 TOP 10
    # Step 5: 按年份分组，计算趋势（升温/降温）
    # Step 6: 再次调用 LLM，把统计好的数据喂给它，让它写押题报告。
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


# 模块三：私有知识库管理
class KnowledgeBaseManager:
    def __init__(self, data_dir="./data", persist_dir="./storage"):
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        
        # 确保 data 目录存在
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)
            print(f"📁 已创建数据目录: {self.data_dir}")

    def list_files(self):
        """列出当前知识库中的所有文件"""
        print("\n" + "="*60)
        print("📂 当前私有知识库文件列表")
        print("="*60)
        
        if not self.data_dir.exists():
            print("❌ 数据目录不存在。")
            return

        files = list(self.data_dir.iterdir())
        # 过滤掉隐藏文件
        files = [f for f in files if not f.name.startswith('.')]
        
        if not files:
            print("   (空)")
            return

        print(f"{'文件名':<40} {'大小':<10} {'年份':<8} {'类型'}")
        print("-" * 60)
        
        for f in sorted(files, key=lambda x: x.name):
            size_mb = f.stat().st_size / 1024 / 1024
            year = extract_year_from_filename(f.name) # 复用主程序的函数
            
            # 简单判断类型
            f_type = "真题" if "真题" in f.name else ("教材" if "教材" in f.name or "讲义" in f.name else "其他")
            
            # 截断长文件名
            display_name = f.name if len(f.name) <= 38 else f.name[:35] + "..."
            
            print(f"{display_name:<40} {size_mb:.2f} MB   {year:<8} {f_type}")
            
        print("-" * 60)
        print(f"💡 总计: {len(files)} 个文件")

    # 复制文件
    def add_file(self, source_path_str):
        """添加新文件到知识库"""
        source_path = Path(source_path_str)
        
        if not source_path.exists():
            print(f"❌ 错误: 找不到文件 '{source_path}'")
            return False
        
        if not source_path.is_file():
            print(f"❌ 错误: '{source_path}' 不是一个文件")
            return False
            
        # 目标路径
        dest_path = self.data_dir / source_path.name
        
        # 检查是否已存在
        if dest_path.exists():
            overwrite = input(f"⚠️ 文件 '{source_path.name}' 已存在，是否覆盖？(y/n): ").strip().lower()
            if overwrite != 'y':
                print("❌ 操作取消。")
                return False
        
        try:
            # 复制文件
            shutil.copy2(source_path, dest_path)
            print(f"✅ 成功添加: {source_path.name}")
            print("⚠️ 注意: 新文件添加后，必须执行 [重建索引] 才能被 AI 检索到！")
            return True
        except Exception as e:
            print(f"❌ 复制失败: {e}")
            return False
    
    # 删除文件
    def delete_file(self, filename):
        """从知识库删除文件"""
        target_path = self.data_dir / filename
        
        if not target_path.exists():
            print(f"❌ 错误: 知识库中找不到文件 '{filename}'")
            return False
        
        confirm = input(f"⚠️ 确认要删除 '{filename}' 吗？此操作不可恢复 (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ 操作取消。")
            return False
            
        try:
            target_path.unlink()
            print(f"✅ 成功删除: {filename}")
            print("⚠️ 注意: 文件删除后，必须执行 [重建索引] 才能生效！")
            return True
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False

    # 删除 ./storage 文件夹
    def rebuild_index(self):
        """删除旧索引并重新构建"""
        print("\n🔄 开始重建索引...")
        
        # 1. 删除旧索引目录
        if self.persist_dir.exists():
            print(f"🗑️ 正在删除旧索引: {self.persist_dir} ...")
            try:
                shutil.rmtree(self.persist_dir)
                print("   ✅ 旧索引已清除。")
            except Exception as e:
                print(f"   ❌ 删除旧索引失败: {e}")
                print("   💡 请手动删除 ./storage 文件夹后重试。")
                return False
        else:
            print("   ℹ️ 未发现旧索引，将直接构建新索引。")

        # 2. 检查数据目录
        files = list(self.data_dir.iterdir())
        files = [f for f in files if not f.name.startswith('.')]
        
        if not files:
            print("❌ 错误: 数据目录为空，无法构建索引。请先添加文件。")
            return False

        # 3. 重新执行构建逻辑 (复用主程序的逻辑，但这里简化处理)
        # 为了不重复代码，我们直接调用主程序的 build_or_load_knowledge_base 逻辑
        # 但由于该函数在全局作用域，我们需要稍微变通一下
        # 最好的方式是：提示用户重启程序或重新运行构建流程
        # 这里我们模拟一个简单的重建过程，或者提示用户
        
        print("\n" + "="*60)
        print("🚀 索引重建指引")
        print("="*60)
        print("由于向量索引构建涉及复杂的模型加载，为了保证稳定性：")
        print("1. 程序将自动退出。")
        print("2. 请重新运行 python main.py。")
        print("3. 系统会自动检测到索引缺失，并基于最新文件重新构建。")
        print("="*60)
        
        # 如果需要强制在当前进程重建，可以引入 build_or_load_knowledge_base
        # 但为了避免上下文冲突，推荐重启方式。
        # 如果用户坚持要在当前进程重建，可以取消下面注释（需确保环境变量一致）
        
        # --- 高级选项：尝试在当前进程重建 (可选) ---
        # print("🔄 正在重新加载文档并构建索引 (这可能需要几分钟)...")
        # try:
        #     # 这里需要重新导入 SimpleDirectoryReader 等，略繁琐
        #     # 建议直接让用户重启，体验更稳定
        #     pass 
        # except Exception as e:
        #     print(f"重建出错: {e}")
        
        return True # 返回 True 表示准备就绪，等待重启

# 辅助函数：提取年份 (如果主程序里有了，这里可以不用重复定义，或者直接从全局引用)
# 为了防止作用域问题，我们在类外部定义一个通用的，或者直接在类里用正则
def extract_year_from_filename(file_path: str) -> str:
    match = re.search(r'\d{4}', file_path)
    return match.group(0) if match else "未知"




if __name__ == "__main__":
    print("🏗️  一建智能备考助手 (本地版) 启动中...")
    print("-" * 40)
    
    # 1. 构建或加载知识库
    index = build_or_load_knowledge_base()
    
    # 初始化分析器
    analyzer = TrendAnalyzer(index)

    # 初始化知识库管理器 (新增)
    kb_manager = KnowledgeBaseManager()

    print("-" * 40)
    print("✅ 准备就绪！请选择功能模式：")
    print("   [1] 🗣️  智能问答 (查教材/查真题)")
    print("   [2] 📈  趋势分析与押题 (模块二)")
    print("   [q]  退出")

    # print("💡 示例问题: '深基坑支护有哪些类型？' 或 '索赔成立的三个条件是什么？'")
    
    # 2. 循环提问
    while True:
        try:
            mode = input("\n👉 请输入模式编号 (1/2/3/q): ").strip()

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
            elif mode == '3':
                print("--- 进入知识库管理模式 ---")
                while True:
                    print("\n请选择操作:")
                    print("   [a] 添加文件 (输入文件路径)")
                    print("   [d] 删除文件")
                    print("   [l] 列出当前文件")
                    print("   [r] 重建索引 (重要!)")
                    print("   [b] 返回主菜单")
                    
                    sub_mode = input("👉 选择 (a/d/l/r/b): ").strip().lower()
                    
                    if sub_mode == 'b':
                        break
                    elif sub_mode == 'l':
                        kb_manager.list_files()
                    elif sub_mode == 'a':
                        path = input("👉 请输入新文件的绝对或相对路径: ").strip()
                        # 处理拖拽文件时可能产生的引号
                        path = path.strip('"').strip("'")
                        kb_manager.add_file(path)
                    elif sub_mode == 'd':
                        kb_manager.list_files() # 先展示再删除
                        fname = input("👉 请输入要删除的完整文件名: ").strip()
                        kb_manager.delete_file(fname)
                    elif sub_mode == 'r':
                        kb_manager.rebuild_index()
                        # 重建索引后，通常需要重启程序才能重新加载 index 对象
                        # 这里我们直接退出，让用户重新运行
                        print("\n💡 索引重建准备就绪。为了使更改生效，程序将退出。")
                        print("请重新运行: python main.py")
                        sys.exit(0)
                    else:
                        print("❌ 无效输入。")
            else:
                print("❌ 无效输入，请输入 1, 2, 3 或 q")
        except KeyboardInterrupt:
            print("\n👋 强制退出，再见！")
            break