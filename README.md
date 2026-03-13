# yijian_ai
使用大模型搭建的一套一级建造师考试系统，包括智能知识问答模块，真题趋势分析与押题引擎，私有知识库管理 

第一部分：核心框架与库 
| 框架/库名称 | 角色比喻 | 主要作用 |
| :--- | :--- | :--- |
| LlamaIndex | 图书管理员 | 核心框架。负责把文档切分、变成向量、存进数据库，并根据问题去数据库里找最相关的片段。 |
| Ollama | 大脑 (LLM) | 本地大模型运行器。它加载了 `qwen2.5:3b`，负责理解问题、总结答案、提取考点。 |
| HuggingFace (`transformers`) | 翻译官 (Embedding) | 提供嵌入模型 (`bge-small-zh`)。负责把中文文字转换成计算机能理解的数字向量（Vector）。 |
| Streamlit | 装修设计师 (UI) | 快速构建网页前端。不用写 HTML/CSS，只用 Python 代码就能画出按钮、输入框、图表。 |
| Pandas | 会计 | 处理表格数据。在知识库管理中用来展示文件列表。 |
| Tqdm | 进度条显示员 | 在命令行或后台循环时，显示绿色的进度条，让你知道程序没卡死。 |
| Pathlib / Shutil | 文件管家 | Python 内置库。负责创建文件夹、复制文件、删除文件、读取路径。 |

第一步：安装基础库
在终端或命令行中运行以下命令：
bash

编辑



pip install streamlit llama-index llama-index-llms-ollama llama-index-embeddings-huggingface pandas tqdm
第二步：安装系统级依赖 (重要)
由于代码中使用了 HuggingFaceEmbedding (通常基于 SentenceTransformers) 和 pandas，在某些系统上可能需要额外的构建工具：
Windows 用户: 通常上述 pip 命令即可，但如果报错，可能需要安装 Microsoft C++ Build Tools。
Linux/Mac 用户: 可能需要安装 python3-dev 或 build-essential。
第三步：确保 Ollama 环境就绪
代码强依赖于本地运行的 Ollama 服务，这不是 Python 库，需要单独安装软件：
下载并安装 Ollama: 访问 https://ollama.com 下载对应系统的安装包。
拉取模型: 代码中指定使用了 qwen2.5:3b 模型和 BAAI/bge-small-zh-v1.5 嵌入模型。
启动 Ollama 后，在终端运行：
bash

编辑



ollama pull qwen2.5:3b
注意: BAAI/bge-small-zh-v1.5 会在代码首次运行时自动通过 HuggingFace 下载（代码中已配置国内镜像 hf-mirror.com），无需手动运行 ollama 命令下载嵌入模型。
3. 代码功能简析
main.py: 是一个命令行交互程序，用于构建知识库、问答、趋势分析和文件管理。
app.py: 是一个基于 Streamlit 的网页版应用，提供了图形化界面来实现相同的功能。
4. 如何运行
安装完成后：
启动 Ollama 服务 (保持后台运行):
bash

编辑



ollama serve
运行网页版 (推荐):
bash

编辑



streamlit run app.py
或者运行命令行版:
bash

编辑



python main.py
提示: 首次运行时，程序会自动下载嵌入模型并构建向量索引，可能需要几分钟时间。请确保在项目目录下创建了 data 文件夹并放入了一些 PDF 或 TXT 文档（如教材、真题），否则程序会提示文件夹为空。
