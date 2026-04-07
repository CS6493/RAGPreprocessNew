零环境依赖・便携版 RAG 预处理代码

项目概述
本代码是 RAG（检索增强生成）前置预处理专用便携版，无需配置虚拟环境、无需手动安装依赖，全自动完成从原始数据集加载、脏数据清洗、Token 级文本分块，到 Contriever 768 维稠密向量生成、BM25+Contriever 双索引构建的全流程。采用三阶段持久化设计，运行中断后可直接从断点续跑，彻底解决长时间预处理任务 “白跑” 问题；支持 NVIDIA GPU 自动加速（无 GPU 则无缝切换 CPU），适配 Windows/Mac/Linux 全平台，适合个人、团队协作或跨设备部署。
核心特性（便携版专属）
✅ 零环境依赖：无需 conda、无需虚拟环境，脚本自动检测并安装所有所需依赖库，新手开箱即用✅ 全自动 GPU 适配：自动识别 NVIDIA 显卡，启用 GPU 加速（速度提升 10~50 倍），无 GPU 则自动切换 CPU，无需手动配置✅ 三阶段落地硬盘：每阶段运行结果自动保存到本地，中断后重新运行即可续跑，不丢失任何进度✅ 脏数据终极清洗：强制转字符串、过滤空值 / 乱码 / 过短文本，彻底杜绝 TypeError、NoneType 等数据格式报错✅ 标准化输出：最终生成的索引文件可直接对接检索模块，完全适配团队协作交付需求✅ 跨平台兼容：支持 Windows、Mac、Linux，无需修改任何代码，直接运行✅ Mock 模式支持：内置检索测试，支持检索组并行开发，无需等待预处理全流程完成
快速开始（全程 1 步，无需配置）
1. 准备工作
电脑安装 Python 3.9 ~ 3.11（官网免费下载，一路下一步默认安装即可）
将代码文件 preprocess_3stage_portable.py 保存到任意文件夹（无需复杂目录结构）
2. 一键运行
打开电脑终端（Windows：CMD/PowerShell；Mac/Linux：Terminal）
进入代码所在文件夹（示例命令：cd 你的代码文件夹路径）
执行以下命令，脚本将自动完成所有操作：
bash
运行
python preprocess_3stage_portable.py
3. 运行反馈（正常状态）
启动后首先显示 🔧 自动检查并安装依赖库...，无需手动干预
随后打印运行设备：✅ 运行设备：CUDA（GPU 可用）或 ✅ 运行设备：CPU
按三阶段顺序自动执行，每阶段完成后打印进度提示
全流程结束后，自动运行检索测试，验证索引可用性

三阶段执行流程（自动顺序执行，断点续跑）
阶段	核心操作	输出文件（保存至 rag_preprocess_output/）	耗时说明
阶段 1	加载 3 个数据集（HotpotQA/PubMedQA/FinanceBench）→ 脏数据清洗 → Token 级分块（512 Token + 10% 重叠）	stage1_chunks.pkl（分块文本）、stage1_metadata.json（元数据）	较快（1~5 分钟）
阶段 2	加载分块文本 → 生成 Contriever 768 维稠密向量 → 归一化处理	stage2_embeddings.npy（向量文件）	最长（GPU：10~30 分钟；CPU：10~15 小时）
阶段 3	加载向量 / 分块 / 元数据 → 构建 BM25 稀疏索引 + Contriever 稠密索引 → 生成最终交付文件	保存至 indexes_optimized_final/（3 个索引文件）	较快（1~3 分钟）

最终交付文件（核心输出）
运行完成后，自动生成 indexes_optimized_final/ 文件夹，包含 3 个可直接对接检索模块的标准文件，无需额外处理：

indexes_optimized_final/
├─ bm25_size_512_overlap_10.pkl    # BM25稀疏检索索引（适配检索组接口）
├─ contriever_optimized.index       # Contriever稠密检索索引（768维，内积等价余弦相似度）
└─ contriever_metadata.json         # 分块元数据（标注每个分块的数据集来源）

运行说明与注意事项
1. 依赖安装说明
脚本自动安装所有所需依赖（torch、faiss-cpu、transformers、datasets 等），无需手动执行pip install
若安装失败（网络问题），可重新运行脚本，将自动重试安装
2. 设备适配说明
GPU 加速：仅支持 NVIDIA 显卡（自动识别，无需安装 CUDA），运行时自动调大批次（BATCH_SIZE=32）
CPU 运行：自动适配，批次调整为 BATCH_SIZE=8，确保不占用过多内存
3. 中断续跑说明
若运行中途中断（关机、断网等），再次执行 python preprocess_3stage_portable.py 即可
脚本会自动读取已保存的阶段文件，从上次完成的阶段继续执行，无需从头跑

4. 常见问题解决
问题 1：Python 版本不兼容 → 安装 Python 3.9~3.11，重新运行
问题 2：GPU 显存不足 → 打开代码，将 BATCH_SIZE = 32 if torch.cuda.is_available() else 8 中的 32 改为 16/8
问题 3：文件找不到 → 确保终端进入了代码所在文件夹，或直接拖入代码文件到终端执行
问题 4：缓存警告 → 无需处理，脚本已配置自动规避缓存路径问题

5. 团队协作注意事项
仅需上传核心文件 preprocess_3stage_portable.py 和本 README.md 到团队仓库
无需上传 rag_preprocess_output/（中间结果）、indexes_optimized_final/（大文件）
提交仓库时，建议使用规范提交信息：feat: 新增零环境依赖版RAG预处理代码

文件结构（简洁清晰，无冗余）

.
├── preprocess_3stage_portable.py    # 核心预处理脚本（唯一需要运行的文件）
├── indexes_optimized_final/         # 自动生成，存储最终索引文件（交付检索组）
└── README.md                        # 操作说明文档（本文件）

检索测试说明
代码全流程执行完成后，会自动运行 Mock 模式检索测试：
测试查询：What is RAG technology?
输出内容：检索排名、分块所属数据集、分块 ID
作用：快速验证索引文件可用性，支持检索组并行开发，无需额外编写测试代码

总结
本代码的核心优势是 “零配置、高稳定、可续跑”，无需任何环境配置经验，任何人拿到文件后，执行一行命令即可完成 RAG 预处理全流程，同时适配团队协作上传、跨设备运行，彻底解决预处理过程中 “环境报错、中断白跑、脏数据报错” 三大痛点。