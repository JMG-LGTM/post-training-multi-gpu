# 官方 IDQL Q/V 路径与审核边界

本文件记录 2026-11-08 的 CC/CPU 审核阶段及官方路径的数据、运行合同。
2026-11-14 的审核分支整理保留训练提交 `3bc0f55` 的 16 个绑定源码文件和全部
vendor 文件字节，不构成重新训练。该分支依赖
[BC/IQL 更新分支](https://github.com/JMG-LGTM/post-training-multi-gpu/tree/review/bc-iql-updates-20261114)。
后续真实 Scene24 双 A100 训练与 rollout 见
[训练审阅](SCENE24_OFFICIAL_TWINQ_REVIEW_20261113.md)；本轮实际 CPU 复核见
[分支拆分审计](BRANCH_SPLIT_AUDIT_20261114.json)。

官方入口 `train_idql_official.py` 实际执行作者的 JAX Q/V 更新函数和网络。
旧入口 `train.py --method critic` 仍是自有 PyTorch PixelIQL。
偏好入口 `train_preference_idql.py` 已移至
[独立历史偏好分支](https://github.com/JMG-LGTM/post-training-multi-gpu/tree/archive/custom-preference-critic-20261114)，不在本分支提供。
自有 PixelIQL 不能称为作者官方实现，也不能将其既有权重改名为官方模型。

## 来源

- 论文：[IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies](https://arxiv.org/abs/2304.10573)。
- 作者仓库：[philippe-eecs/IDQL](https://github.com/philippe-eecs/IDQL)。
- 固定版本：`7582571ce686c77b4ee0f592dd639ab8151295fa`。
- 当前真正导出的 learner 是 `jaxrl5/agents/ddpm_iql_simple/ddpm_iql_learner.py`，
  上游 README 链接还指向旧 `ddpm_iql`；本分支不把两者混为同一个入口。
- 上游文件原样保存，来源清单记录各文件 SHA256。更新函数不翻写成 Torch loss。
  导入适配只隔离无关的 eager imports，不替换参与计算的作者源码。
- 原仓库 `setup.py` 声明 MIT，但该提交没有 tracked LICENSE/NOTICE 文件。
  保留原始 metadata、README 和引用，不伪造一个“上游已有”的许可文本。

## 实际复用与适配

| 部分 | 新路径的处理 |
| --- | --- |
| V 更新 | 直接调用上游 `DDPMIQLLearner.update_v`：target 双 Q 取 min，expectile 回归。 |
| Q 更新 | 直接调用上游 `DDPMIQLLearner.update_q`：`r + discount * masks * V(next)`，双头均值 MSE，随后 target EMA。 |
| Q/V 网络 | 使用上游 MLP、Ensemble、StateActionValue、StateValue；原始向量路径保留官方结构。 |
| 优化器 | Optax Adam；不暗加旧版 AdamW weight decay 或梯度裁剪。 |
| 多卡 | 单个 JAX 进程管理本机设备，梯度先跨设备求均值，再各执行一次相同 Adam 更新。V 同步更新后，Q 才计算 next V。 |
| 行为策略 | 不创建 actor optimizer。冻结 π0.5/BC78 是外部候选策略替代，不能称为复现官方 DDPM actor。 |
| 候选选择 | 用 target 双 Q 的 min，取 argmax；作者官方实现本身支持该规则。N=4 是实验配置，不是上游默认 N=64。 |
| 偏好损失 | 官方路径不加入 preference；旧偏好实验只在独立历史分支保留。 |

## CC 复审后的视觉修正

训练使用作者 DrQ 原 `batched_random_crop`，padding=4，以 edge padding
实现随机平移。每次真实更新先生成一份增强 batch，官方 V 与 Q 更新共同使用它；
当前/下一状态、两相机分别拆分 RNG。多设备把设备序号折入增强 key，避免各卡
重复同一随机流。推进后的 RNG 保存在完整 checkpoint；同一设备布局的恢复逐值可验。
验证、同状态动作排序及普通评分不增强，也不推进训练 RNG。

增强配方进入 adapter 合同，并与配置、源文件 SHA 一起约束恢复。
原始 vector 路径仍 pad0，不改变作者 reference 的参数、Adam、target 或 RNG。
pixel 随机增强的双设备轨迹不要求等于单设备随机轨迹；原数学与 pmean 等价
检查、以及同布局恢复检查分别记录。

像素示例改为64px。CPU 按示例实际构建的参数数为 Q=2,300,308、
V=2,163,135、target Q=2,300,308，训练参数合计4,463,443，actor=0。
这个检查没有参数更新，也不代表 A100 吞吐或显存已经验证。运行时 provenance
会记录实际配置的参数数量，避免沿用其他图像尺寸的估计。

| 配置 | expectile | Q/V 优化器 | 随机平移 | 图像 |
| --- | --- | --- | --- | --- |
| 旧自有 PixelIQL 有效性验证配置 | 0.8 | AdamW，weight decay=0.01 | pad4 | 128px |
| 本分支 CC 审核前版本0332f2e | 0.7 | 官方 Optax Adam，无额外 weight decay/clip | 无 | 128px |
| 2026-11-08修正后的像素示例 | 0.7 | 官方 Optax Adam，无额外 weight decay/clip | 作者原函数 pad4 | 64px |

旧模型的进度相关性或 rollout 成绩不迁移到新路径。新配置必须重新通过下述诊断。

## 只读诊断与 checkpoint 选择

每个验证事件遍历完整唯一 held-out roster，输出 `validation/update_########.json`
及标量 `validation.csv`。全局 V 进度相关性对唯一 transition 统计一次；
共享前缀通过已验证原 episode/chunk 身份恢复全部成功路径 membership，
全局 remaining 使用成功 membership 剩余块数的中位数，逐路径相关性保留完整路径。
失败路径本身不虚构“距离成功”；未定义的相关性不从路径中位数里静默剔除。

两个动作质量检查分别保留：

- 同 root 的人工段与原失败策略段 median online A 比较，原失败段取人工开始之后。
  这是不同状态的启发式比较；有人工根缺少对应原失败段时不放行。
- 真正同状态比较必须在精确原始时刻 t 核验 simulator state、双相机与本体状态相同，
  比较同有效执行长度的人工和原策略动作，分数为 target 双 Q 的 min。
  不使用相近帧、floor8、相同进度或同 root 代替同状态。人工更优只是明确标注的假设。

ESS、成功末块权重占比使用 `exp(beta*(online_min_Q-V))`、cap20、beta3 的假想权重。
这些权重只用于诊断，不改变冻结行为策略或官方 Q/V 训练目标。

默认门槛在开训前固定，并进入完整合同：

| 指标 | 条件 |
| --- | --- |
| 完整成功路径 V/剩余块全局 Spearman | ≤−0.5 |
| 逐成功路径 Spearman 中位数 | ≤−0.3 |
| ESS / 验证样本数 | ≥0.1 |
| 成功末块权重占比 | ≤0.5 |
| 有效同 root 人工 A 更高比例 | >0.5 |
| 真实同状态对中人工 target-min-Q 更高比例 | >0.5 |

后两项是2026-11-08审核时新增的预注册检查；严格大于机会水平不等于统计显著。
报告保留匹配根数、pair数、ties 和缺失原因，不能把少量样本解释为策略成功率保证。
缺真实 pair 包、覆盖不完整、指标未定义、非有限或任何门槛失败，都不选合格 scorer。

常规验证仍按500步运行；完整保存边界（默认2500步、最终步、安全停止步）无论是否
碰巧处于验证间隔，都会计算同一步诊断。只在这些保留的完整 checkpoint 中选择，
固定按全局 rho、逐路径中位 rho、验证 TD loss、最早步数依次排序。
低 TD loss 不能越过未通过的动作质量门槛。

`selection.json` 的 `best_qualified_scorer` 才是通过门槛的候选；没有通过者时为 null，
不回退最后一步。`best_progress_candidate` 只供定位 V 与 Q 的问题，不能当作已合格。
完整诊断和选择状态随 checkpoint 保存；恢复以 CP 内状态为准，不能事后改阈值续训。
CP、评分导出和诊断全部落地后才发布选择指针。实际 NAS 管理接入时须保留或验证
这些证据引用，再处理源文件清理；当前代码未自动启动 NAS 调度。

### 真实同状态包

`diagnostics.same_state_pairs` 显式绑定 JSON。其顶层字段为 schema_version=1、
format=`official_idql_native_same_state_pairs_v1`、dataset_sha256、
dataset_manifest_sha256、label=`human_intervention_assumed_preferred` 和 pairs。
每对包含 pair_id、root_run_id、chosen_run_id、rejected_run_id、state_index、horizon。
文件只描述已有原始数据，不能提供伪造图像或特征代替核验。

loader 从 native manifest 已哈希文件读取精确 t 的原始证据，要求人工接管与其原失败
策略来自同一留出 root，整个动作片段保持各自来源，horizon 在1至8之间且两者一致。
不用8步对齐，因此 t=3/H6 也可以核验。外部特征仅在同 t 有精确缓存观测时使用，
不自动发明特征提取器。未配置的状态为 UNAVAILABLE；这不妨碍保存训练断点，
但不能据此称任何 scorer 已通过 IDQL 评测准入检查。

### 实际数据与行为策略

正式准备必须绑定完整 frozen roster 和归一化，接管及对应原失败保持同一 split。
preflight 记录真实 native 成员数与来源；观察/动作两字段的官方 BC 示教不能伪装成
含奖励、终止和 next observation 的真实 RL 转移。
历史准备描述将 BC52 记为34成功+16接管+2官方示教，将 native critic 记为34+16+16=66。
后续 Scene24 冻结规则实际得到33成功+17失败+16接管，详见训练审阅；以冻结 manifest 为准。
审核意见中的“52+16”须对照真实 manifest，不能为凑68条添加虚假转移。

2026-11-08 的审核阶段没有替用户更换 BC78。CC 当时建议 π0.5 base 或 BC2500、独立留出开局及随机候选
对照，属于当时的后续实验参数选择，未静默落实到训练。该阶段要求在上 GPU 前绑定
实际数据/候选策略，并完成用户确认的100至200步双 A100 短测；后续执行记录见 Scene24 训练审阅。

视觉输入与动作块是 LIBERO/π0.5 的域适配，必须与官方核心分开报告：

- `vector` 明确使用状态向量；只取 proprio 的配置不能冒充双相机输入。
- `external_features` 要求已对齐的当前/下一状态特征、缓存身份、编码器和预处理
  文件 SHA、特征层定义。推理时必须用相同变换。本分支没有自动提取 π0.5 特征。
- `pixels` 显式使用该上游仓库 DrQ 路径的 PixelMultiplexer/D4PGEncoder，
  显式选择 `separate_rgb_views`，把缓存中两张 RGB 图像分成两个 pixel key，
  各自编码，再接官方 Q/V 核心。critic 的编码特征供双 Q 使用，V 有独立编码器。
  这不是作者论文原始的向量观测网络，也不沿用旧自有 CNN。
  示例配置是供审核的视觉适配选项，不能据此宣称已经完成完整论文复现。
- H8 动作块和有效 mask 使用固定、同一套编码用于训练和候选评分。
  已执行动作零填后展平，再附有效时长 mask；尾部 H6 与 padding NaN 必须核验。
- 奖励、折扣单位、真实终止和录制截断沿数据合同，不在适配器里偷偷修改。
  `mask=1` 的录制截断仍需真实下一帧，不能靠重复当前帧伪造。

## 运行方式

使用独立 Python 环境。已验证的 CPU 数值环境为 Python 3.11、JAX/jaxlib
0.5.3、Flax 0.10.2、Optax 0.2.4、Gym 0.26.2。后者是兼容性验证环境，
不是声称原作者 2023 年依赖原封不动适用于当前 Python。
`idql-official` extra 固定的是上述 CPU 基础栈，不包含 CUDA 插件。
目标 A100 的 JAX CUDA 插件、驱动兼容和实际 GPU 短测需要在部署阶段单独核验；
设置 `CUDA_VISIBLE_DEVICES` 本身不代表 CUDA 环境已经就绪。

```bash
python -m pip install -e '.[idql-official,test,plot]'

# 新入口不是 torchrun；先只做数据/配置预检。
CUDA_VISIBLE_DEVICES=2,3 python train_idql_official.py \
  --config configs/official_idql/idql_official_pixels.yaml --data /path/to/native-package --dry-run
```

`--dry-run` 会创建或验证内容寻址的 CPU replay cache；不会创建 Q/V 模型或初始化
JAX 设备。该示例的 frozen roster、归一化和行为策略身份仍需按实际实验填写，
不能直接把未绑定的示例当作已批准的 Scene24 新训练。

多卡路径拒绝 torchrun 的多进程环境，防止每个进程重复创建全部设备训练。
配置需要明确设备数、每设备 batch、总 batch；当前官方路径仅支持累积 1。
双卡 `32 × 2 = 64`，每次 V/Q 更新计数一次，actor 更新数为零。

每个新实验使用独立输出目录。完整断点必须保存 Q/V/target Q、Adam 状态、真实更新数、
采样 RNG、数据划分、归一化和适配器合同；导出的评分权重不能当作完整续训断点。
原 PyTorch `.pt` 不可续入这个新路径。示例里的 checkpoint/验证间隔不代表已经启动训练。

保留分组训练/验证集和全量留出 loss。固定旧成员时应提供完整 frozen roster，
核对 root 成员映射，而不是只检查划分比例。CSV、平滑 PNG/SVG 由实际训练日志生成。
新评分导出是库 API，平台策略注册、π0.5 候选调用和真实 rollout 仍需单独接入和验证。

```bash
# 仅 CPU 数值验证，强制两个虚拟 CPU 设备，不接触 A100。
python tools/verify_official_idql.py --output reports/official_idql_cpu_verification.json

# 保留原 CSV；全记录的 raw 与 EWMA100/500 曲线及 CSV。
python scripts/plot_official_idql_metrics.py RUN/metrics.csv \
  --validation RUN/validation.csv --output-dir RUN/loss_plots
```

完整续训断点在 `checkpoints/update_########`；target-Q 评分导出在独立的
`scorers/update_########`。后者由 `official_idql.scoring.load_scorer` 使用同架构的
单设备 template 及完整合同加载，只提供评分接口，不恢复 Adam，也不能用于续训。
归一化与 adapter 必须同时匹配完整训练合同。复用同一份动作编码处理 H8/H6，
`score_chunks` 接收已适配的观测和 raw/normalized 动作的显式标记。

## CC 应重点复核

1. 固定来源和逐文件 SHA，实际函数对象是否来自作者原文件，有没有本地回退。
2. 官方单设备 reference 对照，不能只用同一套适配代码与自身比较。
3. 多次更新后的 Q/V/target 参数、Adam moments/count 是否与全局 batch reference 一致；
   梯度归约必须在 Adam 前，不能对各卡 Adam 更新后的参数再平均。
4. fresh → 完整断点 → restore → 继续，与不中断运行是否一致。
5. H8/H6、动作 mask、归一化、reward/termination、特征合同在训练/评分之间是否一致。
6. 所有域适配是否显式选择、被合同绑定；是否错误复用了旧 Q 权重、偏好项或旧验证收据。

CPU/虚拟双 CPU 设备检查只能证明对应数学与执行逻辑；不等于双 A100 CUDA 已验证，
更不保证超过 BC78。2026-11-08 阶段先交 CC 审核；本次分支整理也不会提交新服务器训练。

## 历史验证记录（2026-11-08）

以下为包含旧偏好实验的历史工作树验证结果，不能当作当前净分支的测试数量或收据。
历史全仓 **390 项 CPU 测试通过，0 失败、0 跳过**。当时官方路径73项全部收集并执行，
包括原8项独立官方参考：初始化与连续 5 次更新的 Q/V/target/Adam/RNG 逐值一致；双虚拟 CPU 设备
对全局 batch64 连续更新最大绝对差 `3.278255e-6`（atol `2e-6`、rtol `2e-5`），
两副本逐值一致。单设备与双设备完整恢复均逐值一致。

该历史阶段另已实测双 CPU runner 的训练、验证、完整断点、恢复和评分导出全链，
以及 installed-wheel 官方源码加载、全部80文件字节一致、无缓存字节码夹带，
和 loss PNG/SVG/平滑 CSV 的实际输出与异常输入拒绝。
增强恢复、真实原始同状态包核验、缺失证据拒绝、阈值冻结、同一步边界验证和
最佳 scorer 选择均有覆盖。所有模型训练和质量门槛正反例均为合成 CPU 测试，
该历史 CPU 收据不证明真实 Scene24 checkpoint 已通过门槛；后续12500步 scorer 的
实际资格与30000步 rollout 分别见 Scene24 训练审阅。

历史机器可读数据、当时实际源码 SHA、版本和测试界限在
[OFFICIAL_IDQL_VALIDATION.json](OFFICIAL_IDQL_VALIDATION.json)，本次逐字节保留。
其中 runner 哈希早于 `3bc0f55` 的初始化修正，且未包含新增的 `replication.py`；
不能把该收据解释成完整覆盖训练提交或当前拆分分支。历史本地原始收据与全仓
JUnit 归档路径为 `reports/official_idql_*`，未随本分支重新生成。
本轮真实 CPU 测试及 16 个训练源码的字节比对只记入
[分支拆分审计](BRANCH_SPLIT_AUDIT_20261114.json)。这些 CPU 检查不是 GPU 短测
或真实策略成功率的证明。

2026-11-14 本次拆分实际运行8个官方测试文件：84项中83通过、1失败、0跳过。
失败项为双虚拟 CPU pmap 与全局 batch64 的数值对照，131072个数组元素中1个
超出原始 `atol=2e-6, rtol=2e-5`，最大绝对差为 `3.2298267e-6`。
为对照该失败，原始训练 `3bc0f55` 工作树与本审核工作树各只单独重跑该项一次，
在同一 CPU 环境下均出现相同失败；两边源码和测试字节相同。
该对照没有把数值断言改写为通过，也没有放宽容差。首次完整结果、两次单项结果及
原始日志/JUnit 的哈希均记入分支拆分审计；本次没有进行 GPU、正式训练或 rollout。
