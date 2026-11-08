# 官方 IDQL Q/V 路径与审核边界

本分支修正实现来源：新入口 `train_idql_official.py` 实际执行作者的
JAX Q/V 更新函数和网络。旧入口 `train.py --method critic` 与
`train_preference_idql.py` 仍是自有 PyTorch PixelIQL，不能称为作者官方实现，
也不能将其既有权重改名为官方模型。

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
| 偏好损失 | 新路径不加入 preference；旧偏好入口不自动切换为官方实现。 |

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
更不保证超过 BC78。当前分支先交 CC 审核，不会自动提交新服务器训练。

## 本次验证记录

2026-11-08 全仓 **347 项 CPU 测试通过，0 失败、0 跳过**。其中独立官方参考验证
8 项：初始化与连续 5 次更新的 Q/V/target/Adam/RNG 逐值一致；双虚拟 CPU 设备
对全局 batch64 连续更新最大绝对差 `3.278255e-6`（atol `2e-6`、rtol `2e-5`），
两副本逐值一致。单设备与双设备完整恢复均逐值一致。

另已实测双 CPU runner 的训练、验证、完整断点、恢复和评分导出全链，
以及 installed-wheel 官方源码加载、全部80文件字节一致、无缓存字节码夹带，
和 loss PNG/SVG/平滑 CSV 的实际输出与异常输入拒绝。

机器可读数据、实际源码 SHA、版本和测试界限在
[OFFICIAL_IDQL_VALIDATION.json](OFFICIAL_IDQL_VALIDATION.json)。本地原始收据与全仓
JUnit 位于 `reports/official_idql_*`。收据以实际源码 SHA 绑定本次被测工作树；
它们不是 GPU 短测或真实策略成功率的证明。
