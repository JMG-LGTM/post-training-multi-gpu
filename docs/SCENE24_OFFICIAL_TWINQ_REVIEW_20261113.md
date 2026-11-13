# Scene24 L1：官方核心双 Q IDQL 训练审阅（2026-11-13）

这份说明对应已完成的 Scene24 L1 双 A100、30000 次真实 Q/V 更新实验。
实际训练源码提交为 `3bc0f55d4fc753e1e6781930c7495266040d7d28`。
2026-11-13 从该提交逐文件重算 SHA256，与训练 provenance 绑定的全部 16 个源码文件一致。
本说明仅增加实验说明；算法源码保持训练时的版本。

## 从哪里看

- 训练入口：[train_idql_official.py](../train_idql_official.py)。
- Q/V 核心：[core.py](../src/post_training/official_idql/core.py)。
- 数据/训练循环：[runner.py](../src/post_training/official_idql/runner.py)。
- 双卡初始化修正：[replication.py](../src/post_training/official_idql/replication.py)。
- 动作评分与导出：[scoring.py](../src/post_training/official_idql/scoring.py)。
- 来源与适配细节：[OFFICIAL_IDQL.md](OFFICIAL_IDQL.md)。
- CC 意见及回应：[审核意见](../REVIEW_CC_OFFICIAL_IDQL.md)、[回应](OFFICIAL_IDQL_CC_RESPONSE.md)。

作者源码固定为 [philippe-eecs/IDQL @ 7582571](https://github.com/philippe-eecs/IDQL/tree/7582571ce686c77b4ee0f592dd639ab8151295fa)，实际调用 `ddpm_iql_simple.DDPMIQLLearner` 的原始 `update_v`、`update_q`。
V 用 target 双 Q 的 min 做 expectile 回归；Q 的目标为 `r + gamma * mask * V(next)`，随后更新 target Q。
多设备在 Adam 前对梯度 pmean；优化器为 Optax Adam。

π0.5/BC78 候选策略、双相机 DrQ 编码、动作块和动作掩码是本项目的域适配。
因此这版是“作者 Q/V 核心 + 显式 LIBERO/π0.5 适配”，不能称为完整复现论文的 DDPM actor 实验。
本次实际路径没有偏好损失或额外辅助目标；单 Q 的后续实验属于另一份工作树。
仓库中旧 PyTorch PixelIQL 和 preference 入口仍保留，审阅本实验应以以上官方入口为准。

## 实际训练参数

| 项目 | 本次实际值 |
| --- | --- |
| 设备与批量 | 2 张 A100 40GB，每卡 32，总 batch 64，梯度累积 1 |
| 更新预算 | Q、V 各 30000 次真实优化更新；actor 更新 0 |
| 冻结行为策略 | BC15000（BC78），critic 训练不加载 actor、不创建 actor optimizer |
| 训练 seed / 采样 | 7 / 有放回采样，共 1920000 次 transition 抽样 |
| Q/V | 两个 Q 头；hidden dims `[256,256]`；Q/V 学习率均 `3e-4` |
| 目标更新 / expectile | `tau=0.005` / `expectile=0.8` |
| 输入 | 双 RGB 相机，128px；本体状态 8 维 |
| 视觉编码 | 上游 DrQ PixelMultiplexer/D4PG；每相机单独编码，V 有独立编码器 |
| 数据增强 | 作者 random crop，edge pad 4；仅训练，当前/下一状态、相机、设备独立随机流 |
| 动作编码 | 最长 H8、每步 7 维，56 维动作加 8 维有效 mask，共 64 维；尾块保留真实有效长度 |
| 奖励 | 沿用稀疏奖励：普通 chunk `-1`，成功终止 `0`，不归一化 |
| 折扣 / 成功确认 | `gamma=0.99`，按 chunk 折扣；连续成功确认 5 个环境步 |
| 验证 / 完整断点 | 每 500 步验证，验证 batch 32；每 2500 步完整保存 |
| 原始 loss / 曲线 | 逐更新 CSV；raw、EWMA100/500 的 PNG、SVG 和 CSV |

上述参数来自正式训练的有效配置；通用像素示例 YAML 的 64px、expectile 0.7、45000 步、验证 batch 64 不是本实验参数。
旧文档中的 CPU/待部署说明描述当时的审核阶段；本实验另有真实双 A100 短测、完整恢复和正式训练记录。

数据为 66 条真实转移录制，冻结 manifest 中的来源为 33 条自主成功、17 条自主失败、16 条接管。
原采集标签中的一条“成功”未满足连续 5 步成功确认，按冻结规则保留为失败；不为了凑原始 34/16 标签而改奖励。
按 haizhou 哈希阈值 80%/20% 配置分组，实际固定训练/验证成员为 37/13 个 root、4965/1380 个 transition。
接管及对应原失败属于同一 split；归一化只用训练 root，复现实验需沿用完整冻结成员。
两条官方观察/动作示教已通过 BC78 继承，不伪造为 critic 所需的奖励、终止、下一状态转移。

## 训练与测评记录

正式训练已完成 Q/V 各 30000 步，actor 0 步；训练循环耗时约 26 分 34 秒（含验证/保存，不含准备和初始化）。
2500 至 30000 的 12 份完整 checkpoint 均已做 SHA256 核验并归档 NAS。
JAX/jaxlib 0.5.3、Flax 0.10.2、Optax 0.2.4、NumPy 1.26.4；可训练参数数为 21257043。

预注册诊断选择的合格 scorer 是 **12500 步**；真实同状态检查只有两对，门槛通过不代表成功率提升。
用户指定的 **30000 步**最终模型另做诊断 rollout：冻结 BC78、N=4 候选、target `min(Q1,Q2)` 取 argmax，50 条结果为 **34/50（68%），错误 0**；该评测协议的 `success_streak=1`，与训练数据连续成功确认 5 步分开记录。
30000 步没有通过全部预注册诊断，保存的 `qualification_claimed=false` 不因这次 rollout 而改写。
本次没有提交 12500 步的正式 rollout，不能把它写成已经测得的更优策略。

训练与推理的环境分开记录：上述 Q/V 训练实际使用双 A100；该次本地 rollout 使用 RTX5090 跑冻结策略、CPU 跑官方 JAX Q 评分。
本地平台集成和评测管理脚本位于独立实验目录，不包含在这份训练代码中。

## 复现所需输入与配置

审阅算法可直接阅读仓库；复现实验还需另行提供以下原始文件，Git 中不包含数据、模型权重或 NAS 归档：

1. native66 数据包和 manifest，以及完整冻结 split roster。
2. `critic_normalization_frozen.json`（训练 root 归一化）。
3. `heldout_true_same_state_pairs.json` 及可核验的原始 simulator、相机、本体状态证据。
4. 做 rollout 时另提供同一份 BC78 权重及输入/动作协议；续训时需完整 Q/V/target、Adam、计数与 RNG checkpoint，单独 scorer 导出不能续训。

BC78 权重身份为 `pi05-bc-sc24-l1-off52-full-b64-u15000-20261101`，actor SHA256：
`de9f64bd52a216215b75e74ab04d34face7d957ab87a0b92f3927026d0943da1`。

下面是正式训练的有效配置快照；仅把原机器上的 5 个输出/输入路径替换为可移植路径，其他值保持原样。
这些路径对应的私有文件必须另行准备，不能用重新随机划分替代原冻结成员。
保存为 `production_review.json` 后可作为 YAML loader 的合法 JSON 配置输入：

```json
{
  "adapter": {
    "external_features": null,
    "mode": "pixels",
    "pixels": {
      "architecture": "upstream_pixel_multiplexer_d4pg",
      "camera_layout": "separate_rgb_views",
      "encoder_features": [
        32,
        32,
        32,
        32
      ],
      "encoder_filters": [
        3,
        3,
        3,
        3
      ],
      "encoder_strides": [
        2,
        1,
        1,
        1
      ],
      "latent_dim": 50,
      "random_crop_pad": 4
    },
    "vector_source": null
  },
  "backend": "official_idql_jax",
  "cache_dir": "cache/scene24_official_twinq",
  "critic": {
    "critic_lr": 0.0003,
    "expectile": 0.8,
    "hidden_dims": [
      256,
      256
    ],
    "tau": 0.005,
    "value_lr": 0.0003
  },
  "data": {
    "action_dim": 7,
    "action_horizon": 8,
    "frozen_members": "review_inputs/native66_split_members_frozen.json",
    "image_flip_horizontal": false,
    "image_flip_vertical": true,
    "image_size": 128,
    "include_post_success": false,
    "normalization": "review_inputs/critic_normalization_frozen.json",
    "proprio_dim": 8,
    "seed": 7,
    "split_strategy": "haizhou_hash_threshold",
    "validation_fraction": 0.2
  },
  "diagnostics": {
    "beta": 3.0,
    "max_weight": 20.0,
    "same_state_pairs": "review_inputs/heldout_true_same_state_pairs.json",
    "thresholds": {
      "ess_fraction_min": 0.1,
      "human_higher_fraction_min": 0.5,
      "rho_max": -0.5,
      "same_state_higher_fraction_min": 0.5,
      "success_terminal_weight_share_max": 0.5,
      "trajectory_median_max": -0.3
    }
  },
  "frozen_behavior": {
    "actor_sha256": "de9f64bd52a216215b75e74ab04d34face7d957ab87a0b92f3927026d0943da1",
    "policy_id": "pi05-bc-sc24-l1-off52-full-b64-u15000-20261101"
  },
  "output_dir": "outputs/scene24_official_twinq_s30000",
  "reward": {
    "discount": 0.99,
    "discount_unit": "chunk",
    "normalize": false,
    "step_reward": -1.0,
    "success_consecutive_steps": 5,
    "success_reward": 0.0,
    "termination_rule": "success_aware_recording_end_v1",
    "type": "sparse"
  },
  "runtime": {
    "device_count": 2,
    "platform": "gpu"
  },
  "schema_version": 1,
  "training": {
    "checkpoint_interval": 2500,
    "global_batch_size": 64,
    "gradient_accumulation_steps": 1,
    "log_interval": 1,
    "optimizer_updates": 30000,
    "per_device_batch_size": 32,
    "sampling": "replacement",
    "seed": 7,
    "validation_batch_size": 32,
    "validation_interval": 500
  }
}
```

```bash
python train_idql_official.py --config production_review.json --data /path/to/native66_package --dry-run
# 数据与独立 CUDA 环境核验后，在可见的两张卡上运行：
CUDA_VISIBLE_DEVICES=2,3 python train_idql_official.py --config production_review.json --data /path/to/native66_package
```

本入口是单个 JAX 进程管理两张卡，不使用 torchrun。
准备 CUDA 环境与跑真实短测仍是复现步骤，不能把安装 CPU extra 当作 GPU 已就绪。

原始记录索引（现有本地/NAS 实验归档，未将原始大文件上传 Git）：
`idql_official_scene24_L1_bc78fixed_b64_s30000_prepare_20261109/results/provenance.json`、
`results/summary.json`、`ops/control/backup_status.json`、
`evaluation_u30000/ops/result.json`、`evaluation_u30000/ops/finalizer_status.json`。
