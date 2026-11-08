# CC 视觉审核意见落实

对应根目录 `REVIEW_CC_OFFICIAL_IDQL.md` 对0332f2e的审核。本轮只修改代码、配置
示例和CPU验证，没有使用GPU，没有提交正式训练或平台评测。

## 必须改1：视觉训练增强

已在 `official_idql/core.py` 的训练更新外层调用作者原 DrQ
`batched_random_crop`，pad4。当前/下一状态、相机和rank使用独立RNG；
官方V、Q共享同一步增强batch，原函数对象和数学保持。
验证、同状态排序、普通评分不增强也不推进RNG。
配方进入adapter合同，增强RNG进入完整断点，恢复逐值验证。
原vector路径pad0仍与作者原创建器及原jit更新连续5步逐值一致。

## 必须改2：诊断与合格 checkpoint 选择

`official_idql/diagnostics.py`输出完整留出覆盖、全局/逐路径V进度Spearman、
同root人工与原失败段的online A比较、ESS、成功末块权重。
补充元数据恢复去重前共享前缀的成功路径membership，不改训练reward或原缓存。
缺失路径、常数相关、未匹配人工根均不会被静默丢弃以抬高指标。

`official_idql/pairs.py`对真正同状态数据核验：精确t的simulator state、双相机、
本体状态一致，人工与原策略动作具有相同有效执行长度，来自同一留出根。
原始文件哈希、归一化、adapter、数据manifest和pair清单均绑定。
不以同root、近邻帧或相同进度代替同状态。未配置真实包就是UNAVAILABLE。
人工更优是明确标注的假设，pair数量和覆盖随报告保留。

开训前固定门槛：全局rho≤−0.5，路径中位rho≤−0.3，ESS/N≥0.1，
成功末块权重≤0.5，同root人工段更高比例>0.5，真同状态人工分数更高比例>0.5。
这些是critic检查，不保证策略成功率提高；少量pair超过0.5不代表显著性。

`official_idql/selection.py`只在实际保留的完整CP中选择。全部门槛满足后，
固定按rho、路径中位rho、验证TD loss、较早步数排序。缺失/越界的实际指标
不能靠布尔PASS标签绕过。没有合格者就保持null，不回退最后一步。
`best_progress_candidate`单独用于分析，不代表已准入评测。

每个CP边界强制同一步只读诊断，即使CP与验证间隔不整除。
选择与完整诊断入CP metadata，恢复以CP为权威，不能续训时改变阈值。
CP、scorer和证据均落地后才发布 `selection.json`。
已有合格CP不会被后续TD更低、动作排名却失败的CP覆盖。

## 建议项

- 像素示例改64px；实际CPU创建Q/V可训练参数4,463,443，actor=0。
  provenance记录实际参数数量。
- 保留官方expectile0.7、Adam；文档并列旧0.8/AdamW0.01和增强差异。
  旧实验的有效性结论不直接沿用。
- 实际数据以native manifest及完整frozen roster为准。此前BC52含2条只具
  observation/action的官方示教；不能按“52+16”凑成68条虚构RL转移。
  此前native critic是34成功+16接管+16原失败=66，正式准备须再次核对真实成员。
- 用户此前选择的BC78未更换。base/BC2500及新开局、随机候选对照建议保留待定。
- 双A100 CUDA插件、驱动、真实吞吐未验证；正式GPU短测仍需用户授权。
  当前没有自动部署NAS调度或改动其他训练/评测服务。

## 验证

全仓390项CPU测试通过，官方路径73项全部执行，0失败、0跳过。
验证工具逐node核对收集/选择/执行，拒绝继承筛选条件导致的漏跑，
154份源/测试/公共依赖与配置（含80份上游实际源文件）前后SHA一致。
installed-wheel验证保留全部上游源字节，实际新诊断CSV能生成PNG/SVG/EWMA CSV。
所有更新和门槛正反例为合成CPU测试；真实Scene24模型尚未宣称通过门槛。

详见 [实现与接口](OFFICIAL_IDQL.md)、[实际验证记录](OFFICIAL_IDQL_VALIDATION.json)。
