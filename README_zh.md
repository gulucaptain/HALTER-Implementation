# HALTER-Implementation：HALTER 非官方复现

[English](README.md) | 简体中文

复现论文 **From Rollout to Reset: A Graph-Based Harness for Autonomous Long-Horizon Manipulation Evaluation**
（arXiv [2609.19413](https://arxiv.org/abs/2609.19413)）。官方代码尚未开源，本仓库按论文描述重新实现。
论文没有写明的细节（prompt、阈值、任务子目标、两个 X-VLA 策略的分工等）都是我们自己的设定，已在代码注释中标明。

## 系统结构

```
rollout（被测策略，论文用 BESO）
   │  期间以 0.5 Hz 在线构建场景图
   ▼
场景图 G = (物体点云记录, 关系三元组 on/in/under)
   │  GroundedSAM 分割 → 深度反投影 → 多视角融合 → 几何规则推关系；被遮挡物体保留上一次观测
   ▼
LLM（同一个 Qwen3.5-397B，三套 prompt + 参考案例）
   ├─ 打分：完成的子任务比例
   ├─ 规划：用原子复位技能库 K 排出复位序列（符号校验 + 可选重规划）
   └─ 验证：每个技能执行后检查预期效果；全部执行后对照目标初始状态
   ▼
技能执行（每个技能两个 X-VLA 策略，各 30–50 条示教微调）
   │  任一验证失败 → 请求人工
   ▼
指标：Reset SR / 打分准确率 / 验证准确率 / 计划有效率 / 人工介入次数 / 操作员时间
```

| 目录 | 内容 |
|---|---|
| `halter/perception/` | 相机后端（OAK-D、ZED、RealSense、录像回放），GroundedSAM（HF transformers 版），点云融合 |
| `halter/scene_graph/` | 在线场景图构建（后台线程、遮挡记忆），on/in/under 几何推理，抽屉开合状态，文本序列化 |
| `halter/llm/` | LLM 后端：`api`（阿里云百炼/DashScope）、`local_server`（本地 vLLM）、`local_hf`（进程内 transformers）、`mock` |
| `halter/reasoning/` | 三套 prompt、参考案例、打分/规划/验证、计划符号校验 |
| `halter/skills/` | X-VLA 技能执行器（LeRobot），MP-skills 基线骨架（GraspNet + cuRobo） |
| `halter/robot/` | Franka（franky + Robotiq），YAM 接口骨架，LIBERO 封装 |
| `halter/harness/` | 主循环、人工介入接口、指标统计（均值 ± 二项标准误，与论文一致） |
| `halter/sim/` | 符号仿真器：不接硬件、不接模型也能跑通全流程；LIBERO 特权状态生成场景图 |
| `configs/` | 场景（技能库、目标状态、关系阈值）、任务（4 个主任务 + 3 个 held-out）、LLM、参考案例 |
| `scripts/` | 评测入口、LLM 冒烟测试、vLLM 部署、X-VLA 微调、参考案例生成与采集 |

## 安装

```bash
pip install -r requirements.txt                 # 核心依赖（numpy / pyyaml / openai / scipy）
pip install -e ".[perception,xvla,franka]"      # 真机按需安装
```

## 快速开始（无需硬件、无需模型）

```bash
pytest -q tests                                          # 13 个单测
python scripts/run_evaluation.py --config configs/sim.yaml
```

符号仿真 + 规则 LLM（oracle）跑 4 个任务 × 25 个 episode，输出与论文同口径的指标，结果写入 `runs/sim/`。

## Qwen 的两种接入方式

### 1. API（阿里云百炼 Model Studio，OpenAI 兼容）

```bash
export DASHSCOPE_API_KEY=sk-...
python scripts/test_llm.py --llm configs/llm/api_dashscope.yaml          # 与 oracle 对比的冒烟测试
python scripts/run_evaluation.py --config configs/sim.yaml --llm configs/llm/api_dashscope.yaml
```

- `base_url` 默认是北京地域；新加坡地域或新域名在 `configs/llm/api_dashscope.yaml` 中切换。
- **模型 id 请到百炼控制台核对**，默认填的是 `qwen3.5-397b-a17b`。
- 思考模式由顶层参数 `enable_thinking` 控制，默认关闭以降低延迟；打开后会自动切换为流式请求。

### 2. 本地部署（vLLM，OpenAI 兼容服务）

```bash
bash scripts/serve_qwen_vllm.sh                 # 默认 Qwen/Qwen3.5-397B-A17B-FP8，TP=8，端口 8000
python scripts/test_llm.py --llm configs/llm/local_vllm.yaml
```

- 需要 vLLM ≥ 0.17.0，以及 8 张 H200/H100 级别的 GPU（vLLM 官方 recipe 在 8×H200 上验证过）。
- 可通过环境变量 `MODEL` / `TP` / `PORT` 修改配置。
- 思考开关通过 `chat_template_kwargs.enable_thinking` 传入。
- 单卡调试：`MODEL=Qwen/Qwen3-8B TP=1 bash scripts/serve_qwen_vllm.sh`（需要去掉脚本里的 `--language-model-only`），也可以直接用 `configs/llm/local_hf_small.yaml` 走进程内 transformers。

## 真机流程

1. **标定**：每个固定相机标定 `T_world_cam`，腕部相机标定 `T_ee_cam`，保存为 `calib/*.npy`；在 `configs/real_franka.yaml` 里填写机器人 IP 和相机信息。
2. **测量场景**：在 `configs/scenes/*.yaml` 中填写桌面高度、抽屉内部区域、"原位"区域（`pot_home` 等）以及关系阈值。
3. **采示教并训练技能**：每个技能用 `lerobot-record` 录 30–50 条示教，数据集的 task 字符串 = 技能 instruction。然后执行：
   ```bash
   bash scripts/train_xvla_skill.sh <dataset> drawer open_lower_drawer 0
   bash scripts/train_xvla_skill.sh <dataset> drawer open_lower_drawer 1
   ```
4. **参考案例**：仓库自带的参考案例由符号仿真生成，只有关系、没有几何。真机上请用下面的命令采集真实场景图，替换 `configs/reference_cases/<task>.yaml`：
   ```bash
   python scripts/capture_graph.py --config configs/real_franka.yaml --task configs/tasks/drawer_store.yaml --out case.yaml
   ```
   每个任务需要 1 个成功案例和 3 个失败案例，每个技能需要 4 个成功案例。
5. **评测**：
   ```bash
   python scripts/run_evaluation.py --config configs/real_franka.yaml
   ```
   遇到人工介入时，终端会提示操作员复位；每个 episode 结束后可以输入真值标签，用于计算准确率。

## 与论文的差异 / 待确认

- **任务子目标与初始状态**：论文只给出了任务名和 12 个技能名，`configs/tasks/*.yaml` 中的子任务和 3 个 held-out 任务是我们根据技能名推测的。
- **两个 X-VLA 策略**：论文没说明两者的区别。本实现默认 p0 先执行、验证失败重试时换 p1（`two_policy_mode: retry`），两个策略用不同 seed 训练；也可以选 `primary` 或 `alternate`。
- **失败处理**：论文的做法是验证失败即请求人工，本实现默认 `skill_retries: 0` 与之一致。可以打开重试或重规划做消融。
- **计划有效率**：本实现用技能前置条件和效果做符号校验，统计首次规划是否有效。论文对这一指标的口径没有详细说明。
- **场景图**：几何阈值、抽屉内部区域、开合判定等都由我们自己设定，需要在你的场景上调。
- **被测策略**：论文用 BESO；这里提供通用的 LeRobot 策略 rollout（diffusion / ACT / pi0 / X-VLA 等）。
- **YAM 机械臂**：`YAMRobot` 只是接口骨架，需要按你使用的 i2rt SDK 版本补全。
