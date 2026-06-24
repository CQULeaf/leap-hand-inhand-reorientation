# 基于深度强化学习的 LEAP Hand 单轴手内方块重定向

[技术报告网页](https://yexuhang.com/projects/leap-hand-inhand-reorientation/tech-report/) · [项目展示网页](https://yexuhang.com/projects/leap-hand-inhand-reorientation/)

该仓库整理了 LEAP Hand 在 Isaac Lab 中进行单轴手内方块重定向的训练、播放和实机部署代码。任务目标是在无视觉、无触觉反馈条件下，仅依赖电机本体感知历史，让 LEAP Hand 驱动方块持续完成绕 z 轴的分段重定向。

## 项目亮点

1. 基于 Isaac Lab DirectRLEnv 实现 LEAP Hand 方块重定向任务，支持 `rl_games` PPO 训练和 checkpoint 播放。
2. 策略观测仅包含关节位置和上一时刻目标动作历史，不依赖外部相机、触觉或物体真实状态。
3. 环境内置自动域随机化，包括机器人/物体摩擦、关节刚度阻尼、物体质量、动作噪声、动作延迟、初始位姿扰动和外力扰动。
4. 仓库提供真实 LEAP Hand 部署入口，支持 Python SDK 直连 Dynamixel 和 ROS2 两种路径。
5. 工程结构按训练、评估、部署和工具分层，预训练权重统一放在 `pretrained/`。

## 仓库结构

```text
pretrained/                 # 预训练 checkpoint
scripts/
  train/                    # 训练入口
  eval/                     # 仿真播放入口
  deploy/                   # 真实硬件部署包装脚本
  internal/                 # shell 脚本调用的 Python 实现
  tools/                    # 资产检查、任务列表、TensorBoard
source/LEAP_Isaaclab/
  LEAP_Isaaclab/assets/     # LEAP Hand USD 资产
  LEAP_Isaaclab/tasks/      # 任务注册、配置和 DirectRLEnv 实现
  LEAP_Isaaclab/utils/      # ADR、观测处理与 LEAP Hand 工具
  LEAP_Isaaclab/deployment_scripts/
                            # rl_games policy loader 与实机控制器
docker/                     # Isaac Lab 容器入口
tech_report/                # 技术报告 LaTeX 源码；网页版 PDF 由个人网站托管
```

## 环境要求

项目所用环境：

- Ubuntu 22.04 LTS
- NVIDIA GeForce RTX 5060 Laptop GPU
- Isaac Sim `5.1.0`
- Isaac Lab `2.3.0`
- Conda 环境名：`env_isaaclab`
- Isaac Lab 环境内 Python `3.11.15`

本地脚本默认尝试激活 Conda 环境 `env_isaaclab`。如果当前 shell 没有 `conda` 命令，脚本会尝试加载 `/home/tools/anaconda3/etc/profile.d/conda.sh`。

## 资产与 Checkpoint

LEAP Hand USD 资产和预训练权重建议通过 Git LFS 管理。克隆仓库后运行：

```bash
git lfs install
git lfs pull
python3 scripts/tools/check_assets.py
```

预训练权重：

```text
pretrained/leap_hand_reorient.pth
```

如果未显式传入 checkpoint，`scripts/eval/play_stage1.sh` 和 `scripts/deploy/reorient_z.sh` 会默认使用该权重。

## 快速开始

所有命令默认从仓库根目录运行。当前 `.sh` 文件不要求有可执行位，建议统一用 `bash scripts/...` 调用。

```bash
python3 scripts/tools/check_assets.py
find scripts -name '*.sh' -print0 | xargs -0 bash -n
```

列出已注册任务：

```bash
source /home/tools/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH="$PWD/source/LEAP_Isaaclab:${PYTHONPATH:-}"
python scripts/tools/list_tasks.py
```

在仿真中播放预训练策略：

```bash
bash scripts/eval/play_stage1.sh --num-envs 1
```

有限步数播放，用于迁移后 smoke test：

```bash
bash scripts/eval/play_stage1.sh \
  --num-envs 1 \
  --max-steps 20 \
  --headless \
  --physx-smoke-buffers
```

## 训练

调试训练入口：

```bash
bash scripts/train/stage1.sh --profile debug --run-name debug_reorientation
```

本地训练入口：

```bash
bash scripts/train/stage1.sh --profile local-5060
```

从已有 checkpoint 继续训练：

```bash
bash scripts/train/stage1.sh \
  --checkpoint pretrained/leap_hand_reorient.pth \
  --max-iterations 1200 \
  --run-name resume_reorientation
```

训练输出默认写入：

```text
logs/rl_games/leap_hand_reorient/<run-name>/
```

查看 TensorBoard：

```bash
bash scripts/tools/tensorboard_stage1.sh
```

## 真实硬件部署

仓库提供部署脚本，但真实硬件执行必须单独按安全关键步骤处理。部署路径会连接 Dynamixel 电机，并可能设置控制模式、扭矩状态、P/D gain 和电流限制。

离线部署链路 smoke test，不连接电机：

```bash
bash scripts/deploy/reorient_z.sh \
  --checkpoint pretrained/leap_hand_reorient.pth \
  --device cuda:0 \
  --dry-run \
  --max-steps 5
```

真实硬件命令模板：

```bash
bash scripts/deploy/reorient_z.sh \
  --checkpoint pretrained/leap_hand_reorient.pth \
  --port /dev/ttyUSB0 \
  --hz 30 \
  --kp 800 \
  --kd 200 \
  --curr-lim 500 \
  --max-steps 600 \
  --disable-torque-on-exit
```

运行前请确认 LEAP Hand 固定、电源与电流限制、急停方案、串口、checkpoint 和步数上限。

## Citing

本仓库基于 LEAP Hand 与 Isaac Lab 相关开源实现整理。若该代码对你的研究有帮助，请引用：

```bibtex
@article{
    shaw2023leaphand,
    title={LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning},
    author={Shaw, Kenneth and Agarwal, Ananye and Pathak, Deepak},
    journal={Robotics: Science and Systems (RSS)},
    year={2023}
}
```

## Acknowledgements

本项目构建在以下代码基础上：

- LEAP Hand Isaac Lab Implementation - https://github.com/leap-hand/LEAP_Hand_Isaac_Lab
- LEAP Hand IsaacGym Implementation - https://github.com/leap-hand/LEAP_Hand_Sim
- Isaac Lab - https://github.com/isaac-sim/IsaacLab
- RL Games - https://github.com/Denys88/rl_games

**License:** MIT License. Provided as-is, without warranty.
