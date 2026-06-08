# RoadGuard-Vision

基于 Ultralytics YOLO 的道路缺陷检测与可解释维修优先级辅助评估课程设计。系统面向 RDD2022 数据集，提供可复现数据转换、图片检测、ByteTrack 视频巡检、模型对比和中文 Gradio GUI。

## 适用边界

- 系统仅识别 `D00` 纵向裂缝、`D10` 横向裂缝、`D20` 网状裂缝和 `D40` 坑洞。
- 检测框面积只用于估计严重程度，不等于真实道路破损面积。
- ByteTrack 根据连续帧轨迹进行近似去重，不等于工程测量意义上的精确缺陷数量。
- 维修优先级是课程设计中的可解释辅助评分，不是经过道路工程认证的评价标准。

## 数据集

使用 [RDD2022 官方数据集](https://github.com/sekilab/RoadDamageDetector)，纳入日本、印度、捷克、美国、中国摩托车视角和中国无人机视角，暂不使用挪威子集。

转换工具期望先将原始文件整理为：

```text
datasets/RDD2022/
  Japan/
    images/
    annotations/
  India/
    images/
    annotations/
  Czech/
    images/
    annotations/
  United_States/
    images/
    annotations/
  China_MotorBike/
    images/
    annotations/
  China_Drone/
    images/
    annotations/
```

数据转换按子集分别使用固定随机种子 `42` 做 `80% / 10% / 10%` 划分，再合并为统一 YOLO 数据目录。损坏图片、缺失 XML、非法框和 XML 错误会写入问题清单。

## 环境安装

本地复现、运行 Notebook、启动 GUI 和执行测试建议使用独立 conda 环境：

```powershell
conda create -n roadguard-vision python=3.11 -y
conda activate roadguard-vision
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

安装完成后，可先运行测试确认基础逻辑正常：

```powershell
python -m unittest discover -s tests -v
pytest -q
```

启动本地 GUI：

```powershell
python scripts/launch_gui.py --host 127.0.0.1 --port 7860
```

浏览器访问 `http://127.0.0.1:7860`。GUI 会读取 `models/` 或 `weights/<模型>/best.pt` 中的真实训练权重；找不到权重时会显示中文提示，不会回退到官方预训练权重。

## Notebook 顺序

云平台首次训练建议先运行 `notebooks/00_cloud_training.ipynb`。课程流程展示仍按以下顺序执行：

1. `notebooks/01_dataset_review.ipynb`：检查数据来源、类别分布、样例和质量问题。
2. `notebooks/02_voc_to_yolo.ipynb`：转换 VOC XML，划分数据并生成 `data.yaml`。
3. `notebooks/03_train_models.ipynb`：使用同一参数训练 YOLO11 与 YOLO26 模型。
4. `notebooks/04_evaluate_models.ipynb`：汇总测试集指标、权重大小和单图耗时。
5. `notebooks/05_priority_scoring.ipynb`：逐项解释辅助评分。
6. `notebooks/06_gradio_gui.ipynb`：自动发现实际存在的训练权重并启动中文 GUI。

## 华为云 ModelArts Notebook 训练

云端入口面向 NVIDIA CUDA GPU Notebook。昇腾 NPU 需要单独适配，本项目当前不承诺兼容。

1. 创建带 NVIDIA GPU 的 ModelArts Notebook 实例，打开 JupyterLab。
2. 将完整项目压缩包上传并解压，至少保留 `notebooks/`、`src/`、`requirements.txt` 和 `datasets/downloads/`。
3. 确认六个 RDD2022 官方 ZIP 位于 `datasets/downloads/`。华为云 JupyterLab 对不超过 `100 MB` 的文件支持直接上传；超过 `100 MB` 且不超过 `50 GB` 的文件应通过 OBS 中转。RDD2022 压缩包通常较大，建议使用 OBS。
4. 打开 `notebooks/00_cloud_training.ipynb`，直接执行 Run All。无需手动填写 ZIP 路径，也无需切换训练开关。
5. Notebook 会自动安装缺失依赖、发现并解压 ZIP、生成 YOLO 数据集，然后训练 `YOLO11n` 和 `YOLO26n`。
6. 再次执行 Run All 时，会复用已有转换结果，并跳过已存在 `weights/<模型>/best.pt` 的模型，避免重复训练。
7. 正式实验前如需补齐四模型结果，将 `SELECTED_MODELS` 改为四个模型。

官方 RDD2022 归档解压后使用以下布局，转换工具也兼容前文所列的简化布局：

```text
RDD2022/
  Japan/
    train/
      images/
      annotations/
        xmls/
```

如果云实例没有公网访问能力，应提前上传预训练 `.pt` 文件，并将 `MODEL_SPECS` 中的值改为本地路径。训练中断后，可在 `RESUME_PATHS` 中填写对应模型的 `runs/train/<模型>/weights/last.pt`，入口 Notebook 会调用 Ultralytics 断点续训并重新同步最佳权重。

训练完成后的稳定权重位置为：

```text
weights/
  YOLO11n/best.pt
  YOLO26n/best.pt
```

答辩部署时也可以将真实训练权重放入简写目录：

```text
models/
  n11_best.pt
  s11_best.pt
  n26_best.pt
  s26_best.pt
```

GUI 优先读取 `models/` 简写目录，缺失时兼容 `weights/<模型>/best.pt`。模型对比页会展示实际加载路径，并读取 `reports/model_comparison.csv` 中的真实评估指标。没有评估记录的模型显示“未评估”，不存在的模型显示“未训练”。GUI 不会回退到官方预训练权重。

直接运行 `notebooks/06_gradio_gui.ipynb` 即可启动检测平台，也可以在本地 conda 环境中运行 `python scripts/launch_gui.py --host 127.0.0.1 --port 7860`。图片页输出标注图、缺陷明细、整体风险和推理耗时；视频页输出 ByteTrack 近似去重汇总、标注视频预览、标注视频下载和 CSV 巡检明细下载。没有本地模型时平台仍可启动，并显示中文提示。

华为云参考：[上传本地文件到 JupyterLab](https://support.huaweicloud.com/intl/en-us/usermanual-standard-modelarts/modelarts_30_0043.html)。

## 默认实验参数

| 参数 | 默认值 |
|---|---:|
| `imgsz` | `640` |
| `epochs` | `50` |
| `batch` | `8` |
| `seed` | `42` |
| `device` | `0` |

如因 GPU 显存限制降低 `batch`，必须在训练 Notebook 生成的实验记录中保留实际值。仓库不提交数据集、权重、训练输出和 GUI 导出文件。

## 评分规则

```text
缺陷分数 = 类型基础分 + 框面积占比分 + 密集区域加分
```

- 类型基础分：`D00=20`、`D10=25`、`D20=40`、`D40=50`。
- 面积占比分：`<1%=5`、`1%-5%=15`、`>5%=30`。
- 密集区域加分：另一缺陷中心点距离不超过图片对角线 `15%` 时加 `10` 分，每个缺陷最多加一次。
- 优先级：`<40` 为低，`40-69` 为中，`>=70` 为高。
- 模型置信度单独展示，不加入严重程度分数。

## 测试

```powershell
python -m unittest discover -s tests -v
```

安装 `pytest` 后也可以运行：

```powershell
pytest -q
```

## 参考

- [RDD2022 数据归档](https://doi.org/10.6084/m9.figshare.21431547.v1)
- [Ultralytics 训练文档](https://docs.ultralytics.com/modes/train/)
- [Ultralytics 跟踪文档](https://docs.ultralytics.com/modes/track/)
- [Gradio 文档](https://www.gradio.app/docs)
