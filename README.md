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

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

当前源码可以使用标准库测试核心逻辑。训练、可视化和 GUI 需要安装完整依赖。

## Notebook 顺序

1. `notebooks/01_dataset_review.ipynb`：检查数据来源、类别分布、样例和质量问题。
2. `notebooks/02_voc_to_yolo.ipynb`：转换 VOC XML，划分数据并生成 `data.yaml`。
3. `notebooks/03_train_models.ipynb`：使用同一参数训练 YOLO11 与 YOLO26 模型。
4. `notebooks/04_evaluate_models.ipynb`：汇总测试集指标、权重大小和单图耗时。
5. `notebooks/05_priority_scoring.ipynb`：逐项解释辅助评分。
6. `notebooks/06_gradio_gui.ipynb`：注册实际训练权重并启动中文 GUI。

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

