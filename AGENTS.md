# 基于 YOLO 的道路缺陷检测与可解释维修优先级评估系统

## 1. 项目目标

本项目是道路缺陷检测课程设计。系统使用 RDD2022 公开数据集训练 Ultralytics YOLO 模型，识别道路图片和巡检视频中的缺陷，并将检测结果转换为可解释的维修优先级辅助评分。

最终交付形式为中文 Gradio GUI。系统必须形成完整闭环：

1. 使用公开数据训练道路缺陷检测模型。
2. 对比 YOLO11 与 YOLO26 的检测效果和推理效率。
3. 对图片输出缺陷位置、类别、置信度、维修优先级和整体风险。
4. 对视频使用 ByteTrack 对连续帧中的同一缺陷进行近似去重。
5. 对视频输出巡检汇总表、CSV 明细和标注后视频。

维修优先级是基于检测结果生成的可解释辅助评分，不是经过道路工程认证的评价标准。

## 2. 范围边界

### 必须实现

- 使用 Python 完成业务逻辑。
- 使用 Notebook 作为数据审查、数据转换、训练、评估、评分说明和 GUI 启动的主要入口。
- 使用少量 Python 模块复用公共逻辑，避免在多个 Notebook 中复制代码。
- 使用实际训练得到的权重进行 GUI 推理。
- 提供图片检测、视频巡检、模型对比和系统说明四个 GUI 标签页。
- 在 README 或课程报告中说明评分规则的依据、用途和局限性。

### 不纳入当前范围

- 用户登录和权限系统。
- 数据库和历史任务管理。
- GPS 地图和路段定位。
- PDF 报告生成。
- 桌面客户端。
- JavaScript、Java、C++ 等非 Python 业务代码。
- 将检测框面积表述为真实道路破损面积。
- 将辅助评分表述为道路工程认证标准。

## 3. 数据集规范

### 3.1 数据来源

使用 RDD2022 数据集，但暂不使用挪威子集。纳入以下国家、地区和采集视角：

- 日本。
- 印度。
- 捷克。
- 美国。
- 中国摩托车视角。
- 中国无人机视角。

RDD2022 原始标注采用 Pascal VOC XML 格式。训练前必须转换为 YOLO 标签格式。

### 3.2 类别定义

模型只检测以下四种缺陷：

| 类别代码 | 中文名称 |
|---|---|
| `D00` | 纵向裂缝 |
| `D10` | 横向裂缝 |
| `D20` | 网状裂缝 |
| `D40` | 坑洞 |

类别编号必须在数据转换脚本、`data.yaml`、模型推理、GUI 和 CSV 中保持一致。

### 3.3 数据划分

- 按国家或采集视角分别划分数据，再合并结果。
- 使用固定随机种子 `42`。
- 每个子集按 `80% / 10% / 10%` 划分训练集、验证集和测试集。
- 数据划分结果必须可复现。
- 数据转换必须记录损坏图片、缺失 XML、非法框坐标和无法转换的样本。
- 对非法框执行跳过并记录，不得将非法标签写入 YOLO 数据集。

## 4. 目录结构

项目按以下结构实现：

```text
notebooks/
  01_dataset_review.ipynb
  02_voc_to_yolo.ipynb
  03_train_models.ipynb
  04_evaluate_models.ipynb
  05_priority_scoring.ipynb
  06_gradio_gui.ipynb

src/
  data_utils.py
  priority.py
  image_inference.py
  video_patrol.py
  app.py

tests/
```

各 Notebook 的职责如下：

| Notebook | 职责 |
|---|---|
| `01_dataset_review.ipynb` | 展示数据来源、类别分布、样例和数据质量审查结果 |
| `02_voc_to_yolo.ipynb` | 将 XML 转换为 YOLO 标签，划分数据并生成 `data.yaml` |
| `03_train_models.ipynb` | 训练 YOLO11 和 YOLO26 模型，记录参数和权重路径 |
| `04_evaluate_models.ipynb` | 汇总精度、召回率、mAP、速度和模型大小 |
| `05_priority_scoring.ipynb` | 使用样例解释缺陷评分和整体风险计算过程 |
| `06_gradio_gui.ipynb` | 导入 `src/app.py` 并启动最终 GUI |

各 Python 模块的职责如下：

| 模块 | 职责 |
|---|---|
| `src/data_utils.py` | 数据审查、XML 转 YOLO、数据划分和问题清单导出 |
| `src/priority.py` | 缺陷级评分和图片或视频整体风险汇总 |
| `src/image_inference.py` | 图片推理、结果标准化和标注图生成 |
| `src/video_patrol.py` | 视频推理、ByteTrack 去重、标注视频和 CSV 导出 |
| `src/app.py` | Gradio 标签页仪表盘组装 |

## 5. 模型训练与评估

### 5.1 对比模型

正式实验目标为训练并比较以下四个模型：

- `YOLO11n`
- `YOLO11s`
- `YOLO26n`
- `YOLO26s`

最低验收要求为至少完成一个 YOLO11 模型和一个 YOLO26 模型的真实训练。

### 5.2 默认训练参数

四个模型必须使用相同的数据划分和尽可能一致的训练超参数，以保证比较公平。

| 参数 | 默认值 |
|---|---:|
| `imgsz` | `640` |
| `epochs` | `50` |
| `batch` | `8` |
| `seed` | `42` |
| `device` | `0` |

如果 GPU 显存不足，可以降低 `batch`。任何偏离默认参数的情况都必须记录在实验表中。

### 5.3 评估指标

每个模型至少记录以下指标：

- Precision。
- Recall。
- `mAP50`。
- `mAP50-95`。
- 训练后权重文件大小。
- 单张图片推理耗时。

GUI 只能加载实际训练得到的权重。找不到权重时必须显示中文错误信息，不得静默回退到官方预训练权重。

## 6. 可解释维修优先级

### 6.1 缺陷分数

每个检测缺陷的分数计算公式为：

```text
缺陷分数 = 类型基础分 + 面积占比分 + 密集区域加分
```

类型基础分：

| 类别 | 基础分 |
|---|---:|
| `D00` | `20` |
| `D10` | `25` |
| `D20` | `40` |
| `D40` | `50` |

面积占比分：

| 检测框面积 / 图片面积 | 加分 |
|---|---:|
| `< 1%` | `5` |
| `1% - 5%` | `15` |
| `> 5%` | `30` |

密集区域加分：

- 如果同一张图片中另一个缺陷框的中心点距离不超过图片对角线的 `15%`，增加 `10` 分。
- 每个缺陷最多增加一次密集区域分数。

### 6.2 优先级等级

| 总分 | 优先级 |
|---|---|
| `< 40` | 低 |
| `40 - 69` | 中 |
| `>= 70` | 高 |

图片和视频的整体风险取所有缺陷中的最高等级，同时展示低、中、高各等级的数量。视频必须基于 ByteTrack 近似去重后的唯一缺陷计算汇总。

模型置信度必须单独展示，不得加入严重程度分数。检测框面积只是严重程度的近似参考，不等于真实道路破损面积。

## 7. 公共接口

模块至少提供以下接口。后续实现可以增加类型注解和辅助接口，但不得改变这些接口的核心职责。

```python
score_defect(detection, image_shape, nearby_detections) -> dict
summarize_risk(scored_defects) -> dict
analyze_image(image, model, conf_threshold) -> dict
analyze_video(video_path, model, conf_threshold, output_dir) -> dict
build_demo(model_registry) -> gr.Blocks
```

接口职责：

| 接口 | 职责 |
|---|---|
| `score_defect` | 计算单个缺陷的面积占比、评分和等级 |
| `summarize_risk` | 汇总缺陷数量、等级数量和整体风险 |
| `analyze_image` | 执行单图推理、评分并生成标注结果 |
| `analyze_video` | 执行视频推理、ByteTrack 去重和文件导出 |
| `build_demo` | 组装 Gradio 标签页仪表盘 |

## 8. 视频去重与 CSV 导出

视频巡检使用 Ultralytics Track 模式与 ByteTrack。连续帧中具有相同 `track_id` 的目标只统计一次。

对每个唯一 `track_id`：

- 保留置信度最高的检测记录。
- 保留对应的最佳帧索引。
- 使用最佳检测框计算面积占比和维修优先级。
- 在报告中明确说明去重结果是目标跟踪产生的近似统计。

视频 CSV 至少包含以下字段：

```text
track_id,class_code,class_name,confidence,x1,y1,x2,y2,
area_ratio,priority_score,priority_level,best_frame_index
```

导出文件使用时间戳命名，避免覆盖已有结果。

## 9. Gradio GUI

使用中文标签页仪表盘，不实现单页工作台或步骤向导。

### 9.1 图片检测

- 上传图片。
- 选择已训练模型。
- 调整置信度阈值。
- 展示标注图。
- 展示缺陷类别、置信度、面积占比、评分和等级。
- 展示整张图片的整体风险和推理耗时。
- 没有检测到缺陷时显示“未发现缺陷”，整体风险为低。

### 9.2 视频巡检

- 上传道路视频。
- 选择已训练模型。
- 调整置信度阈值。
- 使用 ByteTrack 近似去重。
- 展示巡检汇总表。
- 提供标注视频下载。
- 提供 CSV 明细下载。
- 视频无法解码时返回清晰的中文提示。

### 9.3 模型对比

- 展示四个模型的评估指标表。
- 展示精度与速度对比图。
- 标记推荐模型。
- 保留模型切换功能，便于使用相同素材比较推理结果。

### 9.4 系统说明

- 说明 `D00`、`D10`、`D20` 和 `D40` 的含义。
- 说明维修优先级计算规则。
- 说明框面积、跟踪去重和辅助评分的局限性。
- 提供数据集和框架参考链接。

## 10. 异常处理

- 数据转换遇到损坏图片、缺失 XML 或非法框时，跳过对应样本并写入问题清单。
- GUI 找不到训练权重时，显示中文错误，不加载未经本项目训练的权重。
- 图片没有检测结果时，返回空明细表和低风险汇总。
- 视频无法解码时，停止分析并显示中文错误。
- 视频没有检测结果时，返回空明细表、低风险汇总和明确提示。
- 导出目录不存在时，由 Python 创建目录。
- 生成标注视频或 CSV 失败时，显示具体错误原因。

## 11. 验收清单

- 数据审查 Notebook 展示数据来源、类别统计和样例。
- 数据转换 Notebook 能生成合法 YOLO 标签和 `data.yaml`。
- 数据转换能够跳过并记录损坏图片、缺失 XML 和非法框。
- 至少实际训练一个 YOLO11 模型和一个 YOLO26 模型。
- 正式实验目标包含四个模型的对比结果。
- 评估 Notebook 展示 Precision、Recall、`mAP50`、`mAP50-95`、模型大小和单图推理耗时。
- 评分 Notebook 能使用样例逐项解释分数来源。
- 图片检测页面能显示检测框、中文类别、置信度、缺陷级评分和整体风险。
- 视频巡检页面能按 ByteTrack 唯一 `track_id` 近似去重。
- 视频巡检页面能导出标注视频和 CSV。
- GUI 能切换实际训练得到的 YOLO11 与 YOLO26 权重。
- README 或报告说明辅助评分的适用边界。
- 所有业务逻辑只使用 Python。

## 12. 开发要求

- 保持实现聚焦，不主动加入范围外功能。
- 优先编写小型、单一职责的 Python 函数。
- 为数据转换、评分、风险汇总、空检测结果和视频唯一 ID 汇总编写测试。
- Notebook 负责说明流程、展示实验和调用公共模块；可复用逻辑放入 `src/`。
- 训练、评估和 GUI 使用同一套类别映射。
- 权重、数据集、训练输出和 GUI 导出结果不得提交到源码仓库。
- 文档和课程报告中的结论必须与实际实验数据一致。

## 13. 参考资料

- [AGENTS.md 约定](https://agents.md/)
- [RDD2022 官方仓库](https://github.com/sekilab/RoadDamageDetector)
- [RDD2022 数据归档](https://doi.org/10.6084/m9.figshare.21431547.v1)
- [RDD2022 论文](https://arxiv.org/abs/2209.08538)
- [Ultralytics 训练文档](https://docs.ultralytics.com/modes/train/)
- [Ultralytics 跟踪文档](https://docs.ultralytics.com/modes/track/)
- [YOLO11 文档](https://docs.ultralytics.com/models/yolo11/)
- [YOLO26 文档](https://docs.ultralytics.com/models/yolo26/)
- [Gradio 文档](https://www.gradio.app/docs)
