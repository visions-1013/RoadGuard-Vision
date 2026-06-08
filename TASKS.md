# RoadGuard-Vision 任务清单

本清单记录当前仓库实现状态。`[x]` 表示源码或工作流已经实现；`[ ]` 表示必须在提供 RDD2022 数据、完整 Python 依赖和可用 GPU 后执行的真实实验或手工验收。

## 1. 项目脚手架

- [x] 创建 `src/`、`tests/`、`notebooks/`、`reports/`。
- [x] 创建 `.gitignore`、`requirements.txt` 和 README。
- [x] 忽略数据集、部署模型、权重、训练输出和 GUI 导出结果。
- [x] 创建 `.gitattributes` 和 `.editorconfig`，统一文本换行、编码和缩进规则。
- [x] 安装完整 Python 依赖。
- [ ] 在安装后的 Ultralytics 版本中验证 `yolo11*.pt` 和 `yolo26*.pt` 标识可用。

## 2. 类别与评分配置

- [x] 固定类别顺序：`D00=0`、`D10=1`、`D20=2`、`D40=3`。
- [x] 定义中文名称、基础分、面积阈值、密集阈值和等级阈值。
- [x] 测试类别顺序和唯一性。

## 3. 数据审查与转换

- [x] 实现图片解码检查、VOC XML 读取和问题记录。
- [x] 跳过并记录损坏图片、缺失 XML、非法 XML 和非法框。
- [x] 跳过非目标类别。
- [x] 实现 VOC 框转 YOLO 坐标。
- [x] 实现子集内固定种子 `42` 的 `80% / 10% / 10%` 划分。
- [x] 实现各子集划分合并、问题 CSV 和 `data.yaml` 导出。
- [x] 测试损坏图片、缺失 XML、非法框、坐标转换、可复现划分和类别顺序。
- [ ] 使用完整 RDD2022 数据执行转换。
- [ ] 抽查生成标签坐标均位于 `[0,1]`。

## 4. 数据 Notebook

- [x] 创建 `01_dataset_review.ipynb`，配置六个纳入子集并排除挪威。
- [x] 创建 `02_voc_to_yolo.ipynb`，调用公共转换函数。
- [ ] 使用实际数据运行两个 Notebook 并保存审查输出。

## 5. 维修优先级

- [x] 实现 `score_defect(detection, image_shape, nearby_detections)`。
- [x] 实现 `summarize_risk(scored_defects)`。
- [x] 测试四类基础分、面积边界、密集分最多一次、等级边界和空结果。
- [x] 创建 `05_priority_scoring.ipynb` 展示逐项评分过程。

## 6. 图片推理

- [x] 实现结果标准化、标注和 `analyze_image(image, model, conf_threshold)`。
- [x] 返回标注图、明细、风险汇总、提示和推理耗时。
- [x] 测试空检测和置信度不参与评分。
- [ ] 使用真实训练权重测试有缺陷和无缺陷图片。

## 7. 视频巡检

- [x] 实现 ByteTrack 结果标准化和唯一 `track_id` 汇总。
- [x] 为相同 `track_id` 保留最高置信度记录及其帧索引。
- [x] 实现标注视频、时间戳 CSV 和 `analyze_video(...)`。
- [x] 测试 CSV 字段、轨迹去重、输出目录创建和无法解码提示。
- [ ] 使用真实训练权重测试可解码视频和无缺陷视频。

## 8. 真实训练与评估

- [x] 创建 `03_train_models.ipynb`，集中配置四个模型和统一参数。
- [x] 创建 `04_evaluate_models.ipynb`，汇总 Precision、Recall、`mAP50`、`mAP50-95`、权重大小和单图耗时。
- [x] 在评估 Notebook 中导出 CSV、绘制精度速度图并选择推荐模型。
- [ ] 训练 `YOLO11n`。
- [ ] 训练 `YOLO11s`。
- [ ] 训练 `YOLO26n`。
- [ ] 训练 `YOLO26s`。
- [ ] 确认至少一个 YOLO11 和一个 YOLO26 权重真实存在。
- [ ] 使用实际权重运行评估 Notebook。
- [ ] 核对 README 或课程报告中的实验结论与实际指标一致。

## 9. Gradio GUI

- [x] 实现模型注册表校验，权重缺失时显示中文错误。
- [x] 禁止静默回退到官方预训练权重。
- [x] 实现 `build_demo(model_registry)`。
- [x] 创建图片检测、视频巡检、模型对比和系统说明四个中文标签页。
- [x] 自动发现 `models/` 部署权重并兼容 `weights/` 训练输出权重。
- [x] 模型对比页展示四模型状态、实际加载路径和真实评估指标。
- [x] 视频巡检页展示标注视频预览，并提供标注视频和 CSV 下载。
- [x] 创建 `06_gradio_gui.ipynb`，自动发现实际训练权重并允许零模型启动。
- [x] 安装依赖并手工启动 GUI。
- [x] 手工验证图片、视频、下载和模型切换流程。

## 10. 最终验收

- [x] 使用标准库运行核心单元测试。
- [ ] 安装 `pytest` 后运行 `pytest -q`。
- [ ] 完整执行六个 Notebook。
- [ ] 检查 Git 状态，确认没有提交数据集、权重、训练输出或 GUI 导出结果。
