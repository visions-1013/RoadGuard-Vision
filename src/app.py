"""Chinese Gradio dashboard assembly."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .cloud_training import MODEL_NAMES
from .constants import CLASS_CODE_TO_NAME
from .image_inference import analyze_image
from .video_patrol import analyze_video

DETAIL_COLUMNS = [
    "class_code",
    "class_name",
    "confidence",
    "area_ratio",
    "priority_score",
    "priority_level",
]
DETAIL_HEADERS = ["类别代码", "缺陷类别", "置信度", "框面积占比", "评分", "优先级"]
VIDEO_COLUMNS = [
    "track_id",
    "class_code",
    "class_name",
    "confidence",
    "area_ratio",
    "priority_score",
    "priority_level",
    "best_frame_index",
]
VIDEO_HEADERS = [
    "轨迹 ID",
    "类别代码",
    "缺陷类别",
    "置信度",
    "框面积占比",
    "评分",
    "优先级",
    "最佳帧索引",
]
COMPARISON_HEADERS = [
    "模型",
    "状态",
    "实际加载路径",
    "Precision",
    "Recall",
    "mAP50",
    "mAP50-95",
    "权重大小（MB）",
    "单图耗时（ms）",
    "推荐模型",
]
ULTRALYTICS_CONFIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "ultralytics"


class ModelRegistryError(ValueError):
    """Raised when the GUI registry references an unavailable project weight."""


def validate_model_registry(model_registry: dict) -> dict:
    """Resolve registered weights and reject every unavailable entry."""
    resolved = {}
    for model_name, config in model_registry.items():
        if not isinstance(config, dict) or not config.get("weights"):
            raise ModelRegistryError(f"模型 {model_name} 未配置训练权重")
        weights = Path(config["weights"]).expanduser().resolve()
        if not weights.is_file():
            raise ModelRegistryError(f"找不到训练权重：{weights}")
        resolved[model_name] = {**config, "weights": str(weights)}
    return resolved


@lru_cache(maxsize=None)
def _load_yolo_weight(weights: str):
    """Load one local project weight once per GUI process."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
    ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ModelRegistryError("缺少 ultralytics 依赖，无法加载模型") from error
    return YOLO(weights)


def load_project_model(model_registry: dict, model_name: str):
    """Load a validated local weight without any pretrained-weight fallback."""
    registry = validate_model_registry(model_registry)
    if model_name not in registry:
        raise ModelRegistryError("当前没有可用训练权重，请先将模型放入 models/ 目录")
    return _load_yolo_weight(registry[model_name]["weights"])


def _details_rows(records: list[dict], columns: list[str]) -> list[list]:
    return [[record.get(column) for column in columns] for record in records]


def _risk_markdown(result: dict) -> str:
    summary = result["risk_summary"]
    counts = summary["level_counts"]
    return (
        f"**状态：** {result['message']}  \n"
        f"**整体风险：** {summary['overall_risk']}  \n"
        f"**缺陷数量：** {summary['total']} "
        f"（低：{counts['低']}，中：{counts['中']}，高：{counts['高']}）"
    )


def _comparison_rows(model_registry: dict) -> list[list]:
    rows = []
    for name in MODEL_NAMES:
        config = model_registry.get(name)
        if config is None:
            rows.append([name, "未训练", "", None, None, None, None, None, None, ""])
            continue
        metrics = config.get("metrics", {})
        rows.append(
            [
                name,
                config.get("status", "未评估"),
                config.get("weights", ""),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("map50"),
                metrics.get("map50_95"),
                metrics.get("weight_mb"),
                metrics.get("inference_ms"),
                "是" if config.get("recommended") else "",
            ]
        )
    return rows


def _registry_notice(model_registry: dict) -> str:
    if not model_registry:
        return (
            "**模型状态：** 当前没有可用训练权重。请将本项目训练得到的模型放入 "
            "`models/` 目录，例如 `models/n11_best.pt`。"
        )
    return "**模型状态：** 已自动加载 " + "、".join(model_registry)


def _comparison_plot(model_registry: dict):
    rows = [
        (name, config.get("metrics", {}).get("map50_95"), config.get("metrics", {}).get("inference_ms"))
        for name, config in model_registry.items()
    ]
    rows = [(name, accuracy, speed) for name, accuracy, speed in rows if accuracy is not None and speed is not None]
    if not rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter([speed for _, _, speed in rows], [accuracy for _, accuracy, _ in rows])
    for name, accuracy, speed in rows:
        axis.annotate(name, (speed, accuracy))
    axis.set_xlabel("单张图片推理耗时（ms）")
    axis.set_ylabel("mAP50-95")
    axis.set_title("模型精度与速度对比")
    figure.tight_layout()
    return figure


def build_demo(model_registry) -> "object":
    """Build the four-tab Chinese Gradio dashboard."""
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("缺少 gradio 依赖，无法启动 GUI") from error

    try:
        registry = validate_model_registry(model_registry)
        registry_error = ""
    except ModelRegistryError as error:
        registry = {}
        registry_error = f"**模型配置错误：** {error}"

    model_names = list(registry)
    default_model = model_names[0] if model_names else None

    def run_image(image, model_name, conf_threshold):
        if image is None:
            return None, [], "**错误：** 请先上传图片"
        try:
            model = load_project_model(registry, model_name)
            result = analyze_image(image, model, conf_threshold)
            summary = _risk_markdown(result) + f"  \n**推理耗时：** {result['inference_ms']:.2f} ms"
            return result["annotated_image"], _details_rows(result["details"], DETAIL_COLUMNS), summary
        except Exception as error:
            return None, [], f"**图片检测失败：** {error}"

    def run_video(video_path, model_name, conf_threshold):
        if not video_path:
            return None, None, None, [], "**错误：** 请先上传视频"
        try:
            model = load_project_model(registry, model_name)
            result = analyze_video(video_path, model, conf_threshold, "exports")
            return (
                result["annotated_video"],
                result["annotated_video"],
                result["csv_path"],
                _details_rows(result["details"], VIDEO_COLUMNS),
                _risk_markdown(result),
            )
        except Exception as error:
            return None, None, None, [], f"**视频巡检失败：** {error}"

    with gr.Blocks(
        title="RoadGuard-Vision 道路缺陷检测系统",
    ) as demo:
        gr.Markdown("# 基于 YOLO 的道路缺陷检测与维修优先级辅助评估系统")
        gr.Markdown(_registry_notice(registry), elem_classes=["roadguard-note"])
        if registry_error:
            gr.Markdown(registry_error)
        with gr.Tab("图片检测"):
            with gr.Row():
                image_input = gr.Image(label="上传道路图片")
                image_output = gr.Image(label="标注结果")
            image_model = gr.Dropdown(model_names, value=default_model, label="已训练模型")
            image_conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="置信度阈值")
            image_button = gr.Button("开始检测", variant="primary")
            image_summary = gr.Markdown()
            image_table = gr.Dataframe(headers=DETAIL_HEADERS, label="缺陷明细")
            image_button.click(
                run_image,
                [image_input, image_model, image_conf],
                [image_output, image_table, image_summary],
            )

        with gr.Tab("视频巡检"):
            video_input = gr.Video(label="上传道路巡检视频")
            video_model = gr.Dropdown(model_names, value=default_model, label="已训练模型")
            video_conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="置信度阈值")
            video_button = gr.Button("开始巡检", variant="primary")
            video_summary = gr.Markdown()
            video_preview = gr.Video(label="标注视频预览")
            video_table = gr.Dataframe(headers=VIDEO_HEADERS, label="ByteTrack 近似去重汇总")
            annotated_download = gr.File(label="下载标注视频")
            csv_download = gr.File(label="下载 CSV 明细")
            video_button.click(
                run_video,
                [video_input, video_model, video_conf],
                [
                    video_preview,
                    annotated_download,
                    csv_download,
                    video_table,
                    video_summary,
                ],
            )

        with gr.Tab("模型对比"):
            gr.Dataframe(
                value=_comparison_rows(registry),
                headers=COMPARISON_HEADERS,
                label="真实训练权重评估指标",
            )
            gr.Plot(value=_comparison_plot(registry), label="精度与速度对比图")

        with gr.Tab("系统说明"):
            class_lines = "\n".join(
                f"- `{code}`：{name}" for code, name in CLASS_CODE_TO_NAME.items()
            )
            gr.Markdown(
                f"""\
## 缺陷类别
{class_lines}

## 维修优先级辅助评分
缺陷分数 = 类型基础分 + 框面积占比分 + 密集区域加分。模型置信度单独展示，不加入评分。

## 局限性
- 检测框面积仅作为严重程度的近似参考，不等于真实道路破损面积。
- ByteTrack 根据连续帧目标跟踪进行近似去重，不代表工程测量意义上的精确缺陷数量。
- 维修优先级用于课程设计中的辅助决策展示，不是经过道路工程认证的评价标准。

## 参考链接
- [RDD2022 官方仓库](https://github.com/sekilab/RoadDamageDetector)
- [Ultralytics 训练文档](https://docs.ultralytics.com/modes/train/)
- [Ultralytics 跟踪文档](https://docs.ultralytics.com/modes/track/)
"""
            )
    return demo
