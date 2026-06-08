"""Launch the RoadGuard-Vision Gradio GUI from the project root."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 RoadGuard-Vision 中文 GUI")
    parser.add_argument("--host", default="127.0.0.1", help="Gradio 监听地址")
    parser.add_argument("--port", default=7860, type=int, help="Gradio 监听端口")
    parser.add_argument("--share", action="store_true", help="启用 Gradio share 链接")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = PROJECT_ROOT / "outputs" / "launch_gui.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("booting\n", encoding="utf-8")

    from src.app import build_demo
    from src.cloud_training import MODEL_NAMES, available_trained_weights

    registry = available_trained_weights(
        MODEL_NAMES,
        PROJECT_ROOT / "weights",
        models_root=PROJECT_ROOT / "models",
        metrics_path=PROJECT_ROOT / "reports" / "model_comparison.csv",
    )
    demo = build_demo(registry)
    log_path.write_text(
        f"starting host={args.host} port={args.port} models={list(registry)}\n",
        encoding="utf-8",
    )
    launch_result = demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        prevent_thread_lock=True,
    )
    # Keep the Gradio server object alive while this launcher sleeps.
    _ = launch_result
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"launched result={list(launch_result)}\n")
    demo.block_thread()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log_path = PROJECT_ROOT / "outputs" / "launch_gui.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        raise
