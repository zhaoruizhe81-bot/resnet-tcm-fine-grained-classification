from __future__ import annotations

import io
import platform
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from .history_store import (
    default_history_db_path,
    fetch_history_records,
    fetch_history_stats,
    init_history_db,
    insert_history_record,
)
from .inference_service import (
    CheckpointBundle,
    default_export_name,
    discover_checkpoints,
    load_checkpoint_bundle,
    predict_pil_image,
    predict_uploaded_images,
    predict_video_frames,
    read_checkpoint_metadata,
)
from .project_service import discover_training_runs, get_default_data_root, load_training_run_artifacts, scan_dataset_overview
from .utils import ensure_dir


OUTPUT_ROOT = Path("outputs")
WEBAPP_ROOT = OUTPUT_ROOT / "webapp"
EXPORT_ROOT = WEBAPP_ROOT / "exports"
DEFAULT_DB_PATH = default_history_db_path(OUTPUT_ROOT)


def initialize_page(title: str, icon: str) -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    ensure_dir(WEBAPP_ROOT)
    ensure_dir(EXPORT_ROOT)
    init_history_db(DEFAULT_DB_PATH)
    apply_theme()


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
          background:
            radial-gradient(circle at top left, rgba(194, 151, 92, 0.15), transparent 30%),
            linear-gradient(180deg, #f7f2e8 0%, #f0e6d2 100%);
        }
        .block-container {
          padding-top: 1.5rem;
          padding-bottom: 2rem;
        }
        div[data-testid="stMetricValue"] {
          color: #6b3f1d;
        }
        .app-card {
          background: rgba(255, 255, 255, 0.8);
          border: 1px solid rgba(107, 63, 29, 0.15);
          border-radius: 18px;
          padding: 1rem 1.2rem;
          box-shadow: 0 12px 40px rgba(93, 64, 24, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _write_records_to_csv(records: list[dict[str, Any]]) -> bytes:
    if not records:
        return "".encode("utf-8-sig")
    dataframe = pd.DataFrame(records)
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def _save_export_bytes(filename: str, payload: bytes) -> Path:
    ensure_dir(EXPORT_ROOT)
    target = EXPORT_ROOT / filename
    if target.exists():
        stem = target.stem
        target = EXPORT_ROOT / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{target.suffix}"
    target.write_bytes(payload)
    return target


def _utc_now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _counts_dataframe(counts: dict[str, int]) -> pd.DataFrame:
    dataframe = pd.DataFrame(
        [{"class_name": key, "count": value} for key, value in counts.items()]
    )
    if dataframe.empty:
        return pd.DataFrame(columns=["class_name", "count"])
    return dataframe.sort_values("count", ascending=False)


@st.cache_data(show_spinner=False)
def cached_checkpoints() -> list[str]:
    return [str(path) for path in discover_checkpoints(OUTPUT_ROOT)]


@st.cache_data(show_spinner=False)
def cached_training_runs() -> list[dict[str, Any]]:
    return discover_training_runs(OUTPUT_ROOT)


@st.cache_data(show_spinner=False)
def cached_dataset_overview(data_root: str) -> dict[str, Any]:
    return scan_dataset_overview(data_root)


@st.cache_data(show_spinner=False)
def cached_run_artifacts(run_dir: str) -> dict[str, Any]:
    return load_training_run_artifacts(run_dir)


@st.cache_resource(show_spinner=False)
def cached_model_bundle(checkpoint_path: str, device_preference: str, modified_at: float) -> CheckpointBundle:
    return load_checkpoint_bundle(checkpoint_path, requested_device=device_preference)


def sidebar_model_context(require_model: bool = False) -> dict[str, Any]:
    st.sidebar.markdown("## 模型控制台")
    if st.sidebar.button("刷新模型列表"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    discovered = cached_checkpoints()
    st.session_state.setdefault("checkpoint_source_mode", "scan" if discovered else "manual")
    st.session_state.setdefault("selected_checkpoint_path", discovered[0] if discovered else "")
    st.session_state.setdefault("manual_checkpoint_path", "")
    st.session_state.setdefault("device_preference", "auto")

    source_mode = st.sidebar.radio(
        "模型来源",
        options=["scan", "manual"],
        index=0 if st.session_state["checkpoint_source_mode"] == "scan" else 1,
        format_func=lambda item: "扫描结果" if item == "scan" else "手动输入",
    )
    st.session_state["checkpoint_source_mode"] = source_mode

    selected_checkpoint = ""
    if source_mode == "scan" and discovered:
        default_index = discovered.index(st.session_state["selected_checkpoint_path"]) if st.session_state["selected_checkpoint_path"] in discovered else 0
        selected_checkpoint = st.sidebar.selectbox("已发现模型", discovered, index=default_index)
        st.session_state["selected_checkpoint_path"] = selected_checkpoint
    else:
        selected_checkpoint = st.sidebar.text_input(
            "Checkpoint 路径",
            value=st.session_state["manual_checkpoint_path"],
            placeholder="例如 outputs/resnet18_tcm/best.pt",
        )
        st.session_state["manual_checkpoint_path"] = selected_checkpoint

    device_preference = st.sidebar.selectbox(
        "推理设备",
        options=["auto", "cuda", "cpu"],
        index=["auto", "cuda", "cpu"].index(st.session_state["device_preference"]),
    )
    st.session_state["device_preference"] = device_preference

    checkpoint_path = Path(selected_checkpoint) if selected_checkpoint else None
    checkpoint_valid = bool(checkpoint_path and checkpoint_path.is_file())

    if checkpoint_valid:
        st.sidebar.success(f"已选择模型: {checkpoint_path.parent.name}")
    elif require_model:
        st.sidebar.error("当前页面需要先选择有效的 checkpoint")
    else:
        st.sidebar.info("未选择有效模型时，非识别页面仍可浏览。")

    stats = fetch_history_stats(DEFAULT_DB_PATH)
    st.sidebar.metric("历史记录", stats["total_records"])
    st.sidebar.metric("已发现模型", len(discovered))
    st.sidebar.metric("训练实验", len(cached_training_runs()))

    return {
        "checkpoint_path": str(checkpoint_path) if checkpoint_valid else None,
        "device_preference": device_preference,
        "discovered_checkpoints": discovered,
    }


def get_active_bundle(context: dict[str, Any]) -> CheckpointBundle | None:
    checkpoint_path = context.get("checkpoint_path")
    if not checkpoint_path:
        return None

    target = Path(checkpoint_path)
    return cached_model_bundle(
        str(target),
        context["device_preference"],
        target.stat().st_mtime,
    )


def _current_data_root(bundle: CheckpointBundle | None) -> str:
    if bundle is not None:
        return bundle.config["data"]["root"]
    return get_default_data_root()


def render_home_page() -> None:
    initialize_page("中药分类平台", "🌿")
    context = sidebar_model_context(require_model=False)
    bundle = get_active_bundle(context) if context["checkpoint_path"] else None

    st.title("中药细粒度分类智能识别平台")
    st.caption("基于 ResNet 的中药图片与视频识别、训练结果分析和项目闭环展示系统。")

    stats = fetch_history_stats(DEFAULT_DB_PATH)
    runs = cached_training_runs()
    cols = st.columns(4)
    cols[0].metric("已发现模型", len(context["discovered_checkpoints"]))
    cols[1].metric("训练实验数", len(runs))
    cols[2].metric("历史识别记录", stats["total_records"])
    cols[3].metric("当前模型类别数", len(bundle.class_names) if bundle else 0)

    st.markdown(
        """
        <div class="app-card">
        <h3>平台能力概览</h3>
        <p>这个 Web 应用把原来的 CLI 训练项目升级成了课程展示型系统，包含模型选择、单图识别、批量识别、视频抽帧识别、训练结果看板、数据集概览与历史记录管理。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    link_cols = st.columns(4)
    link_cols[0].page_link("pages/01_image_recognition.py", label="进入单图识别", icon="🖼️")
    link_cols[1].page_link("pages/02_batch_recognition.py", label="进入批量识别", icon="🗂️")
    link_cols[2].page_link("pages/03_video_recognition.py", label="进入视频识别", icon="🎬")
    link_cols[3].page_link("pages/05_training_dashboard.py", label="查看训练看板", icon="📈")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("系统架构")
        st.markdown(
            """
            - 前端：Streamlit 多页面交互界面
            - 服务层：模型加载、图片推理、视频抽帧推理、训练结果读取
            - 存储层：SQLite 历史记录库 + `outputs/` 中的模型与结果文件
            - 同步流程：本地开发 -> GitHub -> Windows `git pull`
            """
        )
    with right:
        st.subheader("当前状态")
        if bundle:
            st.success(f"已激活模型：`{bundle.model_name}`")
            st.write(f"Checkpoint: `{bundle.checkpoint_path}`")
            st.write(f"数据目录：`{bundle.config['data']['root']}`")
            st.write(f"运行设备：`{bundle.device}`")
        else:
            st.warning("尚未选择有效 checkpoint。你仍然可以浏览数据集概览、训练看板和历史记录页面。")


def render_image_page() -> None:
    initialize_page("单图识别", "🖼️")
    context = sidebar_model_context(require_model=True)
    bundle = get_active_bundle(context)

    st.title("单图识别")
    st.caption("上传一张中药图片，输出 Top-K 预测、耗时与结果导出。")

    if bundle is None:
        st.warning("请先在左侧选择有效的 checkpoint。")
        return

    with st.form("image-recognition-form"):
        uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp", "webp"])
        top_k = st.slider("Top-K", min_value=1, max_value=min(10, len(bundle.class_names)), value=5)
        submitted = st.form_submit_button("开始识别")

    if submitted:
        if uploaded_file is None:
            st.error("请先上传图片。")
        else:
            with st.spinner("正在进行图片识别..."):
                payload = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(payload)).convert("RGB")
                result = predict_pil_image(bundle, image, top_k=top_k)
                rows = [
                    {
                        "rank": index,
                        "class_name": item["class_name"],
                        "probability": item["probability"],
                    }
                    for index, item in enumerate(result["predictions"], start=1)
                ]
                csv_bytes = _write_records_to_csv(rows)
                export_path = _save_export_bytes(
                    default_export_name("image_result", uploaded_file.name),
                    csv_bytes,
                )
                insert_history_record(
                    DEFAULT_DB_PATH,
                    created_at=_utc_now_text(),
                    record_type="image",
                    input_name=uploaded_file.name,
                    checkpoint_path=bundle.checkpoint_path,
                    model_name=bundle.model_name,
                    summary=f"Top1: {result['top1_class']} ({result['top1_probability']:.4f})",
                    output_path=str(export_path),
                    duration_seconds=result["duration_seconds"],
                    metadata={
                        "predictions": result["predictions"],
                        "export_path": str(export_path),
                    },
                )
                st.session_state["image_recognition_result"] = {
                    "image_bytes": payload,
                    "rows": rows,
                    "top1_class": result["top1_class"],
                    "top1_probability": result["top1_probability"],
                    "duration_seconds": result["duration_seconds"],
                    "csv_bytes": csv_bytes,
                    "export_path": str(export_path),
                }

    result = st.session_state.get("image_recognition_result")
    if result:
        left, right = st.columns([1, 1.1])
        with left:
            st.image(result["image_bytes"], caption="上传图片", use_container_width=True)
        with right:
            metric_cols = st.columns(3)
            metric_cols[0].metric("Top1 类别", result["top1_class"])
            metric_cols[1].metric("Top1 概率", f"{result['top1_probability']:.4f}")
            metric_cols[2].metric("推理耗时", f"{result['duration_seconds']:.4f}s")
            st.dataframe(pd.DataFrame(result["rows"]), use_container_width=True)
            st.download_button(
                "下载识别结果 CSV",
                data=result["csv_bytes"],
                file_name=Path(result["export_path"]).name,
                mime="text/csv",
            )
            st.caption(f"结果文件已保存到：`{result['export_path']}`")


def render_batch_page() -> None:
    initialize_page("批量图片识别", "🗂️")
    context = sidebar_model_context(require_model=True)
    bundle = get_active_bundle(context)

    st.title("批量图片识别")
    st.caption("一次上传多张图片，输出明细结果、汇总统计和 CSV 导出。")

    if bundle is None:
        st.warning("请先在左侧选择有效的 checkpoint。")
        return

    with st.form("batch-recognition-form"):
        uploaded_files = st.file_uploader(
            "批量上传图片",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
        )
        top_k = st.slider("Top-K", min_value=1, max_value=min(10, len(bundle.class_names)), value=3, key="batch_topk")
        submitted = st.form_submit_button("开始批量识别")

    if submitted:
        if not uploaded_files:
            st.error("请至少上传一张图片。")
        else:
            with st.spinner("正在进行批量识别..."):
                uploads = [(file.name, file.getvalue()) for file in uploaded_files]
                result = predict_uploaded_images(bundle, uploads, top_k=top_k)
                csv_bytes = _write_records_to_csv(result["rows"])
                export_path = _save_export_bytes(
                    default_export_name("batch_result", f"{len(uploaded_files)}_images"),
                    csv_bytes,
                )
                summary = (
                    f"共 {len(result['rows'])} 张，Top1 最多为 {result['dominant_class']} "
                    f"({result['dominant_count']} 张)"
                )
                insert_history_record(
                    DEFAULT_DB_PATH,
                    created_at=_utc_now_text(),
                    record_type="batch_image",
                    input_name=f"{len(uploaded_files)} images",
                    checkpoint_path=bundle.checkpoint_path,
                    model_name=bundle.model_name,
                    summary=summary,
                    output_path=str(export_path),
                    duration_seconds=sum(row["duration_seconds"] for row in result["rows"]),
                    metadata={
                        "rows": result["rows"],
                        "counts": result["counts"],
                        "export_path": str(export_path),
                    },
                )
                st.session_state["batch_recognition_result"] = {
                    "rows": result["rows"],
                    "counts": result["counts"],
                    "dominant_class": result["dominant_class"],
                    "dominant_count": result["dominant_count"],
                    "csv_bytes": csv_bytes,
                    "export_path": str(export_path),
                }

    result = st.session_state.get("batch_recognition_result")
    if result:
        summary_df = _counts_dataframe(result["counts"])
        top_row = result["rows"][0] if result["rows"] else None
        metrics = st.columns(3)
        metrics[0].metric("图片数量", len(result["rows"]))
        metrics[1].metric("主类别", result["dominant_class"])
        metrics[2].metric("主类别张数", result["dominant_count"])

        chart_col, table_col = st.columns([0.9, 1.1])
        with chart_col:
            if not summary_df.empty:
                st.plotly_chart(
                    px.bar(summary_df, x="class_name", y="count", title="批量识别 Top1 统计"),
                    use_container_width=True,
                )
        with table_col:
            st.dataframe(pd.DataFrame(result["rows"]), use_container_width=True)

        st.download_button(
            "下载批量识别 CSV",
            data=result["csv_bytes"],
            file_name=Path(result["export_path"]).name,
            mime="text/csv",
        )
        if top_row:
            st.caption(
                f"第一张图片结果：`{top_row['filename']}` -> `{top_row['top1_class']}` "
                f"({top_row['top1_probability']:.4f})"
            )


def render_video_page() -> None:
    initialize_page("视频识别", "🎬")
    context = sidebar_model_context(require_model=True)
    bundle = get_active_bundle(context)

    st.title("视频识别")
    st.caption("上传视频后按秒抽帧分类，输出逐帧结果、主类别统计和可下载明细。")

    if bundle is None:
        st.warning("请先在左侧选择有效的 checkpoint。")
        return

    with st.form("video-recognition-form"):
        uploaded_video = st.file_uploader("上传视频", type=["mp4", "avi", "mov", "mkv"])
        sample_interval = st.number_input("抽帧间隔（秒）", min_value=0.2, max_value=10.0, value=1.0, step=0.2)
        max_frames = st.slider("最多抽取帧数", min_value=10, max_value=180, value=60, step=10)
        top_k = st.slider("Top-K", min_value=1, max_value=min(10, len(bundle.class_names)), value=3, key="video_topk")
        submitted = st.form_submit_button("开始视频识别")

    if submitted:
        if uploaded_video is None:
            st.error("请先上传视频。")
        else:
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(uploaded_video.getvalue())

            try:
                with st.spinner("正在抽帧并进行视频识别..."):
                    result = predict_video_frames(
                        bundle,
                        temp_path,
                        top_k=top_k,
                        sample_interval_seconds=sample_interval,
                        max_frames=max_frames,
                    )
                csv_bytes = _write_records_to_csv(result["rows"])
                export_path = _save_export_bytes(
                    default_export_name("video_result", uploaded_video.name),
                    csv_bytes,
                )
                summary = (
                    f"采样 {result['sampled_frames']} 帧，主类别 {result['dominant_class']} "
                    f"({result['dominant_count']} 帧)"
                )
                insert_history_record(
                    DEFAULT_DB_PATH,
                    created_at=_utc_now_text(),
                    record_type="video",
                    input_name=uploaded_video.name,
                    checkpoint_path=bundle.checkpoint_path,
                    model_name=bundle.model_name,
                    summary=summary,
                    output_path=str(export_path),
                    duration_seconds=result["processing_duration_seconds"],
                    metadata={
                        "rows": result["rows"],
                        "counts": result["counts"],
                        "sampled_frames": result["sampled_frames"],
                        "frame_interval": result["frame_interval"],
                        "video_duration_seconds": result["video_duration_seconds"],
                        "export_path": str(export_path),
                    },
                )
                st.session_state["video_recognition_result"] = {
                    "video_bytes": uploaded_video.getvalue(),
                    "rows": result["rows"],
                    "counts": result["counts"],
                    "dominant_class": result["dominant_class"],
                    "dominant_count": result["dominant_count"],
                    "sampled_frames": result["sampled_frames"],
                    "video_duration_seconds": result["video_duration_seconds"],
                    "processing_duration_seconds": result["processing_duration_seconds"],
                    "csv_bytes": csv_bytes,
                    "export_path": str(export_path),
                }
            finally:
                if temp_path.exists():
                    temp_path.unlink()

    result = st.session_state.get("video_recognition_result")
    if result:
        st.video(result["video_bytes"])
        metric_cols = st.columns(4)
        metric_cols[0].metric("采样帧数", result["sampled_frames"])
        metric_cols[1].metric("主类别", result["dominant_class"])
        metric_cols[2].metric("视频时长", f"{result['video_duration_seconds']:.2f}s")
        metric_cols[3].metric("处理耗时", f"{result['processing_duration_seconds']:.2f}s")

        summary_df = _counts_dataframe(result["counts"])
        chart_col, table_col = st.columns([0.8, 1.2])
        with chart_col:
            if not summary_df.empty:
                st.plotly_chart(
                    px.bar(summary_df, x="class_name", y="count", title="视频抽帧 Top1 统计"),
                    use_container_width=True,
                )
        with table_col:
            st.dataframe(pd.DataFrame(result["rows"]), use_container_width=True)
        st.download_button(
            "下载视频识别 CSV",
            data=result["csv_bytes"],
            file_name=Path(result["export_path"]).name,
            mime="text/csv",
        )


def render_dataset_page() -> None:
    initialize_page("数据集概览", "📊")
    context = sidebar_model_context(require_model=False)
    bundle = get_active_bundle(context) if context["checkpoint_path"] else None

    data_root = _current_data_root(bundle)
    st.title("数据集概览")
    st.caption(f"当前展示数据目录：`{data_root}`")

    try:
        overview = cached_dataset_overview(data_root)
    except Exception as exc:
        st.error(f"无法读取数据集：{exc}")
        return

    metrics = st.columns(4)
    metrics[0].metric("类别数", overview["class_count"])
    metrics[1].metric("训练集", overview["split_counts"].get("train", 0))
    metrics[2].metric("验证集", overview["split_counts"].get("val", 0))
    metrics[3].metric("测试集", overview["split_counts"].get("test", 0))

    split_df = pd.DataFrame(
        [{"split": split_name, "image_count": image_count} for split_name, image_count in overview["split_counts"].items()]
    )
    if not split_df.empty:
        st.plotly_chart(
            px.bar(split_df, x="split", y="image_count", color="split", title="训练/验证/测试样本分布"),
            use_container_width=True,
        )

    class_df = pd.DataFrame(overview["per_class_rows"])
    if not class_df.empty:
        st.subheader("类别分布表")
        st.dataframe(class_df, use_container_width=True, height=320)

    if overview["sample_images"]:
        st.subheader("样例图片")
        sample_cols = st.columns(3)
        for index, item in enumerate(overview["sample_images"]):
            with sample_cols[index % 3]:
                st.image(item["path"], caption=f"{item['split']} / {item['class_name']}", use_container_width=True)


def render_dashboard_page() -> None:
    initialize_page("训练结果看板", "📈")
    sidebar_model_context(require_model=False)

    st.title("训练结果看板")
    runs = cached_training_runs()
    if not runs:
        st.warning("当前 `outputs/` 下还没有可用的训练实验。")
        return

    run_options = {run["run_name"]: run for run in runs}
    selected_run_name = st.selectbox("选择实验", list(run_options.keys()))
    selected_run = run_options[selected_run_name]
    artifacts = cached_run_artifacts(selected_run["run_dir"])

    history_df = pd.DataFrame(artifacts["history"])
    test_metrics = artifacts["test_metrics"]
    resolved_config = artifacts["resolved_config"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("模型", selected_run["model_name"])
    metric_cols[1].metric("已完成 Epoch", selected_run["epochs_completed"])
    metric_cols[2].metric("最佳验证准确率", f"{selected_run['best_val_accuracy']:.4f}")
    metric_cols[3].metric(
        "测试准确率",
        f"{test_metrics.get('accuracy', 0.0):.4f}" if test_metrics else "N/A",
    )

    if not history_df.empty:
        loss_col, acc_col = st.columns(2)
        with loss_col:
            loss_df = history_df.melt(
                id_vars=["epoch"],
                value_vars=["train_loss", "val_loss"],
                var_name="metric",
                value_name="value",
            )
            st.plotly_chart(
                px.line(loss_df, x="epoch", y="value", color="metric", title="训练/验证 Loss 曲线"),
                use_container_width=True,
            )
        with acc_col:
            acc_df = history_df.melt(
                id_vars=["epoch"],
                value_vars=["train_accuracy", "val_accuracy"],
                var_name="metric",
                value_name="value",
            )
            st.plotly_chart(
                px.line(acc_df, x="epoch", y="value", color="metric", title="训练/验证 Accuracy 曲线"),
                use_container_width=True,
            )
        st.dataframe(history_df.tail(10), use_container_width=True)

    with st.expander("查看实验配置"):
        st.json(resolved_config)


def render_history_page() -> None:
    initialize_page("识别历史", "🕘")
    sidebar_model_context(require_model=False)

    st.title("识别历史记录")
    stats = fetch_history_stats(DEFAULT_DB_PATH)
    metric_cols = st.columns(4)
    metric_cols[0].metric("总记录数", stats["total_records"])
    metric_cols[1].metric("单图识别", stats["by_type"].get("image", 0))
    metric_cols[2].metric("批量识别", stats["by_type"].get("batch_image", 0))
    metric_cols[3].metric("视频识别", stats["by_type"].get("video", 0))

    record_type = st.selectbox("筛选类型", options=["全部", "image", "batch_image", "video"])
    records = fetch_history_records(
        DEFAULT_DB_PATH,
        limit=300,
        record_type=None if record_type == "全部" else record_type,
    )
    if not records:
        st.info("还没有识别历史记录。")
        return

    dataframe = pd.DataFrame(
        [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "record_type": row["record_type"],
                "input_name": row["input_name"],
                "model_name": row["model_name"],
                "summary": row["summary"],
                "duration_seconds": row["duration_seconds"],
                "output_path": row["output_path"],
            }
            for row in records
        ]
    )
    st.dataframe(dataframe, use_container_width=True, height=360)

    selected_id = st.selectbox("查看详情记录 ID", options=[row["id"] for row in records])
    selected_record = next(row for row in records if row["id"] == selected_id)
    st.json(selected_record["metadata"])


def render_system_page() -> None:
    initialize_page("系统信息", "⚙️")
    context = sidebar_model_context(require_model=False)

    st.title("系统信息")
    info_cols = st.columns(4)
    info_cols[0].metric("Python", platform.python_version())
    info_cols[1].metric("操作系统", platform.system())
    info_cols[2].metric("数据库文件", DEFAULT_DB_PATH.name)
    info_cols[3].metric("导出目录", EXPORT_ROOT.name)

    torch_info = {"version": "未安装", "cuda": "unknown"}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda": str(torch.cuda.is_available()),
        }
    except Exception:
        pass

    st.subheader("运行时")
    st.json(
        {
            "python_executable": sys.executable,
            "torch_version": torch_info["version"],
            "cuda_available": torch_info["cuda"],
            "history_db": str(DEFAULT_DB_PATH),
            "export_root": str(EXPORT_ROOT),
        }
    )

    discovered = cached_checkpoints()
    st.subheader("已发现模型")
    if discovered:
        rows = []
        for checkpoint in discovered:
            metadata = read_checkpoint_metadata(checkpoint)
            rows.append(
                {
                    "checkpoint_path": checkpoint,
                    "model_name": metadata["model_name"],
                    "class_count": len(metadata["class_names"]),
                    "image_size": metadata["image_size"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("当前没有发现本地 checkpoint。")

    st.subheader("训练实验")
    runs = cached_training_runs()
    if runs:
        st.dataframe(pd.DataFrame(runs), use_container_width=True)
    else:
        st.info("当前没有可展示的训练实验。")
