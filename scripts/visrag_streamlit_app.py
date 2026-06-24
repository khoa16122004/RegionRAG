from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st
from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset
from PIL import Image, ImageDraw

VISRAG_EVAL_REPOS = {
    "ArxivQA": "openbmb/VisRAG-Ret-Test-ArxivQA",
    "ChartQA": "openbmb/VisRAG-Ret-Test-ChartQA",
    "InfoVQA": "openbmb/VisRAG-Ret-Test-InfoVQA",
    "MP-DocVQA": "openbmb/VisRAG-Ret-Test-MP-DocVQA",
    "PlotQA": "openbmb/VisRAG-Ret-Test-PlotQA",
    "SlideVQA": "openbmb/VisRAG-Ret-Test-SlideVQA",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def list_local_visrag_dirs(root: Path) -> dict[str, str]:
    if not root.exists() or not root.is_dir():
        return {}

    result: dict[str, str] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.startswith("VisRAG-Ret-Test-"):
            short_name = child.name.replace("VisRAG-Ret-Test-", "")
            result[short_name] = str(child)

    if not result and root.name.startswith("VisRAG-Ret-Test-"):
        short_name = root.name.replace("VisRAG-Ret-Test-", "")
        result[short_name] = str(root)

    return result


def _to_dataset_dict(ds: Dataset | DatasetDict) -> DatasetDict:
    if isinstance(ds, DatasetDict):
        return ds
    return DatasetDict({"train": ds})


def _fallback_load_local(path: str) -> DatasetDict:
    base = Path(path)
    parquet_files = [str(p) for p in base.rglob("*.parquet")]
    jsonl_files = [str(p) for p in base.rglob("*.jsonl")]
    json_files = [str(p) for p in base.rglob("*.json")]
    csv_files = [str(p) for p in base.rglob("*.csv")]

    if parquet_files:
        return _to_dataset_dict(load_dataset("parquet", data_files={"data": parquet_files})["data"])
    if jsonl_files:
        return _to_dataset_dict(load_dataset("json", data_files={"data": jsonl_files})["data"])
    if json_files:
        return _to_dataset_dict(load_dataset("json", data_files={"data": json_files})["data"])
    if csv_files:
        return _to_dataset_dict(load_dataset("csv", data_files={"data": csv_files})["data"])

    raise FileNotFoundError(f"No parquet/json/jsonl/csv file found under: {path}")


@st.cache_data(show_spinner=False)
def load_hf_dataset(repo_id: str) -> DatasetDict:
    try:
        ds = load_dataset(repo_id)
        return _to_dataset_dict(ds)
    except ValueError as exc:
        if "Config name is missing" not in str(exc):
            raise

        config_names = get_dataset_config_names(repo_id)
        merged = DatasetDict()
        for config_name in config_names:
            config_ds = load_dataset(repo_id, config_name)
            if isinstance(config_ds, DatasetDict):
                for split_name, split_ds in config_ds.items():
                    merged[f"{config_name}/{split_name}"] = split_ds
            else:
                merged[config_name] = config_ds
        return merged


@st.cache_data(show_spinner=False)
def load_local_dataset(path: str) -> DatasetDict:
    try:
        ds = load_dataset(path)
        return _to_dataset_dict(ds)
    except Exception:
        try:
            config_names = get_dataset_config_names(path)
            merged = DatasetDict()
            for config_name in config_names:
                config_ds = load_dataset(path, config_name)
                if isinstance(config_ds, DatasetDict):
                    for split_name, split_ds in config_ds.items():
                        merged[f"{config_name}/{split_name}"] = split_ds
                else:
                    merged[config_name] = config_ds
            if len(merged) > 0:
                return merged
        except Exception:
            pass

        return _fallback_load_local(path)


def _image_from_value(value: Any, base_dir: Path | None = None) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value

    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            return Image.open(BytesIO(value["bytes"]))
        if "path" in value and value["path"]:
            img_path = Path(value["path"])
            if not img_path.is_absolute() and base_dir is not None:
                img_path = base_dir / img_path
            if img_path.exists() and img_path.suffix.lower() in IMAGE_EXTS:
                return Image.open(img_path)

    if isinstance(value, str):
        img_path = Path(value)
        if not img_path.is_absolute() and base_dir is not None:
            img_path = base_dir / img_path
        if img_path.exists() and img_path.suffix.lower() in IMAGE_EXTS:
            return Image.open(img_path)

    return None


def _extract_boxes(value: Any) -> list[list[float]]:
    boxes: list[list[float]] = []

    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(x, (int, float)) for x in value):
            return [[float(x) for x in value]]
        for item in value:
            boxes.extend(_extract_boxes(item))

    if isinstance(value, dict):
        for key in ("bbox", "box", "boxes", "bboxes"):
            if key in value:
                boxes.extend(_extract_boxes(value[key]))

    return boxes


def extract_images_and_boxes(row: dict[str, Any], base_dir: Path | None = None) -> tuple[list[Image.Image], list[list[float]]]:
    images: list[Image.Image] = []
    boxes: list[list[float]] = []

    for key, value in row.items():
        maybe_img = _image_from_value(value, base_dir=base_dir)
        if maybe_img is not None:
            images.append(maybe_img)

        key_lower = key.lower()
        if any(token in key_lower for token in ("bbox", "box", "region")):
            boxes.extend(_extract_boxes(value))

    return images, boxes


def _draw_boxes(image: Image.Image, boxes: list[list[float]], box_mode: str) -> Image.Image:
    if not boxes:
        return image

    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for i, box in enumerate(boxes):
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = box

        if max(box) <= 1.0:
            x1, x2 = x1 * w, x2 * w
            y1, y2 = y1 * h, y2 * h

        if box_mode == "xywh":
            x2 = x1 + x2
            y2 = y1 + y2
        elif box_mode == "auto" and (x2 <= x1 or y2 <= y1):
            x2 = x1 + x2
            y2 = y1 + y2

        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        color = (255, 0, 0) if i % 2 == 0 else (0, 255, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    return img


def value_to_text(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return str(value)


def main() -> None:
    st.set_page_config(page_title="VisRAG Eval Visualizer", layout="wide")
    st.title("VisRAG Eval Visualizer")
    st.caption("Visualize VisRAG evaluation samples with images, text fields, and optional bboxes.")

    with st.sidebar:
        st.header("Data Source")
        source_type = st.radio("Source", ["HuggingFace", "Local folder"], index=0)
        box_mode = st.selectbox("BBox format", ["auto", "xyxy", "xywh"], index=0)

        local_root_default = Path("data") / "VisRAG"
        local_root = Path(st.text_input("Local root", value=str(local_root_default)))

    ds_dict: DatasetDict | None = None
    base_dir: Path | None = None

    if source_type == "HuggingFace":
        repo_label = st.selectbox("Dataset", list(VISRAG_EVAL_REPOS.keys()), index=0)
        repo_id = VISRAG_EVAL_REPOS[repo_label]

        if st.button("Load dataset", type="primary"):
            with st.spinner(f"Loading {repo_id} ..."):
                ds_dict = load_hf_dataset(repo_id)
                st.session_state["loaded_ds"] = ds_dict
                st.session_state["loaded_base_dir"] = None
                st.session_state["loaded_name"] = repo_id
    else:
        local_map = list_local_visrag_dirs(local_root)
        if local_map:
            picked = st.selectbox("Local dataset", list(local_map.keys()), index=0)
            selected_path = local_map[picked]
        else:
            selected_path = st.text_input(
                "Local dataset path",
                value=str(local_root),
                help="Path to one dataset folder, e.g. data/VisRAG/VisRAG-Ret-Test-ArxivQA",
            )

        if st.button("Load dataset", type="primary"):
            with st.spinner(f"Loading local dataset from {selected_path} ..."):
                ds_dict = load_local_dataset(selected_path)
                st.session_state["loaded_ds"] = ds_dict
                st.session_state["loaded_base_dir"] = selected_path
                st.session_state["loaded_name"] = selected_path

    if "loaded_ds" in st.session_state:
        ds_dict = st.session_state["loaded_ds"]
        loaded_base = st.session_state.get("loaded_base_dir")
        base_dir = Path(loaded_base) if loaded_base else None
        st.success(f"Loaded: {st.session_state.get('loaded_name', 'dataset')}")

    if ds_dict is None:
        st.info("Choose source and click Load dataset.")
        return

    split_names = list(ds_dict.keys())
    split = st.selectbox("Split", split_names, index=0)
    ds = ds_dict[split]

    st.write(f"Rows: {len(ds):,}")
    st.write("Columns:", ", ".join(ds.column_names))

    searchable_cols = [
        c for c in ds.column_names if any(k in c.lower() for k in ("query", "question", "text", "answer"))
    ]
    selected_search_cols = st.multiselect(
        "Search in columns",
        options=searchable_cols if searchable_cols else ds.column_names,
        default=searchable_cols[:2] if searchable_cols else ds.column_names[:2],
    )
    keyword = st.text_input("Keyword filter (optional)").strip().lower()

    if keyword and selected_search_cols:
        matched_indices = []
        for i in range(len(ds)):
            row = ds[i]
            haystack = " ".join(value_to_text(row.get(col, "")) for col in selected_search_cols).lower()
            if keyword in haystack:
                matched_indices.append(i)
    else:
        matched_indices = list(range(len(ds)))

    if not matched_indices:
        st.warning("No sample matched the keyword.")
        return

    st.write(f"Matched rows: {len(matched_indices):,}")

    chosen_pos = st.slider("Sample position", min_value=0, max_value=len(matched_indices) - 1, value=0)
    row_idx = matched_indices[chosen_pos]
    row = ds[row_idx]

    left, right = st.columns([1, 1])

    with left:
        st.subheader(f"Sample #{row_idx}")
        images, boxes = extract_images_and_boxes(row, base_dir=base_dir)
        if images:
            show_boxes = st.checkbox("Overlay bboxes", value=True)
            for img_i, img in enumerate(images):
                out_img = _draw_boxes(img, boxes, box_mode) if show_boxes else img
                st.image(out_img, caption=f"Image {img_i}", use_container_width=True)
        else:
            st.info("No image field detected in this sample.")

    with right:
        st.subheader("Fields")
        for key in sorted(row.keys()):
            value = row[key]
            with st.expander(key, expanded=key.lower() in {"query", "question", "answer"}):
                if isinstance(value, str) and len(value) < 800:
                    st.write(value)
                else:
                    st.code(value_to_text(value), language="json")


if __name__ == "__main__":
    main()
