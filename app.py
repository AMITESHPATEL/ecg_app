"""
ECG Annotation Application (Streamlit edition)
================================================

A lightweight Streamlit tool that lets a clinician review ECG rhythm segments
(one 10-second Lead II strip at a time) and correct mislabeled rhythm
annotations (VT / SVT / Others), with comments, autosave, and resume support.

This is a Streamlit port of the original Gradio app. Behavior is kept as
close as possible to the original:
  - Same dataset acquisition (download from Google Drive if not cached).
  - Same annotations.csv / state.json layout, so existing annotation
    progress from the Gradio version is picked up automatically.
  - Same navigation model (Previous / Next, autosave on every change).
  - Best-effort keyboard shortcuts (ArrowLeft/ArrowRight, 1/2/3). These rely
    on locating Streamlit's rendered buttons/checkboxes by their visible
    text inside the parent document, which is a common trick but is not an
    officially supported Streamlit API -- if a future Streamlit version
    changes its internal markup, shortcuts may stop working. Everything
    else in the app works fine without them; disable via the sidebar toggle
    if they misbehave.

Run:
    streamlit run ecg_annotation_app_streamlit.py

Env vars (same as the Gradio version):
    ECG_DATASET_PATH   local cache directory for the dataset (default ./test_ds)
    ECG_GDRIVE_URL     Google Drive folder URL to download from if not cached

See README.md for full instructions.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gdown
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from datasets import load_from_disk
from azure_sync import (
    download_blob_from_azure,
    get_container_client,
    upload_blob_to_azure,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# Default source of the dataset: a Google Drive folder shared as
# "Anyone with the link" -> "Viewer". Overridable via env var.
DEFAULT_GDRIVE_URL = (
    "https://drive.google.com/drive/u/1/folders/1CZK9OQzsIM0cBwWeZ9RtIdG7LmebpBr9"
)

# Rhythm columns present in the source dataset (used to compute ground truth).
GROUND_TRUTH_COLUMNS = ["VT", "SVT", "AFIB", "AFLT"]

# Labels the clinician can assign. Order here defines the "1 / 2 / 3" shortcuts.
DOCTOR_LABELS = ["VT", "SVT", "Others"]

CSV_COLUMNS = [
    "sample_index",
    "dataset",
    "record",
    "ground_truth",
    "doctor_VT",
    "doctor_SVT",
    "doctor_Others",
    "comments",
    "reviewed",
    "annotated_at",
]

CUSTOM_CSS = """
<style>
.block-container { max-width: 1500px !important; padding-top: 1.5rem !important; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
div[data-testid="stCheckbox"] label p { font-size: 1.05em; }
</style>
"""

# JS injected via a zero-height component. Best-effort keyboard shortcuts --
# see module docstring for caveats.
KEYBOARD_JS = """
<script>
(function () {
  const targetDoc = window.parent.document;
  if (targetDoc.__ecgShortcutsAttached) return;
  targetDoc.__ecgShortcutsAttached = true;

  function clickButtonByText(text) {
    const buttons = targetDoc.querySelectorAll("button");
    for (const btn of buttons) {
      if (btn.innerText.trim() === text) { btn.click(); return true; }
    }
    return false;
  }

  function clickCheckboxByLabel(label) {
    const labels = targetDoc.querySelectorAll('div[data-testid="stCheckbox"] label');
    for (const lbl of labels) {
      if (lbl.innerText.trim() === label) { lbl.click(); return true; }
    }
    return false;
  }

  targetDoc.addEventListener("keydown", function (e) {
    const active = targetDoc.activeElement;
    const tag = active ? active.tagName.toLowerCase() : "";
    // Never hijack keys while the user is typing (comments box, etc.)
    if (tag === "textarea" || tag === "input") return;

    if (e.key === "ArrowRight") {
      clickButtonByText("Next \\u25B6");
    } else if (e.key === "ArrowLeft") {
      clickButtonByText("\\u25C0 Previous");
    } else if (e.key === "1") {
      clickCheckboxByLabel("VT");
    } else if (e.key === "2") {
      clickCheckboxByLabel("SVT");
    } else if (e.key === "3") {
      clickCheckboxByLabel("Others");
    }
  });
})();
</script>
"""


# --------------------------------------------------------------------------
# Dataset acquisition (download from Google Drive, then cache locally)
# --------------------------------------------------------------------------

# Files that mark the root of an HF `save_to_disk()` dataset directory.
_DATASET_MARKER_FILES = {"dataset_info.json", "dataset_dict.json", "state.json"}


def _is_dataset_root(path: Path) -> bool:
    return any((path / marker).exists() for marker in _DATASET_MARKER_FILES)


def _find_dataset_root(path: Path) -> Path:
    """
    Google Drive folder downloads sometimes wrap the dataset in an extra
    directory level (e.g. downloading folder "test_ds" produces
    `<output>/test_ds/dataset_info.json` instead of `<output>/dataset_info.json`).
    Walk down while there's exactly one subdirectory and no dataset marker
    file at the current level, so `load_from_disk` gets the right path either way.
    """
    current = path
    while not _is_dataset_root(current):
        subdirs = [p for p in current.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            current = subdirs[0]
        else:
            break  # ambiguous or already at the right level; let load_from_disk report errors
    return current


def ensure_dataset_available(local_path: Path, gdrive_url: str | None) -> Path:
    """
    Return a local path to the dataset, downloading it from Google Drive first
    if it isn't already cached on disk. This lets the app fetch the dataset at
    runtime instead of committing large binary files to version control.

    The Google Drive folder must be shared as "Anyone with the link" -> "Viewer".
    """
    if local_path.exists() and any(local_path.iterdir()):
        return _find_dataset_root(local_path)  # already downloaded in a prior run

    if not gdrive_url:
        raise FileNotFoundError(
            f"No dataset found at '{local_path}' and no gdrive URL was provided."
        )

    st.write(f"Dataset not found at '{local_path}' -- downloading from Google Drive...")
    local_path.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(url=gdrive_url, output=str(local_path), quiet=False, use_cookies=False)

    root = _find_dataset_root(local_path)
    if not _is_dataset_root(root):
        raise FileNotFoundError(
            f"Downloaded files from Google Drive but couldn't find a dataset at "
            f"'{root}'. Check that the shared folder contains the dataset produced "
            f"by `save_to_disk()` (it should have a dataset_info.json / state.json)."
        )
    return root


# --------------------------------------------------------------------------
# Data / persistence helpers
# --------------------------------------------------------------------------

def ground_truth_labels(row: dict[str, Any]) -> list[str]:
    """Return the list of positive rhythm columns for a dataset row."""
    return [col for col in GROUND_TRUTH_COLUMNS if int(row.get(col, 0) or 0) == 1]


def init_annotations(dataset, csv_path: Path) -> pd.DataFrame:
    """Load an existing annotation CSV, or create a fresh one for the dataset."""
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if len(df) == len(dataset):
            df["reviewed"] = df["reviewed"].astype(bool)
            for label in DOCTOR_LABELS:
                df[f"doctor_{label}"] = df[f"doctor_{label}"].astype(bool)
            df["comments"] = df["comments"].fillna("").astype(str)
            return df
        # Dataset size changed since the CSV was created -- rebuild to stay safe.

    rows = []
    for i in range(len(dataset)):
        row = dataset[i]
        rows.append(
            {
                "sample_index": i,
                "dataset": row.get("dataset", ""),
                "record": row.get("record", ""),
                "ground_truth": ",".join(ground_truth_labels(row)),
                "doctor_VT": False,
                "doctor_SVT": False,
                "doctor_Others": False,
                "comments": "",
                "reviewed": False,
                "annotated_at": "",
            }
        )
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(csv_path, index=False)
    return df


def save_annotations(df: pd.DataFrame, csv_path: Path) -> None:
    df.to_csv(csv_path, index=False)


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(state_path: Path, sample_index: int) -> None:
    state = {
        "last_opened_sample": sample_index,
        "last_saved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    state_path.write_text(json.dumps(state, indent=2))


def first_unreviewed_index(df: pd.DataFrame) -> int:
    """Index of the first sample not yet reviewed, or 0 if all are done."""
    unreviewed = df.index[~df["reviewed"].astype(bool)]
    return int(unreviewed[0]) if len(unreviewed) else 0


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------

def build_ecg_figure(signal: np.ndarray, fs: float, dataset_name: str, record: str) -> go.Figure:
    """Build an interactive, theme-agnostic Plotly strip of a Lead II signal."""
    t = np.arange(len(signal)) / fs if fs else np.arange(len(signal))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=signal,
            mode="lines",
            line=dict(width=1.4, color="#e63946"),
            name="Lead II",
            hovertemplate="t=%{x:.2f}s<br>amp=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{dataset_name} — Record {record} (Lead II)",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (mV)",
        margin=dict(l=55, r=20, t=36, b=36),
        height=360,
        dragmode="pan",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#888888"),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.25)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.25)", zeroline=False)
    return fig


def render_meta_line(idx: int, total: int, row: dict) -> str:
    """Single-line metadata bar: sample counter, dataset, record, HR (ground truth hidden for blind review)."""
    hr = row.get("HR", "?")
    fields = [
        f"<b>Sample</b> {idx + 1} / {total}",
        f"<b>Dataset:</b> {row.get('dataset', '')}",
        f"<b>Record:</b> {row.get('record', '')}",
        f"<b>Heart Rate:</b> {hr} bpm",
    ]
    items = "".join(f'<span style="white-space:nowrap;">{f}</span>' for f in fields)
    return (
        '<div style="font-size:1.05em;padding:8px 12px;border-radius:8px;'
        'background:rgba(128,128,128,0.08);margin-bottom:4px;display:flex;'
        f'flex-wrap:wrap;gap:24px;align-items:center;">{items}</div>'
    )


def render_progress_caption(df: pd.DataFrame) -> str:
    total = len(df)
    reviewed = int(df["reviewed"].astype(bool).sum())
    pct = (reviewed / total * 100) if total else 0.0
    return f"Reviewed {reviewed} / {total} ({pct:.1f}%)"


# --------------------------------------------------------------------------
# Cached, expensive, process-wide resources
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading ECG dataset (downloads once, then cached)...")
def get_dataset():
    dataset_root = ensure_dataset_available(
        Path(os.environ.get("ECG_DATASET_PATH", "./test_ds")),
        os.environ.get("ECG_GDRIVE_URL", DEFAULT_GDRIVE_URL),
    )
    return load_from_disk(str(dataset_root))


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

st.set_page_config(page_title="ECG Annotation Review", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

dataset = get_dataset()
total = len(dataset)

data_dir = Path("./annotation_data")
data_dir.mkdir(parents=True, exist_ok=True)
csv_path = data_dir / "annotations.csv"
state_path = data_dir / "state.json"

# Initialize Azure Blob Storage container client
if "azure_client" not in st.session_state:
    st.session_state.azure_client = get_container_client()

# On startup, download existing annotations.csv from Azure Blob Storage if available
if "azure_initialized" not in st.session_state:
    st.session_state.azure_initialized = True
    if st.session_state.azure_client:
        downloaded = download_blob_from_azure(
            st.session_state.azure_client, "annotations.csv", csv_path
        )
        if not downloaded and csv_path.exists():
            upload_blob_to_azure(st.session_state.azure_client, csv_path, "annotations.csv")

if "df" not in st.session_state:
    st.session_state.df = init_annotations(dataset, csv_path)

if "idx" not in st.session_state:
    saved_state = load_state(state_path)
    start_idx = saved_state.get("last_opened_sample")
    if start_idx is None or not (0 <= start_idx < total):
        start_idx = first_unreviewed_index(st.session_state.df)
    st.session_state.idx = start_idx
    st.session_state.jump_sample_input = start_idx + 1

if "jump_sample_input" not in st.session_state:
    st.session_state.jump_sample_input = st.session_state.idx + 1

df = st.session_state.df
idx = st.session_state.idx


def sync_to_azure(local_csv: Path) -> None:
    """Helper to sync local annotations.csv to Azure Blob Storage if configured."""
    client = st.session_state.get("azure_client")
    if client:
        try:
            upload_blob_to_azure(client, local_csv, "annotations.csv")
        except Exception:
            pass


def persist_sample(sample_idx: int) -> None:
    """Write the current widget state for `sample_idx` into df, save to disk, and sync to Azure."""
    df = st.session_state.df
    mask = df["sample_index"] == sample_idx
    checked = [
        label for label in DOCTOR_LABELS
        if st.session_state.get(f"chk_{label}_{sample_idx}", False)
    ]
    comment = st.session_state.get(f"comment_{sample_idx}", "")
    for label in DOCTOR_LABELS:
        df.loc[mask, f"doctor_{label}"] = label in checked
    df.loc[mask, "comments"] = comment
    df.loc[mask, "reviewed"] = True
    df.loc[mask, "annotated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_annotations(df, csv_path)
    save_state(state_path, sample_idx)
    sync_to_azure(csv_path)


def on_jump_change() -> None:
    val = st.session_state.get("jump_sample_input", st.session_state.idx + 1)
    persist_sample(st.session_state.idx)
    target_idx = max(0, min(int(val) - 1, total - 1))
    st.session_state.idx = target_idx
    st.session_state.jump_sample_input = target_idx + 1
    save_state(state_path, target_idx)


def go_to(offset: int) -> None:
    persist_sample(st.session_state.idx)  # save whatever is on screen before leaving it
    new_idx = min(max(st.session_state.idx + offset, 0), total - 1)
    st.session_state.idx = new_idx
    st.session_state.jump_sample_input = new_idx + 1
    save_state(state_path, new_idx)


# -- Sidebar ----------------------------------------------------------------

with st.sidebar:
    st.subheader("Navigation")
    st.number_input(
        "Jump to sample #",
        min_value=1,
        max_value=total,
        key="jump_sample_input",
        on_change=on_jump_change,
        step=1,
    )
    if st.button("Go", use_container_width=True):
        on_jump_change()
        st.rerun()

    st.divider()
    if st.session_state.get("azure_client"):
        st.caption("☁ **Azure Blob Storage:** Connected & autosyncing")
    else:
        st.caption("💾 **Storage:** Local cache mode")

    st.caption(f"Annotations file: `{csv_path}`")
    st.caption(f"Resume state file: `{state_path}`")
    shortcuts_enabled = st.toggle("Enable keyboard shortcuts (experimental)", value=True)

# -- Main content -------------------------------------------------------

row = dataset[idx]
signal = np.asarray(row["II"], dtype=float)
fs = float(row.get("fs", 500) or 500)

st.markdown(
    render_meta_line(idx, total, row), unsafe_allow_html=True
)

fig = build_ecg_figure(signal, fs, row.get("dataset", ""), row.get("record", ""))
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

col_labels, col_comments = st.columns([2, 3])

with col_labels:
    st.markdown("**Doctor Annotation**  (shortcuts: 1 / 2 / 3)")
    ann_row = df.loc[df["sample_index"] == idx].iloc[0]
    label_cols = st.columns(len(DOCTOR_LABELS))
    for col, label in zip(label_cols, DOCTOR_LABELS):
        with col:
            st.checkbox(
                label,
                value=bool(ann_row[f"doctor_{label}"]),
                key=f"chk_{label}_{idx}",
                on_change=persist_sample,
                args=(idx,),
            )

with col_comments:
    comment_value = ann_row["comments"]
    comment_value = "" if pd.isna(comment_value) else str(comment_value)
    st.text_area(
        "Comments",
        value=comment_value,
        key=f"comment_{idx}",
        height=68,
        placeholder="Optional notes for this sample...",
        on_change=persist_sample,
        args=(idx,),
    )

nav_prev, nav_progress, nav_next = st.columns([1, 4, 1])
with nav_prev:
    st.button("◀ Previous", on_click=go_to, args=(-1,), disabled=(idx <= 0), use_container_width=True)
with nav_progress:
    reviewed = int(df["reviewed"].astype(bool).sum())
    st.progress(reviewed / total if total else 0.0, text=render_progress_caption(df))
with nav_next:
    st.button("Next ▶", on_click=go_to, args=(1,), disabled=(idx >= total - 1), type="primary", use_container_width=True)

if shortcuts_enabled:
    components.html(KEYBOARD_JS, height=0)