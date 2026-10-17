"""DeepGuard research dashboard. Run from the repository root."""
from pathlib import Path
import json
import tempfile
import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from deepguard.utils import config, seed_all
from deepguard.inference import Analyzer
from deepguard.reporting import html_report

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="DeepGuard AI · Forensic Research", page_icon="◈", layout="wide")
st.markdown(f"<style>{(ROOT/'app/styles/theme.css').read_text()}</style>", unsafe_allow_html=True)
cfg = config(ROOT/"configs/default.yaml")
seed_all(cfg["seed"])


@st.cache_resource
def analyzer(version: tuple) -> Analyzer:
    asset = ROOT/"checkpoints/face_landmarker.task"
    return Analyzer(cfg, str(ROOT/"checkpoints"), str(asset) if asset.exists() else None)


st.sidebar.markdown("## ◈ DeepGuard AI")
st.sidebar.caption("MULTIMODAL FORENSIC RESEARCH")
pages = ["Overview", "Analyze Video", "Detection Results", "Forensic Evidence", "Research Laboratory", "Dataset Explorer", "Model Information", "About and Documentation"]
page = st.sidebar.radio("Workspace", pages)
paths = list((ROOT/"checkpoints").glob("*.pt")) + list((ROOT/"checkpoints").glob("*.json"))
version = tuple(sorted((str(p), p.stat().st_mtime_ns) for p in paths))
st.sidebar.caption(f"{len(list((ROOT/'checkpoints').glob('*.pt')))} modality checkpoint(s) available")
st.sidebar.info("Research use. Evidence can be incomplete or wrong. No guaranteed authenticity verdict.")

if page == "Overview":
    st.markdown('<div class="hero"><div class="eyebrow">Evidence before conclusions</div><h1>Inspect the face.<br>Listen to the voice.<br>Question the agreement.</h1><p>A transparent research workspace for visual, audio, temporal and lip-sync evidence. Every score must trace back to a trained model and a documented experiment.</p></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for c, title, text in zip(cols, ["Visual", "Audio", "Temporal", "Lip-sync"], ["EfficientNet face analysis", "Log-Mel CNN", "Embedding sequence LSTM", "Learned alignment evidence"]):
        c.metric(title, "Available" if (ROOT/f"checkpoints/{'sync' if title=='Lip-sync' else title.lower()}.pt").exists() else "Not trained")
        c.caption(text)
    st.subheader("A reproducible investigation")
    st.write("Validate media → extract evidence → gate quality → compare modalities → fuse calibrated evidence → report uncertainty.")
    st.warning("No trained checkpoint is bundled. Dataset access and held-out evaluation are pending. Engineering tests are not detection benchmarks.")

elif page == "Analyze Video":
    st.title("Analyze video")
    st.caption(f"Maximum {cfg['max_upload_mb']} MB · {cfg['max_duration']} seconds · up to 4K. Temporary disk media are deleted after each analysis.")
    upload = st.file_uploader("Upload authorized video", type=["mp4", "mov", "avi", "mkv", "webm"])
    if upload:
        if upload.size > cfg["max_upload_mb"]*1024**2:
            st.error("Upload exceeds the configured limit.")
        else:
            st.video(upload)
            if st.button("Inspect evidence", type="primary"):
                try:
                    st.session_state.pop("report", None)
                    st.session_state.pop("features", None)
                    with st.status("Validating and extracting media…", expanded=True) as status:
                        with tempfile.TemporaryDirectory(prefix="deepguard-upload-") as folder:
                            path = Path(folder)/("upload" + Path(upload.name).suffix.lower())
                            path.write_bytes(upload.getvalue())
                            st.write("Decoding frames and audio; checking face and signal quality.")
                            report, features = analyzer(version).analyze(str(path))
                        st.session_state.report = report
                        st.session_state.features = features
                        status.update(label="Analysis finished; temporary disk media deleted", state="complete")
                    st.info(f"{report['decision']} · {report['seconds']:.2f} seconds. Open Detection Results and Forensic Evidence.")
                except Exception as exc:
                    st.error(f"Analysis could not finish: {exc}")
    if st.button("Clear retained evidence"):
        st.session_state.pop("features", None)
        st.session_state.pop("report", None)
        st.success("Session evidence cleared.")

elif page == "Detection Results":
    st.title("Detection results")
    report = st.session_state.get("report")
    if report is None:
        st.info("Analyze a video to view actual evidence. No sample scores are displayed.")
    else:
        cols = st.columns(3)
        cols[0].metric("Decision", report["decision"])
        confidence = report["calibrated_confidence"]
        cols[1].metric("Calibrated confidence", f"{confidence:.1%}" if confidence is not None else "Unavailable")
        cols[2].metric("Analysis time", f"{report['seconds']:.2f} s")
        for reason in report["reasons"]:
            st.warning(reason)
        if report["probabilities"]:
            table = pd.DataFrame({"modality": list(report["probabilities"]), "evidence score": list(report["probabilities"].values())})
            st.plotly_chart(px.bar(table, x="modality", y="evidence score", range_y=[0,1], template="plotly_dark", color_discrete_sequence=["#60d8d0"]), use_container_width=True)
            st.caption("Modality scores are uncalibrated unless separately validated. Sync is mismatch evidence, not a fake probability.")
        st.write("Visual/audio disagreement:", report["disagreement"])
        st.download_button("Download HTML report", html_report(report), file_name="deepguard-report.html", mime="text/html")
        st.json(report)

elif page == "Forensic Evidence":
    st.title("Forensic evidence")
    f = st.session_state.get("features")
    if f is None:
        st.info("Analyze a video first. Media evidence stays in the current session until cleared.")
    else:
        if len(f["selected_frames"]):
            st.subheader("Sampled frames")
            st.image(list(f["selected_frames"][:4]), width=220)
        if len(f["faces"]):
            st.subheader("Tracked face crops")
            st.image(list(f["faces"][:6]), width=120)
            if "visual" in analyzer(version).models and st.button("Compute visual Grad-CAM"):
                import torch
                from deepguard.explain import gradcam
                face = torch.from_numpy(f["faces"][:1]).permute(0,3,1,2).float()/255
                cam = gradcam(analyzer(version).models["visual"], face)[0].numpy()
                st.plotly_chart(px.imshow(cam, color_continuous_scale="Magma", title="Visual attribution, not proof of manipulation"))
        if f["audio_mels"].size:
            st.plotly_chart(px.imshow(f["audio_mels"][0], origin="lower", aspect="auto", labels={"x":"10 ms audio hops", "y":"Mel band"}, title="Extracted log-Mel spectrogram", color_continuous_scale="Magma"), use_container_width=True)
        if len(f["sync_times"]):
            st.plotly_chart(px.scatter(x=f["sync_times"], y=np.ones(len(f["sync_times"])), labels={"x":"Video-relative timestamp (s)", "y":"Aligned window available"}), use_container_width=True)
            st.caption("Window alignment visualization; validated offset-score curves require a trained synchronizer.")
        else:
            st.info("Landmark-aligned mouth/audio windows unavailable. A licensed landmark asset and usable face/audio evidence are required.")

elif page == "Research Laboratory":
    st.title("Research laboratory")
    results = list((ROOT/"reports/results").rglob("metrics.json"))
    if not results:
        st.info("No held-out evaluation results exist. Train and evaluate before comparing models.")
    else:
        records = [{"experiment": str(p.parent.relative_to(ROOT)), **json.loads(p.read_text())} for p in results]
        table = pd.DataFrame(records)
        st.dataframe(table, use_container_width=True)
        st.caption("Compare only compatible datasets, labels and splits. Inspect provenance and coverage.")

elif page == "Dataset Explorer":
    st.title("Dataset explorer")
    st.warning("FakeAVCeleb access requires owner approval. No dataset is bundled or assumed.")
    manifests = list((ROOT/"data/manifests").glob("*.csv"))
    if manifests:
        selected = st.selectbox("Local manifest", manifests, format_func=lambda p:p.name)
        table = pd.read_csv(selected)
        st.dataframe(table, use_container_width=True)
        if "label" in table:
            st.plotly_chart(px.histogram(table, x="label", color="split" if "split" in table else None, template="plotly_dark"), use_container_width=True)
    else:
        st.info("No local manifests yet. See DATASET_GUIDE.md for the required metadata contract.")

elif page == "Model Information":
    st.title("Models and provenance")
    try:
        a = analyzer(version)
        if not a.models:
            st.warning("Model not trained. Predictions are unavailable.")
        for kind, meta in a.meta.items():
            with st.expander(kind):
                st.json({k:v for k,v in meta.items() if k != "state_dict"})
    except Exception as exc:
        st.error(f"Checkpoint validation failed: {exc}")
    st.write("Visual: EfficientNet-B0 · Audio: 2D CNN · Temporal: LSTM · Lip-sync: two-stream 3D CNN · Fusion: regularized logistic regression with quality and availability covariates.")

else:
    st.title("Methodology and documentation")
    docs = sorted(ROOT.glob("*.md")) + sorted((ROOT/"docs").glob("*.md"))
    selected = st.selectbox("Read a document", docs, format_func=lambda p:p.stem.replace("_", " ").title())
    st.markdown(selected.read_text())
    st.caption("Do not use this research project as the sole basis for financial, legal or identity decisions.")
