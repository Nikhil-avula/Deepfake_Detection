"""Accessible video-only research interface: plain language, real results, clear limits."""
from pathlib import Path
import json
import tempfile
import html
import logging
import traceback
import shutil
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from deepguard.utils import ROOT,config,seed_all
from deepguard.inference import Analyzer
from deepguard.reporting import html_report

st.set_page_config(page_title="DeepGuard · Video Check",page_icon="◈",layout="wide",initial_sidebar_state="expanded")
st.markdown(f"<style>{(ROOT/'app/styles/theme.css').read_text()}</style>",unsafe_allow_html=True)
cfg=config();seed_all(cfg["seed"],cfg["threads"])
MODEL=ROOT/"checkpoints/visual.json"
SUMMARY=ROOT/"reports/results/summary.json"


@st.cache_resource
def detector(version):
    return Analyzer(cfg,str(MODEL))


def go(page):st.session_state.page=page


def clear():
    for key in ["result","evidence","video_name"]:st.session_state.pop(key,None)
    st.session_state.upload_version=st.session_state.get("upload_version",0)+1


def summary():
    return json.loads(SUMMARY.read_text()) if SUMMARY.exists() and model_ready() else None


@st.cache_data
def model_configuration(version):
    try:return json.loads(MODEL.read_text()).get('config',{})
    except (OSError,ValueError):return {}


def model_ready():
    if not MODEL.exists():return False
    settings=model_configuration(MODEL.stat().st_mtime_ns)
    keys=['face_detector','frames','image_size','min_faces','min_face_fraction',
          'max_multiple_fraction','min_sharpness','dominant_face_area_ratio','video_sampler']
    return bool(settings) and all(settings.get(key)==cfg.get(key) for key in keys)


def show_result():
    result=st.session_state.get("result")
    if result is None:
        st.info("Your result will appear here after you check a video.")
        st.button("Choose a video",on_click=go,args=("Check a video",),type="primary")
        return
    title=result["decision"]
    descriptions={"Likely fake":"The model found visual patterns associated with manipulated faces in its small training dataset. Verify the source before drawing a conclusion.",
        "Setup required":"The application is missing a required program or its trained model file. This is a computer setup issue, not a judgment about your video. Follow the instructions below before trying again.",
        "Likely real":"The model found fewer of the visual patterns it learned to associate with manipulated faces. This does not prove the video is authentic.",
        "Not sure":"The model cannot give a clear answer for this video. Check the original source or use an independent verification method.",
        "Cannot analyze":"We could not obtain enough usable face samples from this video. The specific reason and next step appear below."}
    if title=="Not sure":
        descriptions[title]=("The video was analyzed, but its visual patterns differ from those this model learned. The model needs broader training and independent testing before it can reliably judge videos like this."
            if result.get('unfamiliar_features') else
            "The video was analyzed, but the model found mixed evidence for real and fake. Its score is in the uncertain range. Rechecking the same file will not make the model more certain.")
    style="fake" if title=="Likely fake" else "real" if title=="Likely real" else "unsure"
    st.markdown(f'<div class="decision {style}"><div class="eyebrow">Your video check</div><h2>{html.escape(title)}</h2><p>{descriptions[title]}</p></div>',unsafe_allow_html=True)
    for reason in result["reasons"]:st.warning(reason)
    if title=="Cannot analyze" and "frames_checked" in result:
        st.info(f"Decoded {result['frames_checked']} frames and found {result.get('faces_found',0)} face samples. The sampled frames below help show whether the person is too far away, turned away, obscured, or one of several similarly sized faces.")
    left,middle,right=st.columns(3)
    score=result.get("fake_score")
    left.metric("Model's fake-pattern score",f"{score:.0%}" if score is not None else "Not available")
    middle.metric("Face samples checked",result.get("faces_found",0))
    right.metric("Time taken",f"{result['seconds']:.1f} seconds")
    st.caption("A higher score means the model leans toward fake. This score is not the probability that a video is truly fake. Close or unfamiliar results are marked Not sure.")
    performance=summary()
    if performance:
        st.info(f"Pilot limitation: {performance['test_metrics']['accuracy']:.0%} accuracy on {performance['test_evaluated']} analyzable held-out videos. Treat this as a research result, not verification.")
    if result["membership"]=="training video":st.info("This is a training example. Use a held-out or independently sourced video for a meaningful test.")
    elif result["membership"]=="held-out test video":st.info("This exact file was reserved for testing and was not used to fit the classifier.")
    st.download_button("Download your report",html_report(result),file_name="deepguard-video-report.html",mime="text/html")
    media=st.session_state.get("evidence")
    if media is not None:
        with st.expander("See what the model checked",expanded=True):
            if len(media["faces"]):
                st.write("Face samples from your video")
                st.image(list(media["faces"][:6]),width=130)
            if len(media["frames"]):
                st.write("A few sampled video frames")
                st.image(list(media["frames"]),width=230)
            st.caption("These images show the visual input, not proof that any region was forged. When background people appear, only a clearly larger foreground face is checked; similarly sized faces are skipped.")
    with st.expander("Technical report"):
        st.json(result)


st.sidebar.markdown('<div class="brand">◈ Deep<span>Guard</span></div><div class="eyebrow">A clearer look at video</div>',unsafe_allow_html=True)
st.sidebar.write("")
pages=["Start","Check a video","Your result","How reliable is it?","Our dataset","About this project"]
page=st.sidebar.radio("Explore",pages,key="page")
st.sidebar.divider()
st.sidebar.caption("VISUAL DETECTION ONLY")
st.sidebar.write("Looks for learned patterns in faces. Sound is not used.")
st.sidebar.caption("Trained pilot available" if model_ready() else "Model update required" if MODEL.exists() else "Trained model file missing")
st.sidebar.button("Clear video and result",on_click=clear,use_container_width=True)

if page=="Start":
    st.markdown('<section class="hero"><div class="eyebrow">Video deepfake detection</div><h1>Does this video<br>deserve a second look?</h1><p>Upload a video to check for visual signs of a manipulated face. Get a plain-language result, see the face samples checked, and understand when the model is unsure.</p></section>',unsafe_allow_html=True)
    st.button("Check a video →",type="primary",on_click=go,args=("Check a video",))
    st.write("")
    for col,number,title,text in zip(st.columns(3),["01","02","03"],["Choose a video","We check the face","Understand the result"],["Use a short video with one clearly visible person.","The model checks face samples from across the video.","See Likely real, Likely fake, Not sure, or guidance if the video cannot be analyzed."]):
        col.markdown(f'<div class="feature"><div class="number">{number}</div><h3>{title}</h3><p>{text}</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="note">Built as a student research pilot. No automated result can guarantee that a video is real or fake.</div>',unsafe_allow_html=True)
    data=summary()
    if data:
        a,b,c,d=st.columns(4)
        a.metric("Videos provided",data["received_videos"])
        b.metric("Videos held out for testing",data["test_total"])
        c.metric("Test videos with usable faces",data["test_evaluated"])
        d.metric("Measured pilot accuracy",f"{data['test_metrics']['accuracy']:.0%}")
        st.caption("Held-out examples are kept separate from training. Open How reliable is it? to see actual results and limitations.")
    else:st.info("Training or evaluation is still in progress. Predictions become available only when a fitted model exists.")

elif page=="Check a video":
    st.markdown('<div class="eyebrow">Start a check</div>',unsafe_allow_html=True)
    st.title("Upload your video")
    st.write("Choose a video where one main person's face is visible for several moments. Small background faces are allowed. The video can be silent.")
    st.caption(f"MP4, MOV, AVI, MKV or WebM · up to {cfg['max_upload_mb']} MB · up to {cfg['max_duration']} seconds · up to 4K")
    setup_issues=[]
    if not shutil.which("ffprobe"):
        setup_issues.append("FFprobe is missing from PATH. Install FFmpeg including ffprobe.exe, then restart the terminal and this application.")
    if not MODEL.exists():
        setup_issues.append("The trained model file checkpoints/visual.json is missing. Update your checkout with git pull, or restore the trained model file, then restart the application.")
    elif not model_ready():
        setup_issues.append('The model and video settings are from different versions. Wait for training to complete, or update the model and configuration together.')
    for issue in setup_issues:st.error("Setup required: "+issue)
    upload=st.file_uploader("Choose a video file",type=["mp4","mov","avi","mkv","webm"],key=f"upload-{st.session_state.get('upload_version',0)}")
    if upload:
        if upload.size>cfg["max_upload_mb"]*1024**2:st.error("This file is too large. Choose a shorter or smaller video.")
        else:
            st.video(upload)
            if st.button("Check this video",type="primary",disabled=bool(setup_issues)):
                st.session_state.pop("result",None);st.session_state.pop("evidence",None)
                try:
                    with st.status("Reading the video and checking face samples…",expanded=True) as status:
                        with tempfile.TemporaryDirectory(prefix="deepguard-user-") as folder:
                            path=Path(folder)/("video"+Path(upload.name).suffix.lower())
                            path.write_bytes(upload.getvalue())
                            retry=ROOT/cfg.get('retry_model','checkpoints/coverage_retry.json')
                            version=(MODEL.stat().st_mtime_ns if MODEL.exists() else 0,retry.stat().st_mtime_ns if retry.exists() else 0)
                            result,evidence=detector(version).analyze(path)
                        st.session_state.result=result;st.session_state.evidence=evidence
                        status.update(label="Check complete. Temporary video file deleted.",state="complete")
                    show_result()
                except Exception:
                    logging.getLogger(__name__).exception("Video analysis failed")
                    status.update(label="The video check stopped because of an application error.",state="error")
                    st.error("The check could not finish. Please try again or ask the project team to inspect the application log.")
                    with st.expander("Error details for troubleshooting"):
                        details=traceback.format_exc()
                        st.caption("Share this report with the project team. It may contain local file paths; review it before sharing.")
                        st.code(details,language="text")
                        st.download_button("Download error report",details,file_name="deepguard-error.txt",mime="text/plain")
    st.caption("The temporary video on disk is deleted after checking. Your preview and extracted images remain in this session until you clear them or the session ends.")

elif page=="Your result":
    st.title("Your video result")
    show_result()

elif page=="How reliable is it?":
    st.markdown('<div class="eyebrow">Evidence, not promises</div>',unsafe_allow_html=True)
    st.title("How reliable is this model?")
    data=summary()
    if not data:st.info("Evaluation has not finished. No example accuracy is displayed.")
    else:
        m=data["test_metrics"]
        st.warning("These results describe a small held-out part of the uploaded dataset. They do not establish accuracy on arbitrary videos from the internet.")
        a,b,c,d=st.columns(4)
        a.metric("Test accuracy",f"{m['accuracy']:.1%}")
        b.metric("Fake videos detected",f"{m['recall']:.1%}")
        c.metric("Real videos flagged as fake",f"{m['false_positive_rate']:.1%}")
        d.metric("Test coverage",f"{data['test_coverage']:.1%}")
        st.caption(f"Based on {m['videos']} analyzed test videos from {m['source_groups']} source groups. Metrics use a fixed 50% decision boundary; the application also abstains for unclear inputs.")
        if data.get('evaluation_scope'):st.info('This evaluation uses '+data['evaluation_scope']+'. Augmented copies are excluded from the primary test.')
        cm=pd.DataFrame(m["confusion_matrix"],index=["Actually real","Actually fake"],columns=["Predicted real","Predicted fake"])
        st.subheader("Where did it get things right or wrong?")
        st.plotly_chart(px.imshow(cm,text_auto=True,color_continuous_scale="Teal",labels={"x":"Model prediction","y":"Provided label"},aspect="auto",template="plotly_dark"),use_container_width=True)
        ci=m["accuracy_group_bootstrap_95_ci"]
        st.write(f"The estimated accuracy interval is {ci[0]:.0%}–{ci[1]:.0%}. With so few source groups, this estimate is unstable.")
        comparison=ROOT/"reports/results/comparison.json"
        if comparison.exists():
            st.subheader("Recorded model evaluation")
            records=json.loads(comparison.read_text())
            st.dataframe(pd.DataFrame([{"Model":r["model"],"Accuracy":r["test"]["accuracy"],"F1":r["test"]["f1"],"ROC-AUC":r["test"]["roc_auc"]} for r in records]),hide_index=True,use_container_width=True)
        robust=ROOT/"reports/results/robustness/summary.json"
        if robust.exists():
            st.subheader("What happens when picture quality changes?")
            st.dataframe(pd.DataFrame(json.loads(robust.read_text())),hide_index=True,use_container_width=True)
        with st.expander("Detailed metrics and test procedure"):
            st.json(data)
            st.write("Model settings were selected on validation data. Calibration used a separate split. Original/fake filename pairs stay together. Subject/donor identities were not supplied, so an identity-disjoint test is not certified.")
        predictions=ROOT/"reports/results/heldout_predictions.csv"
        if predictions.exists():st.download_button("Download held-out predictions",predictions.read_bytes(),"heldout-predictions.csv","text/csv")
        st.button("Read the full evaluation report",on_click=go,args=("About this project",))

elif page=="Our dataset":
    st.title("The videos behind the pilot")
    inventory=ROOT/"experiments/dataset_inventory.json"
    if not inventory.exists():st.info("Dataset inspection is still in progress.")
    else:
        inv=json.loads(inventory.read_text())
        st.write("The project uses real and fake videos explicitly labeled by the project owner. Augmented versions are linked to their source videos and do not count as new independent sources.")
        a,b,c=st.columns(3)
        a.metric("Real videos",inv["class_counts"].get("0",0));b.metric("Fake videos",inv["class_counts"].get("1",0));c.metric("Basic decode failures",inv["corrupt_files"])
        trained=summary()
        if trained:
            st.success(f"Face checks completed: {trained.get('pipeline_usable_videos',trained['usable_videos'])} of {trained['received_videos']} videos had enough suitable face samples.")
            st.caption(f"{trained.get('pipeline_excluded_videos',trained['excluded_videos'])} videos still lack suitable face evidence. Uploading the same files again does not add new examples.")
            if trained.get('balanced_class_counts'):
                st.subheader('Equal real and fake counts after quality checks')
                counts=pd.DataFrame(trained['balanced_class_counts']).T.rename(columns={'0':'Real','1':'Fake'})
                st.dataframe(counts.rename(index={'train':'Learning','val':'Choosing settings','calibration':'Adjusting scores','test':'Reserved testing'}),use_container_width=True)
                st.caption('Only learning uses augmentations. Reserved testing includes new sources and separate historical monitoring. Some readable files are held back to keep both classes equal.')
                if trained.get('retry_classifier_training'):
                    retry_counts=trained['retry_classifier_training']['balanced_class_counts']['train']
                    st.info(f"The table describes the standard classifier. A second classifier, trained with {retry_counts['0']} Real and {retry_counts['1']} Fake videos, retries only videos with insufficient face evidence. Existing scored results are preserved.")
        if inv.get('originals_per_class'):
            st.info(f"Before quality filtering: {inv['originals_per_class']} original files and {inv['augmentations_per_class']} augmented files per class. {inv.get('reserved_files',0)} additional files are retained outside this balanced selection.")
        manifest=ROOT/"data/manifests/splits.csv"
        if manifest.exists() and not (trained and trained.get('balanced_class_counts')):
            frame=pd.read_csv(manifest)
            counts=frame.groupby(["split","label"]).size().unstack(fill_value=0).rename(columns={0:"Real",1:"Fake"})
            st.subheader("Why we keep some videos separate")
            st.dataframe(counts.rename(index={"train":"Learning","val":"Choosing settings","calibration":"Adjusting scores","test":"Final test"}),use_container_width=True)
        st.info("Original/fake pairs stay in one group. Dataset origin and participant identities were not independently verified. No raw uploaded videos are published through this page.")
        st.write("A small dataset can teach a useful pilot, but more diverse independently sourced examples are needed before making broad accuracy claims.")

else:
    st.title("About DeepGuard")
    st.write("DeepGuard is a video-only deepfake detection project. A user uploads a video; the system samples facial images, applies a trained visual classifier and explains the result in plain language.")
    st.subheader("What the result means")
    st.write("Likely real and Likely fake describe model evidence. Not sure means the score or visual input does not support a clear answer. Cannot analyze means usable facial evidence was missing or the file could not be read.")
    st.subheader("Project documentation")
    docs=sorted(ROOT.glob("*.md"))+sorted((ROOT/"docs").glob("*.md"))
    selected=st.selectbox("Choose a document",docs,format_func=lambda p:p.stem.replace("_"," ").title())
    with st.expander("Read selected document",expanded=True):st.markdown(selected.read_text())
st.markdown('<div class="footer">DeepGuard · Student visual-detection research · A model result is a reason to investigate, not proof.</div>',unsafe_allow_html=True)
