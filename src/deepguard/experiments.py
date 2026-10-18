"""Real held-out upload-route evaluation and fixed visual robustness measurements."""
from __future__ import annotations
import tempfile
import time
from pathlib import Path
import pandas as pd
import numpy as np
from .data import assert_no_leakage
from .inference import Analyzer
from .robustness import perturb
from .evaluation import evaluate
from .utils import write_json


def evaluate_robustness(cfg: dict,model: str="checkpoints/visual.json",manifest: str="data/manifests/splits.csv") -> list[dict]:
    frame=pd.read_csv(manifest,dtype={"source_group":str});assert_no_leakage(frame)
    heldout=frame[frame.split=="test"]
    analyzer=Analyzer(cfg,model)
    summaries=[]
    for condition in ["original","compression","blur","darkness"]:
        rows=[]
        with tempfile.TemporaryDirectory(prefix="deepguard-heldout-") as folder:
            for _,row in heldout.iterrows():
                path=row.path
                if condition!="original":
                    path=str(Path(folder)/row.file)
                    perturb(row.path,path,condition)
                result,_=analyzer.analyze(path)
                rows.append({"file":row.file,"label":int(row.label),"source_group":row.source_group,
                    "condition":condition,"decision":result["decision"],"fake_score":result["fake_score"],
                    "seconds":result["seconds"],"unfamiliar_features":result.get("unfamiliar_features",False),
                    "reason":" ".join(result["reasons"])})
        table=pd.DataFrame(rows);output=Path("reports/results/robustness")/condition;output.mkdir(parents=True,exist_ok=True)
        table.to_csv(output/"predictions.csv",index=False)
        valid=table.fake_score.notna()
        clear=table.decision.isin(["Likely real","Likely fake"])
        stats=evaluate(table.label[valid].to_numpy(),table.fake_score[valid].to_numpy(),table.source_group[valid].to_numpy(),str(output)) if valid.any() else None
        selective=float(np.mean((table.decision[clear]=="Likely fake").astype(int).to_numpy()==table.label[clear].to_numpy())) if clear.any() else None
        summaries.append({"Condition":condition,"Reserved videos":len(table),"Analyzed":int(valid.sum()),
                          "Coverage":float(valid.mean()),"Accuracy at 0.5":stats["accuracy"] if stats else None,
                          "Clear answers":int(clear.sum()),"Accuracy on clear answers":selective,
                          "Mean check seconds":float(table.seconds.mean())})
        write_json("reports/results/robustness/summary.json",summaries)
    write_json("reports/results/robustness/protocol.json",{"source":"Same held-out source pairs as primary evaluation",
        "conditions":{"compression":"H264 CRF35","blur":"Gaussian sigma1.2","darkness":"brightness -0.12"},
        "not_external":"Transformed held-out uploads, not an independent real-world dataset",
        "accuracy_note":"Raw-score accuracy at 0.5 and application abstention are reported separately"})
    return summaries
