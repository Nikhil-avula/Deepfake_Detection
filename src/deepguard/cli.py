"""Reproduce the video-only dataset, model and held-out evaluation."""
import argparse
import json
import logging
from pathlib import Path
from .utils import config,seed_all,write_json,provenance


def main():
    p=argparse.ArgumentParser(description="DeepGuard video-only detection")
    p.add_argument("--config",default=None)
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("hardware")
    a=sub.add_parser("prepare-dataset");a.add_argument("manifest")
    a.add_argument("--output",default="data/manifests/splits.csv")
    a=sub.add_parser("import-uploads");a.add_argument("index");a.add_argument("--destination",default="data/raw/uploaded")
    a=sub.add_parser("train");a.add_argument("manifest",default="data/manifests/splits.csv",nargs="?")
    a.add_argument("--model",default="checkpoints/visual.json");a.add_argument("--results",default="reports/results")
    a=sub.add_parser("analyze");a.add_argument("video");a.add_argument("output");a.add_argument("--model",default="checkpoints/visual.json")
    a=sub.add_parser("robustness");a.add_argument("--model",default="checkpoints/visual.json")
    args=p.parse_args();cfg=config(args.config);seed_all(cfg["seed"],cfg["threads"])
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s")
    if args.command=="hardware":print(json.dumps(provenance(),indent=2))
    elif args.command=="prepare-dataset":
        from .data import prepare_labeled_dataset
        frame=prepare_labeled_dataset(args.manifest,args.output,cfg)
        print(json.dumps({"videos":len(frame),"splits":frame.split.value_counts().to_dict(),"output":args.output},indent=2))
    elif args.command=="import-uploads":
        from .data import import_uploads,split_pairs
        frame=import_uploads(args.index,args.destination,cfg,"data/manifests/inventory.csv")
        if frame.error.ne("").any():raise ValueError("Inspect corrupt files before splitting")
        split_pairs(frame,cfg["seed"]).to_csv("data/manifests/splits.csv",index=False)
    elif args.command=="train":
        from .training import train
        print(json.dumps(train(args.manifest,cfg,args.model,args.results),indent=2))
    elif args.command=="analyze":
        from .inference import Analyzer
        from .reporting import html_report
        result,_=Analyzer(cfg,args.model).analyze(args.video)
        write_json(args.output,result);Path(args.output).with_suffix(".html").write_text(html_report(result))
        print(json.dumps(result,indent=2))
    elif args.command=="robustness":
        from .experiments import evaluate_robustness
        evaluate_robustness(cfg,args.model)


if __name__=="__main__":main()
