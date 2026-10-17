"""Command-line research workflow. Run `deepguard --help`."""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import config, write_json, provenance, seed_all


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepGuard AI research workflow")
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hardware")
    p = sub.add_parser("download")
    p.add_argument("url"); p.add_argument("output"); p.add_argument("--sha256", required=True)
    p = sub.add_parser("inventory")
    p.add_argument("manifest"); p.add_argument("output")
    p = sub.add_parser("split")
    p.add_argument("manifest"); p.add_argument("output")
    p = sub.add_parser("prepare")
    p.add_argument("manifest"); p.add_argument("--landmark-asset")
    p = sub.add_parser("baseline")
    p.add_argument("manifest"); p.add_argument("output")
    p = sub.add_parser("train")
    p.add_argument("kind", choices=["visual", "audio", "temporal", "sync"])
    p.add_argument("manifest"); p.add_argument("output")
    p.add_argument("--visual-checkpoint"); p.add_argument("--landmark-asset")
    p = sub.add_parser("predict")
    p.add_argument("manifest"); p.add_argument("checkpoint"); p.add_argument("output")
    p.add_argument("--split", default="test"); p.add_argument("--visual-checkpoint"); p.add_argument("--landmark-asset")
    p = sub.add_parser("calibrate")
    p.add_argument("checkpoint"); p.add_argument("predictions"); p.add_argument("output")
    p = sub.add_parser("evaluate")
    p.add_argument("predictions"); p.add_argument("output")
    p = sub.add_parser("sync-evaluate")
    p.add_argument("manifest"); p.add_argument("checkpoint"); p.add_argument("asset"); p.add_argument("output")
    p = sub.add_parser("collect")
    p.add_argument("manifest"); p.add_argument("split"); p.add_argument("output")
    p.add_argument("--checkpoints", default="checkpoints"); p.add_argument("--landmark-asset")
    p = sub.add_parser("fusion-fit")
    p.add_argument("training"); p.add_argument("calibration"); p.add_argument("output")
    p = sub.add_parser("fusion-evaluate")
    p.add_argument("predictions"); p.add_argument("checkpoint"); p.add_argument("output")
    p = sub.add_parser("analyze")
    p.add_argument("video"); p.add_argument("output"); p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--landmark-asset")
    p = sub.add_parser("perturb")
    p.add_argument("video"); p.add_argument("output"); p.add_argument("kind", choices=["compression", "blur", "lighting", "noise"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = config(args.config)
    seed_all(cfg["seed"])
    from . import data, training, evaluation, fusion
    if args.command == "hardware":
        import json
        print(json.dumps(provenance(), indent=2))
    elif args.command == "download":
        data.download(args.url, args.output, args.sha256)
    elif args.command == "inventory":
        data.inventory(args.manifest, args.output)
    elif args.command == "split":
        frame = data.read_manifest(args.manifest)
        # Refuse known corrupt inventory rows and hash all files to group duplicates.
        if "error" in frame and frame.error.ne("").any():
            raise ValueError("Remove/repair documented corrupt files before splitting")
        from .utils import sha256
        frame["sha256"] = [sha256(p) for p in frame.path]
        result = data.split_manifest(frame, cfg["seed"])
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(result.groupby(["split", "label"]).size().to_string())
    elif args.command == "prepare":
        from .media import preprocess
        frame = data.read_manifest(args.manifest)
        for path in frame.path:
            preprocess(path, cfg, "data/processed", args.landmark_asset)
    elif args.command == "baseline":
        training.baseline(args.manifest, cfg, args.output)
    elif args.command == "train":
        training.train(args.manifest, args.kind, cfg, args.output, args.visual_checkpoint, args.landmark_asset)
    elif args.command == "predict":
        training.predict_manifest(args.manifest, args.checkpoint, args.output, args.split, args.visual_checkpoint, args.landmark_asset)
    elif args.command == "calibrate":
        training.calibrate(args.checkpoint, args.predictions, args.output)
    elif args.command == "evaluate":
        table = pd.read_csv(args.predictions)
        if "split" not in table or set(table.split) != {"test"}:
            raise ValueError("Evaluate only held-out test predictions")
        meta_path = Path(args.predictions+".meta.json")
        if not meta_path.exists():
            raise ValueError("Prediction provenance sidecar required")
        import json
        meta = json.loads(meta_path.read_text())
        if meta["kind"] == "sync":
            raise ValueError("Sync needs aligned/mismatched pair evaluation, not overall fake labels")
        valid = table.label.isin([0, 1])
        evaluation.evaluate(table.label[valid].to_numpy(), table.probability[valid].to_numpy(), args.output,
                            table["group_id" if "group_id" in table else "identities"][valid].to_numpy())
        write_json(Path(args.output)/"runtime.json", {"mean_forward_seconds": float(table.seconds.mean()), **meta})
    elif args.command == "collect":
        fusion.collect(args.manifest, cfg, args.checkpoints, args.split, args.output, args.landmark_asset)
    elif args.command == "sync-evaluate":
        from .sync_evaluation import assess_sync
        assess_sync(args.manifest,args.checkpoint,args.asset,args.output)
    elif args.command == "fusion-fit":
        fusion.fit(args.training, args.calibration, args.output)
    elif args.command == "fusion-evaluate":
        fusion.assess(args.predictions, args.checkpoint, args.output)
    elif args.command == "analyze":
        from .inference import Analyzer
        from .reporting import html_report
        report, _ = Analyzer(cfg, args.checkpoints, args.landmark_asset).analyze(args.video)
        write_json(args.output, report)
        Path(args.output).with_suffix(".html").write_text(html_report(report), encoding="utf-8")
    elif args.command == "perturb":
        from .robustness import perturb
        perturb(args.video, args.output, args.kind)


if __name__ == "__main__":
    main()
