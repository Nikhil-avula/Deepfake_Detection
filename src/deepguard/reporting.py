"""Portable, escaped analysis reports without retaining uploaded video files."""
import html
import json


def html_report(report: dict) -> str:
    decision=html.escape(report["decision"])
    details=html.escape(json.dumps(report,indent=2,allow_nan=False))
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>DeepGuard video report</title>
<style>body{{font:16px system-ui;max-width:850px;margin:3rem auto;padding:1rem;color:#102a38}}
pre{{white-space:pre-wrap;background:#eef5f7;padding:1rem;border-radius:12px}}</style>
<h1>DeepGuard · Video check</h1><h2>{decision}</h2><p>Visual evidence from a small research pilot.
A score is not proof of authenticity. Unknown-source performance has not been established.</p>
<pre>{details}</pre></html>"""
