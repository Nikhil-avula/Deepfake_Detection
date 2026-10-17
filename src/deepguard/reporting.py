"""Escaped, portable HTML reports; uploaded media are never embedded."""
import html
import json


def html_report(report: dict) -> str:
    content = html.escape(json.dumps(report, indent=2, allow_nan=False))
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><title>DeepGuard analysis</title>"
            "<style>body{font:16px system-ui;max-width:900px;margin:3rem auto;padding:1rem}"
            "pre{white-space:pre-wrap;background:#eef4f6;padding:1rem}</style>"
            "<h1>DeepGuard AI · Analysis report</h1><p>Research evidence, not proof of authenticity. "
            "Confidence is available only with separately calibrated fusion weights.</p><pre>" + content + "</pre></html>")
