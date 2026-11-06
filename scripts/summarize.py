#!/usr/bin/env python
"""Re-compute metrics from episode logs: python scripts/summarize.py runs/real/*.jsonl"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from halter.harness import metrics  # noqa: E402

eps = metrics.load(sys.argv[1:])
print(json.dumps({"overall": metrics.summarize(eps), "per_task": metrics.per_task(eps)}, indent=2))
