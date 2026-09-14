"""Offline process-protocol fixture, not a model or tokenizer qualification."""

import json
import sys

value = json.load(sys.stdin)
request = value["request"]
tokens = max(1, len(request["input"].split()))
if value["operation"] == "count_input_tokens":
    result = {"input_tokens": tokens}
elif value["operation"] == "generate":
    result = {
        "output": "offline fixture answer",
        "input_tokens": tokens,
        "output_tokens": 3,
    }
else:
    raise ValueError("unknown operation")
print(json.dumps(result))
