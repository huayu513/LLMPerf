"""Apply validated native SGLang options without shell evaluation."""
from __future__ import annotations

import json
import os
import sys


def transform(command, contract):
    removed = contract.get("remove_flags", {})
    result = []
    index = 0
    while index < len(command):
        token = command[index]
        if token in removed:
            index += 2 if removed[token] else 1
        else:
            result.append(token)
            index += 1
    return result + list(contract.get("append", []))


if __name__ == "__main__":
    contract = json.loads(os.environ["S1_SGLANG_NATIVE"])
    for argument in transform(sys.argv[1:], contract):
        sys.stdout.buffer.write(str(argument).encode() + b"\0")
