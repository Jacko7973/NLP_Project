import json
from typing import Iterable

def load_jsonl(file_path: str) -> Iterable[dict]:
    with open(file_path, "r") as f:
        for line in f:
            yield json.loads(line.strip())
