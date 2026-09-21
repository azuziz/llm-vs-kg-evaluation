import json
import pathlib
import sys
from typing import Any, Iterable, Set

def walk(x: Any) -> Iterable[Any]:
    if isinstance(x, dict):
        for v in x.values():
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)
    else:
        yield x

def extract_biolink_predicates(spec: dict) -> Set[str]:
    preds = set()
    for v in walk(spec):
        if isinstance(v, str) and v.startswith("biolink:"):
            preds.add(v)
    return preds

def main():
    if len(sys.argv) != 2:
        print("Usage: python 02_extract_predicates.py <kg_name>")
        print("Example: python 02_extract_predicates.py ctd")
        sys.exit(2)

    kg = sys.argv[1]
    inpath = pathlib.Path("simple_specs") / f"{kg}.simple_spec.json"
    if not inpath.exists():
        raise FileNotFoundError(f"Missing {inpath}. Run 01_fetch_simple_specs.py first.")

    spec = json.loads(inpath.read_text())
    preds = sorted(extract_biolink_predicates(spec))

    outdir = pathlib.Path("predicates")
    outdir.mkdir(exist_ok=True)

    (outdir / f"{kg}.predicates.txt").write_text("\n".join(preds) + "\n")
    (outdir / f"{kg}.predicates.json").write_text(json.dumps(preds, indent=2))

    print(f"{kg}: {len(preds)} predicates")
    print(f"Saved: predicates/{kg}.predicates.txt and predicates/{kg}.predicates.json")

if __name__ == "__main__":
    main()
