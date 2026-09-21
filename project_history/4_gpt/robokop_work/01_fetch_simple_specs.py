import json
import pathlib
import re
import sys
import requests

BASE = "https://robokop-automat.apps.renci.org"
OUTDIR = pathlib.Path("simple_specs")
OUTDIR.mkdir(exist_ok=True)

def discover_simple_spec_slugs(openapi_path: str) -> list[str]:
    text = pathlib.Path(openapi_path).read_text(encoding="utf-8", errors="ignore")
    # find paths like: /<slug>/simple_spec:
    slugs = sorted(set(re.findall(r"^\\s*/([^/\\s]+)/simple_spec\\s*:", text, flags=re.MULTILINE)))
    return slugs

def fetch_json(url: str) -> dict:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def main():
    openapi_path = "openapi.yml"
    if not pathlib.Path(openapi_path).exists():
        print("Missing openapi.yml. Run:")
        print('  curl -sS "https://robokop-automat.apps.renci.org/openapi.yml" -o openapi.yml')
        sys.exit(2)

    slugs = discover_simple_spec_slugs(openapi_path)
    print(f"Discovered {len(slugs)} services with /simple_spec")

    failures = []
    for slug in slugs:
        url = f"{BASE}/{slug}/simple_spec"
        try:
            print(f"Fetching {slug}: {url}")
            data = fetch_json(url)
            outpath = OUTDIR / f"{slug}.simple_spec.json"
            outpath.write_text(json.dumps(data, indent=2))
            print(f"  saved -> {outpath}")
        except Exception as e:
            failures.append((slug, str(e)))
            print(f"  FAILED -> {slug}: {e}")

    if failures:
        print("\nFailures summary:")
        for slug, err in failures:
            print(f"  {slug}: {err}")
        print("\nThis is usually OK (some services may be down or not expose simple_spec).")

if __name__ == "__main__":
    main()
