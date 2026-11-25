"""Download the overhead-RGB subset of Nutrition5k from its public GCS bucket.

Mirrors the bucket layout under ``--root``::

    <root>/metadata/dish_metadata_cafe{1,2}.csv, ingredients_metadata.csv
    <root>/dish_ids/README, dish_ids_*.txt, splits/{rgb,depth}_{train,test}_ids.txt
    <root>/imagery/realsense_overhead/<dish_id>/rgb.png   (+ depth_raw.png with --depth)

Skips the side-angle videos, which are most of the 181 GB archive.
Re-running only fetches missing files.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "nutrition5k_dataset"
PREFIX = "nutrition5k_dataset/"
LIST_API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
FILE_URL = f"https://storage.googleapis.com/{BUCKET}/"
OVERHEAD = "imagery/realsense_overhead/"


def list_objects(prefix: str) -> list[str]:
    """Names (relative to PREFIX) of every object under ``prefix``."""
    names: list[str] = []
    token = None
    while True:
        query = {"prefix": PREFIX + prefix, "fields": "nextPageToken,items(name)"}
        if token:
            query["pageToken"] = token
        with urllib.request.urlopen(f"{LIST_API}?{urllib.parse.urlencode(query)}") as r:
            page = json.load(r)
        names += [item["name"].removeprefix(PREFIX) for item in page.get("items", [])]
        token = page.get("nextPageToken")
        if not token:
            return names


def fetch(name: str, root: Path, retries: int = 3) -> bool:
    """Download one object to ``root/name``. Returns False if it already existed."""
    dest = root / name
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    url = FILE_URL + urllib.parse.quote(PREFIX + name)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                tmp.write_bytes(r.read())
            tmp.replace(dest)
            return True
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/nutrition5k"))
    parser.add_argument("--depth", action="store_true", help="also download depth_raw.png")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    wanted = {"rgb.png"} | ({"depth_raw.png"} if args.depth else set())
    names = list_objects("metadata/") + list_objects("dish_ids/")
    names += [n for n in list_objects(OVERHEAD) if n.rsplit("/", 1)[-1] in wanted]
    print(f"{len(names)} files under {args.root}", file=sys.stderr)

    done = new = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for future in as_completed(pool.submit(fetch, n, args.root) for n in names):
            new += future.result()
            done += 1
            if done % 250 == 0 or done == len(names):
                print(f"  {done}/{len(names)} ({new} downloaded)", file=sys.stderr)


if __name__ == "__main__":
    main()
