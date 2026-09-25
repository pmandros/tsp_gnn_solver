"""Download TSPLIB95 (symmetric and asymmetric) into data/.

    python scripts/download_benchmarks.py [--data-dir data]

The Fu et al. TSP500/1000/10000 test files are distributed through the
authors' and later papers' repositories (Att-GCRN+MCTS, DIMES, DIFUSCO) rather
than a stable URL, so they are not fetched here; put them at
``data/fu/tsp{500,1000,10000}_test_concorde.txt`` (see BENCHMARKS.md).
"""

import argparse
import gzip
import io
import os
import shutil
import tarfile
import urllib.request

TSPLIB = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95"
ARCHIVES = {
    "tsplib": f"{TSPLIB}/tsp/ALL_tsp.tar.gz",
    "tsplib_atsp": f"{TSPLIB}/atsp/ALL_atsp.tar.gz",
}


def fetch(url: str) -> bytes:
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def extract(blob: bytes, dest: str):
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            data = tar.extractfile(m).read()
            if name.endswith(".gz"):  # the archives hold individually gzipped files
                data, name = gzip.decompress(data), name[:-3]
            with open(os.path.join(dest, name), "wb") as f:
                f.write(data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()
    for sub, url in ARCHIVES.items():
        dest = os.path.join(args.data_dir, sub)
        if os.path.isdir(dest) and os.listdir(dest):
            print(f"{dest} exists, skipping")
            continue
        try:
            extract(fetch(url), dest)
        except Exception as e:
            shutil.rmtree(dest, ignore_errors=True)
            print(f"failed to fetch {url}: {e}")
            continue
        print(f"{dest}: {len(os.listdir(dest))} files")


if __name__ == "__main__":
    main()
