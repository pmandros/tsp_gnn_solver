"""Download benchmark data (and optionally a Concorde binary) into data/.

    python scripts/download_benchmarks.py [--data-dir data] [--no-fu] [--concorde tools/concorde]

- TSPLIB95 symmetric and asymmetric instances, plus the published optimal
  tours. The script tries the Heidelberg site first and falls back to a
  pinned GitHub mirror (pdrozdowski/TSPLib.Net).
- The Fu et al. TSP500/1000/10000 test files, from the authors' repository
  (Spider-scnu/TSP) at a pinned commit.
- With ``--concorde PATH``, the prebuilt Linux x86-64 Concorde binary that
  pyconcorde ships (jvkersch/pyconcorde), at a pinned commit. Point
  ``CONCORDE_BIN`` at it afterwards.
"""

import argparse
import gzip
import io
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.request

TSPLIB = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95"
ARCHIVES = {
    "tsplib": f"{TSPLIB}/tsp/ALL_tsp.tar.gz",
    "tsplib_atsp": f"{TSPLIB}/atsp/ALL_atsp.tar.gz",
}
MIRROR = ("https://github.com/pdrozdowski/TSPLib.Net", "5cb1449963fa56176c062ff806eb831dcbc07c54")
MIRROR_DIRS = {"tsplib": "TSPLIB95/tsp", "tsplib_atsp": "TSPLIB95/atsp"}
FU_RAW = "https://raw.githubusercontent.com/Spider-scnu/TSP/729c52e92693f077ad9761aa3803327758c7d435"
FU_FILES = {
    500: "MCTS-CPUver/tsp-200-500-1000/tsp500_test_concorde.txt",
    1000: "MCTS-CPUver/tsp-200-500-1000/tsp1000_test_concorde.txt",
    10000: "MCTS-CPUver/tsp-10000/tsp10000_test_concorde.txt",
}
CONCORDE_URL = ("https://raw.githubusercontent.com/jvkersch/pyconcorde/a573c1b73244f0dc7cf88ddd18ccfa4c65919974/"
                "external/pyconcorde-build/binaries/linux/concorde")


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


def from_mirror(data_dir: str):
    url, commit = MIRROR
    with tempfile.TemporaryDirectory() as tmp:
        print(f"cloning {url}")
        subprocess.run(["git", "clone", "-q", url, tmp], check=True)
        subprocess.run(["git", "-C", tmp, "checkout", "-q", commit], check=True)
        for sub, rel in MIRROR_DIRS.items():
            dest = os.path.join(data_dir, sub)
            os.makedirs(dest, exist_ok=True)
            for name in os.listdir(os.path.join(tmp, rel)):
                src = os.path.join(tmp, rel, name)
                if name.endswith(".gz"):
                    with gzip.open(src, "rb") as f, open(os.path.join(dest, name[:-3]), "wb") as g:
                        g.write(f.read())
                else:
                    shutil.copy(src, dest)
            print(f"{dest}: {len(os.listdir(dest))} files")


def get_tsplib(data_dir: str):
    todo = [s for s in ARCHIVES if not (os.path.isdir(os.path.join(data_dir, s)) and os.listdir(os.path.join(data_dir, s)))]
    if not todo:
        print("TSPLIB already present")
        return
    try:
        for sub in todo:
            extract(fetch(ARCHIVES[sub]), os.path.join(data_dir, sub))
            print(f"{sub}: {len(os.listdir(os.path.join(data_dir, sub)))} files")
    except Exception as e:
        print(f"TSPLIB site failed ({e}); using the GitHub mirror")
        from_mirror(data_dir)


def get_fu(data_dir: str):
    dest = os.path.join(data_dir, "fu")
    os.makedirs(dest, exist_ok=True)
    for n, rel in FU_FILES.items():
        path = os.path.join(dest, f"tsp{n}_test_concorde.txt")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(fetch(f"{FU_RAW}/{rel}"))
        print(f"{path}")


def get_concorde(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(fetch(CONCORDE_URL))
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"{path}  (export CONCORDE_BIN={os.path.abspath(path)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--no-tsplib", action="store_true")
    p.add_argument("--no-fu", action="store_true")
    p.add_argument("--concorde", metavar="PATH", help="also download the prebuilt Linux Concorde binary to PATH")
    args = p.parse_args()
    if not args.no_tsplib:
        get_tsplib(args.data_dir)
    if not args.no_fu:
        get_fu(args.data_dir)
    if args.concorde:
        get_concorde(args.concorde)


if __name__ == "__main__":
    main()
