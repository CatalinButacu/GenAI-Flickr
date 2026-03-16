"""Quick script to inspect AMASS and ARCTIC data formats."""
import numpy as np
import os

def inspect_amass():
    root = "data/AMASS"
    for dirpath, dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".npz"):
                path = os.path.join(dirpath, f)
                d = np.load(path, allow_pickle=True)
                print(f"=== AMASS: {path} ===")
                print(f"Keys: {list(d.keys())}")
                for k in sorted(d.keys()):
                    v = d[k]
                    shape = getattr(v, "shape", "scalar")
                    dtype = getattr(v, "dtype", type(v).__name__)
                    print(f"  {k}: shape={shape} dtype={dtype}")
                return

def inspect_arctic():
    root = "data/ARCTIC/unpack/raw_seqs"
    for dirpath, dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".npy"):
                path = os.path.join(dirpath, f)
                d = np.load(path, allow_pickle=True)
                print(f"\n=== ARCTIC: {path} ===")
                if isinstance(d, np.ndarray) and d.dtype == object:
                    item = d.item()
                    if isinstance(item, dict):
                        print(f"Keys: {list(item.keys())}")
                        for k in sorted(item.keys()):
                            v = item[k]
                            if isinstance(v, np.ndarray):
                                print(f"  {k}: shape={v.shape} dtype={v.dtype}")
                            elif isinstance(v, dict):
                                print(f"  {k}: dict with keys {list(v.keys())}")
                                for kk, vv in v.items():
                                    if isinstance(vv, np.ndarray):
                                        print(f"    {kk}: shape={vv.shape} dtype={vv.dtype}")
                            else:
                                print(f"  {k}: {type(v).__name__} = {v}")
                    else:
                        print(f"  type: {type(item).__name__}")
                else:
                    print(f"  shape={d.shape} dtype={d.dtype}")
                return

def inspect_pahoi():
    root = "data/PAHOI"
    for dirpath, dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".txt") and "description" not in dirpath.lower():
                continue
            if f.endswith(".txt"):
                path = os.path.join(dirpath, f)
                print(f"\n=== PAHOI text: {path} ===")
                with open(path, encoding="utf-8", errors="replace") as fh:
                    print(fh.read()[:500])
                return

if __name__ == "__main__":
    inspect_amass()
    inspect_arctic()
    inspect_pahoi()
