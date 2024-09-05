import os
import h5py
import json
import sys
from typed_cap import Cap


class Args:
    # @alias=d
    datasets: list[str]

    # @alias=o
    output: str
    """output json filepath"""


def remove_duplicates(data: dict[str, list[str]]) -> dict[str, list[str]]:
    keys: set[str] = set()
    unique: dict[str, list[str]] = {}
    repeat: dict[str, int] = {}
    for split, items in data.items():
        unique[split] = []
        for item in items:
            if item in keys:
                d_name = item.split("/")[1]
                repeat[d_name] = repeat.get(d_name, 0) + 1
            else:
                keys.add(item)
                unique[split].append(item)

    for i, (split, cnt) in enumerate(repeat.items()):
        if i == 0:
            print("[WARN] repeat:")
        print(f"\t{split}: {cnt}")

    return unique


def show_data_details(data: dict[str, list[str]], title: str = "unknown"):
    tot = 0
    print(f"[INFO] {title} details:")
    for split, items in data.items():
        print(f"{split:>15}: {len(items)}")
        tot += len(items)
    print(f"{'total':>15}: {tot}")


if __name__ == "__main__":
    cap = Cap(Args)
    parsed = cap.parse()
    args = parsed.val
    argv = parsed.args

    data: dict[str, list[str]] = {}

    for fp in argv:
        file_size = os.path.getsize(fp) // 1024 // 1024
        print(f"[INFO] indexing {fp} ({file_size} MB)")

        items = []
        with h5py.File(fp, "r+") as h5f:
            for dset in args.datasets:
                dset = dset.upper()
                if dset in h5f:
                    sub = h5f[dset]
                    if isinstance(sub, h5py.Group):
                        keys = list(sub.keys())
                        keys = [os.path.join(f"/{dset}", p) for p in keys]
                        items += keys
        data[os.path.basename(fp)] = items
        print(f"       got {len(items)} items")

    show_data_details(data, "raw")
    data_unique = remove_duplicates(data)
    show_data_details(data_unique, "unique")

    with open(args.output, "w") as f:
        json.dump(data_unique, f, indent=4)
