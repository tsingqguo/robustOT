import os
import pathlib
import re
import sys

bl = [".git", "__pycache__"]
mods = ["data", "datasets", "models", "options", "util", "build"]

T = 0


class Rewrite:
    filepath: str
    modified: bool

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.modified = False

    def exec(self):
        if pathlib.Path(self.filepath).suffix != ".py":
            return
        f = open(self.filepath, "r+")
        t = f.read()
        ls = t.split("\n")

        def mod(ln: int, text: str):
            global T
            ls[ln] = text
            if not self.modified:
                T += 1
                self.modified = True

        for i, l in enumerate(ls):
            r = re.compile(r"from\s(?P<mod>[^\.|\s]{1,})(?P<left>.*)$")
            m = r.match(l)
            if m != None:
                if m.group("mod") not in mods:
                    pass
                else:
                    mod(i, f'from pix2pix.{m.group("mod")}{m.group("left")}')
                    continue
            r = re.compile(r"import\s(?P<mod>[^\.|\s]{1,})(?P<left>.*)$")
            m = r.match(l)
            if m != None:
                if m.group("mod") not in mods:
                    pass
                else:
                    # mod(i, f'import pix2pix.{m.group("mod")}{m.group("left")}')
                    mod(i, f'from pix2pix import {m.group("mod")}')
                    continue
            r = re.compile(r'^(?P<p>.*)importlib\.import_module\((?!")(?P<s>.*)')
            m = r.match(l)
            if m != None:
                mod(i, f'{m.group("p")}importlib.import_module("pix2pix." + {m.group("s")}')
        if self.modified:
            print(f"rewriting with {self.filepath}")
            t = "\n".join(ls)
            f.seek(0)
            f.write(t)


def rewrite(fp: str):
    r = Rewrite(fp)
    r.exec()


def scan(p: str):
    for f in os.listdir(p):
        j = os.path.join(p, f)
        if os.path.isdir(j):
            if f not in bl:
                scan(j)
        else:
            rewrite(j)


if __name__ == "__main__":
    try:
        root = sys.argv[1]
    except Exception:
        root = "libs/pix2pix"
    if os.path.isdir(root):
        scan(root)
    else:
        rewrite(root)
    print(f"rewrote total {T} files")
