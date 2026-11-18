"""List or selectively download files from the authors' OSF project (osf.io/rb5tn).

Unlike the repo's download_from_osf.py, this lets you download only the files you need,
and it shows sizes before downloading anything.

Usage (run from the MDA folder):
  python scripts/osf_fetch.py --list --save results/osf_listing.tsv
  python scripts/osf_fetch.py --match evaluation --match join_order_full      # all substrings must occur
  python scripts/osf_fetch.py --match evaluation/ --any /physical_plan/ --any /scan_costs_percentiles/
                                                                             # and at least one --any
  python scripts/osf_fetch.py --match runs.zip --extract runs/raw/imdb/join_order_full/
                                             # keep the zip, unpack only the members under these prefixes

A downloaded .zip is unpacked completely (nested zips too) and then deleted, unless --no_unzip or --extract is given.
With --extract (one or more prefixes, repeatable) the zip is kept and only members starting with one of the prefixes
are unpacked (an unpacked member that is a zip, e.g. datasets/imdb.zip, is unpacked in turn); if the zip is already
there, nothing is downloaded and the members are unpacked from it (so a later run can add prefixes).
"""
import argparse
import os
import sys
import zipfile

import osfclient

from lcm_recheck.config import DATA_ROOT, OSF_PROJECT


def walk(project_id):
    storage = osfclient.OSF().project(project_id).storage()   # anonymous access; the project is public
    for f in storage.files:                                   # walks all folders recursively
        yield f.path.strip("/"), (f.size or 0), f


def unzip_recursively(path):
    folder = os.path.dirname(path)
    with zipfile.ZipFile(path) as z:
        z.extractall(folder)
    os.remove(path)
    for root, _, files in os.walk(folder):
        for name in files:
            if name.endswith(".zip"):
                unzip_recursively(os.path.join(root, name))


def unzip_nested(path):
    """Unpack zip `path` next to itself, delete it, and do the same for the zips it contained (only those)."""
    folder = os.path.dirname(path)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        z.extractall(folder)
    os.remove(path)
    print(f"  unpacked {os.path.basename(path)} ({len(names)} entries)")
    for n in names:
        if n.endswith(".zip") and not n.startswith("__MACOSX/"):
            unzip_nested(os.path.join(folder, n))


def extract_members(path, prefixes):
    """Unpack the members of zip `path` that start with one of `prefixes` next to it; keep the zip.

    An extracted member that is itself a zip (datasets.zip holds datasets/imdb.zip) is unpacked and deleted.
    """
    folder = os.path.dirname(path)
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.startswith("__MACOSX/") and n.startswith(tuple(prefixes))]
        for n in names:
            z.extract(n, folder)
        n_files = sum(not n.endswith("/") for n in names)
        mb = sum(z.getinfo(n).file_size for n in names) / 1e6
    print(f"  extracted {n_files} files ({mb:.1f} MB) under {', '.join(prefixes)}"
          if n_files else f"  no member of {os.path.basename(path)} starts with {', '.join(prefixes)}")
    for n in names:
        if n.endswith(".zip"):
            unzip_nested(os.path.join(folder, n))
    return n_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=OSF_PROJECT, help="OSF project id")
    ap.add_argument("--list", action="store_true", help="only list files and sizes")
    ap.add_argument("--save", help="with --list: also write the listing to this TSV file")
    ap.add_argument("--match", action="append", default=[], help="substring the path must contain (repeatable)")
    ap.add_argument("--any", action="append", default=[],
                    help="substring of which at least one must occur (repeatable; combined with --match)")
    ap.add_argument("--dest", default=DATA_ROOT, help="download root (same layout as the repo expects)")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--no_unzip", action="store_true", help="keep .zip files as downloaded")
    ap.add_argument("--extract", action="extend", nargs="+", default=[],
                    help="keep each .zip and unpack only members starting with one of these prefixes, e.g. "
                         "runs/raw/imdb/join_order_full/ (repeatable); an existing zip is not downloaded again")
    a = ap.parse_args()

    print("Reading the file list from OSF (can take a few minutes)...", file=sys.stderr)
    files = [(p, s, f) for p, s, f in walk(a.project)
             if all(m in p for m in a.match) and (not a.any or any(m in p for m in a.any))]
    total_mb = sum(s for _, s, _ in files) / 1e6

    if a.list:
        lines = [f"{s / 1e6:12.1f} MB\t{p}" for p, s, _ in files]
        print("\n".join(lines))
        print(f"{len(files)} files, {total_mb:.1f} MB in total")
        if a.save:
            os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
            with open(a.save, "w") as out:
                out.write("size_mb\tpath\n" + "\n".join(f"{s / 1e6:.1f}\t{p}" for p, s, _ in files) + "\n")
        return

    if not files:
        sys.exit("No file matches. Run with --list first and adjust --match.")
    print(f"{len(files)} files, {total_mb:.1f} MB will be downloaded to {a.dest}")
    if not a.yes and input("Continue? [y/N] ").strip().lower() != "y":
        sys.exit("Aborted.")

    for p, s, f in files:
        target = os.path.join(a.dest, p)
        if a.extract and target.endswith(".zip") and os.path.exists(target):
            print(f"exists, not downloaded again: {p}")
            extract_members(target, a.extract)
            continue
        if os.path.exists(target) or os.path.exists(target + ".done"):
            print(f"exists, skipping: {p}")
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        print(f"downloading {p} ({s / 1e6:.1f} MB)")
        with open(target, "wb") as fp:
            f.write_to(fp)
        if target.endswith(".zip") and a.extract:
            extract_members(target, a.extract)
        elif target.endswith(".zip") and not a.no_unzip:
            print("  unzipping...")
            unzip_recursively(target)
            open(target + ".done", "w").close()      # marker so a re-run does not download it again
    print("Done.")


if __name__ == "__main__":
    main()