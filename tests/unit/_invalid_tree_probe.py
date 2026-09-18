#!/usr/bin/env python
"""Standalone probe: write, then read back, a single tree whose 'sections'
node indices have optionally been shifted off the library's 0-based
convention, and report whether that round-trip succeeded.

Run as a subprocess rather than imported in-process: a tree with invalid
node indices currently makes neuroh5 abort the whole process (SIGABRT)
instead of raising a catchable Python exception -- see
test_exception_safety.py for the same reasoning applied to other entry
points (which *are* exception-safe). Isolating the probe in a subprocess
means a crash here fails only the probe, not the whole pytest run.

Usage: python _invalid_tree_probe.py --path FILE [--shift N] [--gid G] [--n-pts N]

--shift shifts every node index in a valid, correctly-0-based tree
(from _neuroh5_testing.make_tree) by N: 0 (default) leaves it valid;
+1 reproduces a tree imported from a standard 1-based SWC file without
the neurotrees_import -n -1 offset (the exact mistake that originally
motivated this probe).

Prints "OK <gid>" and exits 0 if the round-trip succeeded. Any other
outcome (clean exception -> nonzero exit with a traceback, or a hard
abort -> nonzero exit with no "OK" line) means it was rejected.
"""
import argparse

import numpy as np
from mpi4py import MPI

from neuroh5.io import append_cell_trees, read_trees
from _neuroh5_testing import create_populations_file, make_tree


def shift_section_node_indices(sections, shift):
    """Add `shift` to every node index in a valid `sections` array (as
    produced by _neuroh5_testing.make_tree), leaving the num_sections and
    per-section node-count header values untouched.

    Computed in a wider signed dtype and cast back down with `.astype`
    (which wraps) rather than an in-place uint16 add (which numpy now
    raises OverflowError on for an out-of-range scalar like shift=-1) --
    a negative shift should reach neuroh5 as the huge wrapped uint16
    value real 0-based-minus-one data would actually contain on disk,
    not fail before ever leaving Python.
    """
    out = sections.copy()
    ptr = 1
    while ptr < len(out):
        count = int(out[ptr])
        ptr += 1
        shifted = out[ptr : ptr + count].astype(np.int64) + shift
        out[ptr : ptr + count] = shifted.astype(out.dtype)
        ptr += count
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--gid", type=int, default=0)
    p.add_argument("--n-pts", type=int, default=6)
    p.add_argument("--shift", type=int, default=0)
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    create_populations_file(args.path, [("GC", 0, 1, 0)])

    tree = make_tree(args.n_pts, args.gid)
    if args.shift:
        tree["sections"] = shift_section_node_indices(tree["sections"], args.shift)

    append_cell_trees(args.path, "GC", {args.gid: tree}, comm=comm)
    comm.barrier()
    g, n_nodes = read_trees(args.path, "GC", comm=comm)
    got = dict(g)
    assert args.gid in got, f"gid {args.gid} missing from read-back trees"
    print(f"OK {args.gid}")


if __name__ == "__main__":
    main()
