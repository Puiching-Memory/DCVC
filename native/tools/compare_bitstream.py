#!/usr/bin/env python3
"""Compare official PyTorch bitstream vs native container (NAL structure).

When NN engines are enabled, also compares payload bytes for bit-exactness.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def read_uint_adaptive(data: bytes, pos: int):
    a3 = data[pos]
    pos += 1
    if (a3 >> 7) == 0:
        return a3, pos
    a2 = data[pos]
    pos += 1
    if (a3 >> 6) == 0x02:
        return ((a3 & 0x3F) << 8) + a2, pos
    a3 &= 0x3F
    a1 = data[pos]
    a0 = data[pos + 1]
    pos += 2
    return (a3 << 24) + (a2 << 16) + (a1 << 8) + a0, pos


def parse_nals(buf: bytes):
    pos = 0
    nals = []
    while pos < len(buf):
        flag = buf[pos]
        pos += 1
        nal = flag >> 4
        sps_id = flag & 0x0F
        if nal == 0:  # SPS
            h, pos = read_uint_adaptive(buf, pos)
            w, pos = read_uint_adaptive(buf, pos)
            f = buf[pos]
            pos += 1
            nals.append({"type": "SPS", "w": w, "h": h, "ec": (f >> 2) & 1, "ada": f & 1})
        elif nal in (1, 2):
            qp = buf[pos]
            pos += 1
            ln, pos = read_uint_adaptive(buf, pos)
            payload = buf[pos : pos + ln]
            pos += ln
            nals.append({"type": "I" if nal == 1 else "P", "qp": qp, "payload": payload})
        else:
            raise ValueError(f"unknown nal {nal} at {pos}")
    return nals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--payload", action="store_true", help="require identical payloads")
    args = ap.parse_args()
    na = parse_nals(args.a.read_bytes())
    nb = parse_nals(args.b.read_bytes())
    if len(na) != len(nb):
        print(f"nal count {len(na)} vs {len(nb)}", file=sys.stderr)
        sys.exit(1)
    for i, (a, b) in enumerate(zip(na, nb)):
        if a["type"] != b["type"]:
            print(f"[{i}] type {a['type']} vs {b['type']}", file=sys.stderr)
            sys.exit(1)
        if a["type"] == "SPS":
            for k in ("w", "h", "ec", "ada"):
                if a[k] != b[k]:
                    print(f"SPS {k}: {a[k]} vs {b[k]}", file=sys.stderr)
                    sys.exit(1)
        else:
            if a["qp"] != b["qp"]:
                print(f"[{i}] qp {a['qp']} vs {b['qp']}", file=sys.stderr)
                sys.exit(1)
            if args.payload and a["payload"] != b["payload"]:
                print(f"[{i}] payload mismatch len {len(a['payload'])} vs {len(b['payload'])}",
                      file=sys.stderr)
                sys.exit(1)
    print(f"OK: {len(na)} NALs match structure")


if __name__ == "__main__":
    main()
