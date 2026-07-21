#!/usr/bin/env python3
"""Make intra_analysis_standard.onnx support dynamic spatial resolution.

The model was originally converted from a fixed 256x256 ONNX; its only
resolution-coupled part is the leading pixel_unshuffle (factor 8), which was
traced as two Reshapes with baked-in constant shapes:
    reshape1: [1,3,256,256] -> [-1, 3, 32, 8, 32, 8]   (32 = 256/8)
    reshape2: [...]        -> [-1, 192, 32, 32]

This script replaces those two static shape constants with a small subgraph that
computes the shapes dynamically from the input (Shape -> Gather H,W -> Div by 8
-> Concat). Every Conv weight and every other op is left untouched, so the model
is bit-identical to the original at 256x256 and now runs at any H,W (multiple of
8, i.e. any multiple of 64 for the full intra pipeline).
"""
import argparse
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _const_scalar(name, value, dtype=TensorProto.INT64):
    arr = np.array([value], dtype=(np.int64 if dtype == TensorProto.INT64 else np.float32))
    return helper.make_node('Constant', [], [name], value=numpy_helper.from_array(arr))


def patch_model(model_path, out_path):
    m = onnx.load(model_path)
    g = m.graph

    # 1) Make the spatial dims of input 'in0' dynamic (batch stays 1).
    in_vi = {vi.name: vi for vi in g.input}
    if 'in0' not in in_vi:
        raise RuntimeError("input 'in0' not found")
    shape = in_vi['in0'].type.tensor_type.shape
    # dim0 = 1, dim1 = 3 (keep concrete), dim2/3 -> dynamic
    shape.dim[0].dim_value = 1
    shape.dim[1].dim_value = 3
    shape.dim[2].Clear()
    shape.dim[2].dim_param = 'h'
    shape.dim[3].Clear()
    shape.dim[3].dim_param = 'w'

    # 2) Locate the pixel_unshuffle Reshape->Transpose->Reshape and the two
    #    Constant nodes feeding the Reshapes. Identify them by their value.
    const_nodes = [n for n in g.node if n.op_type == 'Constant']
    reshape_nodes = [n for n in g.node if n.op_type == 'Reshape']

    def const_val(n):
        for a in n.attribute:
            if a.name == 'value':
                return numpy_helper.to_array(a.t).tolist()
        return None

    c_unshuffle = None   # [-1, 3, 32, 8, 32, 8]
    c_final = None       # [-1, 192, 32, 32]
    for n in const_nodes:
        v = const_val(n)
        if v == [-1, 3, 32, 8, 32, 8]:
            c_unshuffle = n
        elif v == [-1, 192, 32, 32]:
            c_final = n
    if c_unshuffle is None or c_final is None:
        raise RuntimeError("could not find pixel_unshuffle constant nodes")

    # The reshape nodes that consume these constants.
    r1 = next(n for n in reshape_nodes if c_unshuffle.output[0] in n.input)
    r2 = next(n for n in reshape_nodes if c_final.output[0] in n.input)

    # 3) Build a dynamic shape-computation subgraph.
    #    shape1 = [-1, 3, H/8, 8, W/8, 8]
    #    shape2 = [-1, 192, H/8, W/8]
    pre = []
    # Length-1 int64 constants (rank 1 so Concat axis=0 is valid).
    pre.append(_const_scalar('mone', -1))
    pre.append(_const_scalar('three', 3))
    pre.append(_const_scalar('eight', 8))
    pre.append(_const_scalar('ch192', 192))
    pre.append(_const_scalar('starts2', 2))
    pre.append(_const_scalar('starts3', 3))
    pre.append(_const_scalar('ends2', 3))
    pre.append(_const_scalar('ends3', 4))
    pre.append(_const_scalar('axes0', 0))

    shp = 'in0_shape'                       # Shape(in0) -> [1,3,H,W] (rank 1)
    pre.append(helper.make_node('Shape', ['in0'], [shp], shp))
    # Slice keeps rank-1 length-1 tensors (Gather would produce scalars).
    hname, wname = 'in0_H', 'in0_W'         # [H], [W]
    pre.append(helper.make_node('Slice', [shp, 'starts2', 'ends2', 'axes0'], [hname], 'slice_H'))
    pre.append(helper.make_node('Slice', [shp, 'starts3', 'ends3', 'axes0'], [wname], 'slice_W'))
    h8, w8 = 'in0_H8', 'in0_W8'             # [H/8], [W/8]
    pre.append(helper.make_node('Div', [hname, 'eight'], [h8], 'div_H8'))
    pre.append(helper.make_node('Div', [wname, 'eight'], [w8], 'div_W8'))

    s1 = 'unshuffle_shape'                  # [-1, 3, H/8, 8, W/8, 8]
    pre.append(helper.make_node(
        'Concat', ['mone', 'three', h8, 'eight', w8, 'eight'], [s1], 'concat_shape1', axis=0))
    s2 = 'final_shape'                      # [-1, 192, H/8, W/8]
    pre.append(helper.make_node(
        'Concat', ['mone', 'ch192', h8, w8], [s2], 'concat_shape2', axis=0))

    # 4) Rewire the two Reshape nodes to consume the dynamic shape tensors,
    #    and splice the new subgraph in place of the old Constants.
    r1.input[1] = s1
    r2.input[1] = s2

    new_nodes = []
    inserted = False
    for n in g.node:
        if n is c_unshuffle or n is c_final:
            continue
        if not inserted and (n is r1):
            new_nodes.extend(pre)
            inserted = True
        new_nodes.append(n)
    if not inserted:
        raise RuntimeError("failed to splice dynamic subgraph")
    del g.node[:]
    g.node.extend(new_nodes)

    # 5) Run shape inference so internal/again output dims are populated.
    try:
        m = onnx.shape_inference.infer_shapes(m)
    except Exception as e:
        print('shape_inference warning:', e)

    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    print('Saved dynamic model ->', out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(ROOT, 'onnx', 'models', 'intra_analysis_standard.onnx'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'onnx', 'models', 'intra_analysis_standard.onnx'))
    ap.add_argument('--backup', action='store_true', default=True,
                    help='keep a .static256.bak copy of the original (default on)')
    args = ap.parse_args()

    if args.backup and os.path.abspath(args.model) == os.path.abspath(args.out):
        bak = args.model + '.static256.bak'
        if not os.path.exists(bak):
            import shutil
            shutil.copy(args.model, bak)
            print('Backed up original ->', bak)
    patch_model(args.model, args.out)


if __name__ == '__main__':
    main()
