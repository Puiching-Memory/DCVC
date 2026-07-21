import os
import re
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

# Repo root (DCVC/). This script lives in <root>/onnx/python/.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _pt_key_to_onnx_init(pt_key):
    """Map PyTorch state-dict key to ONNX initializer name.
    Examples:
        enc.enc_1.dc.0.weight   -> enc_1.dc0.weight
        enc.enc_1.ffn.0.weight  -> enc_1.ffn0.weight
        enc.enc_2.0.dc.0.weight -> enc_2.0.dc0.weight
        enc.enc_2.6.weight      -> enc_2.6.weight
    """
    k = pt_key.replace('enc.enc_', 'enc_')
    k = re.sub(r'\.dc\.(\d)\.', r'.dc\1.', k)
    k = re.sub(r'\.ffn\.(\d)\.', r'.ffn\1.', k)
    return k


def _replace_initializers_with_checkpoint_weights(initializers, checkpoint_path=None):
    """Replace FP16-rounded ONNX initializers with the FP32 checkpoint weights."""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(ROOT, 'checkpoints', 'cvpr2025_image.pth.tar')
    if not os.path.exists(checkpoint_path):
        print(f'Checkpoint {checkpoint_path} not found, keeping FP16-rounded weights')
        return initializers

    try:
        import sys
        sys.path.insert(0, ROOT)
        os.environ.setdefault('SUPPRESS_CUSTOM_KERNEL_WARNING', '1')
        from src.utils.common import get_state_dict
        sd = get_state_dict(checkpoint_path)
    except Exception as e:
        print(f'Failed to load checkpoint weights: {e}, keeping FP16-rounded weights')
        return initializers

    # Build map from ONNX initializer name -> PyTorch tensor
    onnx_to_pt = {}
    for pt_key, tensor in sd.items():
        oname = _pt_key_to_onnx_init(pt_key)
        onnx_to_pt[oname] = tensor

    replaced = 0
    new_inits = []
    for init in initializers:
        if init.name not in onnx_to_pt:
            new_inits.append(init)
            continue
        pt = onnx_to_pt[init.name]
        arr = pt.detach().cpu().numpy().astype(np.float32)
        new_inits.append(numpy_helper.from_array(arr, name=init.name))
        replaced += 1
    print(f'Replaced {replaced} initializers with FP32 checkpoint weights')
    return new_inits


def convert_dcvc_depthconv_to_standard_ops(model_path, out_path, checkpoint_path=None):
    m = onnx.load(model_path)

    # Convert all initializers to FP32.
    init_by_name = {}
    new_inits = []
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        new_inits.append(numpy_helper.from_array(arr, name=init.name))
        init_by_name[init.name] = arr

    # Change graph I/O to FP32 so the CPU EP runs everything in FP32.
    def set_fp32(vi):
        vi.type.tensor_type.elem_type = TensorProto.FLOAT

    for vi in m.graph.input:
        set_fp32(vi)
    for vi in m.graph.output:
        set_fp32(vi)
    for vi in m.graph.value_info:
        set_fp32(vi)

    new_nodes = []
    for node in m.graph.node:
        if node.op_type != 'DcvcDepthConv':
            new_nodes.append(node)
            continue

        x = node.input[0]
        dc0_w, dc0_b = node.input[1], node.input[2]
        dc2_w, dc2_b = node.input[3], node.input[4]
        dc3_w, dc3_b = node.input[5], node.input[6]
        ffn0_w, ffn0_b = node.input[7], node.input[8]
        ffn2_w, ffn2_b = node.input[9], node.input[10]
        has_adaptor = len(node.input) == 13
        if has_adaptor:
            ad_w, ad_b = node.input[11], node.input[12]
        out = node.output[0]
        base = node.name + '/'

        # C_out is the number of output channels of this block.
        c_out = int(init_by_name[dc0_w].shape[0])
        half = 2 * c_out
        four = 4 * c_out

        # Scalar 4.0 used by WSiLU activations.
        four_name = base + 'four'
        new_inits.append(numpy_helper.from_array(
            np.array(4.0, dtype=np.float32), name=four_name))
        init_by_name[four_name] = np.array(4.0, dtype=np.float32)

        cur = x
        if has_adaptor:
            ad_out = base + 'adaptor_out'
            new_nodes.append(helper.make_node(
                'Conv', [cur, ad_w, ad_b], [ad_out], base + 'adaptor', kernel_shape=[1, 1]))
            cur = ad_out

        # dc0: 1x1 conv
        t0 = base + 'dc0_out'
        new_nodes.append(helper.make_node(
            'Conv', [cur, dc0_w, dc0_b], [t0], base + 'dc0', kernel_shape=[1, 1]))

        # WSiLU: t0 * sigmoid(4 * t0)
        t0_mul = base + 'dc0_mul4'
        new_nodes.append(helper.make_node(
            'Mul', [t0, four_name], [t0_mul], base + 'dc0_mul4'))
        sig = base + 'dc0_sig'
        new_nodes.append(helper.make_node(
            'Sigmoid', [t0_mul], [sig], base + 'dc0_sig'))
        t1 = base + 'dc0_act'
        new_nodes.append(helper.make_node(
            'Mul', [t0, sig], [t1], base + 'dc0_wsilu'))

        # dc2: depthwise 3x3 conv
        t2 = base + 'dc2_out'
        new_nodes.append(helper.make_node(
            'Conv', [t1, dc2_w, dc2_b], [t2], base + 'dc2',
            kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=c_out))

        # dc3: 1x1 conv
        t3 = base + 'dc3_out'
        new_nodes.append(helper.make_node(
            'Conv', [t2, dc3_w, dc3_b], [t3], base + 'dc3', kernel_shape=[1, 1]))

        # residual: dc_out + x
        dc_add = base + 'dc_add'
        new_nodes.append(helper.make_node(
            'Add', [t3, cur], [dc_add], base + 'dc_add'))

        # ffn0: 1x1 conv, 4*C_out channels
        f0 = base + 'ffn0_out'
        new_nodes.append(helper.make_node(
            'Conv', [dc_add, ffn0_w, ffn0_b], [f0], base + 'ffn0', kernel_shape=[1, 1]))

        # WSiLUChunkAdd: silu(f0), split into two halves, add.
        # First half: channels [0, 2*C_out)
        f1 = base + 'f1'
        starts1, ends1, axes1, steps1 = base + 's1', base + 'e1', base + 'a1', base + 'p1'
        new_inits.append(numpy_helper.from_array(np.array([0], dtype=np.int64), name=starts1))
        new_inits.append(numpy_helper.from_array(np.array([half], dtype=np.int64), name=ends1))
        new_inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes1))
        new_inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name=steps1))
        new_nodes.append(helper.make_node(
            'Slice', [f0, starts1, ends1, axes1, steps1], [f1], base + 'slice1'))

        # Second half: channels [2*C_out, 4*C_out)
        f2 = base + 'f2'
        starts2, ends2, axes2, steps2 = base + 's2', base + 'e2', base + 'a2', base + 'p2'
        new_inits.append(numpy_helper.from_array(np.array([half], dtype=np.int64), name=starts2))
        new_inits.append(numpy_helper.from_array(np.array([four], dtype=np.int64), name=ends2))
        new_inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes2))
        new_inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name=steps2))
        new_nodes.append(helper.make_node(
            'Slice', [f0, starts2, ends2, axes2, steps2], [f2], base + 'slice2'))

        # WSiLU on each half
        f1_mul = base + 'f1_mul4'
        new_nodes.append(helper.make_node(
            'Mul', [f1, four_name], [f1_mul], base + 'f1_mul4'))
        f1_sig = base + 'f1_sig'
        new_nodes.append(helper.make_node(
            'Sigmoid', [f1_mul], [f1_sig], base + 'f1_sig'))
        f1_act = base + 'f1_act'
        new_nodes.append(helper.make_node(
            'Mul', [f1, f1_sig], [f1_act], base + 'f1_wsilu'))

        f2_mul = base + 'f2_mul4'
        new_nodes.append(helper.make_node(
            'Mul', [f2, four_name], [f2_mul], base + 'f2_mul4'))
        f2_sig = base + 'f2_sig'
        new_nodes.append(helper.make_node(
            'Sigmoid', [f2_mul], [f2_sig], base + 'f2_sig'))
        f2_act = base + 'f2_act'
        new_nodes.append(helper.make_node(
            'Mul', [f2, f2_sig], [f2_act], base + 'f2_wsilu'))

        chunk_add = base + 'chunk_add'
        new_nodes.append(helper.make_node(
            'Add', [f1_act, f2_act], [chunk_add], base + 'chunk_add'))

        # ffn2: 1x1 conv, back to C_out channels
        f2_out = base + 'ffn2_out'
        new_nodes.append(helper.make_node(
            'Conv', [chunk_add, ffn2_w, ffn2_b], [f2_out], base + 'ffn2', kernel_shape=[1, 1]))

        # residual: ffn_out + dc_add
        ffn_add = base + 'ffn_add'
        new_nodes.append(helper.make_node(
            'Add', [f2_out, dc_add], [ffn_add], base + 'ffn_add'))

        # The current model only contains DcvcDepthConv (no shortcut variant).
        # If shortcut is needed, add another Add here.
        final = ffn_add
        if final != out:
            new_nodes.append(helper.make_node(
                'Identity', [final], [out], base + 'out'))

    m.graph.ClearField('node')
    m.graph.node.extend(new_nodes)
    new_inits = _replace_initializers_with_checkpoint_weights(new_inits, checkpoint_path)
    m.graph.ClearField('initializer')
    m.graph.initializer.extend(new_inits)

    onnx.save(m, out_path)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert DCVC intra_analysis ONNX to standard CPU-runnable ops')
    parser.add_argument('--model', default=os.path.join(ROOT, 'native', 'assets', 'onnx', 'intra_analysis.onnx'),
                        help='Path to original FP16 custom-op ONNX model')
    parser.add_argument('--out', default=os.path.join(ROOT, 'onnx', 'intra_analysis_standard.onnx'),
                        help='Output path for the standard-op FP32 model')
    parser.add_argument('--checkpoint', default=os.path.join(ROOT, 'checkpoints', 'cvpr2025_image.pth.tar'),
                        help='Path to the FP32 PyTorch checkpoint; if unavailable, FP16-rounded weights are kept')
    args = parser.parse_args()
    convert_dcvc_depthconv_to_standard_ops(args.model, args.out, args.checkpoint)
