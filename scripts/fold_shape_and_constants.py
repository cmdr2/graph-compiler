#!/usr/bin/env python3
"""
fold_shape_and_constants.py

Usage:
  python fold_shape_and_constants.py --input model.onnx --output model.folded.onnx \
    --input-name input_tensor --input-shape 1,128,512,512
"""

import argparse
import onnx
import numpy as np
from onnx import helper, numpy_helper, shape_inference
import traceback


# -----------------------
# Small numpy evaluator for common ops
# -----------------------
def eval_node_numpy(node, inputs):
    """Evaluate a single ONNX node with numpy arrays.
    inputs: dict name->np.array for the node's inputs
    Returns list of numpy arrays corresponding to node.outputs
    Raises NotImplementedError for unsupported op.
    """
    op = node.op_type
    inp = [inputs[name] for name in node.input]
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    print(f"\nFolding operation: {op} ({node.name})")
    print(f"Input names: {node.input}")
    print(f"Input data: {[str(i)[:100] for i in inp]}")

    if op != "Shape":
        print(f"Input dtypes: {[i.dtype for i in inp]}")

    output = None
    if op == "Constant":
        # Find the value attribute and convert to numpy array
        for a in node.attribute:
            if a.name == "value":
                output = [numpy_helper.to_array(a.t)]
        if output is None:
            raise ValueError("Constant node without 'value' attribute")
    elif op == "Identity":
        output = [inp[0]]
    elif op == "Gather":
        # onnx Gather: data, indices, axis (attr) default 0
        data = inp[0]
        indices = inp[1]
        axis = attrs.get("axis", 0)
        output = [np.take(data, indices, axis=axis)]
    elif op == "Unsqueeze":
        axes = attrs.get("axes")
        if axes is None:
            # Newer opset (13+): axes is a second input
            if len(inp) >= 2:
                axes = inp[1].tolist() if hasattr(inp[1], "tolist") else list(inp[1])
            else:
                raise NotImplementedError("Unsqueeze with no axes attribute or input")
        arr = inp[0]
        for ax in sorted(axes):
            arr = np.expand_dims(arr, ax)
        output = [arr]
    elif op == "Squeeze":
        axes = attrs.get("axes")
        arr = inp[0]
        if axes is None:
            # Newer opset (13+): axes is an optional second input
            if len(inp) >= 2:
                axes = inp[1].tolist() if hasattr(inp[1], "tolist") else list(inp[1])
                for ax in sorted(axes, reverse=True):
                    arr = np.squeeze(arr, axis=ax)
                output = [arr]
            else:
                # No axes specified: squeeze all single dims
                output = [np.squeeze(arr)]
        else:
            for ax in sorted(axes, reverse=True):
                arr = np.squeeze(arr, axis=ax)
            output = [arr]
    elif op == "Concat":
        axis = attrs.get("axis", 0)
        output = [np.concatenate(inp, axis=axis)]
    elif op == "Shape":
        shape_tuple = inp[0]
        output = [np.array(shape_tuple, dtype=np.int64)]
    elif op == "Reshape":
        data = inp[0]
        # reshape target shape may be second input or attr
        if len(inp) >= 2:
            shape_tensor = inp[1]
            target = tuple([int(x) for x in shape_tensor.tolist()])
            output = [np.reshape(data, target)]
        else:
            raise NotImplementedError("Reshape missing shape input")
    elif op == "Slice":
        # support both attribute-based (old) and input-based (opset 10+) Slice
        arr = inp[0]
        if len(inp) >= 3:
            # Modern ONNX: starts, ends, axes, steps as inputs
            starts = inp[1].tolist()
            ends = inp[2].tolist()
            axes = inp[3].tolist() if len(inp) >= 4 else list(range(len(starts)))
            steps = inp[4].tolist() if len(inp) >= 5 else [1] * len(starts)
        else:
            # Old ONNX: attributes
            starts = attrs.get("starts")
            ends = attrs.get("ends")
            axes = attrs.get("axes", list(range(len(starts))))
            steps = attrs.get("steps", [1] * len(axes))
        # Build slices
        sl = [slice(None)] * arr.ndim
        for i, ax in enumerate(axes):
            s = starts[i]
            e = ends[i]
            st = steps[i]
            sl[ax] = slice(s, e, st)
        output = [arr[tuple(sl)]]
    elif op in ("Add", "Sub", "Mul", "Div"):
        a = inp[0]
        b = inp[1]
        if op == "Add":
            output = [np.add(a, b)]
        elif op == "Sub":
            output = [np.subtract(a, b)]
        elif op == "Mul":
            output = [np.multiply(a, b)]
        elif op == "Div":
            output = [np.divide(a, b)]
    elif op == "Sqrt":
        a = inp[0]
        output = [np.sqrt(a)]
    elif op == "Cast":
        to = attrs.get("to")  # onnx enum
        # map to numpy dtype
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.UINT8: np.uint8,
            onnx.TensorProto.INT8: np.int8,
            onnx.TensorProto.UINT16: np.uint16,
            onnx.TensorProto.INT16: np.int16,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
            onnx.TensorProto.BOOL: np.bool_,
            onnx.TensorProto.FLOAT16: np.float16,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.UINT32: np.uint32,
            onnx.TensorProto.UINT64: np.uint64,
        }
        dt = dtype_map.get(to)
        if dt is None:
            raise NotImplementedError(f"Cast to dtype {to} not supported")
        output = [inp[0].astype(dt)]
    elif op == "Transpose":
        perm = attrs.get("perm", None)
        if perm is None:
            output = [np.transpose(inp[0])]
        else:
            output = [np.transpose(inp[0], axes=perm)]
    elif op == "Tile":
        repeats = inp[1]
        output = [np.tile(inp[0], tuple(repeats.tolist()))]
    elif op == "ConstantOfShape":
        # input is shape tensor, attr 'value' optionally present (a tensor)
        shape = tuple([int(x) for x in inp[0].tolist()])
        fill_val = attrs.get("value")
        if fill_val is not None:
            # attr 'value' is a TensorProto; convert:
            if isinstance(fill_val, onnx.TensorProto):
                fill_arr = numpy_helper.to_array(fill_val)
                scalar = fill_arr.flatten()[0]
                dtype = fill_arr.dtype
            else:
                scalar = fill_val
                dtype = None
        else:
            scalar = 0
            dtype = np.float32  # ONNX default for ConstantOfShape is float32
        # Create array with proper dtype
        if dtype is not None:
            output = [np.full(shape, scalar, dtype=dtype)]
        else:
            output = [np.full(shape, scalar)]
    # Add more ops as needed...
    else:
        # Not implemented op
        raise NotImplementedError(f"Op {op} not implemented in numpy evaluator")
    print(f"Output: {[str(o)[:100] for o in output]}")  # Print first 100 chars to avoid huge output
    print(f"Output dtypes: {[o.dtype for o in output]}")
    return output


# -----------------------
# Helpers to access shapes collected by shape_inference
# -----------------------
def collect_value_shapes(model):
    """Return dict mapping value_name -> tuple(shape dims ints or None)."""
    m = model
    shapes = {}

    def record(vi):
        if vi is None or not vi.type.HasField("tensor_type"):
            return
        t = vi.type.tensor_type
        if not t.HasField("shape"):
            return
        dims = []
        for d in t.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            else:
                dims.append(None)
        shapes[vi.name] = tuple(dims)

    for vi in m.graph.value_info:
        record(vi)
    for vi in m.graph.input:
        record(vi)
    for vi in m.graph.output:
        record(vi)
    return shapes


# -----------------------
# Main transform
# -----------------------
def fold_model(input_path, output_path, input_name, input_shape):
    model = onnx.load(input_path)

    # set input shape for the provided input name
    found = False
    for inp in model.graph.input:
        if inp.name == input_name:
            found = True
            # Overwrite the shape dims with concrete values
            tt = inp.type.tensor_type
            del tt.shape.dim[:]
            for s in input_shape:
                dim = tt.shape.dim.add()
                if s is None:
                    dim.dim_param = "unk"
                else:
                    dim.dim_value = int(s)
            break
    if not found:
        raise ValueError(f"Input name '{input_name}' not found in model inputs")

    # Run ONNX shape inference
    inferred = shape_inference.infer_shapes(model)
    shapes = collect_value_shapes(inferred)

    # Build lookup of constant nodes -> numpy arrays
    consts = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    arr = numpy_helper.to_array(a.t)
                    consts[node.output[0]] = arr

    # Unified constant and Shape folding loop
    # Keep iterating until nothing more can be folded
    total_nodes_folded = 0
    total_shape_nodes_folded = 0
    passes = 0
    prev_graph_changed = True
    # Track created cast outputs to avoid SSA violations
    created_casts = {}  # Maps: original_input_name -> cast_output_name

    while prev_graph_changed:
        passes += 1
        curr_shape_count = sum(1 for node in model.graph.node if node.op_type == "Shape")
        print(f"Info: Folding pass {passes}. Remaining Shape nodes: {curr_shape_count}...")

        new_nodes = []
        graph_changed = False

        for node in model.graph.node:
            # Handle nodes with no inputs
            if len(node.input) == 0:
                new_nodes.append(node)
                continue

            # add a cast Node for starts, ends, axis (if available) of Slice to int64
            if node.op_type == "Slice":
                for i in range(1, min(4, len(node.input))):
                    inp_name = node.input[i]
                    if inp_name not in consts:
                        continue

                    arr = consts[inp_name]
                    if not np.issubdtype(arr.dtype, np.integer) or arr.dtype != np.int64:
                        # Check if we already created a cast for this input
                        if inp_name in created_casts:
                            # Reuse the existing cast output
                            cast_output = created_casts[inp_name]
                            print(f"Reusing existing Cast output for '{inp_name}': {cast_output}")
                        else:
                            # Create new cast node
                            print("Adding Cast node for Slice input to int64")
                            cast_output = f"{inp_name}_casted_to_int64"
                            cast_node = helper.make_node(
                                "Cast",
                                inputs=[inp_name],
                                outputs=[cast_output],
                                name=f"Cast_{inp_name}_to_int64",
                                to=onnx.TensorProto.INT64,
                            )
                            new_nodes.append(cast_node)
                            # Track this cast to avoid duplicates
                            created_casts[inp_name] = cast_output
                        # Update the input to the Slice node
                        node.input[i] = cast_output

            if node.op_type == "Shape":
                # If the input shape is known, we can fold it, else skip
                src = node.input[0]
                shape_tuple = shapes.get(src)
                if shape_tuple is None or any(d is None for d in shape_tuple):
                    new_nodes.append(node)
                    continue
            else:
                # Check if all inputs are constants (or empty string for optional inputs)
                all_const = all(name == "" or name in consts for name in node.input)

                if not all_const:
                    new_nodes.append(node)
                    continue

            # Try to evaluate and fold the node
            try:
                if node.op_type == "Shape":
                    shape_tuple = shapes.get(node.input[0])
                    inputs_map = {node.input[0]: shape_tuple}
                else:
                    inputs_map = {n: consts[n] for n in node.input if n != ""}
                # Pass shapes context for Shape op (and any future ops that need it)
                outs = eval_node_numpy(node, inputs_map)

                # Success: create Constant nodes for each output
                for i, out_name in enumerate(node.output):
                    if not out_name:
                        continue
                    arr = outs[i]
                    # Ensure dtype convertible to ONNX types; prefer int64 for ints
                    if np.issubdtype(arr.dtype, np.integer):
                        arr = arr.astype(np.int64)
                    tensor = numpy_helper.from_array(arr, name=out_name)
                    const_node = helper.make_node(
                        "Constant", name=f"Folded_{out_name}", inputs=[], outputs=[out_name], value=tensor
                    )
                    new_nodes.append(const_node)
                    consts[out_name] = arr

                # Track statistics
                total_nodes_folded += 1
                if node.op_type == "Shape":
                    total_shape_nodes_folded += 1
                    print(f"Info: Folded Shape node '{node.name or node.output[0]}' with shape {outs[0].tolist()}")

                graph_changed = True
                # Original node is removed (not appended)

            except (NotImplementedError, ValueError) as e:
                # Skip folding unsupported ops or ops with missing requirements
                # (e.g., Shape with unknown dimensions)
                if node.op_type != "Constant":  # Don't spam for constants
                    print(f"Info: Skipping {node.op_type} node: {e}")
                new_nodes.append(node)
            except Exception as e:
                # Evaluation failed; skip folding
                print(
                    f"Error: Error folding {node.op_type} node {node.name or node.output[0] if node.output else 'unknown'}: {e}"
                )
                traceback.print_exc()
                new_nodes.append(node)

        # Update graph with new nodes
        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)

        # Recompute shapes after each pass
        inferred = shape_inference.infer_shapes(model)
        shapes = collect_value_shapes(inferred)

        # Safety limit
        if passes > 200:
            raise RuntimeError("Error: Reached folding pass limit (200); stopping.")

        prev_graph_changed = graph_changed

    # Convert Reshape nodes with 0 or -1 in shape to explicit values
    print("\nResolving Reshape nodes with 0 or -1 in shape...")
    reshape_nodes_fixed = 0
    for node in model.graph.node:
        if node.op_type == "Reshape" and len(node.input) >= 2:
            shape_input_name = node.input[1]
            output_shape = shapes.get(node.output[0])

            # If shape input is constant and output shape is known, update it
            if shape_input_name in consts and output_shape and all(d is not None for d in output_shape):
                shape_array = consts[shape_input_name]
                if 0 in shape_array or -1 in shape_array:
                    print(
                        f"Info: Fixing Reshape '{node.name or node.output[0]}': {shape_array.tolist()} -> {list(output_shape)}"
                    )
                    new_shape = np.array(output_shape, dtype=np.int64)
                    consts[shape_input_name] = new_shape
                    # Update the Constant node
                    for cn in model.graph.node:
                        if cn.op_type == "Constant" and shape_input_name in cn.output:
                            tensor = numpy_helper.from_array(new_shape, name=shape_input_name)
                            cn.ClearField("attribute")
                            cn.attribute.add().CopyFrom(helper.make_attribute("value", tensor))
                            reshape_nodes_fixed += 1
                            break

    print(f"Fixed {reshape_nodes_fixed} Reshape nodes with explicit shapes")

    # Remove dead nodes (constants that are no longer used)
    # Build set of all used values
    used_values = set()
    # Collect outputs used by nodes
    for node in model.graph.node:
        for inp in node.input:
            if inp:  # Skip empty strings (optional inputs)
                used_values.add(inp)
    # Collect graph outputs
    for out in model.graph.output:
        used_values.add(out.name)

    # Filter nodes: keep only those whose outputs are used
    new_nodes = []
    dead_nodes = 0
    for node in model.graph.node:
        # A node is kept if any of its outputs is used
        is_used = any(out in used_values for out in node.output if out)
        if is_used:
            new_nodes.append(node)
        else:
            dead_nodes += 1
    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)

    # Clean up duplicate initializers: keep last occurrence for a name
    seen = {}
    new_inits = []
    for init in model.graph.initializer:
        seen[init.name] = init
    for name, init in seen.items():
        new_inits.append(init)
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(new_inits)

    # Optionally run shape inference again to refresh value_info
    final = shape_inference.infer_shapes(model)

    onnx.save(final, output_path)
    print(f"\nSaved folded model to {output_path}")
    print(f"Total nodes folded: {total_nodes_folded} (including {total_shape_nodes_folded} Shape nodes)")
    print(f"Removed {dead_nodes} dead nodes.")
    print(f"Total passes: {passes}")
    return final


# -----------------------
# CLI
# -----------------------
def parse_shape(s):
    if s.strip() == "":
        return []
    parts = s.split(",")
    out = []
    for p in parts:
        p = p.strip()
        if p == "?" or p.lower() == "none" or p == "":
            out.append(None)
        else:
            out.append(int(p))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Input ONNX model path")
    p.add_argument("--output", "-o", required=True, help="Output ONNX model path")
    p.add_argument("--input-name", required=True, help="Name of model input to set the known shape on")
    p.add_argument(
        "--input-shape",
        required=True,
        help="Comma-separated concrete dims, e.g. 1,128,512,512 (use ? for unknown dims)",
    )
    args = p.parse_args()
    shape = parse_shape(args.input_shape)
    fold_model(args.input, args.output, args.input_name, shape)
