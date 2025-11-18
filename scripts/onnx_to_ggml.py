#!/usr/bin/env python3
"""
Convert ONNX model to C++ code using ggml function calls.

This script reads an ONNX model and generates a C++ file that:
1. Defines a getGraph() function that constructs the computation graph using ggml_onnx_* functions
2. Defines a predict() function that executes the graph and prints the output
"""

import argparse
import onnx
from onnx import numpy_helper
import sys
from pathlib import Path


def sanitize_name(name):
    """Sanitize tensor/node names to be valid C++ identifiers."""
    # Replace invalid characters with underscores
    sanitized = name.replace(".", "_").replace("/", "_").replace(":", "_")
    sanitized = sanitized.replace("-", "_").replace("[", "_").replace("]", "_")
    sanitized = sanitized.replace(" ", "_").replace("(", "_").replace(")", "_")

    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = "tensor_" + sanitized

    return sanitized


def replace_diffusers_names(name):
    return name
    name = name.replace("attentions.0.to_k", "attentions.0.key")
    name = name.replace("attentions.0.to_q", "attentions.0.query")
    name = name.replace("attentions.0.to_v", "attentions.0.value")
    name = name.replace("attentions.0.to_out.0", "attentions.0.proj_attn")
    return name


def get_ggml_type(elem_type):
    """Map ONNX data type to ggml type."""
    type_map = {
        1: "GGML_TYPE_F32",  # FLOAT
        2: "GGML_TYPE_I8",  # UINT8 -> closest approximation
        3: "GGML_TYPE_I8",  # INT8
        6: "GGML_TYPE_I32",  # INT32
        7: "GGML_TYPE_I64",  # INT64
        10: "GGML_TYPE_F16",  # FLOAT16
        11: "GGML_TYPE_F64",  # DOUBLE -> no exact match, using F64 if available
    }
    return type_map.get(elem_type, "GGML_TYPE_F32")


def normalize_dims_to_4d(dims):
    if len(dims) >= 4:
        return dims
    # Prepend 1s to make it 4D
    return [1] * (4 - len(dims)) + dims


def generate_cpp_code(model, output_path, print_values=False):
    """Generate C++ code from ONNX model.

    Creates two files:
    - <name>_graph.h: Contains the getGraph() function
    - <name>.cpp: Contains all boilerplate code and includes the graph header

    Args:
        model: ONNX model to convert
        output_path: Path for the output C++ file
        print_values: If True, generates code to print intermediate tensor values
    """
    graph = model.graph

    # Collect initializers (weights/constants)
    initializers = {init.name: init for init in graph.initializer}

    # Map to track tensor name -> variable name
    tensor_vars = {}

    # Analyze graph to detect which initializers need reshaping for broadcasting
    # Maps: initializer_name -> (op_type, role, required_shape_in_ggml_order)
    reshape_info = {}

    # Track Conv weights that need dimension reversal
    # Maps: weight_name -> (op_type, role, original_dims)
    conv_weights = {}

    # Track Reshape shape tensors that need dimension reversal
    # Maps: shape_tensor_name -> (op_type, role, original_dims)
    reshape_shapes = {}

    # Track Unsqueeze/Squeeze axes tensors that need axis conversion
    # Maps: axes_tensor_name -> (op_type, input_tensor_name)
    unsqueeze_squeeze_axes = {}

    # Track Slice axes tensors that need axis conversion
    # Maps: axes_tensor_name -> (op_type, input_tensor_name)
    slice_axes = {}

    # Build a map of tensor names to their shapes (in ONNX order) for rank calculation
    # We'll populate this from initializers and graph value_info
    tensor_shapes = {}

    # Add initializer shapes
    for init_name, init in initializers.items():
        tensor_shapes[init_name] = list(init.dims)

    # Add graph input shapes
    for input_info in graph.input:
        if input_info.name not in tensor_shapes:  # Don't override initializers
            shape = []
            if input_info.type.HasField("tensor_type"):
                tensor_type = input_info.type.tensor_type
                if tensor_type.HasField("shape"):
                    for dim in tensor_type.shape.dim:
                        if dim.HasField("dim_value"):
                            shape.append(dim.dim_value)
                        else:
                            shape.append(-1)  # Dynamic dimension
            if shape:
                tensor_shapes[input_info.name] = shape

    # Add shapes from value_info (intermediate tensors)
    for value_info in graph.value_info:
        if value_info.name not in tensor_shapes:
            shape = []
            if value_info.type.HasField("tensor_type"):
                tensor_type = value_info.type.tensor_type
                if tensor_type.HasField("shape"):
                    for dim in tensor_type.shape.dim:
                        if dim.HasField("dim_value"):
                            shape.append(dim.dim_value)
                        else:
                            shape.append(-1)  # Dynamic dimension
            if shape:
                tensor_shapes[value_info.name] = shape

    for node in graph.node:
        op_type = node.op_type

        # Conv: weight (1st input after input) needs dimension reversal
        # ONNX Conv weight: [out_channels, in_channels, kH, kW]
        # GGML Conv weight: [kW, kH, in_channels, out_channels]
        if op_type == "Conv" and len(node.input) >= 2 and node.input[1]:
            weight_name = node.input[1]
            if weight_name in initializers:
                init = initializers[weight_name]
                if len(init.dims) == 4:  # Conv2D weight should be 4D
                    conv_weights[weight_name] = ("Conv", "weight", list(init.dims))

        # Reshape: shape tensor (2nd input) needs dimension reversal
        # ONNX shape: [N, C, H, W] -> GGML shape: [W, H, C, N]
        if op_type == "Reshape" and len(node.input) >= 2 and node.input[1]:
            shape_name = node.input[1]
            if shape_name in initializers:
                init = initializers[shape_name]
                # Shape is a 1D tensor containing the target dimensions
                if len(init.dims) == 1:
                    reshape_shapes[shape_name] = ("Reshape", "shape", list(init.dims))

        # Unsqueeze/Squeeze: axes tensor (2nd input) needs axis position conversion
        # ONNX axes refer to output positions, need to convert to GGML order
        if op_type in ["Unsqueeze", "Squeeze"] and len(node.input) >= 2 and node.input[1]:
            axes_name = node.input[1]
            input_name = node.input[0]
            unsqueeze_squeeze_axes[axes_name] = (op_type, input_name)

        # Slice: axes tensor (4th input) needs axis position conversion
        # ONNX axes refer to positions in ONNX order, need to convert to GGML order
        if op_type == "Slice" and len(node.input) >= 4 and node.input[3]:
            axes_name = node.input[3]
            input_name = node.input[0]
            slice_axes[axes_name] = (op_type, input_name)

        # Conv: 3rd input (bias) needs reshaping from [C] to [1, 1, C, 1] in ONNX/logical order
        # ONNX order is [N, C, H, W], so [1, C, 1, 1] means N=1, C=channels, H=1, W=1
        # GGML order is [W, H, C, N], so we want [W=1, H=1, C=channels, N=1]
        if op_type == "Conv" and len(node.input) >= 3 and node.input[2]:
            bias_name = node.input[2]
            if bias_name in initializers:
                init = initializers[bias_name]
                if len(init.dims) == 1:  # Only reshape if it's 1D
                    channels = init.dims[0]
                    # Store channel count; we'll generate the right shape later
                    reshape_info[bias_name] = ("Conv", "bias", [1, channels, 1, 1], channels)

        # BatchNormalization: inputs 1-4 (scale, bias, mean, var) need reshaping from [C] to [1, 1, C, 1]
        if op_type == "BatchNormalization":
            param_names = ["scale", "bias", "mean", "var"]
            for i, param_role in enumerate(param_names, start=1):
                if len(node.input) > i and node.input[i]:
                    param_name = node.input[i]
                    if param_name in initializers:
                        init = initializers[param_name]
                        if len(init.dims) == 1:  # Only reshape if it's 1D
                            channels = init.dims[0]
                            reshape_info[param_name] = (
                                "BatchNormalization",
                                param_role,
                                [1, channels, 1, 1],
                                channels,
                            )

    # Track constants that should be converted to tensors
    # These are constants used as inputs to specific operators
    constant_tensors = {}  # Maps: constant_name -> (dims, data_type, values)

    # Operators whose constant inputs should become tensors
    operators_with_tensor_constants = {
        "InstanceNormalization",
        "Mul",
        "Div",
        "Add",
        "Sub",
        "RandomNormalLike",
        "Transpose",
        "MatMul",
    }

    # First pass: identify Constant nodes and extract their values
    constant_node_outputs = {}  # Maps: output_name -> node
    for node in graph.node:
        if node.op_type == "Constant":
            if len(node.output) > 0:
                const_name = node.output[0]
                constant_node_outputs[const_name] = node

    # Second pass: determine which constants are used by target operators
    constants_used_by_target_ops = set()
    constants_used_as_reshape_shapes = set()  # Track constants used as Reshape shape inputs
    constants_used_as_slice_axes = {}  # Track constants used as Slice axes: name -> input_tensor_name
    for node in graph.node:
        if node.op_type in operators_with_tensor_constants:
            for inp in node.input:
                if inp in constant_node_outputs:
                    constants_used_by_target_ops.add(inp)
        # Track constants used as Reshape shape (2nd input)
        if node.op_type == "Reshape" and len(node.input) >= 2 and node.input[1]:
            shape_input = node.input[1]
            if shape_input in constant_node_outputs:
                constants_used_as_reshape_shapes.add(shape_input)
        # Track constants used as Slice axes (4th input)
        if node.op_type == "Slice" and len(node.input) >= 4 and node.input[3]:
            axes_input = node.input[3]
            input_tensor = node.input[0]
            if axes_input in constant_node_outputs:
                constants_used_as_slice_axes[axes_input] = input_tensor

    # Third pass: process Constant nodes to extract constant tensors
    for node in graph.node:
        if node.op_type == "Constant":
            if len(node.output) > 0:
                const_name = node.output[0]
                should_be_tensor = const_name in constants_used_by_target_ops

                # Look for 'value' attribute
                for attr in node.attribute:
                    if attr.name == "value":
                        if attr.type == onnx.AttributeProto.TENSOR:
                            # Extract tensor data
                            tensor = attr.t
                            data_type = tensor.data_type
                            dims = list(tensor.dims)

                            # Get the actual values
                            if data_type == 1:  # FLOAT
                                if tensor.float_data:
                                    values = list(tensor.float_data)
                                else:
                                    values = list(numpy_helper.to_array(tensor).flatten())
                            elif data_type in [6, 7]:  # INT32, INT64
                                if tensor.int64_data:
                                    values = list(tensor.int64_data)
                                elif tensor.int32_data:
                                    values = list(tensor.int32_data)
                                else:
                                    values = list(numpy_helper.to_array(tensor).flatten())
                            else:
                                values = list(numpy_helper.to_array(tensor).flatten())

                            # If this constant is used by target operators, convert to tensor
                            # Note: scalars (dims=[]) should be converted to 1D tensors with single element
                            if should_be_tensor:
                                if len(dims) == 0:
                                    # Scalar: convert to 1D tensor with single element
                                    constant_tensors[const_name] = ([1], data_type, values)
                                else:
                                    constant_tensors[const_name] = (dims, data_type, values)

    # Determine output paths
    from pathlib import Path

    output_path_obj = Path(output_path)
    output_stem = output_path_obj.stem
    output_dir = output_path_obj.parent

    # Generate <name>_graph.h and <name>.cpp
    graph_header_path = output_dir / f"{output_stem}_graph.h"

    # Start building the graph header file
    graph_lines = []
    graph_lines.append("#pragma once")
    graph_lines.append("")
    graph_lines.append('#include "ggml.h"')
    graph_lines.append('#include "ggml-onnx.h"')
    graph_lines.append("")

    # Start building the main C++ code
    cpp_lines = []
    if print_values:
        cpp_lines.append("#define _COMPILER_PRINT_TENSOR_VALUES")
        cpp_lines.append("")

    cpp_lines.append('#include "ggml.h"')
    cpp_lines.append('#include "ggml-onnx.h"')
    cpp_lines.append('#include "ggml-cpu.h"')
    cpp_lines.append("")
    cpp_lines.append("#ifdef GGML_USE_CUDA")
    cpp_lines.append('#include "ggml-cuda.h"')
    cpp_lines.append("#endif")
    cpp_lines.append("")
    cpp_lines.append("#include <vector>")
    cpp_lines.append("#include <string>")
    cpp_lines.append("#include <iostream>")
    cpp_lines.append("#include <cstring>")
    cpp_lines.append("#include <unordered_map>")
    cpp_lines.append("#include <unordered_set>")
    cpp_lines.append("")
    cpp_lines.append('#include "safetensors.hpp"')
    cpp_lines.append(f'#include "{graph_header_path.name}"')
    cpp_lines.append("")

    # Add global variables
    cpp_lines.append("// Global state")
    cpp_lines.append("ggml_backend_t backend = NULL;")
    cpp_lines.append("ggml_gallocr_t allocr = NULL;")
    cpp_lines.append("ggml_context* ctx_weights = NULL;")
    cpp_lines.append("")

    # Merge constant tensors into initializers for unified handling
    # We'll add them to initializers dict but track them separately for data initialization
    all_tensors_to_declare = list(initializers.keys()) + list(constant_tensors.keys())

    # print("Initializers: ", initializers.keys())
    # print("Constants: ", constant_tensors.keys())

    # Add weight tensor declarations in cpp file and extern declarations in header
    if all_tensors_to_declare:
        cpp_lines.append("// Weight tensors")
        graph_lines.append("// External weight tensor declarations")
        for init_name in all_tensors_to_declare:
            var_name = sanitize_name(init_name)
            cpp_lines.append(f"ggml_tensor* {var_name} = NULL;")
            graph_lines.append(f"extern ggml_tensor* {var_name};")
        cpp_lines.append("")
        graph_lines.append("")

        # Add tensor map for dynamic loading
        cpp_lines.append("// Tensor map for weight loading")
        cpp_lines.append("std::unordered_map<std::string, struct ggml_tensor*> tensor_map;")
        cpp_lines.append("")

    # Generate backend initialization
    cpp_lines.append("void init_backend() {")
    cpp_lines.append("#ifdef GGML_USE_CUDA")
    cpp_lines.append('    fprintf(stderr, "%s: using CUDA backend\\n", __func__);')
    cpp_lines.append("    backend = ggml_backend_cuda_init(0); // init device 0")
    cpp_lines.append("    if (!backend) {")
    cpp_lines.append('        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\\n", __func__);')
    cpp_lines.append("    }")
    cpp_lines.append("#endif")
    cpp_lines.append("")
    cpp_lines.append("    if (!backend) {")
    cpp_lines.append("        backend = ggml_backend_cpu_init();")
    cpp_lines.append("    }")
    cpp_lines.append("}")
    cpp_lines.append("")

    # Generate memory allocator initialization
    cpp_lines.append("void init_mem_allocator() {")
    cpp_lines.append("    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));")
    cpp_lines.append("}")
    cpp_lines.append("")

    # Generate function to load initializers
    cpp_lines.append("// Load model weights and constants")
    cpp_lines.append("void load_weights(const std::string& weights_file) {")
    cpp_lines.append(f"    // Create context for weights")
    cpp_lines.append(f"    int num_weight_tensors = {len(all_tensors_to_declare)};")
    cpp_lines.append("    ctx_weights = ggml_init({")
    cpp_lines.append("        /*.mem_size   =*/ ggml_tensor_overhead() * num_weight_tensors,")
    cpp_lines.append("        /*.mem_buffer =*/ NULL,")
    cpp_lines.append("        /*.no_alloc   =*/ true,")
    cpp_lines.append("    });")
    cpp_lines.append("")
    cpp_lines.append("    // Define weight tensors and populate tensor map")

    # Process regular initializers
    for init_name in initializers.keys():
        var_name = sanitize_name(init_name)
        init = initializers[init_name]
        dims = list(init.dims)  # ONNX dimension order

        # Check if this initializer needs reshaping for broadcasting
        if init_name in reshape_info:
            op_type, role, onnx_display_dims, channels = reshape_info[init_name]
            cpp_lines.append(
                f"    // {init_name} - {op_type} {role} (reshaped from {dims} to {onnx_display_dims} for broadcasting)"
            )
            # For Conv/BatchNorm bias: we want GGML shape [1, 1, channels, 1] which means W=1, H=1, C=channels, N=1
            # ggml_new_tensor_4d takes (ne0, ne1, ne2, ne3) = (W, H, C, N)
            cpp_lines.append(
                f"    {var_name} = ggml_new_tensor_4d(ctx_weights, {get_ggml_type(init.data_type)}, 1, 1, {channels}, 1);"
            )
            init_name = replace_diffusers_names(init_name)
            cpp_lines.append(f'    tensor_map["{init_name}"] = {var_name};')
            continue  # Skip the normal dimension handling below
        # Check if this is a Conv weight that needs dimension reversal
        elif init_name in conv_weights:
            op_type, role, original_dims = conv_weights[init_name]
            # ONNX Conv weight: [out_channels, in_channels, kH, kW]
            # GGML Conv weight: [kW, kH, in_channels, out_channels] (fully reversed)
            cpp_lines.append(f"    // {init_name} - {op_type} {role} (ONNX {original_dims} -> GGML reversed)")
            cpp_lines.append(
                f"    {var_name} = ggml_new_tensor_4d(ctx_weights, {get_ggml_type(init.data_type)}, {dims[3]}, {dims[2]}, {dims[1]}, {dims[0]});"
            )
            init_name = replace_diffusers_names(init_name)
            cpp_lines.append(f'    tensor_map["{init_name}"] = {var_name};')
            continue  # Skip the normal dimension handling below
        # Check if this is a Reshape shape tensor that needs data reversal
        elif init_name in reshape_shapes:
            op_type, role, original_dims = reshape_shapes[init_name]
            # The shape tensor contains dimension values as DATA, not as tensor shape
            # We need to reverse the VALUES inside the tensor
            # First create the tensor normally
            cpp_lines.append(f"    // {init_name} - {op_type} {role} (reversing shape values from ONNX to GGML)")
            cpp_lines.append(
                f"    {var_name} = ggml_new_tensor_1d(ctx_weights, {get_ggml_type(init.data_type)}, {dims[0]});"
            )
            init_name = replace_diffusers_names(init_name)
            cpp_lines.append(f'    tensor_map["{init_name}"] = {var_name};')
            # Note: We'll need to reverse the actual data when loading from safetensors
            # This will be handled in the data loading section
            continue  # Skip the normal dimension handling below
        else:
            cpp_lines.append(f"    // {init_name}")

        # Normalize to 4D for consistent broadcasting in GGML
        # ONNX broadcasts from the right, so [128,1,1] -> [1,128,1,1]
        dims_4d = normalize_dims_to_4d(dims)

        # Add comment if dimensions were padded
        if len(dims) != len(dims_4d):
            cpp_lines.append(f"    // Original ONNX shape {dims} -> normalized to {dims_4d}")

        # GGML uses REVERSED dimension order: ONNX [N,C,H,W] -> GGML [W,H,C,N]
        cpp_lines.append(f"    // ONNX {dims_4d} -> GGML reversed")
        cpp_lines.append(
            f"    {var_name} = ggml_new_tensor_4d(ctx_weights, {get_ggml_type(init.data_type)}, {dims_4d[3]}, {dims_4d[2]}, {dims_4d[1]}, {dims_4d[0]});"
        )

        # Add to tensor map
        init_name = replace_diffusers_names(init_name)
        cpp_lines.append(f'    tensor_map["{init_name}"] = {var_name};')

    # Process constant tensors
    for const_name, (dims, data_type, values) in constant_tensors.items():
        var_name = sanitize_name(const_name)
        cpp_lines.append(f"    // {const_name} - Constant tensor")

        # Normalize to 4D for consistent broadcasting in GGML
        dims_4d = normalize_dims_to_4d(dims)

        # Add comment if dimensions were padded
        if len(dims) != len(dims_4d):
            cpp_lines.append(f"    // Original ONNX shape {dims} -> normalized to {dims_4d}")

        # GGML uses REVERSED dimension order: ONNX [N,C,H,W] -> GGML [W,H,C,N]
        cpp_lines.append(f"    // ONNX {dims_4d} -> GGML reversed")
        cpp_lines.append(
            f"    {var_name} = ggml_new_tensor_4d(ctx_weights, {get_ggml_type(data_type)}, {dims_4d[3]}, {dims_4d[2]}, {dims_4d[1]}, {dims_4d[0]});"
        )

        # Add to tensor map
        const_name = replace_diffusers_names(const_name)
        cpp_lines.append(f'    tensor_map["{const_name}"] = {var_name};')

    cpp_lines.append("")
    cpp_lines.append("    // Allocate memory for weight tensors")
    cpp_lines.append("    ggml_backend_alloc_ctx_tensors(ctx_weights, backend);")
    cpp_lines.append("")

    # Initialize constant tensor values
    if constant_tensors:
        cpp_lines.append("    // Initialize constant tensor values")
        for const_name, (dims, data_type, values) in constant_tensors.items():
            var_name = sanitize_name(const_name)

            # Generate array of values
            if data_type == 1:  # FLOAT
                vals_str = ", ".join([f"{v}f" for v in values])
                cpp_lines.append(f"    {{")
                cpp_lines.append(f"        float {var_name}_data[] = {{{vals_str}}};")
                cpp_lines.append(
                    f"        ggml_backend_tensor_set({var_name}, {var_name}_data, 0, sizeof({var_name}_data));"
                )
                cpp_lines.append(f"    }}")
            elif data_type in [6, 7]:  # INT32, INT64
                vals_str = ", ".join([str(int(v)) for v in values])
                cpp_lines.append(f"    {{")
                cpp_lines.append(f"        int32_t {var_name}_data[] = {{{vals_str}}};")
                cpp_lines.append(
                    f"        ggml_backend_tensor_set({var_name}, {var_name}_data, 0, sizeof({var_name}_data));"
                )
                cpp_lines.append(f"    }}")
        cpp_lines.append("")

    cpp_lines.append("    // Load weight data from safetensors file")
    cpp_lines.append('    // std::cout << "Loading weights from: " << weights_file << std::endl;')
    cpp_lines.append("    std::unordered_set<std::string> loaded_tensors;  // Track which tensors were loaded")
    cpp_lines.append(
        "    safetensors::load_from_file(weights_file, [&loaded_tensors](const std::string& key, const std::string& dtype, const std::vector<uint64_t>& shape, const std::vector<uint8_t>& tensor_data) {"
    )
    cpp_lines.append(
        '        // std::cout << "Read tensor: " << key << ", size: " << tensor_data.size() << " bytes" << std::endl;'
    )
    cpp_lines.append("")
    cpp_lines.append("        auto it = tensor_map.find(key);")
    cpp_lines.append("        if (it != tensor_map.end()) {")
    cpp_lines.append("            ggml_tensor* tensor = it->second;")
    cpp_lines.append("            loaded_tensors.insert(key);  // Mark as loaded")
    cpp_lines.append("")
    # Add special handling for reshape shape tensors
    if reshape_shapes:
        reshape_keys = [sanitize_name(k).replace("_", ".") for k in reshape_shapes.keys()]
        reshape_keys_str = " || ".join([f'key == "{k}"' for k in reshape_shapes.keys()])
        cpp_lines.append("            // Special handling for Reshape shape tensors - reverse the dimension values")
        cpp_lines.append(f"            if ({reshape_keys_str}) {{")
        cpp_lines.append("                // Reverse the shape values from ONNX to GGML order")
        cpp_lines.append('                if (dtype == "I64") {')
        cpp_lines.append(
            "                    const int64_t* onnx_shape = reinterpret_cast<const int64_t*>(tensor_data.data());"
        )
        cpp_lines.append("                    size_t num_dims = tensor_data.size() / sizeof(int64_t);")
        cpp_lines.append("                    std::vector<int64_t> ggml_shape(num_dims);")
        cpp_lines.append("                    for (size_t i = 0; i < num_dims; ++i) {")
        cpp_lines.append("                        ggml_shape[i] = onnx_shape[num_dims - 1 - i];")
        cpp_lines.append("                    }")
        cpp_lines.append(
            "                    ggml_backend_tensor_set(tensor, ggml_shape.data(), 0, tensor_data.size());"
        )
        cpp_lines.append("                } else {")
        cpp_lines.append(
            "                    ggml_backend_tensor_set(tensor, tensor_data.data(), 0, ggml_nbytes(tensor));"
        )
        cpp_lines.append("                }")
        cpp_lines.append("            } else {")
        cpp_lines.append("                ggml_backend_tensor_set(tensor, tensor_data.data(), 0, ggml_nbytes(tensor));")
        cpp_lines.append("            }")
    else:
        cpp_lines.append("            ggml_backend_tensor_set(tensor, tensor_data.data(), 0, ggml_nbytes(tensor));")
    cpp_lines.append("        } else {")
    cpp_lines.append('            std::cout << "Warning: Unknown tensor key: " << key << std::endl;')
    cpp_lines.append("        }")
    cpp_lines.append("    });")
    cpp_lines.append('    // std::cout << "Weights loaded successfully" << std::endl;')
    cpp_lines.append("")

    # Generate code to initialize missing weight-bias pairs based on naming patterns
    cpp_lines.append("    // Initialize missing weight-bias pairs with default values")
    cpp_lines.append("    // Following PyTorch's approach: kaiming_uniform for weights, uniform for biases")
    cpp_lines.append('    std::cout << "Checking for missing weight-bias pairs..." << std::endl;')
    cpp_lines.append("")
    cpp_lines.append("    // Look for .weight and .bias pairs in the tensor map")
    cpp_lines.append("    for (const auto& [key, tensor] : tensor_map) {")
    cpp_lines.append("        // Check if this is a weight tensor (ends with .weight)")
    cpp_lines.append('        if (key.length() > 7 && key.substr(key.length() - 7) == ".weight") {')
    cpp_lines.append("            // Check if weight was not loaded from safetensors")
    cpp_lines.append("            if (loaded_tensors.find(key) == loaded_tensors.end()) {")
    cpp_lines.append('                std::cout << "  Initializing missing weight: " << key << std::endl;')
    cpp_lines.append("                int64_t fan_in, fan_out;")
    cpp_lines.append("                calculate_fan_in_fan_out(tensor, fan_in, fan_out);")
    cpp_lines.append("                kaiming_uniform_init(tensor, fan_in);")
    cpp_lines.append("            }")
    cpp_lines.append("            ")
    cpp_lines.append("            // Check for corresponding bias")
    cpp_lines.append('            std::string bias_key = key.substr(0, key.length() - 7) + ".bias";')
    cpp_lines.append("            auto bias_it = tensor_map.find(bias_key);")
    cpp_lines.append("            if (bias_it != tensor_map.end()) {")
    cpp_lines.append("                // Bias exists, check if it was loaded")
    cpp_lines.append("                if (loaded_tensors.find(bias_key) == loaded_tensors.end()) {")
    cpp_lines.append('                    std::cout << "  Initializing missing bias: " << bias_key << std::endl;')
    cpp_lines.append("                    int64_t fan_in, fan_out;")
    cpp_lines.append("                    calculate_fan_in_fan_out(tensor, fan_in, fan_out);")
    cpp_lines.append("                    bias_uniform_init(bias_it->second, fan_in);")
    cpp_lines.append("                }")
    cpp_lines.append("            }")
    cpp_lines.append("        }")
    cpp_lines.append("    }")
    cpp_lines.append("")

    cpp_lines.append("}")
    cpp_lines.append("")

    # Generate getGraph function in the graph header file
    graph_lines.append("// Build the computation graph")
    graph_lines.append("ggml_cgraph* getGraph(ggml_context* ctx, ggml_tensor* input) {")
    # Find input tensors
    inputs = {inp.name: inp for inp in graph.input if inp.name not in initializers}

    # Assign input tensor variable
    if len(inputs) == 1:
        input_name = list(inputs.keys())[0]
        var_name = sanitize_name(input_name)
        tensor_vars[input_name] = var_name
        if var_name != "input":
            graph_lines.append(f"    // Input: {input_name}")
            graph_lines.append(f"    auto {var_name} = input;")
            graph_lines.append("")
    else:
        graph_lines.append("    // TODO: Handle multiple inputs")
        for i, (input_name, inp) in enumerate(inputs.items()):
            var_name = sanitize_name(input_name)
            tensor_vars[input_name] = var_name
            graph_lines.append(f"    auto {var_name} = input; // {input_name}")
        graph_lines.append("")

    # Reference weight tensors (already created globally)
    for init_name in initializers.keys():
        var_name = sanitize_name(init_name)
        tensor_vars[init_name] = var_name

    # Reference constant tensors that were converted to weight tensors
    for const_name in constant_tensors.keys():
        var_name = sanitize_name(const_name)
        tensor_vars[const_name] = var_name

    # Track constant values (not tensors)
    constant_values = {}  # Maps: constant_name -> (type, value_str, value_data)

    # Fourth pass: process Constant nodes for inline constants (not converted to tensors)
    for node in graph.node:
        if node.op_type == "Constant":
            if len(node.output) > 0:
                const_name = node.output[0]

                # Skip if already converted to tensor
                if const_name in constant_tensors:
                    continue

                # Look for 'value' attribute
                for attr in node.attribute:
                    if attr.name == "value":
                        if attr.type == onnx.AttributeProto.TENSOR:
                            # Extract tensor data
                            tensor = attr.t
                            data_type = tensor.data_type
                            dims = list(tensor.dims)

                            # Get the actual values
                            if data_type == 1:  # FLOAT
                                if tensor.float_data:
                                    values = list(tensor.float_data)
                                else:
                                    values = list(numpy_helper.to_array(tensor).flatten())
                            elif data_type in [6, 7]:  # INT32, INT64
                                if tensor.int64_data:
                                    values = list(tensor.int64_data)
                                elif tensor.int32_data:
                                    values = list(tensor.int32_data)
                                else:
                                    values = list(numpy_helper.to_array(tensor).flatten())
                            else:
                                values = list(numpy_helper.to_array(tensor).flatten())

                            # Determine C++ type and value based on shape
                            # If dims is empty (scalar) or not present, create scalar constant
                            # If dims is present (even [1]), create vector constant
                            if len(dims) == 0:
                                # True scalar constant (no shape)
                                if data_type == 1:  # FLOAT
                                    const_type = "float"
                                    value_str = f"{values[0]}f"
                                elif data_type in [6, 7]:  # INT32, INT64
                                    const_type = "int64_t"
                                    value_str = f"{int(values[0])}"
                                else:
                                    const_type = "float"
                                    value_str = f"{float(values[0])}f"
                                constant_values[const_name] = (const_type, value_str, values[0])
                            else:
                                # Vector/array constant (has shape, even if shape is [1])
                                if data_type == 1:  # FLOAT
                                    const_type = "std::vector<float>"
                                    vals_str = ", ".join([f"{v}f" for v in values])
                                elif data_type in [6, 7]:  # INT32, INT64
                                    const_type = "std::vector<int64_t>"
                                    # Special handling for Reshape shape constants - reverse the values
                                    if const_name in constants_used_as_reshape_shapes:
                                        # Reverse from ONNX to GGML order
                                        values_reversed = list(reversed(values))
                                        vals_str = ", ".join([str(int(v)) for v in values_reversed])
                                    # Special handling for Unsqueeze/Squeeze axes - convert axis positions
                                    elif const_name in unsqueeze_squeeze_axes:
                                        op_type, input_name = unsqueeze_squeeze_axes[const_name]
                                        # Get input tensor rank
                                        input_rank = len(tensor_shapes.get(input_name, []))
                                        if input_rank == 0:
                                            # Unknown rank, assume it's at least 1D
                                            input_rank = 1

                                        # Calculate output rank
                                        num_axes = len(values)
                                        if op_type == "Unsqueeze":
                                            output_rank = input_rank + num_axes
                                        else:  # Squeeze
                                            output_rank = input_rank - num_axes

                                        # Convert each axis: ggml_axis = (output_rank - 1) - onnx_axis
                                        values_converted = [(output_rank - 1) - int(v) for v in values]
                                        vals_str = ", ".join([str(v) for v in values_converted])
                                        # Store conversion info for comment later
                                        constant_values[const_name] = (
                                            const_type,
                                            f"{{{vals_str}}}",
                                            values,
                                            f"{op_type} axes (ONNX {[int(v) for v in values]} -> GGML {values_converted}, input_rank={input_rank}, output_rank={output_rank})",
                                        )
                                    # Special handling for Slice axes - convert axis positions
                                    elif const_name in constants_used_as_slice_axes:
                                        input_name = constants_used_as_slice_axes[const_name]
                                        # Get input tensor rank
                                        input_rank = len(tensor_shapes.get(input_name, []))
                                        if input_rank == 0:
                                            # Unknown rank, assume it's at least 4D (common for vision models)
                                            input_rank = 4

                                        # For Slice, output rank equals input rank
                                        # Convert each axis: ggml_axis = (input_rank - 1) - onnx_axis
                                        values_converted = [(input_rank - 1) - int(v) for v in values]
                                        vals_str = ", ".join([str(v) for v in values_converted])
                                        # Store conversion info for comment later
                                        constant_values[const_name] = (
                                            const_type,
                                            f"{{{vals_str}}}",
                                            values,
                                            f"Slice axes (ONNX {[int(v) for v in values]} -> GGML {values_converted}, input_rank={input_rank})",
                                        )
                                    else:
                                        vals_str = ", ".join([str(int(v)) for v in values])
                                else:
                                    const_type = "std::vector<float>"
                                    vals_str = ", ".join([f"{float(v)}f" for v in values])
                                value_str = f"{{{vals_str}}}"
                                constant_values[const_name] = (const_type, value_str, values)
                    elif attr.name == "value_float":
                        const_type = "float"
                        value_str = f"{attr.f}f"
                        constant_values[const_name] = (const_type, value_str, attr.f)
                    elif attr.name == "value_int":
                        const_type = "int64_t"
                        value_str = f"{attr.i}"
                        constant_values[const_name] = (const_type, value_str, attr.i)

    # Process each node in the graph
    graph_lines.append("    // Graph operations")
    for node in graph.node:
        op_type = node.op_type
        node_name = node.name if node.name else f"{op_type}_node"

        # Skip Constant nodes - we generate variables for them inline or convert to tensors
        if op_type == "Constant":
            if len(node.output) > 0:
                const_name = node.output[0]

                # If this constant was converted to a tensor, skip it
                if const_name in constant_tensors:
                    continue

                # Otherwise, generate inline constant
                if const_name in constant_values:
                    const_val_tuple = constant_values[const_name]
                    const_type = const_val_tuple[0]
                    value_str = const_val_tuple[1]
                    # Check if there's a conversion comment (4th element)
                    has_comment = len(const_val_tuple) > 3

                    var_name = sanitize_name(const_name)
                    graph_lines.append(f"    // Node: {node_name} (Op: {op_type})")
                    # Add note if this is a Reshape shape constant that was reversed
                    if const_name in constants_used_as_reshape_shapes:
                        graph_lines.append(f"    // Reshape shape constant (reversed from ONNX to GGML order)")
                    # Add note if this is an Unsqueeze/Squeeze axes constant that was converted
                    elif has_comment:
                        graph_lines.append(f"    // {const_val_tuple[3]}")
                    graph_lines.append(f"    const {const_type} {var_name} = {value_str};")
                    tensor_vars[const_name] = var_name
                    graph_lines.append("")
            continue

        graph_lines.append(f"    // Node: {node_name} (Op: {op_type})")

        # Get input variables
        input_vars = []
        for inp in node.input:
            if inp in tensor_vars:
                input_vars.append(tensor_vars[inp])
            else:
                # Handle missing inputs (might be optional)
                sanitized = sanitize_name(inp) if inp else "nullptr"
                input_vars.append(sanitized)

        # Get output variable names
        output_vars = []
        for out in node.output:
            var_name = sanitize_name(out)
            tensor_vars[out] = var_name
            output_vars.append(var_name)

        # Generate function call
        func_name = f"ggml_onnx_{op_type.lower()}"

        # Create output variable(s)
        if len(output_vars) == 1:
            # Join input variables, but only add them if there are any
            if input_vars:
                inputs_str = ", " + ", ".join(input_vars)
            else:
                inputs_str = ""

            # Process attributes and create variables for them
            attr_vars = []
            for attr in node.attribute:
                attr_name = attr.name
                attr_var_name = f"{output_vars[0]}_{attr_name}"

                if attr.type == onnx.AttributeProto.INT:
                    graph_lines.append(f"    int64_t {attr_var_name} = {attr.i};")
                    attr_vars.append(attr_var_name)
                elif attr.type == onnx.AttributeProto.FLOAT:
                    graph_lines.append(f"    float {attr_var_name} = {attr.f}f;")
                    attr_vars.append(attr_var_name)
                elif attr.type == onnx.AttributeProto.INTS:
                    vals = list(attr.ints)
                    # Special handling for Reshape 'shape' attribute
                    # Need to reverse the dimensions from ONNX to GGML order
                    if op_type == "Reshape" and attr_name == "shape" and len(vals) > 0:
                        # Reverse the shape dimensions for GGML
                        vals_reversed = list(reversed(vals))
                        vals_str = ", ".join(map(str, vals_reversed))
                        graph_lines.append(f"    // Reshape shape: ONNX {list(vals)} -> GGML {vals_reversed}")
                        graph_lines.append(f"    std::vector<int64_t> {attr_var_name} = {{{vals_str}}};")
                    # Special handling for Pad 'pads' attribute
                    # ONNX format: [x1_begin, x2_begin, ..., xn_begin, x1_end, x2_end, ..., xn_end]
                    # The ggml_onnx_pad function handles the conversion internally, so pass as-is
                    elif op_type == "Pad" and attr_name == "pads" and len(vals) > 0:
                        vals_str = ", ".join(map(str, vals))
                        graph_lines.append(
                            f"    // Pad pads (ONNX format - will be converted inside ggml_onnx_pad): {list(vals)}"
                        )
                        graph_lines.append(f"    std::vector<int64_t> {attr_var_name} = {{{vals_str}}};")
                    # Special handling for Unsqueeze/Squeeze 'axes' attribute
                    # Need to convert axis positions from ONNX to GGML order
                    # ONNX axes refer to output positions in ONNX order [N,C,H,W]
                    # GGML uses reversed order [W,H,C,N], so axis positions need conversion
                    elif (op_type in ["Unsqueeze", "Squeeze"]) and attr_name == "axes" and len(vals) > 0:
                        # Determine the output rank to know how to convert axes
                        # For Unsqueeze: output_rank = input_rank + len(axes)
                        # For Squeeze: output_rank = input_rank - len(axes)
                        # We'll assume 4D output for now (most common case)
                        output_rank = 4
                        # Convert each axis: ggml_axis = (output_rank - 1) - onnx_axis
                        vals_converted = [output_rank - 1 - v for v in vals]
                        vals_str = ", ".join(map(str, vals_converted))
                        graph_lines.append(f"    // {op_type} axes: ONNX {list(vals)} -> GGML {vals_converted}")
                        graph_lines.append(f"    std::vector<int64_t> {attr_var_name} = {{{vals_str}}};")
                    else:
                        vals_str = ", ".join(map(str, vals))
                        graph_lines.append(f"    std::vector<int64_t> {attr_var_name} = {{{vals_str}}};")
                    attr_vars.append(attr_var_name)
                elif attr.type == onnx.AttributeProto.FLOATS:
                    vals = ", ".join([f"{v}f" for v in attr.floats])
                    graph_lines.append(f"    std::vector<float> {attr_var_name} = {{{vals}}};")
                    attr_vars.append(attr_var_name)
                elif attr.type == onnx.AttributeProto.STRING:
                    graph_lines.append(f'    std::string {attr_var_name} = "{attr.s.decode()}";')
                    attr_vars.append(attr_var_name)

            # Add attribute variables to function call
            if attr_vars:
                attrs_str = ", " + ", ".join(attr_vars)
            else:
                attrs_str = ""

            if print_values:
                graph_lines.append(f'    std::cout << "\\nInserting {op_type} node: {node_name}\\n";')
            graph_lines.append(f"    auto {output_vars[0]} = {func_name}(ctx{inputs_str}{attrs_str});")

            # Set tensor name for debugging/printing if print_values is enabled
            if print_values and len(node.output) > 0:
                original_output_name = node.output[0]
                # Use snprintf to safely copy the name, respecting GGML_MAX_NAME
                graph_lines.append(f"    if ({output_vars[0]}) {{")
                graph_lines.append(
                    f'        snprintf({output_vars[0]}->name, sizeof({output_vars[0]}->name), "{original_output_name}");'
                )
                graph_lines.append(f"    }}")
        else:
            graph_lines.append(f"    // Multiple outputs from {op_type}")
            for out_var in output_vars:
                graph_lines.append(f"    ggml_tensor* {out_var} = nullptr; // TODO: Handle multi-output ops")

        graph_lines.append("")

    # Find output tensor
    outputs = {out.name: out for out in graph.output}

    graph_lines.append("    // Build forward graph")
    graph_lines.append("    ggml_cgraph* gf = ggml_new_graph(ctx);")

    if len(outputs) == 1:
        output_name = list(outputs.keys())[0]
        output_var = tensor_vars.get(output_name, sanitize_name(output_name))
        graph_lines.append(f"    // Graph output: {output_name}")
        graph_lines.append(f"    ggml_build_forward_expand(gf, {output_var});")
    else:
        graph_lines.append("    // Multiple outputs")
        for output_name in outputs.keys():
            output_var = tensor_vars.get(output_name, sanitize_name(output_name))
            graph_lines.append(f"    ggml_build_forward_expand(gf, {output_var}); // {output_name}")

    graph_lines.append("")
    graph_lines.append("    return gf;")
    graph_lines.append("}")
    graph_lines.append("")

    # Generate predict function
    cpp_lines.append("// Execute the graph and print output")
    cpp_lines.append("void predict(float* input_data) {")
    cpp_lines.append("    // Create context for computation")
    cpp_lines.append("    ggml_init_params params = {")
    cpp_lines.append(
        "        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),"
    )
    cpp_lines.append("        /*.mem_buffer =*/ NULL,")
    cpp_lines.append("        /*.no_alloc   =*/ true,")
    cpp_lines.append("    };")
    cpp_lines.append("    ggml_context* ctx = ggml_init(params);")
    cpp_lines.append("")
    cpp_lines.append("    // Create input tensor with graph's expected shape")
    cpp_lines.append("    // Note: Converting ONNX dimension order (NCHW) to GGML order (WHCN)")

    if len(inputs) == 1:
        input_info = list(inputs.values())[0]
        input_name = list(inputs.keys())[0]
        shape = input_info.type.tensor_type.shape
        dims = [d.dim_value if d.dim_value > 0 else -1 for d in shape.dim]
        elem_type = input_info.type.tensor_type.elem_type
        ggml_type = get_ggml_type(elem_type)

        cpp_lines.append(f"    // Input '{input_name}' ONNX shape: {dims}")

        # Filter out dynamic dimensions (-1) for the actual tensor creation
        static_dims = [d for d in dims if d > 0]

        if len(static_dims) == 0:
            cpp_lines.append("    // ERROR: All dimensions are dynamic, cannot create tensor")
            cpp_lines.append("    ggml_tensor* input = nullptr;")
        elif len(dims) == 1:
            dim_str = str(dims[0]) if dims[0] > 0 else "1 /* dynamic */"
            cpp_lines.append(f"    ggml_tensor* input = ggml_new_tensor_1d(ctx, {ggml_type}, {dim_str});")
        elif len(dims) == 2:
            # 2D: ONNX [dim0, dim1] -> GGML [dim1, dim0]
            dim0_str = str(dims[0]) if dims[0] > 0 else "1 /* dynamic */"
            dim1_str = str(dims[1]) if dims[1] > 0 else "1 /* dynamic */"
            cpp_lines.append(f"    ggml_tensor* input = ggml_new_tensor_2d(ctx, {ggml_type}, {dim1_str}, {dim0_str});")
        elif len(dims) == 3:
            # 3D: ONNX [dim0, dim1, dim2] -> GGML [dim2, dim1, dim0]
            dim0_str = str(dims[0]) if dims[0] > 0 else "1 /* dynamic */"
            dim1_str = str(dims[1]) if dims[1] > 0 else "1 /* dynamic */"
            dim2_str = str(dims[2]) if dims[2] > 0 else "1 /* dynamic */"
            cpp_lines.append(
                f"    ggml_tensor* input = ggml_new_tensor_3d(ctx, {ggml_type}, {dim2_str}, {dim1_str}, {dim0_str});"
            )
        elif len(dims) == 4:
            # 4D: ONNX [N, C, H, W] -> GGML [W, H, C, N]
            dim0_str = str(dims[0]) if dims[0] > 0 else "1 /* dynamic */"
            dim1_str = str(dims[1]) if dims[1] > 0 else "1 /* dynamic */"
            dim2_str = str(dims[2]) if dims[2] > 0 else "1 /* dynamic */"
            dim3_str = str(dims[3]) if dims[3] > 0 else "1 /* dynamic */"
            cpp_lines.append(
                f"    ggml_tensor* input = ggml_new_tensor_4d(ctx, {ggml_type}, {dim3_str}, {dim2_str}, {dim1_str}, {dim0_str});"
            )
        else:
            # For higher dimensions, use generic tensor creation
            dims_str = ", ".join([str(d) if d > 0 else "1" for d in dims])
            cpp_lines.append(f"    int64_t input_dims[] = {{{dims_str}}};")
            cpp_lines.append(f"    ggml_tensor* input = ggml_new_tensor(ctx, {ggml_type}, {len(dims)}, input_dims);")

    else:
        cpp_lines.append("    // TODO: Handle multiple inputs")
        for input_name, input_info in inputs.items():
            shape = input_info.type.tensor_type.shape
            dims = [d.dim_value if d.dim_value > 0 else -1 for d in shape.dim]
            cpp_lines.append(f"    // Input '{input_name}' shape: {dims}")
        cpp_lines.append("    ggml_tensor* input = nullptr;")

    cpp_lines.append("")
    cpp_lines.append("    // Build computation graph")
    cpp_lines.append("    ggml_cgraph* gf = getGraph(ctx, input);")
    cpp_lines.append("")
    cpp_lines.append("    // Print input tensor shape for debugging")
    cpp_lines.append(
        '    std::cout << "Input tensor dims: [" << input->ne[0] << ", " << input->ne[1] << ", " << input->ne[2] << ", " << input->ne[3] << "]" << std::endl;'
    )
    cpp_lines.append("")
    cpp_lines.append("    // Allocate tensors")
    cpp_lines.append("    ggml_gallocr_alloc_graph(allocr, gf);")
    cpp_lines.append("")
    cpp_lines.append("    // Set input data")
    cpp_lines.append("    ggml_backend_tensor_set(input, input_data, 0, ggml_nbytes(input));")
    cpp_lines.append("")
    cpp_lines.append('    std::cout << std::endl << "Starting graph computation..." << std::endl;')
    cpp_lines.append("    // Compute the graph")
    cpp_lines.append("    ggml_backend_graph_compute(backend, gf);")
    cpp_lines.append("")

    # Add code to print intermediate values if enabled
    if print_values:
        cpp_lines.append("    // Print input tensor values")
        cpp_lines.append('    std::cout << "\\nInput Tensor Values:\\n";')
        cpp_lines.append(
            '    std::cout << "================================================================================" << std::endl;'
        )
        cpp_lines.append('    print_tensor_values("input", input);')
        cpp_lines.append("    std::cout << std::endl;")
        cpp_lines.append("")
        cpp_lines.append("    // Print intermediate tensor values")
        cpp_lines.append('    std::cout << "\\nIntermediate Tensor Values:\\n";')
        cpp_lines.append(
            '    std::cout << "================================================================================" << std::endl;'
        )
        cpp_lines.append("")
        cpp_lines.append("    // Print all nodes in the graph")
        cpp_lines.append("    int n_nodes = ggml_graph_n_nodes(gf);")
        cpp_lines.append("    for (int i = 0; i < n_nodes; i++) {")
        cpp_lines.append("        ggml_tensor* node = ggml_graph_node(gf, i);")
        cpp_lines.append("        if (node && node->name[0] != '\\0') {")
        cpp_lines.append("            print_tensor_values(node->name, node);")
        cpp_lines.append("            std::cout << std::endl;")
        cpp_lines.append("        }")
        cpp_lines.append("    }")
        cpp_lines.append("")
    cpp_lines.append("    // Get output tensor")
    cpp_lines.append("    ggml_tensor* result_node = ggml_graph_node(gf, -1);  // get the last node in the graph")
    cpp_lines.append("")
    cpp_lines.append("    // Print output data")
    cpp_lines.append('    std::cout << "\\nOutput Tensor Values:\\n";')
    cpp_lines.append(
        '    std::cout << "================================================================================" << std::endl;'
    )
    cpp_lines.append('    print_tensor_values("output", result_node);')
    cpp_lines.append("    std::cout << std::endl;")
    cpp_lines.append("")
    cpp_lines.append("    // Free context")
    cpp_lines.append("    ggml_free(ctx);")
    cpp_lines.append("}")
    cpp_lines.append("")

    # Generate main function
    cpp_lines.append("int main(int argc, char* argv[]) {")
    cpp_lines.append("    // Check command line arguments")
    cpp_lines.append("    if (argc < 2) {")
    cpp_lines.append('        std::cerr << "Usage: " << argv[0] << " <weights_file.sft>" << std::endl;')
    cpp_lines.append("        return 1;")
    cpp_lines.append("    }")
    cpp_lines.append("")
    cpp_lines.append("    std::string weights_file = argv[1];")
    cpp_lines.append("")
    cpp_lines.append("    // Initialize backend and allocator")
    cpp_lines.append("    init_backend();")
    cpp_lines.append("    init_mem_allocator();")
    cpp_lines.append("")
    cpp_lines.append("    // Load model weights from safetensors file")
    cpp_lines.append("    load_weights(weights_file);")
    cpp_lines.append("")
    cpp_lines.append("    // Create sample input data")

    if len(inputs) == 1:
        input_info = list(inputs.values())[0]
        shape = input_info.type.tensor_type.shape
        dims = [d.dim_value if d.dim_value > 0 else 1 for d in shape.dim]
        total_elements = []
        for d in dims:
            total_elements.append(str(d))
        total_elements_str = " * ".join(total_elements)
        cpp_lines.append(f"    // Input shape: {dims}")
        cpp_lines.append(
            f"    std::vector<float> input_data({total_elements_str}, 0.0f);  // TODO: Set actual input values"
        )
    else:
        cpp_lines.append("    std::vector<float> input_data(1, 0.0f);  // TODO: Set actual input values")

    cpp_lines.append("")
    cpp_lines.append("    // Run prediction")
    cpp_lines.append("    predict(input_data.data());")
    cpp_lines.append("")
    cpp_lines.append("    // Clean up")
    cpp_lines.append("    ggml_gallocr_free(allocr);")
    cpp_lines.append("    ggml_free(ctx_weights);")
    cpp_lines.append("    ggml_backend_free(backend);")
    cpp_lines.append("")
    cpp_lines.append("    return 0;")
    cpp_lines.append("}")
    cpp_lines.append("")

    # Write both files
    # Write the graph header file
    with open(graph_header_path, "w") as f:
        f.write("\n".join(graph_lines))

    # Write the main cpp file
    with open(output_path, "w") as f:
        f.write("\n".join(cpp_lines))

    print(f"Generated C++ files:")
    print(f"  - Graph header: {graph_header_path}")
    print(f"  - Main file: {output_path}")
    print(f"  - Input tensors: {len(inputs)}")
    print(f"  - Output tensors: {len(outputs)}")
    print(f"  - Operations: {len(graph.node)}")
    print(f"  - Initializers: {len(initializers)}")
    print(f"  - Constant tensors: {len(constant_tensors)}")


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to C++ code with ggml function calls")
    parser.add_argument("input", type=str, help="Input ONNX model file")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output C++ file (default: <input_name>_ggml.cpp)"
    )
    parser.add_argument(
        "--print-values", action="store_true", help="Generate code to print intermediate tensor values during execution"
    )

    args = parser.parse_args()

    # Load ONNX model
    try:
        model = onnx.load(args.input)
        onnx.checker.check_model(model)
        print(f"Loaded ONNX model: {args.input}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}", file=sys.stderr)
        return 1

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_ggml.cpp"

    # Generate C++ code
    try:
        generate_cpp_code(model, output_path, print_values=args.print_values)
        output_path_obj = Path(output_path)
        graph_header_name = f"{output_path_obj.stem}_graph.h"
        print(f"\nSuccess! Generated files:")
        print(f"  - {output_path}")
        print(f"  - {output_path_obj.parent / graph_header_name}")
        print(f"\nNext steps:")
        print(f"  1. Ensure safetensors.hpp is in your include path")
        print(f"  2. Implement ggml_onnx_* functions in ggml-onnx.h")
        print(f"  3. Export your model weights to safetensors format (e.g., model.sft)")
        print(f"  4. Compile with: g++ -o model {output_path} -lggml -I<ggml_include_path>")
        print(f"  5. Run: ./model <weights_file.sft>")
        return 0
    except Exception as e:
        print(f"Error generating C++ code: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
