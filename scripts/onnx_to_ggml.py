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


def generate_cpp_code(model, output_path):
    """Generate C++ code from ONNX model.

    Creates two files:
    - <name>_graph.h: Contains the getGraph() function
    - <name>.cpp: Contains all boilerplate code and includes the graph header
    """
    graph = model.graph

    # Collect initializers (weights/constants)
    initializers = {init.name: init for init in graph.initializer}

    # Map to track tensor name -> variable name
    tensor_vars = {}

    # Analyze graph to detect which initializers need reshaping for broadcasting
    # Maps: initializer_name -> (op_type, role, required_shape_in_ggml_order)
    reshape_info = {}

    for node in graph.node:
        op_type = node.op_type

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
    operators_with_tensor_constants = {"InstanceNormalization", "Mul", "Div", "Add", "Sub", "RandomNormalLike"}

    # First pass: identify Constant nodes and extract their values
    constant_node_outputs = {}  # Maps: output_name -> node
    for node in graph.node:
        if node.op_type == "Constant":
            if len(node.output) > 0:
                const_name = node.output[0]
                constant_node_outputs[const_name] = node

    # Second pass: determine which constants are used by target operators
    constants_used_by_target_ops = set()
    for node in graph.node:
        if node.op_type in operators_with_tensor_constants:
            for inp in node.input:
                if inp in constant_node_outputs:
                    constants_used_by_target_ops.add(inp)

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
        dims = list(init.dims)

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
            cpp_lines.append(f'    tensor_map["{init_name}"] = {var_name};')
            continue  # Skip the normal dimension handling below
        else:
            cpp_lines.append(f"    // {init_name}")

        if len(dims) == 1:
            cpp_lines.append(
                f"    {var_name} = ggml_new_tensor_1d(ctx_weights, {get_ggml_type(init.data_type)}, {dims[0]});"
            )
        elif len(dims) == 2:
            cpp_lines.append(
                f"    {var_name} = ggml_new_tensor_2d(ctx_weights, {get_ggml_type(init.data_type)}, {dims[1]}, {dims[0]});"
            )
        elif len(dims) == 3:
            cpp_lines.append(
                f"    {var_name} = ggml_new_tensor_3d(ctx_weights, {get_ggml_type(init.data_type)}, {dims[2]}, {dims[1]}, {dims[0]});"
            )
        elif len(dims) == 4:
            cpp_lines.append(
                f"    {var_name} = ggml_new_tensor_4d(ctx_weights, {get_ggml_type(init.data_type)}, {dims[3]}, {dims[2]}, {dims[1]}, {dims[0]});"
            )
        else:
            dims_str = ", ".join(map(str, dims))
            cpp_lines.append(f"    int64_t {var_name}_dims[] = {{{dims_str}}};")
            cpp_lines.append(
                f"    {var_name} = ggml_new_tensor(ctx_weights, {get_ggml_type(init.data_type)}, {len(dims)}, {var_name}_dims);"
            )

        # Add to tensor map
        cpp_lines.append(f'    tensor_map["{init_name}"] = {var_name};')

    # Process constant tensors
    for const_name, (dims, data_type, values) in constant_tensors.items():
        var_name = sanitize_name(const_name)
        cpp_lines.append(f"    // {const_name} - Constant tensor")

        if len(dims) == 1:
            cpp_lines.append(
                f"    {var_name} = ggml_new_tensor_1d(ctx_weights, {get_ggml_type(data_type)}, {dims[0]});"
            )
        elif len(dims) == 2:
            cpp_lines.append(
                f"    {var_name} = ggml_new_tensor_2d(ctx_weights, {get_ggml_type(data_type)}, {dims[1]}, {dims[0]});"
            )
        elif len(dims) == 3:
            cpp_lines.append(
                f"    {var_name} = ggml_new_tensor_3d(ctx_weights, {get_ggml_type(data_type)}, {dims[2]}, {dims[1]}, {dims[0]});"
            )
        elif len(dims) == 4:
            cpp_lines.append(
                f"    {var_name} = ggml_new_tensor_4d(ctx_weights, {get_ggml_type(data_type)}, {dims[3]}, {dims[2]}, {dims[1]}, {dims[0]});"
            )
        else:
            dims_str = ", ".join(map(str, dims))
            cpp_lines.append(f"    int64_t {var_name}_dims[] = {{{dims_str}}};")
            cpp_lines.append(
                f"    {var_name} = ggml_new_tensor(ctx_weights, {get_ggml_type(data_type)}, {len(dims)}, {var_name}_dims);"
            )

        # Add to tensor map
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
    cpp_lines.append(
        "    safetensors::load_from_file(weights_file, [](const std::string& key, const std::string& dtype, const std::vector<uint64_t>& shape, const std::vector<uint8_t>& tensor_data) {"
    )
    cpp_lines.append(
        '        // std::cout << "Read tensor: " << key << ", size: " << tensor_data.size() << " bytes" << std::endl;'
    )
    cpp_lines.append("")
    cpp_lines.append("        auto it = tensor_map.find(key);")
    cpp_lines.append("        if (it != tensor_map.end()) {")
    cpp_lines.append("            ggml_tensor* tensor = it->second;")
    cpp_lines.append("            ggml_backend_tensor_set(tensor, tensor_data.data(), 0, ggml_nbytes(tensor));")
    cpp_lines.append("        } else {")
    cpp_lines.append('            std::cout << "Warning: Unknown tensor key: " << key << std::endl;')
    cpp_lines.append("        }")
    cpp_lines.append("    });")
    cpp_lines.append('    // std::cout << "Weights loaded successfully" << std::endl;')
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
                    const_type, value_str, _ = constant_values[const_name]
                    var_name = sanitize_name(const_name)
                    graph_lines.append(f"    // Node: {node_name} (Op: {op_type})")
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

            # Add attributes if any
            attrs = []
            for attr in node.attribute:
                attr_name = attr.name
                if attr.type == onnx.AttributeProto.INT:
                    attrs.append(f"{attr_name}={attr.i}")
                elif attr.type == onnx.AttributeProto.FLOAT:
                    attrs.append(f"{attr_name}={attr.f}")
                elif attr.type == onnx.AttributeProto.INTS:
                    vals = ", ".join(map(str, attr.ints))
                    attrs.append(f"{attr_name}=[{vals}]")
                elif attr.type == onnx.AttributeProto.FLOATS:
                    vals = ", ".join(map(str, attr.floats))
                    attrs.append(f"{attr_name}=[{vals}]")
                elif attr.type == onnx.AttributeProto.STRING:
                    attrs.append(f'{attr_name}="{attr.s.decode()}"')

            if attrs:
                graph_lines.append(f'    // Attributes: {", ".join(attrs)}')

            graph_lines.append(f"    auto {output_vars[0]} = {func_name}(ctx{inputs_str});")
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

    if len(inputs) == 1:
        input_info = list(inputs.values())[0]
        input_name = list(inputs.keys())[0]
        shape = input_info.type.tensor_type.shape
        dims = [d.dim_value if d.dim_value > 0 else -1 for d in shape.dim]
        elem_type = input_info.type.tensor_type.elem_type
        ggml_type = get_ggml_type(elem_type)

        cpp_lines.append(f"    // Input '{input_name}' shape: {dims}")

        # Filter out dynamic dimensions (-1) for the actual tensor creation
        static_dims = [d for d in dims if d > 0]

        if len(static_dims) == 0:
            cpp_lines.append("    // ERROR: All dimensions are dynamic, cannot create tensor")
            cpp_lines.append("    ggml_tensor* input = nullptr;")
        elif len(dims) == 1:
            dim_str = str(dims[0]) if dims[0] > 0 else "1 /* dynamic */"
            cpp_lines.append(f"    ggml_tensor* input = ggml_new_tensor_1d(ctx, {ggml_type}, {dim_str});")
        elif len(dims) == 2:
            dim0_str = str(dims[0]) if dims[0] > 0 else "1 /* dynamic */"
            dim1_str = str(dims[1]) if dims[1] > 0 else "1 /* dynamic */"
            cpp_lines.append(f"    ggml_tensor* input = ggml_new_tensor_2d(ctx, {ggml_type}, {dim1_str}, {dim0_str});")
        elif len(dims) == 3:
            dim0_str = str(dims[0]) if dims[0] > 0 else "1 /* dynamic */"
            dim1_str = str(dims[1]) if dims[1] > 0 else "1 /* dynamic */"
            dim2_str = str(dims[2]) if dims[2] > 0 else "1 /* dynamic */"
            cpp_lines.append(
                f"    ggml_tensor* input = ggml_new_tensor_3d(ctx, {ggml_type}, {dim2_str}, {dim1_str}, {dim0_str});"
            )
        elif len(dims) == 4:
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
    cpp_lines.append("    // Allocate tensors")
    cpp_lines.append("    ggml_gallocr_alloc_graph(allocr, gf);")
    cpp_lines.append("")
    cpp_lines.append("    // Set input data")
    cpp_lines.append("    ggml_backend_tensor_set(input, input_data, 0, ggml_nbytes(input));")
    cpp_lines.append("")
    cpp_lines.append("    // Compute the graph")
    cpp_lines.append("    ggml_backend_graph_compute(backend, gf);")
    cpp_lines.append("")
    cpp_lines.append("    // Get output tensor")
    cpp_lines.append("    ggml_tensor* result_node = ggml_graph_node(gf, -1);  // get the last node in the graph")
    cpp_lines.append("")
    cpp_lines.append("    // Read result data")
    cpp_lines.append("    int64_t n = ggml_nelements(result_node);")
    cpp_lines.append("    std::vector<float> result_data(n);")
    cpp_lines.append("    ggml_backend_tensor_get(result_node, result_data.data(), 0, ggml_nbytes(result_node));")
    # cpp_lines.append("")
    # cpp_lines.append("    // Print output shape")
    # cpp_lines.append('    std::cout << "Output shape: ";')
    # cpp_lines.append("    for (int i = 0; i < result_node->n_dims; i++) {")
    # cpp_lines.append('        std::cout << result_node->ne[i] << " ";')
    # cpp_lines.append("    }")
    # cpp_lines.append("    std::cout << std::endl;")
    cpp_lines.append("")
    cpp_lines.append("    // Print output data (first 10 elements)")
    cpp_lines.append('    std::cout << "Output data: ";')
    cpp_lines.append("    for (int64_t i = 0; i < std::min(n, (int64_t)10); i++) {")
    cpp_lines.append('        std::cout << result_data[i] << ", ";')
    cpp_lines.append("    }")
    cpp_lines.append("    if (n > 10) {")
    cpp_lines.append('        std::cout << "...";')
    cpp_lines.append("    }")
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
        total_elements = 1
        for d in dims:
            total_elements *= d
        cpp_lines.append(f"    // Input shape: {dims}")
        cpp_lines.append(
            f"    std::vector<float> input_data({total_elements}, 1.0f);  // TODO: Set actual input values"
        )
    else:
        cpp_lines.append("    std::vector<float> input_data(1, 1.0f);  // TODO: Set actual input values")

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
        generate_cpp_code(model, output_path)
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
