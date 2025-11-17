#!/usr/bin/env python3
"""
Script to run ONNX models with zero-filled inputs.
Accepts an ONNX file and input shape, then runs inference.
"""

import argparse
import numpy as np
import onnxruntime as ort
import onnx
import tempfile
import os

# Suppress ONNX Runtime warnings
ort.set_default_logger_severity(3)  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal


def parse_shape(shape_str):
    """
    Parse a shape string like "1,3,224,224" into a tuple of integers.

    Args:
        shape_str: String representation of shape (e.g., "1,3,224,224")

    Returns:
        Tuple of integers representing the shape
    """
    try:
        return tuple(int(dim) for dim in shape_str.split(","))
    except ValueError as e:
        raise ValueError(
            f"Invalid shape format: {shape_str}. Use comma-separated integers (e.g., '1,3,224,224')"
        ) from e


def run_onnx_model(onnx_file, input_shape, print_values=False):
    """
    Run an ONNX model with zero-filled input.

    Args:
        onnx_file: Path to the ONNX model file
        input_shape: Tuple representing the shape of the input tensor
        print_values: Whether to print intermediate tensor values
    """
    print(f"Loading ONNX model from: {onnx_file}")

    # Create an ONNX Runtime session
    sess_options = ort.SessionOptions()
    if print_values:
        # Enable intermediate outputs for printing
        sess_options.enable_profiling = False
    session = ort.InferenceSession(onnx_file, sess_options=sess_options, providers=["CPUExecutionProvider"])

    # Print model information
    print("\nModel Information:")
    print(f"  Inputs:")
    for inp in session.get_inputs():
        print(f"    - Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

    print(f"  Outputs:")
    for out in session.get_outputs():
        print(f"    - Name: {out.name}, Shape: {out.shape}, Type: {out.type}")

    # Deduce input name from the model
    input_tensor = session.get_inputs()[0]
    input_name = input_tensor.name
    print(f"\nUsing input name: {input_name}")

    # Create zero-filled input
    print(f"\nCreating zero-filled input:")
    print(f"  Shape: {input_shape}")

    input_data = np.zeros(input_shape, dtype=np.float32)
    print(f"  Data type: {input_data.dtype}")
    print(f"  Total elements: {input_data.size}")

    # Run inference
    print("\nRunning inference...")

    if print_values:
        # Load the ONNX model and add intermediate outputs
        model = onnx.load(onnx_file)

        # Run shape inference to get type information for all tensors
        print("\nInferring shapes and types for intermediate tensors...")
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as e:
            print(f"Warning: Shape inference failed: {e}")
            print("Proceeding without full type information...")

        # Collect all intermediate tensor names (outputs of each node)
        intermediate_names = []
        for node in model.graph.node:
            for output in node.output:
                intermediate_names.append(output)

        # Get original output names
        original_output_names = [out.name for out in model.graph.output]

        # Add intermediate tensors as model outputs
        print(f"Adding {len(intermediate_names)} intermediate outputs to model...")

        # Build a lookup of tensor name to type info
        tensor_type_map = {}
        for vi in model.graph.value_info:
            tensor_type_map[vi.name] = vi
        for inp in model.graph.input:
            tensor_type_map[inp.name] = inp
        for out in model.graph.output:
            tensor_type_map[out.name] = out

        # Add intermediate tensors as outputs
        for intermediate_name in intermediate_names:
            if intermediate_name not in original_output_names:
                # Check if we already have this as an output
                already_output = any(out.name == intermediate_name for out in model.graph.output)
                if not already_output:
                    if intermediate_name in tensor_type_map:
                        # Use the existing type information
                        model.graph.output.append(tensor_type_map[intermediate_name])
                    else:
                        # Fallback: create without type info (let ONNX infer it)
                        new_output = onnx.helper.make_empty_tensor_value_info(intermediate_name)
                        model.graph.output.append(new_output)

        # Save modified model to a temporary location
        temp_model_path = os.path.join(tempfile.gettempdir(), "temp_model_with_intermediates.onnx")
        onnx.save(model, temp_model_path)

        # Create new session with modified model
        session_with_intermediates = ort.InferenceSession(
            temp_model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )

        # Run inference with all outputs
        all_outputs = session_with_intermediates.run(None, {input_name: input_data})
        all_output_names = [out.name for out in session_with_intermediates.get_outputs()]

        # Build a mapping from output tensor names to nodes and their inputs
        tensor_to_node = {}
        for node in model.graph.node:
            for output in node.output:
                tensor_to_node[output] = node

        # Build a mapping from tensor names to their output values
        tensor_value_map = {}
        for name, output in zip(all_output_names, all_outputs):
            tensor_value_map[name] = output

        # Build a mapping of initializers (constants)
        initializer_map = {}
        for init in model.graph.initializer:
            initializer_map[init.name] = init

        # Print intermediate values
        print("\nIntermediate Tensor Values:")
        print("=" * 80)
        for name, output in zip(all_output_names, all_outputs):
            if output is not None:
                # Get the node that produces this output
                node = tensor_to_node.get(name)
                op_type = node.op_type if node else "UNKNOWN"

                flattened = output.flatten()
                num_elements = min(10, len(flattened))

                print(f"\n{name} ({op_type}):")

                # Print inputs if the node exists
                if node and len(node.input) > 0:
                    print(f"  Input:")
                    for input_name in node.input:
                        if input_name in tensor_value_map:
                            input_value = tensor_value_map[input_name]
                            input_node = tensor_to_node.get(input_name)
                            input_op_type = input_node.op_type if input_node else "NONE"

                            input_flattened = input_value.flatten()
                            input_num_elements = min(10, len(input_flattened))

                            print(f"    {input_name} ({input_op_type}):")
                            print(f"      Shape: {input_value.shape}, Type: {input_value.dtype}")
                            print(
                                f"      Data (first {input_num_elements}): {input_flattened[:input_num_elements].tolist()}"
                            )
                        elif input_name in initializer_map:
                            # This is a constant/initializer
                            init = initializer_map[input_name]
                            init_array = onnx.numpy_helper.to_array(init)
                            init_flattened = init_array.flatten()
                            init_num_elements = min(10, len(init_flattened))

                            print(f"    {input_name} (CONST):")
                            print(f"      Shape: {init_array.shape}, Type: {init_array.dtype}")
                            print(
                                f"      Data (first {init_num_elements}): {init_flattened[:init_num_elements].tolist()}"
                            )
                        else:
                            # Input not found
                            print(f"    {input_name} (UNKNOWN): <not available>")

                print(f"  Shape: {output.shape}, Type: {output.dtype}")
                print(f"  Data (first {num_elements}): {flattened[:num_elements].tolist()}")
        print("=" * 80)

        # Clean up temp file
        try:
            os.remove(temp_model_path)
        except:
            pass

        # Extract only original outputs
        outputs = [all_outputs[all_output_names.index(name)] for name in original_output_names]
    else:
        outputs = session.run(None, {input_name: input_data})

    # Print outputs
    print("\nInference Results:")
    output_names = [out.name for out in session.get_outputs()]
    for i, (name, output) in enumerate(zip(output_names, outputs)):
        print(f"\n  Output {i}: {name}")
        print(f"    Shape: {output.shape}")
        print(f"    Data type: {output.dtype}")
        print(f"    Min value: {output.min()}")
        print(f"    Max value: {output.max()}")
        print(f"    Mean value: {output.mean()}")

        # Print a sample of the output (first few elements)
        if output.size <= 10:
            print(f"    Values: {output.flatten()}")
        else:
            print(f"    First 10 values: {output.flatten()[:30]}")

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Run an ONNX model with zero-filled inputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.onnx --input-shape 1,3,224,224
  %(prog)s vae_folded.onnx --input-shape 1,4,64,64
  %(prog)s model.onnx --input-shape 1,3,224,224 --print-values
        """,
    )

    parser.add_argument("onnx_file", type=str, help="Path to the ONNX model file")

    parser.add_argument(
        "--input-shape",
        type=str,
        required=True,
        help='Shape of the input tensor as comma-separated integers (e.g., "1,3,224,224")',
    )

    parser.add_argument(
        "--print-values",
        action="store_true",
        help="Print first 10 elements of each intermediate tensor during execution",
    )

    args = parser.parse_args()

    # Parse the input shape
    try:
        input_shape = parse_shape(args.input_shape)
    except ValueError as e:
        parser.error(str(e))

    # Run the model
    try:
        run_onnx_model(args.onnx_file, input_shape, print_values=args.print_values)
        print("\nInference completed successfully!")
    except Exception as e:
        print(f"\nError running model: {e}")
        raise


if __name__ == "__main__":
    main()
