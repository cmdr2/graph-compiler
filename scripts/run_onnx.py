#!/usr/bin/env python3
"""
Script to run ONNX models with zero-filled inputs.
Accepts an ONNX file and input shape, then runs inference.
"""

import argparse
import numpy as np
import onnxruntime as ort


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


def run_onnx_model(onnx_file, input_shape):
    """
    Run an ONNX model with zero-filled input.

    Args:
        onnx_file: Path to the ONNX model file
        input_shape: Tuple representing the shape of the input tensor
    """
    print(f"Loading ONNX model from: {onnx_file}")

    # Create an ONNX Runtime session
    session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])

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
            print(f"    First 10 values: {output.flatten()[:10]}")

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Run an ONNX model with zero-filled inputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.onnx --input-shape 1,3,224,224
  %(prog)s vae_folded.onnx --input-shape 1,4,64,64
        """,
    )

    parser.add_argument("onnx_file", type=str, help="Path to the ONNX model file")

    parser.add_argument(
        "--input-shape",
        type=str,
        required=True,
        help='Shape of the input tensor as comma-separated integers (e.g., "1,3,224,224")',
    )

    args = parser.parse_args()

    # Parse the input shape
    try:
        input_shape = parse_shape(args.input_shape)
    except ValueError as e:
        parser.error(str(e))

    # Run the model
    try:
        run_onnx_model(args.onnx_file, input_shape)
        print("\n✓ Inference completed successfully!")
    except Exception as e:
        print(f"\n✗ Error running model: {e}")
        raise


if __name__ == "__main__":
    main()
