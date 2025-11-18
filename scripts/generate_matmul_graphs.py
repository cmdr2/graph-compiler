import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

"Debugging script to generate ONNX graphs for MatMul operations with various input configurations."


def create_constant_tensor(name, shape, start_value=0):
    """Create a constant tensor with increasing values starting from start_value."""
    total_elements = np.prod(shape)
    data = np.arange(start_value, start_value + total_elements, dtype=np.float32).reshape(shape)
    return helper.make_tensor(name=name, data_type=TensorProto.FLOAT, dims=shape, vals=data.flatten().tolist())


def create_graph_1():
    """
    Graph 1:
    Transpose_0: Input shape [1, 2, 8], Perm [0, 2, 1]
    Transpose_1: Input shape [2, 2], Perm [1, 0]
    MatMul: A = Transpose_0, B = Transpose_1
    """
    # Create constant tensors
    const_0 = create_constant_tensor("const_0", [1, 2, 8], start_value=0)
    const_1 = create_constant_tensor("const_1", [2, 2], start_value=100)

    # Create constant nodes
    const_node_0 = helper.make_node("Constant", inputs=[], outputs=["input_0"], value=const_0)

    const_node_1 = helper.make_node("Constant", inputs=[], outputs=["input_1"], value=const_1)

    # Create Transpose_0 node: [1, 2, 8] -> [1, 8, 2]
    transpose_0 = helper.make_node("Transpose", inputs=["input_0"], outputs=["transpose_0_output"], perm=[0, 2, 1])

    # Create Transpose_1 node: [2, 2] -> [2, 2]
    transpose_1 = helper.make_node("Transpose", inputs=["input_1"], outputs=["transpose_1_output"], perm=[1, 0])

    # Create MatMul node
    matmul = helper.make_node("MatMul", inputs=["transpose_0_output", "transpose_1_output"], outputs=["output"])

    # Create a dummy input (not used in the graph, just for ONNX runtime compatibility)
    dummy_input = helper.make_tensor_value_info("dummy_input", TensorProto.FLOAT, [1])

    # Create graph
    graph_def = helper.make_graph(
        [const_node_0, const_node_1, transpose_0, transpose_1, matmul],
        "graph_1",
        [dummy_input],  # Add dummy input
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 2])],
    )

    # Create model
    model = helper.make_model(graph_def, producer_name="matmul_graph_generator", ir_version=7)
    model.opset_import[0].version = 13

    return model


def create_graph_2():
    """
    Graph 2:
    MatMul: A shape [1, 1, 8, 2], B shape [1, 1, 2, 8]
    """
    # Create constant tensors
    const_a = create_constant_tensor("const_a", [1, 1, 8, 2], start_value=0)
    const_b = create_constant_tensor("const_b", [1, 1, 2, 8], start_value=200)

    # Create constant nodes
    const_node_a = helper.make_node("Constant", inputs=[], outputs=["input_a"], value=const_a)

    const_node_b = helper.make_node("Constant", inputs=[], outputs=["input_b"], value=const_b)

    # Create MatMul node
    matmul = helper.make_node("MatMul", inputs=["input_a", "input_b"], outputs=["output"])

    # Create a dummy input (not used in the graph, just for ONNX runtime compatibility)
    dummy_input = helper.make_tensor_value_info("dummy_input", TensorProto.FLOAT, [1])

    # Create graph
    graph_def = helper.make_graph(
        [const_node_a, const_node_b, matmul],
        "graph_2",
        [dummy_input],  # Add dummy input
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 8, 8])],
    )

    # Create model
    model = helper.make_model(graph_def, producer_name="matmul_graph_generator", ir_version=7)
    model.opset_import[0].version = 13

    return model


def create_graph_3():
    """
    Graph 3:
    Transpose_0: Input shape [1, 8, 1, 2], Perm [0, 2, 1, 3]
    MatMul: A shape [1, 1, 8, 8], B = Transpose_0
    """
    # Create constant tensors
    const_a = create_constant_tensor("const_a", [1, 1, 8, 8], start_value=0)
    const_b = create_constant_tensor("const_b", [1, 8, 1, 2], start_value=300)

    # Create constant nodes
    const_node_a = helper.make_node("Constant", inputs=[], outputs=["input_a"], value=const_a)

    const_node_b = helper.make_node("Constant", inputs=[], outputs=["input_b"], value=const_b)

    # Create Transpose_0 node: [1, 8, 1, 2] -> [1, 1, 8, 2]
    transpose_0 = helper.make_node("Transpose", inputs=["input_b"], outputs=["transpose_0_output"], perm=[0, 2, 1, 3])

    # Create MatMul node
    matmul = helper.make_node("MatMul", inputs=["input_a", "transpose_0_output"], outputs=["output"])

    # Create a dummy input (not used in the graph, just for ONNX runtime compatibility)
    dummy_input = helper.make_tensor_value_info("dummy_input", TensorProto.FLOAT, [1])

    # Create graph
    graph_def = helper.make_graph(
        [const_node_a, const_node_b, transpose_0, matmul],
        "graph_3",
        [dummy_input],  # Add dummy input
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 8, 2])],
    )

    # Create model
    model = helper.make_model(graph_def, producer_name="matmul_graph_generator", ir_version=7)
    model.opset_import[0].version = 13

    return model


def main():
    # Generate and save Graph 1
    print("Generating Graph 1...")
    graph_1 = create_graph_1()
    onnx.checker.check_model(graph_1)
    onnx.save(graph_1, "A.onnx")
    print("Saved A.onnx")

    # Generate and save Graph 2
    print("Generating Graph 2...")
    graph_2 = create_graph_2()
    onnx.checker.check_model(graph_2)
    onnx.save(graph_2, "B.onnx")
    print("Saved B.onnx")

    # Generate and save Graph 3
    print("Generating Graph 3...")
    graph_3 = create_graph_3()
    onnx.checker.check_model(graph_3)
    onnx.save(graph_3, "C.onnx")
    print("Saved C.onnx")

    print("\nAll graphs generated successfully!")


if __name__ == "__main__":
    main()
