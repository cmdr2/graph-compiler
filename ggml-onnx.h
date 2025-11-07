#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// ONNX operator implementations using ggml functions

// Conv - 2D Convolution
// Maps to ggml_conv_2d
static inline ggml_tensor* ggml_onnx_conv(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight,
                                          ggml_tensor* bias) {
    // ggml_conv_2d expects: input, kernel, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
    // Default parameters: stride=1, padding=1, dilation=1
    int s0 = 1;  // stride_h
    int s1 = 1;  // stride_w
    int p0 = 1;  // padding_h
    int p1 = 1;  // padding_w
    int d0 = 1;  // dilation_h
    int d1 = 1;  // dilation_w

    ggml_tensor* conv_result = ggml_conv_2d(ctx, weight, input, s0, s1, p0, p1, d0, d1);

    // Add bias if provided
    if (bias != NULL) {
        conv_result = ggml_add(ctx, conv_result, bias);
    }

    return conv_result;
}

// Relu - Rectified Linear Unit
// Maps to ggml_relu
static inline ggml_tensor* ggml_onnx_relu(ggml_context* ctx, ggml_tensor* input) { return ggml_relu(ctx, input); }

// GlobalAveragePool - Global Average Pooling
// Averages across spatial dimensions (H, W)
static inline ggml_tensor* ggml_onnx_globalaveragepool(ggml_context* ctx, ggml_tensor* input) {
    // For a 4D tensor [N, C, H, W], compute mean over H and W dimensions
    // ggml_mean reduces the last dimension, so we need to handle this carefully

    // Pool over height (dimension 1 in ggml's layout which is width in ONNX)
    ggml_tensor* pooled_h = ggml_pool_2d(ctx, input, GGML_OP_POOL_AVG,
                                         input->ne[0],  // kernel width = input width
                                         input->ne[1],  // kernel height = input height
                                         input->ne[0],  // stride width
                                         input->ne[1],  // stride height
                                         0, 0);         // padding

    return pooled_h;
}

// Flatten - Flatten tensor to 2D
// Maps to ggml_reshape or ggml_view
static inline ggml_tensor* ggml_onnx_flatten(ggml_context* ctx, ggml_tensor* input) {
    // Flatten all dimensions except the batch dimension
    // Input: [N, C, H, W] -> Output: [N, C*H*W]
    int64_t batch_size = input->ne[3];                                    // N (in ggml layout: [W, H, C, N])
    int64_t flattened_size = input->ne[0] * input->ne[1] * input->ne[2];  // W*H*C

    ggml_tensor* flattened = ggml_reshape_2d(ctx, input, flattened_size, batch_size);

    return flattened;
}

// Gemm - General Matrix Multiplication
// Y = alpha * A * B + beta * C
// Maps to ggml_mul_mat and ggml_add
static inline ggml_tensor* ggml_onnx_gemm(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, ggml_tensor* c) {
    // Default ONNX Gemm: Y = A * B^T + C (transB=1 is common)
    // ggml_mul_mat computes: a * b^T
    ggml_tensor* result = ggml_mul_mat(ctx, b, a);

    // Add bias if provided
    if (c != NULL) {
        result = ggml_add(ctx, result, c);
    }

    return result;
}

// MatMul - Matrix Multiplication
// Maps to ggml_mul_mat
static inline ggml_tensor* ggml_onnx_matmul(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    return ggml_mul_mat(ctx, b, a);
}

// Add - Element-wise addition
static inline ggml_tensor* ggml_onnx_add(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    return ggml_add(ctx, a, b);
}

// Sub - Element-wise subtraction
static inline ggml_tensor* ggml_onnx_sub(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    return ggml_sub(ctx, a, b);
}

// Mul - Element-wise multiplication
static inline ggml_tensor* ggml_onnx_mul(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    return ggml_mul(ctx, a, b);
}

// Div - Element-wise division
static inline ggml_tensor* ggml_onnx_div(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    return ggml_div(ctx, a, b);
}

// Sqrt - Element-wise square root
static inline ggml_tensor* ggml_onnx_sqrt(ggml_context* ctx, ggml_tensor* input) { return ggml_sqrt(ctx, input); }

// Tanh - Hyperbolic tangent
static inline ggml_tensor* ggml_onnx_tanh(ggml_context* ctx, ggml_tensor* input) { return ggml_tanh(ctx, input); }

// Sigmoid - Sigmoid activation
static inline ggml_tensor* ggml_onnx_sigmoid(ggml_context* ctx, ggml_tensor* input) { return ggml_sigmoid(ctx, input); }

// Softmax - Softmax activation
static inline ggml_tensor* ggml_onnx_softmax(ggml_context* ctx, ggml_tensor* input) {
    return ggml_soft_max(ctx, input);
}

// BatchNormalization - Batch normalization
static inline ggml_tensor* ggml_onnx_batchnormalization(ggml_context* ctx, ggml_tensor* input, ggml_tensor* scale,
                                                        ggml_tensor* bias, ggml_tensor* mean, ggml_tensor* var) {
    // Y = (X - mean) / sqrt(var + epsilon) * scale + bias
    // This is a simplified version - full implementation would need epsilon parameter

    ggml_tensor* normalized = ggml_sub(ctx, input, mean);
    ggml_tensor* std = ggml_sqrt(ctx, var);
    normalized = ggml_div(ctx, normalized, std);
    normalized = ggml_mul(ctx, normalized, scale);
    normalized = ggml_add(ctx, normalized, bias);

    return normalized;
}

// MaxPool - Max Pooling
static inline ggml_tensor* ggml_onnx_maxpool(ggml_context* ctx, ggml_tensor* input) {
    // Default 2x2 pooling with stride 2
    int kernel_h = 2;
    int kernel_w = 2;
    int stride_h = 2;
    int stride_w = 2;
    int padding = 0;

    return ggml_pool_2d(ctx, input, GGML_OP_POOL_MAX, kernel_w, kernel_h, stride_w, stride_h, padding, padding);
}

// AveragePool - Average Pooling
static inline ggml_tensor* ggml_onnx_averagepool(ggml_context* ctx, ggml_tensor* input) {
    // Default 2x2 pooling with stride 2
    int kernel_h = 2;
    int kernel_w = 2;
    int stride_h = 2;
    int stride_w = 2;
    int padding = 0;

    return ggml_pool_2d(ctx, input, GGML_OP_POOL_AVG, kernel_w, kernel_h, stride_w, stride_h, padding, padding);
}

// Transpose - Transpose tensor
static inline ggml_tensor* ggml_onnx_transpose(ggml_context* ctx, ggml_tensor* input) {
    // Transpose last two dimensions (matrix transpose)
    return ggml_transpose(ctx, input);
}

// Reshape - Reshape tensor
static inline ggml_tensor* ggml_onnx_reshape(ggml_context* ctx, ggml_tensor* input, ggml_tensor* shape) {
    // Note: This is a simplified version
    // Full implementation would need to read shape tensor
    return input;  // Placeholder
}

// Concat - Concatenate tensors
static inline ggml_tensor* ggml_onnx_concat(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    // Concatenate along dimension 0
    return ggml_concat(ctx, a, b, 0);
}

// Slice - Slice tensor
static inline ggml_tensor* ggml_onnx_slice(ggml_context* ctx, ggml_tensor* input) {
    // Placeholder - needs start, end, axes parameters
    return input;
}

// Clip - Clip tensor values to [min, max]
static inline ggml_tensor* ggml_onnx_clip(ggml_context* ctx, ggml_tensor* input) {
    return ggml_clamp(ctx, input, -1.0f, 1.0f);  // Default range
}

// LeakyRelu - Leaky ReLU activation
static inline ggml_tensor* ggml_onnx_leakyrelu(ggml_context* ctx, ggml_tensor* input) {
    return ggml_leaky_relu(ctx, input, 0.01f, true);  // Default alpha=0.01
}

// Gelu - GELU activation
static inline ggml_tensor* ggml_onnx_gelu(ggml_context* ctx, ggml_tensor* input) { return ggml_gelu(ctx, input); }

// LayerNormalization - Layer normalization
static inline ggml_tensor* ggml_onnx_layernormalization(ggml_context* ctx, ggml_tensor* input, ggml_tensor* scale,
                                                        ggml_tensor* bias) {
    return ggml_norm(ctx, input, 1e-5f);  // Simplified version
}

// ReduceMean - Reduce mean across dimensions
static inline ggml_tensor* ggml_onnx_reducemean(ggml_context* ctx, ggml_tensor* input) { return ggml_mean(ctx, input); }

// ReduceSum - Reduce sum across dimensions
static inline ggml_tensor* ggml_onnx_reducesum(ggml_context* ctx, ggml_tensor* input) { return ggml_sum(ctx, input); }

// Pow - Element-wise power
static inline ggml_tensor* ggml_onnx_pow(ggml_context* ctx, ggml_tensor* base, ggml_tensor* exponent) {
    // Note: ggml may not have direct pow, use sqr for power of 2
    return ggml_sqr(ctx, base);  // Simplified for exponent=2
}

// Exp - Element-wise exponential
static inline ggml_tensor* ggml_onnx_exp(ggml_context* ctx, ggml_tensor* input) { return ggml_exp(ctx, input); }

// Log - Element-wise natural logarithm
static inline ggml_tensor* ggml_onnx_log(ggml_context* ctx, ggml_tensor* input) { return ggml_log(ctx, input); }

// Abs - Element-wise absolute value
static inline ggml_tensor* ggml_onnx_abs(ggml_context* ctx, ggml_tensor* input) { return ggml_abs(ctx, input); }

// Neg - Element-wise negation
static inline ggml_tensor* ggml_onnx_neg(ggml_context* ctx, ggml_tensor* input) { return ggml_neg(ctx, input); }

// Erf - Error function
static inline ggml_tensor* ggml_onnx_erf(ggml_context* ctx, ggml_tensor* input) {
    // Note: ggml may not have erf directly
    // This is used in GELU approximation
    return input;  // Placeholder
}

// Cast - Cast tensor to different type
static inline ggml_tensor* ggml_onnx_cast(ggml_context* ctx, ggml_tensor* input) {
    // Placeholder - type conversion would be needed
    return input;
}

// Unsqueeze - Add dimensions
static inline ggml_tensor* ggml_onnx_unsqueeze(ggml_context* ctx, ggml_tensor* input) {
    // Placeholder - needs axes parameter
    return input;
}

// Squeeze - Remove dimensions
static inline ggml_tensor* ggml_onnx_squeeze(ggml_context* ctx, ggml_tensor* input) {
    // Placeholder - needs axes parameter
    return input;
}

// Pad - Pad tensor
static inline ggml_tensor* ggml_onnx_pad(ggml_context* ctx, ggml_tensor* input) {
    // Use ggml_pad with default padding
    return ggml_pad(ctx, input, 0, 0, 0, 0);
}

// Gather - Gather elements
static inline ggml_tensor* ggml_onnx_gather(ggml_context* ctx, ggml_tensor* data, ggml_tensor* indices) {
    // Placeholder - needs axis parameter
    return data;
}

// Split - Split tensor
static inline ggml_tensor* ggml_onnx_split(ggml_context* ctx, ggml_tensor* input) {
    // Placeholder - returns multiple outputs
    return input;
}

#ifdef __cplusplus
}
#endif
