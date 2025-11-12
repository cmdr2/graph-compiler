#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "ggml.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline void ggml_onnx_compute_slice_params(int rank, const std::vector<int64_t>& dim_sizes,
                                                  const std::vector<int64_t>& starts, const std::vector<int64_t>& ends,
                                                  const std::vector<int64_t>& axes, const std::vector<int64_t>& steps,
                                                  std::vector<int64_t>& eff_starts, std::vector<int64_t>& eff_ends,
                                                  std::vector<int64_t>& eff_steps);

// ONNX operator implementations using ggml functions
// These are C++ inline functions (not C), so we can use C++ features like overloading and templates

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
// Accepts a vector of dimensions for the new shape
static inline ggml_tensor* ggml_onnx_reshape(ggml_context* ctx, ggml_tensor* input, const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return input;
    }

    if (shape.size() == 1) {
        return ggml_reshape_1d(ctx, input, shape[0]);
    } else if (shape.size() == 2) {
        return ggml_reshape_2d(ctx, input, shape[0], shape[1]);
    } else if (shape.size() == 3) {
        return ggml_reshape_3d(ctx, input, shape[0], shape[1], shape[2]);
    } else if (shape.size() == 4) {
        return ggml_reshape_4d(ctx, input, shape[0], shape[1], shape[2], shape[3]);
    }

    // Fallback for shapes with more than 4 dimensions
    return input;
}

// Concat - Concatenate tensors
// Base case: 2 tensors
static inline ggml_tensor* ggml_onnx_concat(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    // Concatenate along dimension 0
    return ggml_concat(ctx, a, b, 0);
}

// Variadic template: 3 or more tensors (chains calls recursively)
template <typename... Args>
static inline ggml_tensor* ggml_onnx_concat(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, Args... rest) {
    ggml_tensor* temp = ggml_concat(ctx, a, b, 0);
    return ggml_onnx_concat(ctx, temp, rest...);
}

// Slice - Slice tensor along multiple axes (ONNX Slice operator)
// This is the main implementation that accepts vectors for starts, ends, axes, and steps
// Following ONNX spec: https://onnx.ai/onnx/operators/onnx__Slice.html
static inline ggml_tensor* ggml_onnx_slice(ggml_context* ctx, ggml_tensor* input, const std::vector<int64_t>& starts,
                                           const std::vector<int64_t>& ends, const std::vector<int64_t>& axes = {},
                                           const std::vector<int64_t>& steps = {}) {
    int r = ggml_n_dims(input);  // rank of input

    // Get dimension sizes
    std::vector<int64_t> dim_sizes(r);
    for (int i = 0; i < r; i++) {
        dim_sizes[i] = input->ne[i];
    }

    // Compute effective slice parameters
    std::vector<int64_t> eff_starts, eff_ends, eff_steps;
    ggml_onnx_compute_slice_params(r, dim_sizes, starts, ends, axes, steps, eff_starts, eff_ends, eff_steps);

    // Calculate new dimensions and create view
    std::vector<int64_t> new_dims(r);
    size_t offset = 0;
    size_t stride = ggml_element_size(input);

    for (int i = 0; i < r; i++) {
        int64_t length = eff_ends[i] - eff_starts[i];
        if (eff_steps[i] < 0) {
            length = eff_starts[i] - eff_ends[i];
        }
        new_dims[i] = std::max<int64_t>(0, (length + std::abs(eff_steps[i]) - 1) / std::abs(eff_steps[i]));

        // Calculate offset
        offset += eff_starts[i] * stride;
        stride *= input->ne[i];
    }

    // Create view with new dimensions
    ggml_tensor* result;
    if (r == 1) {
        result = ggml_view_1d(ctx, input, new_dims[0], offset);
    } else if (r == 2) {
        result = ggml_view_2d(ctx, input, new_dims[0], new_dims[1], input->nb[1], offset);
    } else if (r == 3) {
        result = ggml_view_3d(ctx, input, new_dims[0], new_dims[1], new_dims[2], input->nb[1], input->nb[2], offset);
    } else if (r == 4) {
        result = ggml_view_4d(ctx, input, new_dims[0], new_dims[1], new_dims[2], new_dims[3], input->nb[1],
                              input->nb[2], input->nb[3], offset);
    } else {
        // Fallback for higher dimensions
        result = ggml_view_tensor(ctx, input);
    }

    return result;
}

// Clip - Clip tensor values to [min, max]
static inline ggml_tensor* ggml_onnx_clip(ggml_context* ctx, ggml_tensor* input, float min_val, float max_val) {
    return ggml_clamp(ctx, input, min_val, max_val);
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
    // Normalize the input and apply scale and bias
    ggml_tensor* normalized = ggml_norm(ctx, input, 1e-5f);
    if (scale != NULL) {
        normalized = ggml_mul(ctx, normalized, scale);
    }
    if (bias != NULL) {
        normalized = ggml_add(ctx, normalized, bias);
    }
    return normalized;
}

// ReduceMean - Reduce mean across dimensions
static inline ggml_tensor* ggml_onnx_reducemean(ggml_context* ctx, ggml_tensor* input) { return ggml_mean(ctx, input); }

// ReduceSum - Reduce sum across dimensions
static inline ggml_tensor* ggml_onnx_reducesum(ggml_context* ctx, ggml_tensor* input) { return ggml_sum(ctx, input); }

// Pow - Element-wise power
// Version with scalar exponent
static inline ggml_tensor* ggml_onnx_pow(ggml_context* ctx, ggml_tensor* base, float exp_val) {
    // Handle special cases
    if (exp_val == 2.0f) {
        return ggml_sqr(ctx, base);
    } else if (exp_val == 0.5f) {
        return ggml_sqrt(ctx, base);
    } else if (exp_val == 1.0f) {
        return base;  // x^1 = x
    }

    // For general case: pow(base, exp) = exp(exp * log(base))
    ggml_tensor* log_base = ggml_log(ctx, base);
    ggml_tensor* scaled = ggml_scale(ctx, log_base, exp_val);
    return ggml_exp(ctx, scaled);
}

// Version with tensor exponent (for when exponent is not a constant)
static inline ggml_tensor* ggml_onnx_pow(ggml_context* ctx, ggml_tensor* base, ggml_tensor* exponent) {
    // For tensor exponents, use: e^(exponent * ln(base))
    // pow(base, exp) = exp(exp * log(base))
    ggml_tensor* log_base = ggml_log(ctx, base);
    ggml_tensor* product = ggml_mul(ctx, exponent, log_base);
    return ggml_exp(ctx, product);
}

// Exp - Element-wise exponential
static inline ggml_tensor* ggml_onnx_exp(ggml_context* ctx, ggml_tensor* input) { return ggml_exp(ctx, input); }

// Log - Element-wise natural logarithm
static inline ggml_tensor* ggml_onnx_log(ggml_context* ctx, ggml_tensor* input) { return ggml_log(ctx, input); }

// Abs - Element-wise absolute value
static inline ggml_tensor* ggml_onnx_abs(ggml_context* ctx, ggml_tensor* input) { return ggml_abs(ctx, input); }

// Neg - Element-wise negation
static inline ggml_tensor* ggml_onnx_neg(ggml_context* ctx, ggml_tensor* input) { return ggml_neg(ctx, input); }

// Identity - Return input unchanged
static inline ggml_tensor* ggml_onnx_identity(ggml_context* ctx, ggml_tensor* input) {
    // Identity operation just returns the input
    (void)ctx;  // Unused
    return input;
}

// Erf - Error function
static inline ggml_tensor* ggml_onnx_erf(ggml_context* ctx, ggml_tensor* input) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // erf(x) ≈ tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
    // Simplified approximation using available ggml operations
    ggml_tensor* x3 = ggml_mul(ctx, ggml_sqr(ctx, input), input);
    ggml_tensor* scaled = ggml_scale(ctx, x3, 0.044715f);
    ggml_tensor* sum = ggml_add(ctx, input, scaled);
    ggml_tensor* scaled_sum = ggml_scale(ctx, sum, sqrtf(2.0f / M_PI));
    return ggml_tanh(ctx, scaled_sum);
}

// Cast - Cast tensor to different type
static inline ggml_tensor* ggml_onnx_cast(ggml_context* ctx, ggml_tensor* input) {
    // Use ggml_cast to convert type
    // Default to F32 if no specific type is provided
    // Full implementation would read the target type from ONNX attributes
    return ggml_cast(ctx, input, GGML_TYPE_F32);
}

// InstanceNormalization - Instance normalization
static inline ggml_tensor* ggml_onnx_instancenormalization(ggml_context* ctx, ggml_tensor* input, ggml_tensor* scale,
                                                           ggml_tensor* bias) {
    // Instance normalization: normalize each instance (per channel) independently
    // Y = scale * (X - mean) / sqrt(variance + epsilon) + bias
    ggml_tensor* normalized = ggml_norm(ctx, input, 1e-5f);
    if (scale != NULL) {
        normalized = ggml_mul(ctx, normalized, scale);
    }
    if (bias != NULL) {
        normalized = ggml_add(ctx, normalized, bias);
    }
    return normalized;
}

// RandomNormalLike - Create a tensor with random normal distribution, with same shape as input
static inline ggml_tensor* ggml_onnx_randomnormallike(ggml_context* ctx, ggml_tensor* input) {
    // Create a tensor with the same shape as input, filled with random normal values
    // ggml doesn't have built-in random initialization, so create a tensor with same shape
    // In practice, this would need to be filled with random values after creation
    return ggml_dup_tensor(ctx, input);
}

// Unsqueeze - Add dimensions
// Accepts a vector of axes where dimensions of size 1 should be inserted
static inline ggml_tensor* ggml_onnx_unsqueeze(ggml_context* ctx, ggml_tensor* input,
                                               const std::vector<int64_t>& axes) {
    if (axes.empty()) {
        return input;
    }

    int ndims = ggml_n_dims(input);

    // Normalize negative axes and sort them
    std::vector<int64_t> normalized_axes;
    for (int64_t axis : axes) {
        int64_t norm_axis = axis < 0 ? axis + ndims + axes.size() : axis;
        normalized_axes.push_back(norm_axis);
    }
    std::sort(normalized_axes.begin(), normalized_axes.end());

    // Build new shape by inserting 1s at specified axes
    std::vector<int64_t> new_shape;
    int input_idx = 0;
    int axes_idx = 0;
    int new_ndims = ndims + axes.size();

    for (int i = 0; i < new_ndims; i++) {
        if (axes_idx < normalized_axes.size() && i == normalized_axes[axes_idx]) {
            // Insert dimension of size 1
            new_shape.push_back(1);
            axes_idx++;
        } else {
            // Copy dimension from input
            if (input_idx < ndims) {
                new_shape.push_back(input->ne[input_idx]);
                input_idx++;
            }
        }
    }

    // Apply reshape based on final dimensionality
    if (new_shape.size() == 1) {
        return ggml_reshape_1d(ctx, input, new_shape[0]);
    } else if (new_shape.size() == 2) {
        return ggml_reshape_2d(ctx, input, new_shape[0], new_shape[1]);
    } else if (new_shape.size() == 3) {
        return ggml_reshape_3d(ctx, input, new_shape[0], new_shape[1], new_shape[2]);
    } else if (new_shape.size() == 4) {
        return ggml_reshape_4d(ctx, input, new_shape[0], new_shape[1], new_shape[2], new_shape[3]);
    }

    // Fallback
    return input;
}

// Squeeze - Remove dimensions
// Accepts a vector of axes to remove (must have size 1)
// If axes is empty, removes all dimensions of size 1
static inline ggml_tensor* ggml_onnx_squeeze(ggml_context* ctx, ggml_tensor* input,
                                             const std::vector<int64_t>& axes = {}) {
    int ndims = ggml_n_dims(input);
    std::vector<int64_t> new_shape;

    if (axes.empty()) {
        // Remove all dimensions of size 1
        for (int i = 0; i < ndims; i++) {
            if (input->ne[i] != 1) {
                new_shape.push_back(input->ne[i]);
            }
        }
    } else {
        // Normalize negative axes
        std::vector<int64_t> normalized_axes;
        for (int64_t axis : axes) {
            int64_t norm_axis = axis < 0 ? axis + ndims : axis;
            if (norm_axis >= 0 && norm_axis < ndims) {
                normalized_axes.push_back(norm_axis);
            }
        }

        // Remove only the specified axes (if they have size 1)
        for (int i = 0; i < ndims; i++) {
            bool should_remove = false;
            for (int64_t axis : normalized_axes) {
                if (i == axis && input->ne[i] == 1) {
                    should_remove = true;
                    break;
                }
            }
            if (!should_remove) {
                new_shape.push_back(input->ne[i]);
            }
        }
    }

    // Ensure at least 1D
    if (new_shape.empty()) {
        new_shape.push_back(1);
    }

    // Apply reshape based on final dimensionality
    if (new_shape.size() == 1) {
        return ggml_reshape_1d(ctx, input, new_shape[0]);
    } else if (new_shape.size() == 2) {
        return ggml_reshape_2d(ctx, input, new_shape[0], new_shape[1]);
    } else if (new_shape.size() == 3) {
        return ggml_reshape_3d(ctx, input, new_shape[0], new_shape[1], new_shape[2]);
    } else if (new_shape.size() == 4) {
        return ggml_reshape_4d(ctx, input, new_shape[0], new_shape[1], new_shape[2], new_shape[3]);
    }

    // Fallback
    return input;
}

// Pad - Pad tensor
// Accepts a vector of padding values [p0, p1, p2, p3]
// If constant_value is provided, it specifies the padding value (note: ggml_pad doesn't support this yet)
static inline ggml_tensor* ggml_onnx_pad(ggml_context* ctx, ggml_tensor* input, const std::vector<int64_t>& pads,
                                         float constant_value = 0.0f) {
    // Ensure we have at least 4 padding values (pad with zeros if needed)
    int p0 = pads.size() > 0 ? pads[0] : 0;
    int p1 = pads.size() > 1 ? pads[1] : 0;
    int p2 = pads.size() > 2 ? pads[2] : 0;
    int p3 = pads.size() > 3 ? pads[3] : 0;

    // Note: ggml_pad doesn't support custom padding values, but we include the parameter for API compatibility
    // The constant_value would need to be applied in a post-processing step if needed
    (void)constant_value;  // Unused for now

    return ggml_pad(ctx, input, p0, p1, p2, p3);
}

// Gather - Gather elements
static inline ggml_tensor* ggml_onnx_gather(ggml_context* ctx, ggml_tensor* data, ggml_tensor* indices) {
    // Gather elements from data using indices
    // ggml_get_rows is similar to gather operation along axis 0
    return ggml_get_rows(ctx, data, indices);
}

// Helper function for slice logic - computes effective starts, ends, and steps for each dimension
static inline void ggml_onnx_compute_slice_params(int rank, const std::vector<int64_t>& dim_sizes,
                                                  const std::vector<int64_t>& starts, const std::vector<int64_t>& ends,
                                                  const std::vector<int64_t>& axes, const std::vector<int64_t>& steps,
                                                  std::vector<int64_t>& eff_starts, std::vector<int64_t>& eff_ends,
                                                  std::vector<int64_t>& eff_steps) {
    // Initialize effective starts, ends, and steps for all dimensions
    eff_starts.assign(rank, 0);
    eff_ends = dim_sizes;
    eff_steps.assign(rank, 1);

    // Determine axes to slice
    std::vector<int64_t> eff_axes;
    if (axes.empty()) {
        // If axes are omitted, set to [0, ..., starts.size()-1]
        for (size_t i = 0; i < starts.size(); i++) {
            eff_axes.push_back(i);
        }
    } else {
        eff_axes = axes;
        // Handle negative axes
        for (size_t i = 0; i < eff_axes.size(); i++) {
            if (eff_axes[i] < 0) {
                eff_axes[i] += rank;
            }
        }
    }

    // Set effective steps
    std::vector<int64_t> actual_steps;
    if (steps.empty()) {
        actual_steps = std::vector<int64_t>(starts.size(), 1);
    } else {
        actual_steps = steps;
    }

    // Apply starts, ends, and steps to the appropriate axes
    for (size_t i = 0; i < starts.size(); i++) {
        if (i >= eff_axes.size()) break;
        int64_t axis = eff_axes[i];
        if (axis < 0 || axis >= rank) continue;

        int64_t dim = dim_sizes[axis];
        int64_t step = actual_steps[i];

        // Adjust negative starts and ends
        int64_t start = starts[i];
        int64_t end = ends[i];

        if (start < 0) start += dim;
        if (end < 0) end += dim;

        // Clamp starts based on step direction
        if (step > 0) {
            start = std::max<int64_t>(0, std::min<int64_t>(start, dim));
            end = std::max<int64_t>(0, std::min<int64_t>(end, dim));
        } else {
            start = std::max<int64_t>(0, std::min<int64_t>(start, dim - 1));
            end = std::max<int64_t>(-1, std::min<int64_t>(end, dim - 1));
        }

        eff_starts[axis] = start;
        eff_ends[axis] = end;
        eff_steps[axis] = step;
    }
}
