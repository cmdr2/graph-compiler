# graph-compiler

*Work in progress!*

Experimental project. Not ready for real use. I haven't written any optimization passes yet (hence terrible performance).

## Usage

1. Generate an intermediate ONNX file, with a fixed input shape (e.g. `1x3x512x512`):

```py
python scripts/fold_shape_and_constants.py -i INPUT_FILE.onnx -o intermediate.onnx --input-shape 1,3,512,512
```

2. Generate ggml code (C++) from the intermediate ONNX file:

```py
python scripts/onnx_to_ggml.py intermediate.onnx -o example.cpp
```

This will generate `example.cpp` and `example_graph.h`.

3. Edit `CMakeLists.txt` and add these lines to the bottom:

```cmake
add_executable(example example.cpp example_graph.h ${SRC_FILES})
target_include_directories(example PRIVATE src)
```

4. Compile the generated files. This step requires `cmake` and a working C++ compiler like `g++` or `cl`.

```bash
cmake -B build
cmake --build build --config Release
```

**Tip:** You can also compile for CUDA/ROCm/Metal etc by adding the [required flag](https://github.com/ggml-org/ggml/). Example: `cmake -B build -DGGML_CUDA=1` to configure CUDA compilation.

5. Run the compiled model. You need to specify the safetensors file containing the weights for the model:

```bash
./build/bin/Release/example example_weights.safetensors
```

## Models compiled successfully

Numerically accurate compilation works for:
* TinyCNN
* VAE of Stable Diffusion 1.5
