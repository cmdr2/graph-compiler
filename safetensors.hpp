/*
MIT License

A safetensors file has the following structure:
- the first 8 bytes contain a 64-bit integer (little endian), which stores the size of the header string (N)
- the next N bytes contain the header string in the JSON format
- the rest of the bytes contain the raw tensor data in C-style row-major order. The byte order matches the host system.

The JSON header contains a key called "__metadata__", as well as one key per parameter. Each parameter contains:
"dtype", "shape", and "data_offsets". "dtype" is one of "F32", "I64" etc (numpy format), "shape" is an int array, and
"data_offsets" is an int array containing the start and end position in the tensor data.

E.g.
header = {
  "__metadata__": {
    "description": "Optional file metadata"
  },
  "tensor_name": {
    "dtype": "F32",
    "shape": [3, 224, 224],
    "data_offsets": [0, 602112]
  }
}
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include "json.hpp"

using json = nlohmann::json;

namespace safetensors {
    using Callback = std::function<void(const std::string&, const std::string&, const std::vector<uint64_t>&, const std::vector<uint8_t>&)>;

    void load_from_file(const std::string& filename, Callback callback) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open the safetensors file!");
        }

        // read the header size (8 bytes, little endian)
        uint64_t header_size;
        file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
        
        // read the json header
        std::vector<char> header_data(header_size);
        file.read(header_data.data(), header_size);

        json header = json::parse(header_data.begin(), header_data.end());

        // read the tensors
        for (const auto& [key, tensor_info] : header.items()) {
            if (key == "__metadata__") continue;

            const auto& offsets = tensor_info["data_offsets"];
            const auto& dtype = tensor_info["dtype"].get<std::string>();
            // const std::vector<uint32_t> shape;
            const auto& shape = tensor_info["shape"].get<std::vector<uint64_t>>();
            uint64_t start = offsets[0];
            uint64_t end = offsets[1];
            size_t size = end - start;

            std::vector<uint8_t> tensor_data(size);
            file.seekg(8 + header_size + start);
            file.read(reinterpret_cast<char*>(tensor_data.data()), size);

            callback(key, dtype, shape, tensor_data);
        }
    }
}
