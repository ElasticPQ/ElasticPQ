#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <faiss/impl/io.h>

namespace faiss {
struct Index;
struct VectorTransform;
}

namespace epq {

struct CountingIOWriter : faiss::IOWriter {
    size_t bytes_written = 0;

    size_t operator()(const void* ptr, size_t size, size_t nitems) override {
        (void)ptr;
        bytes_written += size * nitems;
        return nitems;
    }
};

template <typename T>
inline void write_scalar(faiss::IOWriter& writer, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto written = writer(&value, sizeof(T), 1);
    if (written != 1) {
        throw std::runtime_error("failed to write scalar");
    }
}

template <typename T>
inline void write_vector_data(
        faiss::IOWriter& writer,
        const T* data,
        size_t count) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (count == 0) {
        return;
    }
    const auto written = writer(data, sizeof(T), count);
    if (written != count) {
        throw std::runtime_error("failed to write vector data");
    }
}

template <typename T>
inline void write_vector(faiss::IOWriter& writer, const std::vector<T>& values) {
    write_scalar<uint64_t>(writer, static_cast<uint64_t>(values.size()));
    write_vector_data(writer, values.data(), values.size());
}

inline void write_string(faiss::IOWriter& writer, const std::string& value) {
    write_scalar<uint64_t>(writer, static_cast<uint64_t>(value.size()));
    write_vector_data(writer, value.data(), value.size());
}

size_t serialized_faiss_index_bytes(const faiss::Index& index);
size_t serialized_faiss_vector_transform_bytes(const faiss::VectorTransform& transform);

}  // namespace epq
