#include "epq/serialization_size.h"

#include <faiss/VectorTransform.h>
#include <faiss/index_io.h>

namespace epq {

size_t serialized_faiss_index_bytes(const faiss::Index& index) {
    CountingIOWriter writer;
    faiss::write_index(&index, &writer);
    return writer.bytes_written;
}

size_t serialized_faiss_vector_transform_bytes(
        const faiss::VectorTransform& transform) {
    CountingIOWriter writer;
    faiss::write_VectorTransform(&transform, &writer);
    return writer.bytes_written;
}

}  // namespace epq
