// ==============================================================================
// Aegis Zero-GC Flat Arena PyTorch C10 Dispatcher Operator
// Reference: PyTorch RFC-0036 (pytorch/rfcs#110)
// Hardware Target: Zero-copy DMA directly into fixed ATen Tensor buffers
// ==============================================================================

#include <torch/extension.h>
#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <stdint.h>
#include <cstring>
#include <tuple>

// Native C flat arena batch extraction with zero allocation (in-place)
void aegis_c10_extract_batch(
    torch::Tensor dataset_tokens,
    torch::Tensor batch_indices,
    int64_t batch_size,
    int64_t block_size,
    torch::Tensor out_x,
    torch::Tensor out_y
) {
    TORCH_CHECK(dataset_tokens.is_contiguous(), "dataset_tokens must be contiguous");
    TORCH_CHECK(batch_indices.is_contiguous(), "batch_indices must be contiguous");
    TORCH_CHECK(out_x.is_contiguous(), "out_x must be contiguous");
    TORCH_CHECK(out_y.is_contiguous(), "out_y must be contiguous");

    const uint16_t* src_dataset = reinterpret_cast<const uint16_t*>(dataset_tokens.data_ptr());
    const int64_t* indices = reinterpret_cast<const int64_t*>(batch_indices.data_ptr());
    int64_t* x_ptr = reinterpret_cast<int64_t*>(out_x.data_ptr());
    int64_t* y_ptr = reinterpret_cast<int64_t*>(out_y.data_ptr());

    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t start_idx = indices[b];
        const uint16_t* src = &src_dataset[start_idx];
        int64_t* dst_x = &x_ptr[b * block_size];
        int64_t* dst_y = &y_ptr[b * block_size];

        for (int64_t t = 0; t < block_size; ++t) {
            dst_x[t] = static_cast<int64_t>(src[t]);
            dst_y[t] = static_cast<int64_t>(src[t + 1]);
        }
    }
}

// Direct zero-copy ATen tensor view wrapping pre-pinned host memory via torch::from_blob
std::tuple<torch::Tensor, torch::Tensor> aegis_c10_from_blob_batch(
    torch::Tensor dataset_tokens,
    torch::Tensor batch_indices,
    int64_t batch_size,
    int64_t block_size,
    int64_t host_x_ptr,
    int64_t host_y_ptr
) {
    TORCH_CHECK(dataset_tokens.is_contiguous(), "dataset_tokens must be contiguous");
    TORCH_CHECK(batch_indices.is_contiguous(), "batch_indices must be contiguous");

    const uint16_t* src_dataset = reinterpret_cast<const uint16_t*>(dataset_tokens.data_ptr());
    const int64_t* indices = reinterpret_cast<const int64_t*>(batch_indices.data_ptr());
    int64_t* dst_x = reinterpret_cast<int64_t*>(host_x_ptr);
    int64_t* dst_y = reinterpret_cast<int64_t*>(host_y_ptr);

    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t start_idx = indices[b];
        const uint16_t* src = &src_dataset[start_idx];
        int64_t* row_x = &dst_x[b * block_size];
        int64_t* row_y = &dst_y[b * block_size];

        for (int64_t t = 0; t < block_size; ++t) {
            row_x[t] = static_cast<int64_t>(src[t]);
            row_y[t] = static_cast<int64_t>(src[t + 1]);
        }
    }

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto tx = torch::from_blob(dst_x, {batch_size, block_size}, options);
    auto ty = torch::from_blob(dst_y, {batch_size, block_size}, options);
    return std::make_tuple(tx, ty);
}

TORCH_LIBRARY(aegis, m) {
    m.def("extract_batch(Tensor dataset, Tensor indices, int batch_size, int block_size, Tensor(a!) out_x, Tensor(b!) out_y) -> ()");
    m.impl("extract_batch", c10::DispatchKey::CPU, TORCH_FN(aegis_c10_extract_batch));
    m.def("from_blob_batch(Tensor dataset, Tensor indices, int batch_size, int block_size, int host_x_ptr, int host_y_ptr) -> (Tensor, Tensor)");
    m.impl("from_blob_batch", c10::DispatchKey::CPU, TORCH_FN(aegis_c10_from_blob_batch));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("extract_batch", &aegis_c10_extract_batch, "Aegis Zero-GC Flat Arena In-Place Batch Extractor (C10 Dispatcher)");
    m.def("from_blob_batch", &aegis_c10_from_blob_batch, "Aegis Zero-Copy torch::from_blob Batch Extractor (C10 Dispatcher)");
}
