# Aegis Zero-GC PyTorch C10 Dispatcher Extension

This module provides native C++ PyTorch operator bindings for RFC-0036, integrating the Aegis 64-byte flat arena with `c10::Dispatcher` and `TORCH_LIBRARY`.

### Architectural Purpose

1. **Zero Tensor Allocation:** Eliminates intermediate Python object creation and ATen dynamic heap reallocations.
2. **C10 Dispatcher Integration:** Registers `aegis::extract_batch` directly on the CPU dispatch key for native PyTorch graph compatibility.
3. **In-Place Device Transfers:** Extracted pinned host tensors are transferred directly into pre-allocated device buffers via `.copy_(..., non_blocking=True)`.

### Installation

```bash
cd pytorch_feeder
pip install -e .
```

### Direct Usage

```python
import torch
import aegis_c10_feeder

# Pre-allocated pinned host memory buffers
out_x = torch.empty((batch_size, block_size), dtype=torch.long, pin_memory=True)
out_y = torch.empty((batch_size, block_size), dtype=torch.long, pin_memory=True)
indices = torch.empty(batch_size, dtype=torch.long)

# Call via native C10 dispatcher operator
torch.ops.aegis.extract_batch(dataset_tensor, indices, batch_size, block_size, out_x, out_y)
```
