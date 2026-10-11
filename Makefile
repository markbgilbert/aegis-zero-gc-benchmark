# ==============================================================================
# AEGIS NATIVE C++20 / CUDA NANO-GPT TRAINING ENGINE MAKEFILE (AL-DEV-40)
# Target Architecture: NVIDIA GeForce RTX 5060 Laptop GPU (sm_120 / Blackwell)
# Strict Compliance: Zero em-dash and en-dash rule enforced.
# ==============================================================================

NVCC := /opt/cuda/bin/nvcc
CXXFLAGS := -O3 -std=c++20 --use_fast_math -arch=native -Xcompiler -Wall,-Wextra,-Wno-unused-parameter
INCLUDES := -Iinclude/cudnn-frontend/include -Illmc
LDFLAGS_FP32 := -lcublas -lcuda -ldl
LDFLAGS_BF16 := -lcublas -lcublasLt -lcudnn -lnvrtc -lnvidia-ml -lcuda -ldl

TARGET_FP32 := aegis_trainer
SRC_FP32 := aegis_trainer.cu

TARGET_BF16 := aegis_trainer_bf16
SRC_BF16 := aegis_trainer_bf16.cu llmc/cudnn_att.cpp

all: $(TARGET_BF16) $(TARGET_FP32)

$(TARGET_BF16): $(SRC_BF16)
	$(NVCC) $(CXXFLAGS) -DENABLE_BF16 -DENABLE_CUDNN $(INCLUDES) $(SRC_BF16) $(LDFLAGS_BF16) -o $(TARGET_BF16)

$(TARGET_FP32): $(SRC_FP32)
	$(NVCC) $(CXXFLAGS) $(SRC_FP32) $(LDFLAGS_FP32) -o $(TARGET_FP32)

clean:
	rm -f $(TARGET_FP32) $(TARGET_BF16)

.PHONY: all clean
