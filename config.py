import torch_xla.core.xla_model as xm

# 전역 TPU 디바이스 설정
DEVICE = xm.xla_device()