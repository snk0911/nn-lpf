import torch
import pywt
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("wavelet list:", pywt.wavelist())