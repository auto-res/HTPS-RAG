import torch

if torch.cuda.is_available():
    print("CUDA is available. You can use GPU acceleration.")
    print(f"Number of CUDA devices available: {torch.cuda.device_count()}")
    print(f"CUDA Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. Operations will run on CPU.")

print(torch.__version__)
print(torch.cuda.get_arch_list())
print(torch.version.cuda)