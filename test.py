import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from model import Image_coding

# Load model weights
model = Image_coding(50,512, 1024)
checkpoint_path = '/media/zll/d/Repo/ljh/Gaussian-splatting-for-hsi/checkpoints_quant/Houston/GaussianImage_Cholesky_hsi_inter_50000_35000/Houston_12/gaussian_model.best.pth.tar'  # Replace with your model weight path
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()  # Set to evaluation mode

# Wrapper class for FLOPs calculation
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Assume if_training is False for inference
        return self.model(x, if_training=False)

wrapped_model = ModelWrapper(model)

# 2. Calculate the number of model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_parameters = count_parameters(model)
print(f"Model Parameters: {num_parameters / 1e6:.2f}M")

# 3. Calculate FLOPs
with torch.no_grad():
    # Move the wrapped model to CUDA
    wrapped_model = wrapped_model.to('cuda')

    # Calculate FLOPs
    flops, params = get_model_complexity_info(
        wrapped_model, (50, 256, 256), as_strings=True, print_per_layer_stat=True
    )
    print(f"FLOPs: {flops}, Parameters: {params}")

# 4. Calculate FPS (optional, requires GPU support)
def calculate_fps(model, input_size=(50, 256, 256), iterations=150):
    if torch.cuda.is_available():
        model.to('cuda')
        input_data = torch.randn(1, *input_size).cuda()
        model.eval()

        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_data, if_training=False)  # Assume if_training is False

        # Timing
        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(input_data, if_training=False)  # Assume if_training is False
        end_time = time.time()

        elapsed_time = end_time - start_time
        fps = iterations / elapsed_time
        print(f"Model FPS: {fps:.2f}")
    else:
        print("CUDA is not available. FPS calculation requires a GPU.")

# Uncomment the line below to calculate FPS if you have a GPU available
calculate_fps(model)
