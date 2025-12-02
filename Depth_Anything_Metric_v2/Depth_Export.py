import torch
import shutil
import sys
import os
import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from depth_config import EXPORT_MODEL_ENCODER_TYPE, EXPORT_DEPTH_INPUT_SIZE, MAX_DEPTH

# =================================================================================
# CONFIGURATION
# =================================================================================

# --- Paths ---
model_path = "/home/DakeQQ/Downloads/depth_anything_v2_metric_hypersim_vits.pth"
depth_metric_path = "/home/DakeQQ/Downloads/Depth-Anything-V2-main/metric_depth/depth_anything_v2"
output_path = "./Depth_Anything_Metric_V2.onnx"
test_image_path = "./test.jpg"

# --- BEV Configuration ---
OUTPUT_BEV = True                # True for output the bird eye view occupancy map.
SOBEL_KERNEL_SIZE = 3            # [3, 5] set for gradient map.
DEFAULT_GRAD_THRESHOLD = 0.5     # Set a appropriate value for detected object.
BEV_WIDTH_METERS = 10.0          # The max width in the image.
BEV_DEPTH_METERS = MAX_DEPTH     # The max depth in the image.
BEV_ROI_START_RATIO = 0.5        # Start position for ROI (0.0 = top, 1.0 = bottom). 0.5 = middle (bottom half).
OP_SET = 17                      # ONNX Runtime opset.


# =================================================================================
# HELPER FUNCTIONS
# =================================================================================

def get_sobel_kernels(kernel_size):
    """Return Sobel kernels for gradient computation (PyTorch format)."""
    kernels = {
        3: (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32),
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        ),
        5: (
            torch.tensor([[-1, -2, 0, 2, 1], [-2, -3, 0, 3, 2], [-3, -5, 0, 5, 3],
                          [-2, -3, 0, 3, 2], [-1, -2, 0, 2, 1]], dtype=torch.float32),
            torch.tensor([[-1, -2, -3, -2, -1], [-2, -3, -5, -3, -2], [0, 0, 0, 0, 0],
                          [2, 3, 5, 3, 2], [1, 2, 3, 2, 1]], dtype=torch.float32)
        )
    }
    if kernel_size not in kernels:
        raise ValueError(f"Unsupported kernel size: {kernel_size}. Use 3 or 5.")
    return kernels[kernel_size]


def create_normalized_uv_grid(width, height, aspect_ratio):
    """Create normalized UV coordinate grid for the view plane."""
    denom = (1 + aspect_ratio ** 2) ** 0.5
    scale_factor = lambda size: (size - 1) / size

    span_x = aspect_ratio / denom * scale_factor(width)
    span_y = scale_factor(height) / denom

    u = torch.linspace(-span_x, span_x, width, dtype=torch.float32)
    v = torch.linspace(-span_y, span_y, height, dtype=torch.float32)

    v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
    return torch.stack([u_grid, v_grid], dim=-1)


# =================================================================================
# OPTIMIZED MODEL WRAPPER WITH BINARY BEV
# =================================================================================

class DepthAnythingV2Wrapper(torch.nn.Module):
    def __init__(self, model, input_size, bev_width_meters, bev_depth_meters,
                 max_depth, sobel_kernel_size=3, roi_start_ratio=0.5):
        super().__init__()
        self.model = model
        self.h, self.w = input_size
        self.max_depth = max_depth
        self.roi_start_ratio = roi_start_ratio

        # 1. Combined Sobel Kernel (2 output channels: dx, dy)
        sobel_x, sobel_y = get_sobel_kernels(sobel_kernel_size)
        sobel_combined = torch.stack([sobel_x, sobel_y], dim=0).unsqueeze(1)
        self.register_buffer('sobel_kernels', sobel_combined)
        self.sobel_padding = sobel_kernel_size // 2

        # 2. Pre-compute ROI bounds based on configurable start ratio
        self.h_start = int(self.h * roi_start_ratio)
        self.h_end = self.h - self.sobel_padding
        self.w_start = self.sobel_padding
        self.w_end = self.w - self.sobel_padding

        self.roi_h = self.h_end - self.h_start
        self.roi_w = self.w_end - self.w_start

        # 3. Pre-compute and pre-scale UV grid
        aspect_ratio = self.w / self.h
        uv_grid = create_normalized_uv_grid(self.w, self.h, aspect_ratio)
        uv_roi = uv_grid[self.h_start:self.h_end, self.w_start:self.w_end, 0]

        # Pre-scale by BEV width to avoid runtime multiplication
        bev_scale_x = self.w / bev_width_meters
        uv_roi_scaled = uv_roi * bev_scale_x
        self.register_buffer('uv_roi_scaled', uv_roi_scaled)

        # 4. Pre-compute BEV transformation constants
        self.bev_w = self.w
        self.bev_h = self.h
        self.register_buffer('bev_offset_x', torch.tensor(self.w / 2.0, dtype=torch.float32).view(1, -1))
        self.register_buffer('bev_scale_z', torch.tensor(self.h / bev_depth_meters, dtype=torch.float32).view(1, -1))
        self.register_buffer('bev_map', torch.zeros([self.bev_h, self.bev_w], dtype=torch.uint8))

        # 5. Pre-allocate BEV grid size
        self.bev_grid_size = self.bev_h * self.bev_w

    def forward(self, image, threshold):
        # Base model inference
        depth = self.model(image)

        # Resize if needed
        if depth.shape[-2:] != (self.h, self.w):
            depth = torch.nn.functional.interpolate(
                depth,
                size=(self.h, self.w),
                mode='bilinear',
                align_corners=False
            )

        if OUTPUT_BEV:
            # ===================================================================
            # OPTIMIZED BEV GENERATION
            # ===================================================================

            # 1. Single-pass gradient computation (fused Sobel)
            gradients = torch.nn.functional.conv2d(
                depth,
                self.sobel_kernels,
                padding=self.sobel_padding
            )

            # 2. Fused gradient magnitude + square root
            grad_mag = torch.sqrt(
                gradients[:, 0:1].square() + gradients[:, 1:2].square()
            )

            # 3. Extract ROI in single slice operation
            grad_roi = grad_mag[:, :, self.h_start:self.h_end, self.w_start:self.w_end].squeeze()
            depth_roi = depth[:, :, self.h_start:self.h_end, self.w_start:self.w_end].squeeze()

            # 4. Binary mask (threshold directly)
            mask = grad_roi > threshold  # [H_roi, W_roi]

            # 5. Calculate BEV indices (fused operations)
            # X index: pre-scaled UV grid * depth + offset
            x_idx = (self.uv_roi_scaled * depth_roi + self.bev_offset_x).long()

            # Z index: depth * scale
            z_idx = (depth_roi * self.bev_scale_z).long()

            # 6. Clamp indices in-place
            x_idx.clamp_(0, self.bev_w - 1)
            z_idx.clamp_(0, self.bev_h - 1)

            # Only update positions where mask is True
            self.bev_map[z_idx[mask], x_idx[mask]] = 1

            # 7. Flip and reshape
            bev_map = torch.flip(self.bev_map, dims=[0])

            return depth, bev_map

        return depth


# =================================================================================
# 1. SETUP AND EXPORT
# =================================================================================
print("Copying modified model files...")
if not os.path.exists(depth_metric_path):
    print(f"\nNot found: {depth_metric_path}")

# Copy necessary files
shutil.copy("./modeling_modified/dpt.py", os.path.join(depth_metric_path, "dpt.py"))
shutil.copy("./modeling_modified/dinov2.py", os.path.join(depth_metric_path, "dinov2.py"))
shutil.copy("./modeling_modified/mlp.py", os.path.join(depth_metric_path, "dinov2_layers/mlp.py"))
shutil.copy("./modeling_modified/patch_embed.py", os.path.join(depth_metric_path, "dinov2_layers/patch_embed.py"))
shutil.copy("./modeling_modified/attention.py", os.path.join(depth_metric_path, "dinov2_layers/attention.py"))
shutil.copy("./depth_config.py", "./modeling_modified/depth_config.py")
sys.path.append(os.path.dirname(os.path.abspath(depth_metric_path)))
print("Files copied.")

from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

print("Loading PyTorch model...")
base_model = DepthAnythingV2(**{**model_configs[EXPORT_MODEL_ENCODER_TYPE], 'max_depth': MAX_DEPTH})
base_model.load_state_dict(torch.load(model_path, map_location='cpu'))
base_model.eval()

# Get input size from config
_, _, input_h, input_w = EXPORT_DEPTH_INPUT_SIZE

# Wrap model
wrapped_model = DepthAnythingV2Wrapper(
    base_model,
    input_size=(input_h, input_w),
    bev_width_meters=BEV_WIDTH_METERS,
    bev_depth_meters=BEV_DEPTH_METERS,
    max_depth=MAX_DEPTH,
    sobel_kernel_size=SOBEL_KERNEL_SIZE,
    roi_start_ratio=BEV_ROI_START_RATIO
).to('cpu').eval()

print(f"Model wrapped. Input resolution: {input_h}x{input_w}")
print(f"ROI start ratio: {BEV_ROI_START_RATIO} (0.0=top, 1.0=bottom)")
print(f"Processing region: rows {int(input_h * BEV_ROI_START_RATIO)} to {input_h}")
print(f"Depth output: float32 in meters [0-{MAX_DEPTH}]")
print(f"BEV output: Binary uint8 (0 or 1)")

# ONNX Export
dummy_image = torch.ones(EXPORT_DEPTH_INPUT_SIZE, dtype=torch.uint8)
dummy_threshold = torch.tensor([DEFAULT_GRAD_THRESHOLD], dtype=torch.float32)

print(f"Exporting to {output_path}...")
input_names = ['image', 'threshold']
output_names = ['depth_map', 'bev_map'] if OUTPUT_BEV else ['depth_map']

with torch.inference_mode():
    torch.onnx.export(
        wrapped_model,
        (dummy_image, dummy_threshold),
        output_path,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        opset_version=OP_SET,
        dynamo=False
    )
print("✅ ONNX export complete.")

# =================================================================================
# 2. INFERENCE TEST
# =================================================================================

print("\nStarting ONNX inference test...")
try:
    ort_session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
except Exception as e:
    print(f"Error loading ONNX: {e}")
    sys.exit()

# Prepare Input
raw_img = cv2.imread(test_image_path)
if raw_img is None:
    print(f"Error: Could not read {test_image_path}")
    sys.exit()

rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
resized_img = cv2.resize(rgb_img, (input_w, input_h))
input_tensor = np.expand_dims(resized_img, axis=0).transpose(0, 3, 1, 2).astype(np.uint8)

in_name = ort_session.get_inputs()
out_name = ort_session.get_outputs()
in_name = [in_name[i].name for i in range(len(in_name))]
out_name = [out_name[i].name for i in range(len(out_name))]

# Run Inference
input_feed = {
    in_name[0]: input_tensor,
}

if len(in_name) > 1:
    input_feed[in_name[1]] = np.array([DEFAULT_GRAD_THRESHOLD], dtype=np.float32)

print("Running inference...")
start = time.time()
results = ort_session.run(None, input_feed)
elapsed = time.time() - start
print(f'⏱️  Time: {elapsed:.3f}s')

depth_out_float32 = results[0][0, 0]
bev_out = results[1] if OUTPUT_BEV else None

print(f"\nDepth output dtype: {depth_out_float32.dtype}, shape: {depth_out_float32.shape}")
print(f"Depth range (meters): [{depth_out_float32.min():.3f}, {depth_out_float32.max():.3f}]")

# =================================================================================
# 3. VISUALIZATION (ALL SAME SIZE)
# =================================================================================

print("\n📊 Visualizing...")
roi_start_line = int(input_h * BEV_ROI_START_RATIO)

# Create figure with uniform subplot sizes
fig = plt.figure(figsize=(18, 6))

# 1. Input Image (resized to match depth map size)
ax1 = plt.subplot(1, 3, 1)
plt.imshow(resized_img)
plt.axhline(y=roi_start_line, color='r', linestyle='--', linewidth=2)
plt.text(10, roi_start_line - 10, 'IGNORED REGION', color='red', fontweight='bold', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
plt.title(f'Input Image ({input_h}x{input_w})', fontsize=12, fontweight='bold')
plt.axis('off')
ax1.set_aspect('equal')

# 2. Depth Map (same size as input)
ax2 = plt.subplot(1, 3, 2)
im = plt.imshow(depth_out_float32, cmap='turbo', vmin=0, vmax=MAX_DEPTH, aspect='equal')
plt.axhline(y=roi_start_line, color='w', linestyle='--', linewidth=2)
plt.title(f'Depth Map (float32 meters)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(im, label='Depth (meters)', fraction=0.046, pad=0.04)
plt.axis('off')
ax2.set_aspect('equal')

# 3. BEV Map (resized to match other visualizations)
if bev_out is not None:
    ax3 = plt.subplot(1, 3, 3)
    # Resize BEV to match input dimensions for uniform display
    bev_resized = cv2.resize(bev_out.astype(np.uint8), (input_w, input_h), interpolation=cv2.INTER_NEAREST)

    extent = [-BEV_WIDTH_METERS / 2, BEV_WIDTH_METERS / 2, 0, BEV_DEPTH_METERS]
    plt.imshow(bev_resized, cmap='Greys', interpolation='nearest', extent=extent, vmin=0, vmax=1, aspect='auto')
    plt.plot(0, 0, 'ro', markersize=10, label='Camera', zorder=5)
    plt.title(f'BEV Occupancy Map (Binary)', fontsize=12, fontweight='bold')
    plt.xlabel('Lateral (m)', fontsize=10)
    plt.ylabel('Longitudinal (m)', fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, color='green', linestyle='--', alpha=0.3)
    ax3.set_aspect('equal')

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("OUTPUT INFORMATION:")
print("=" * 70)
print(f"ROI Configuration: Starting at {BEV_ROI_START_RATIO * 100:.0f}% from top")
print(f"  - Processing rows: {roi_start_line} to {input_h} (out of {input_h})")
print(f"  - To change: Adjust BEV_ROI_START_RATIO (0.0=use full image, 0.5=bottom half, 0.75=bottom quarter)")
print(f"\nDepth output: float32 in meters, range [0, {MAX_DEPTH}]")
print(f"BEV format: Binary uint8 (0 = free space, 1 = occupied)")
print(f"All visualization maps: {input_h}x{input_w} pixels")
print("=" * 70)
print("✅ Done.")
