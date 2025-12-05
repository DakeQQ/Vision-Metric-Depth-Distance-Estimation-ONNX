import os
import sys
import cv2
import time
import torch
import shutil
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt

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
DEFAULT_GRAD_THRESHOLD = 0.01    # Set a appropriate value for detected object.
BEV_WIDTH_METERS = MAX_DEPTH     # The max width in the image.
BEV_DEPTH_METERS = MAX_DEPTH     # The max depth in the image.
BEV_ROI_START_RATIO = 0.5        # Start position for ROI (0.0 = top, 1.0 = bottom).
OP_SET = 17                      # ONNX Runtime opset.


# =================================================================================
# HELPER FUNCTIONS
# =================================================================================

def get_sobel_kernels(kernel_size):
    """Return Sobel kernels for gradient computation."""
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

    sobel_x, sobel_y = kernels[kernel_size]
    return sobel_x.view(1, 1, kernel_size, kernel_size), sobel_y.view(1, 1, kernel_size, kernel_size)


def create_normalized_uv_grid(width, height, aspect_ratio):
    """Create normalized UV coordinate grid for the view plane."""
    denom = (1 + aspect_ratio ** 2) ** 0.5
    scale_factor = lambda size: (size - 1) / size

    span_x = aspect_ratio / denom * scale_factor(width)
    span_y = scale_factor(height) / denom

    u = torch.linspace(-span_x, span_x, width, dtype=torch.float32)
    v = torch.linspace(-span_y, span_y, height, dtype=torch.float32)

    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
    return torch.stack([u_grid, v_grid], dim=-1)


# =================================================================================
# OPTIMIZED MODEL WRAPPER
# =================================================================================

class DepthAnythingV2Wrapper(torch.nn.Module):
    def __init__(self, model, input_size, bev_width_meters, bev_depth_meters,
                 max_depth, sobel_kernel_size=3, roi_start_ratio=0.5):
        super().__init__()
        self.model = model
        self.h, self.w = input_size
        self.max_depth = max_depth
        self.bev_roi_start_ratio = roi_start_ratio

        # Setup Sobel kernels
        self.sobel_x, self.sobel_y = get_sobel_kernels(sobel_kernel_size)
        self.sobel_padding = sobel_kernel_size // 2

        # Setup UV grid (u and v coordinates)
        self._setup_uv_grid()

        # Setup BEV parameters
        self._setup_bev_parameters(bev_width_meters, bev_depth_meters, sobel_kernel_size)

    def _setup_uv_grid(self):
        """Setup UV projection grid."""
        aspect_ratio = self.w / self.h
        projection_uv = create_normalized_uv_grid(self.w, self.h, aspect_ratio)
        # Permute to (1, 2, H, W) -> Channel 0 is U, Channel 1 is V
        self.projection_uv = projection_uv.permute(2, 0, 1).unsqueeze(0)

    def _setup_bev_parameters(self, bev_width_meters, bev_depth_meters, sobel_kernel_size):
        """Setup BEV map parameters and ROI bounds."""
        bev_h, bev_w = self.h, self.w

        self.register_buffer('bev_w', torch.tensor([bev_w], dtype=torch.int32))
        self.register_buffer('bev_h', torch.tensor([bev_h], dtype=torch.int64))

        # Pre-calculate scalars for projection
        self.register_buffer('bev_scale_x', torch.tensor([bev_w / bev_width_meters], dtype=torch.float32))
        self.register_buffer('bev_offset_x', torch.tensor([bev_w / 2.0], dtype=torch.float32))
        self.register_buffer('bev_scale_z', torch.tensor([bev_h / bev_depth_meters], dtype=torch.float32))

        # Define ROI
        ratio = max(0.0, min(1.0, self.bev_roi_start_ratio))
        self.h_start = int(bev_h * ratio)
        self.h_end = bev_h - self.sobel_padding
        self.w_start = self.sobel_padding
        self.w_end = bev_w - self.sobel_padding

        # Slice UV grids for ROI
        # U-coordinate for BEV projection: (1, ROI_H, ROI_W) -> Flatten
        self.uv_roi_u_flat = self.projection_uv[:, 0, self.h_start:self.h_end, self.w_start:self.w_end].reshape(-1)

        # V-coordinate for Height calculation: (1, 1, ROI_H, ROI_W) - Keep spatial dims for multiplication
        self.uv_roi_v = self.projection_uv[:, 1:2, self.h_start:self.h_end, self.w_start:self.w_end]

        # Buffer for BEV map
        self.register_buffer('bev_flat_buffer', torch.zeros(bev_h * bev_w, dtype=torch.uint8))

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
            # 1. Extract ROI Depth
            depth_roi = depth[..., self.h_start:self.h_end, self.w_start:self.w_end]

            # 2. Calculate Height Proxy (H = Depth * v)
            height_map_roi = depth_roi * self.uv_roi_v

            # 3. Compute Gradients on HEIGHT MAP
            dx = torch.nn.functional.conv2d(height_map_roi, self.sobel_x, padding=0)
            dy = torch.nn.functional.conv2d(height_map_roi, self.sobel_y, padding=0)
            gradient_map = dx ** 2 + dy ** 2

            # 4. Pad back to ROI size (Matches Reference.py)
            # Padding order: (left, right, top, bottom)
            pad_size = self.sobel_padding
            gradient_map = torch.nn.functional.pad(
                gradient_map, 
                (pad_size, pad_size, pad_size, pad_size), 
                mode='constant', 
                value=0
            )

            # 5. Flatten
            depth_roi_flat = depth_roi.reshape(-1)
            grad_roi_flat = gradient_map.reshape(-1)

            # 6. Create Binary Mask
            mask_flat = (grad_roi_flat > threshold).to(torch.uint8)

            # 7. Compute BEV Indices
            x_idx = (self.uv_roi_u_flat * depth_roi_flat * self.bev_scale_x + self.bev_offset_x).int()
            z_idx = (depth_roi_flat * self.bev_scale_z).int()

            # 8. Scatter Accumulate
            linear_idx = z_idx * self.bev_w + x_idx
            self.bev_flat_buffer.scatter_add_(0, linear_idx, mask_flat)

            # 9. Reshape
            bev_map = self.bev_flat_buffer.view(self.bev_h, self.bev_w)

            return depth.squeeze(), bev_map

        return depth.squeeze()


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

roi_start_pixel = int(input_h * BEV_ROI_START_RATIO)

print(f'\nModel Configuration:')
print(f'  Sobel kernel: {SOBEL_KERNEL_SIZE}x{SOBEL_KERNEL_SIZE}')
print(f'  Output resolution: {input_h}x{input_w}')
print(f'  BEV ROI start ratio: {BEV_ROI_START_RATIO} (pixel row {roi_start_pixel}/{input_h})')
print(f'  BEV range: {BEV_WIDTH_METERS}m (width) × {BEV_DEPTH_METERS}m (depth)')
print(f'  Depth output: float32 in meters [0-{MAX_DEPTH}]')
print(f'  BEV output: Binary int8 (0 or 1)')

# ONNX Export
dummy_image = torch.ones(EXPORT_DEPTH_INPUT_SIZE, dtype=torch.uint8)
dummy_threshold = torch.tensor([DEFAULT_GRAD_THRESHOLD], dtype=torch.float32)

print(f"\nExporting to {output_path}...")
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
print("✅ ONNX export complete.\n")

# =================================================================================
# 2. INFERENCE TEST
# =================================================================================

print("Starting ONNX inference test...")
try:
    session_opts = ort.SessionOptions()
    session_opts.log_severity_level = 4
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_session = ort.InferenceSession(
        output_path,
        sess_options=session_opts,
        providers=['CPUExecutionProvider']
    )
except Exception as e:
    print(f"Error loading ONNX: {e}")
    sys.exit()

print(f"ONNX Runtime Providers: {ort_session.get_providers()}")

# Prepare Input
raw_img = cv2.imread(test_image_path)
if raw_img is None:
    print(f"❌ Error: Could not read image at {test_image_path}")
    sys.exit(1)

rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
resized_img = cv2.resize(rgb_img, (input_w, input_h))
input_tensor = np.expand_dims(resized_img, axis=0).transpose(0, 3, 1, 2).astype(np.uint8)

# Build input dictionary
input_names_ort = [inp.name for inp in ort_session.get_inputs()]
output_names_ort = [out.name for out in ort_session.get_outputs()]

input_feed = {input_names_ort[0]: input_tensor}
if 'threshold' in input_names_ort:
    input_feed['threshold'] = np.array([DEFAULT_GRAD_THRESHOLD], dtype=np.float32)

# Run Inference
print(f"\nRunning inference on {test_image_path}...")
start = time.time()
results = ort_session.run(output_names_ort, input_feed)
elapsed = time.time() - start
print(f'⏱️  Inference time: {elapsed:.3f} seconds')

depth_out = results[0]
bev_map = results[1] if len(output_names_ort) > 1 else None

print(f"\nDepth output dtype: {depth_out.dtype}, shape: {depth_out.shape}")
print(f"Depth range (meters): [{depth_out.min():.3f}, {depth_out.max():.3f}]")

if bev_map is not None:
    print(f"BEV output dtype: {bev_map.dtype}, shape: {bev_map.shape}")
    print(f"BEV value range: [{bev_map.min()}, {bev_map.max()}]")

# =================================================================================
# 3. VISUALIZATION
# =================================================================================

print("\n📊 Generating visualization...")

# Create figure with fixed aspect ratio subplots
fig = plt.figure(figsize=(18, 6))

# Input Image
ax1 = plt.subplot(1, 3, 1)
ax1.imshow(resized_img)
ax1.axhline(y=roi_start_pixel, color='r', linestyle='--', linewidth=2)
ax1.text(10, roi_start_pixel - 10, 'IGNORED REGION', color='red', fontweight='bold', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax1.set_title(f'Input Image ({input_w}×{input_h})')
ax1.axis('off')

# Depth Map
ax2 = plt.subplot(1, 3, 2)
im = ax2.imshow(depth_out, cmap='turbo')
ax2.axhline(y=roi_start_pixel, color='w', linestyle='--', linewidth=2)
ax2.set_title(f'Depth Map (ROI starts at y={roi_start_pixel})')
plt.colorbar(im, ax=ax2, label='Depth (m)', fraction=0.046, pad=0.04)
ax2.axis('off')

if bev_map is not None:
    ax3 = plt.subplot(1, 3, 3)
    # origin='lower' ensures 0m is at the bottom
    ax3.imshow(bev_map * 255, cmap='Greys', extent=[-BEV_WIDTH_METERS / 2, BEV_WIDTH_METERS / 2, 0, BEV_DEPTH_METERS], origin='lower')
    ax3.set_title("BEV Occupancy (Height-based)")

    # Add Grid
    ax3.grid(True, which='both', color='green', linestyle='--', linewidth=0.5, alpha=0.5)

    # Add Labels
    ax3.set_xlabel('Lateral Distance (m)')
    ax3.set_ylabel('Longitudinal Distance (m)')

    # Add Camera Position
    # Plot a red dot at (0,0) to represent the camera
    ax3.plot(0, 0, 'ro', markersize=8, label='Camera Position')
    ax3.legend(loc='upper right')


plt.tight_layout()
plt.show()
print("✅ Visualization complete.")

print("\n" + "=" * 70)
print("OUTPUT INFORMATION:")
print("=" * 70)
print(f"ROI Configuration: Starting at {BEV_ROI_START_RATIO * 100:.0f}% from top")
print(f"  - Processing rows: {roi_start_pixel} to {input_h} (out of {input_h})")
print(f"  - To change: Adjust BEV_ROI_START_RATIO (0.0=use full image, 0.5=bottom half)")
print(f"\nDepth output: float32 in meters, range [0, {MAX_DEPTH}]")
print(f"BEV format: Binary int8 (0 = free space, 1 = occupied)")
print(f"Threshold: {DEFAULT_GRAD_THRESHOLD} (gradient magnitude threshold)")
print("=" * 70)
print("✅ Done.")
