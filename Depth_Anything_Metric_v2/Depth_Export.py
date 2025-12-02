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
DEFAULT_GRAD_THRESHOLD = 0.09    # Set a appropriate value for detected object.
BEV_WIDTH_METERS = 10.0          # The max width in the image.
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
    span_y = scale_factor(height) / denomimport torch
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
DEFAULT_GRAD_THRESHOLD = 0.09    # Set a appropriate value for detected object.
BEV_WIDTH_METERS = 10.0          # The max width in the image.
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
# OPTIMIZED MODEL WRAPPER WITH REFERENCE.PY BEV APPROACH
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

        # Setup UV grid FIRST (before BEV parameters)
        self._setup_uv_grid()

        # Setup BEV parameters (needs self.projection_uv from above)
        self._setup_bev_parameters(bev_width_meters, bev_depth_meters, sobel_kernel_size)

        # Zero padding buffers for gradient map
        self.zeros_w = torch.zeros([1, self.h // 2 - (sobel_kernel_size - 1) // 2, (sobel_kernel_size - 1) // 2], dtype=torch.float32)
        self.zeros_h = torch.zeros([1, (sobel_kernel_size - 1) // 2, self.w - (sobel_kernel_size - 1) * 2], dtype=torch.float32)


    def _setup_uv_grid(self):
        """Setup UV projection grid (u-coordinate only)."""
        aspect_ratio = self.w / self.h
        projection_uv = create_normalized_uv_grid(self.w, self.h, aspect_ratio)
        self.projection_uv = projection_uv.permute(2, 0, 1)[0].unsqueeze(0)

    def _setup_bev_parameters(self, bev_width_meters, bev_depth_meters, sobel_kernel_size):
        """Setup BEV map parameters and ROI bounds based on BEV_ROI_START_RATIO."""
        bev_h, bev_w = self.h, self.w

        self.register_buffer('bev_w', torch.tensor(bev_w, dtype=torch.int64))
        self.register_buffer('bev_h', torch.tensor(bev_h, dtype=torch.int64))

        # Pre-calculate scalars for projection
        self.register_buffer('bev_scale_x', torch.tensor(bev_w / bev_width_meters, dtype=torch.float32))
        self.register_buffer('bev_offset_x', torch.tensor(bev_w / 2.0, dtype=torch.float32))
        self.register_buffer('bev_scale_z', torch.tensor(bev_h / bev_depth_meters, dtype=torch.float32))

        # Define ROI based on BEV_ROI_START_RATIO
        ratio = max(0.0, min(1.0, self.bev_roi_start_ratio))
        self.h_start = int(bev_h * ratio)
        self.h_end = bev_h - self.sobel_padding
        self.w_start = self.sobel_padding
        self.w_end = bev_w - self.sobel_padding

        # Pre-slice and flatten the UV ROI to save time in forward pass
        # Shape: (1, ROI_H, ROI_W) -> (ROI_H * ROI_W)
        self.uv_roi_flat = self.projection_uv[:, self.h_start:self.h_end, self.w_start:self.w_end].reshape(-1)

        # Pre-allocate buffer for the flat BEV map
        # We use int8 for scatter_add, then convert to binary
        self.register_buffer('bev_flat_buffer', torch.zeros(bev_h * bev_w, dtype=torch.int8))

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
        depth = depth.squeeze(0)
        if OUTPUT_BEV:
            # ===================================================================
            # BEV GENERATION - Following Reference.py approach
            # ===================================================================

            # 1. Compute gradient map
            depth_roi_flat = depth[..., self.h_start:self.h_end, self.w_start:self.w_end]

            dx = torch.nn.functional.conv2d(depth_roi_flat, self.sobel_x, padding=0)
            dy = torch.nn.functional.conv2d(depth_roi_flat, self.sobel_y, padding=0)
            gradient_map = dx ** 2 + dy ** 2
            gradient_map = torch.cat([self.zeros_h, gradient_map, self.zeros_h], dim=-2)
            gradient_map = torch.cat([self.zeros_w, gradient_map, self.zeros_w], dim=-1)

            # 2. Extract ROI and Flatten immediately
            depth_roi_flat = depth_roi_flat.reshape(-1)
            grad_roi_flat = gradient_map.reshape(-1)

            # 3. Create Binary Mask (Element-wise comparison)
            mask_flat = (grad_roi_flat > threshold).to(torch.int8)

            # 4. Compute BEV Indices for ALL pixels in ROI (Vectorized)
            x_idx = (self.uv_roi_flat * depth_roi_flat * self.bev_scale_x + self.bev_offset_x).long()
            z_idx = (depth_roi_flat * self.bev_scale_z).long()

            # 5. Clamp indices to stay within BEV map bounds
            x_idx = x_idx.clamp(0, self.bev_w - 1)
            z_idx = z_idx.clamp(0, self.bev_h - 1)

            # 6. Compute Linear Indices for 1D Scatter
            linear_idx = z_idx * self.bev_w + x_idx

            # 7. Scatter Accumulate
            self.bev_flat_buffer.scatter_add_(0, linear_idx, mask_flat)

            # 8. Reshape and Binarize
            bev_map = self.bev_flat_buffer.view(self.bev_h, self.bev_w)

            # Flip to match coordinate system (bottom-up) and binarize
            bev_map = torch.flip(bev_map, dims=[0])
            bev_map = torch.clamp(bev_map, min=0, max=1)

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

depth_out = results[0][0]
bev_out = results[1] if len(output_names_ort) > 1 else None

print(f"\nDepth output dtype: {depth_out.dtype}, shape: {depth_out.shape}")
print(f"Depth range (meters): [{depth_out.min():.3f}, {depth_out.max():.3f}]")

if bev_out is not None:
    print(f"BEV output dtype: {bev_out.dtype}, shape: {bev_out.shape}")
    print(f"BEV value range: [{bev_out.min()}, {bev_out.max()}]")

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

if bev_out is not None:
    # BEV Occupancy
    ax3 = plt.subplot(1, 3, 3)
    extent = [-BEV_WIDTH_METERS / 2, BEV_WIDTH_METERS / 2, 0, BEV_DEPTH_METERS]
    ax3.imshow(bev_out, cmap='Greys', interpolation='bilinear', extent=extent, aspect='auto')
    ax3.plot(0, 0, 'ro', markersize=8, label='Camera Position')
    ax3.set_title(f'BEV Occupancy Map ({input_w}×{input_h})')
    ax3.set_xlabel('Lateral Distance (m)')
    ax3.set_ylabel('Longitudinal Distance (m)')
    ax3.legend()
    ax3.grid(True, color='green', linestyle='--', alpha=0.3)

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


    u = torch.linspace(-span_x, span_x, width, dtype=torch.float32)
    v = torch.linspace(-span_y, span_y, height, dtype=torch.float32)

    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
    return torch.stack([u_grid, v_grid], dim=-1)


# =================================================================================
# OPTIMIZED MODEL WRAPPER WITH REFERENCE.PY BEV APPROACH
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

        # Setup UV grid FIRST (before BEV parameters)
        self._setup_uv_grid()

        # Setup BEV parameters (needs self.projection_uv from above)
        self._setup_bev_parameters(bev_width_meters, bev_depth_meters, sobel_kernel_size)

        # Zero padding buffers for gradient map
        self.zeros_w = torch.zeros([1, self.h // 2 - (sobel_kernel_size - 1) // 2, (sobel_kernel_size - 1) // 2], dtype=torch.float32)
        self.zeros_h = torch.zeros([1, (sobel_kernel_size - 1) // 2, self.w - (sobel_kernel_size - 1) * 2], dtype=torch.float32)


    def _setup_uv_grid(self):
        """Setup UV projection grid (u-coordinate only)."""
        aspect_ratio = self.w / self.h
        projection_uv = create_normalized_uv_grid(self.w, self.h, aspect_ratio)
        self.projection_uv = projection_uv.permute(2, 0, 1)[0].unsqueeze(0)

    def _setup_bev_parameters(self, bev_width_meters, bev_depth_meters, sobel_kernel_size):
        """Setup BEV map parameters and ROI bounds based on BEV_ROI_START_RATIO."""
        bev_h, bev_w = self.h, self.w

        self.register_buffer('bev_w', torch.tensor(bev_w, dtype=torch.int64))
        self.register_buffer('bev_h', torch.tensor(bev_h, dtype=torch.int64))

        # Pre-calculate scalars for projection
        self.register_buffer('bev_scale_x', torch.tensor(bev_w / bev_width_meters, dtype=torch.float32))
        self.register_buffer('bev_offset_x', torch.tensor(bev_w / 2.0, dtype=torch.float32))
        self.register_buffer('bev_scale_z', torch.tensor(bev_h / bev_depth_meters, dtype=torch.float32))

        # Define ROI based on BEV_ROI_START_RATIO
        ratio = max(0.0, min(1.0, self.bev_roi_start_ratio))
        self.h_start = int(bev_h * ratio)
        self.h_end = bev_h - self.sobel_padding
        self.w_start = self.sobel_padding
        self.w_end = bev_w - self.sobel_padding

        # Pre-slice and flatten the UV ROI to save time in forward pass
        # Shape: (1, ROI_H, ROI_W) -> (ROI_H * ROI_W)
        self.uv_roi_flat = self.projection_uv[:, self.h_start:self.h_end, self.w_start:self.w_end].reshape(-1)

        # Pre-allocate buffer for the flat BEV map
        # We use int8 for scatter_add, then convert to binary
        self.register_buffer('bev_flat_buffer', torch.zeros(bev_h * bev_w, dtype=torch.int8))

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
        depth = depth.squeeze(0)
        if OUTPUT_BEV:
            # 1. Compute gradient map
            depth_roi_flat = depth[..., self.h_start:self.h_end, self.w_start:self.w_end]

            dx = torch.nn.functional.conv2d(depth_roi_flat, self.sobel_x, padding=0)
            dy = torch.nn.functional.conv2d(depth_roi_flat, self.sobel_y, padding=0)
            gradient_map = dx ** 2 + dy ** 2
            gradient_map = torch.cat([self.zeros_h, gradient_map, self.zeros_h], dim=-2)
            gradient_map = torch.cat([self.zeros_w, gradient_map, self.zeros_w], dim=-1)

            # 2. Extract ROI and Flatten immediately
            depth_roi_flat = depth_roi_flat.reshape(-1)
            grad_roi_flat = gradient_map.reshape(-1)

            # 3. Create Binary Mask (Element-wise comparison)
            mask_flat = (grad_roi_flat > threshold).to(torch.int8)

            # 4. Compute BEV Indices for ALL pixels in ROI (Vectorized)
            x_idx = (self.uv_roi_flat * depth_roi_flat * self.bev_scale_x + self.bev_offset_x).long()
            z_idx = (depth_roi_flat * self.bev_scale_z).long()

            # 5. Clamp indices to stay within BEV map bounds
            x_idx = x_idx.clamp(0, self.bev_w - 1)
            z_idx = z_idx.clamp(0, self.bev_h - 1)

            # 6. Compute Linear Indices for 1D Scatter
            linear_idx = z_idx * self.bev_w + x_idx

            # 7. Scatter Accumulate
            self.bev_flat_buffer.scatter_add_(0, linear_idx, mask_flat)

            # 8. Reshape and Binarize
            bev_map = self.bev_flat_buffer.view(self.bev_h, self.bev_w)

            # Flip to match coordinate system (bottom-up) and binarize
            bev_map = torch.flip(bev_map, dims=[0])
            bev_map = torch.clamp(bev_map, min=0, max=1)

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

depth_out = results[0][0]
bev_out = results[1] if len(output_names_ort) > 1 else None

print(f"\nDepth output dtype: {depth_out.dtype}, shape: {depth_out.shape}")
print(f"Depth range (meters): [{depth_out.min():.3f}, {depth_out.max():.3f}]")

if bev_out is not None:
    print(f"BEV output dtype: {bev_out.dtype}, shape: {bev_out.shape}")
    print(f"BEV value range: [{bev_out.min()}, {bev_out.max()}]")

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

if bev_out is not None:
    # BEV Occupancy
    ax3 = plt.subplot(1, 3, 3)
    extent = [-BEV_WIDTH_METERS / 2, BEV_WIDTH_METERS / 2, 0, BEV_DEPTH_METERS]
    ax3.imshow(bev_out, cmap='Greys', interpolation='bilinear', extent=extent, aspect='auto')
    ax3.plot(0, 0, 'ro', markersize=8, label='Camera Position')
    ax3.set_title(f'BEV Occupancy Map ({input_w}×{input_h})')
    ax3.set_xlabel('Lateral Distance (m)')
    ax3.set_ylabel('Longitudinal Distance (m)')
    ax3.legend()
    ax3.grid(True, color='green', linestyle='--', alpha=0.3)

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

