import sys
import cv2
import torch
import time
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_PATH = '/home/DakeQQ/Downloads/MoGe-main'                   # https://github.com/microsoft/MoGe
MODEL_PATH = '/home/DakeQQ/Downloads/moge-2-vits-normal/model.pt'   # Only support MoGe_V2
ONNX_MODEL_PATH = '/home/DakeQQ/Downloads/MoGe_ONNX/MoGe.onnx'
TEST_IMAGE_PATH = "./test.jpg"

INPUT_IMAGE_SIZE = [720, 1280]   # Input image shape [Height, Width].
NUM_TOKENS = 3600                # Larger is finer but slower.
FOCAL = None                     # Set None for auto else fixed.

OUTPUT_BEV = True                # True for output the bird eye view occupancy map.
SOBEL_KERNEL_SIZE = 3            # [3, 5] set for gradient map.
DEFAULT_GRAD_THRESHOLD = 0.09    # Set a appropriate value for detected object.
BEV_WIDTH_METERS = 10.0          # The max width in the image.
BEV_DEPTH_METERS = 10.0          # The max depth in the image.
BEV_ROI_START_RATIO = 0.5        # Start position for ROI (0.0 = top, 1.0 = bottom).
OP_SET = 17                      # ONNX Runtime opset.

if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from moge.model.v2 import MoGeModel

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

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


def solve_optimal_focal_shift(uv_flat, xy, z, num_iters=15):
    shift = torch.zeros(1, dtype=torch.float32)
    z_shifted = z
    # Gauss-Newton iterations
    for _ in range(num_iters):
        xy_proj = xy / z_shifted

        focal = (xy_proj * uv_flat).sum() / ((xy_proj ** 2).sum() + 1e-6)
        residual = focal * xy_proj - uv_flat
        jacobian = -focal * xy / (z_shifted ** 2 + 1e-6)

        delta_shift = (jacobian * residual).sum() / ((jacobian ** 2).sum() + 1e-6)
        shift -= delta_shift
        z_shifted = z + shift
    return shift


def solve_optimal_shift(uv_flat, xy, z, focal, num_iters=15):
    shift = torch.zeros(1, dtype=torch.float32)
    focal_xy = focal * xy
    z_shifted = z
    # Gauss-Newton iterations
    for _ in range(num_iters):
        focal_xy_z_shifted = focal_xy / z_shifted
        residual = focal_xy_z_shifted - uv_flat
        jacobian = focal_xy_z_shifted / z_shifted

        delta_shift = (-jacobian * residual).sum() / ((jacobian ** 2).sum() + 1e-6)
        shift -= delta_shift
        z_shifted = z + shift
    return shift


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


# ==============================================================================
# MODEL WRAPPER
# ==============================================================================

class MoGeV2(torch.nn.Module):
    def __init__(self, model, output_image_size, num_tokens, sobel_kernel_size=3, bev_roi_start_ratio=0.5):
        super().__init__()
        self.model = model
        self.output_image_size = output_image_size
        self.aspect_ratio = output_image_size[1] / output_image_size[0]
        self.bev_roi_start_ratio = bev_roi_start_ratio

        # Calculate internal processing resolution
        sqrt_tokens_aspect = num_tokens ** 0.5
        self.base_h = int(sqrt_tokens_aspect / self.aspect_ratio ** 0.5)
        self.base_w = int(sqrt_tokens_aspect * self.aspect_ratio ** 0.5)
        self.base_hw = (self.base_h, self.base_w)

        patch_size = model.encoder.backbone.patch_embed.patch_size
        self.internal_size = [self.base_h * patch_size[0], self.base_w * patch_size[1]]

        # Normalization constants
        self.inv_image_std_255 = 1.0 / (255.0 * model.encoder.image_std)
        self.image_mean_inv_std = model.encoder.image_mean / model.encoder.image_std

        # Setup positional embeddings
        self._setup_pos_embeddings()

        # Precompute UV grids
        self._setup_uv_grids()

        # Setup Sobel kernels
        self.sobel_x, self.sobel_y = get_sobel_kernels(sobel_kernel_size)
        self.sobel_padding = sobel_kernel_size // 2

        # Setup BEV parameters
        self._setup_bev_parameters()

        self.zeros_w = torch.zeros([1, output_image_size[0] // 2 - (sobel_kernel_size - 1) // 2, (sobel_kernel_size - 1) // 2], dtype=torch.float32)
        self.zeros_h = torch.zeros([1, (sobel_kernel_size - 1) // 2, output_image_size[1] - (sobel_kernel_size - 1) * 2], dtype=torch.float32)

    def _setup_pos_embeddings(self):
        """Prepare positional embeddings for the model."""
        pos_embed = self.model.encoder.backbone.pos_embed
        class_pos_embed = pos_embed[:, :1, :]
        patch_pos_embed = pos_embed[:, 1:, :]

        M = int((pos_embed.shape[1] - 1) ** 0.5)
        interpolate_offset = self.model.encoder.backbone.interpolate_offset
        scale_factor = (
            (self.base_h + interpolate_offset) / M,
            (self.base_w + interpolate_offset) / M
        )

        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, -1).permute(0, 3, 1, 2),
            mode="bicubic", antialias=False, align_corners=False, scale_factor=scale_factor
        ).permute(0, 2, 3, 1).flatten(1, 2)

        self.pos_embed = torch.cat([class_pos_embed, patch_pos_embed], dim=1)
        self.model.encoder.backbone.num_register_tokens += 1

    def _setup_uv_grids(self):
        """Precompute UV grids for different resolution levels."""
        self.uv_grids = []

        # Multi-scale UV grids
        for level in range(5):
            scale = 2 ** level
            uv = create_normalized_uv_grid(
                self.base_w * scale,
                self.base_h * scale,
                self.aspect_ratio
            )
            self.uv_grids.append(uv.permute(2, 0, 1).unsqueeze(0))

        # Low-res UV for focal recovery
        uv = create_normalized_uv_grid(self.base_w, self.base_h, self.aspect_ratio)
        uv_lr = torch.nn.functional.interpolate(
            uv.permute(2, 0, 1).unsqueeze(0),
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        )
        self.uv_grids.append(uv_lr.permute(0, 2, 3, 1).reshape(-1, 2))

        # Projection UV grid (u-coordinate only)
        projection_uv = create_normalized_uv_grid(
            self.output_image_size[1],
            self.output_image_size[0],
            self.aspect_ratio
        )
        self.projection_uv = projection_uv.permute(2, 0, 1)[0].unsqueeze(0)

    def _setup_bev_parameters(self):
        """Setup BEV map parameters and ROI bounds based on BEV_ROI_START_RATIO."""
        bev_h, bev_w = self.output_image_size

        self.register_buffer('bev_w', torch.tensor(bev_w, dtype=torch.int64))
        self.register_buffer('bev_h', torch.tensor(bev_h, dtype=torch.int64))
        
        # Pre-calculate scalars for projection
        self.register_buffer('bev_scale_x', torch.tensor(bev_w / BEV_WIDTH_METERS, dtype=torch.float32))
        self.register_buffer('bev_offset_x', torch.tensor(bev_w / 2.0, dtype=torch.float32))
        self.register_buffer('bev_scale_z', torch.tensor(bev_h / BEV_DEPTH_METERS, dtype=torch.float32))

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
        # We use float for scatter_add, then convert to binary
        self.register_buffer('bev_flat_buffer', torch.zeros(bev_h * bev_w, dtype=torch.int8))

    def recover_focal_shift(self, points, focal=None, downsample_size=(64, 64)):
        points_lr = torch.nn.functional.interpolate(
            points.permute(0, 3, 1, 2),
            size=downsample_size,
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1).reshape(-1, 3)

        xy, z = torch.split(points_lr, [2, 1], dim=-1)

        if focal:
            return solve_optimal_shift(self.uv_grids[-1], xy, z, focal)
        else:
            return solve_optimal_focal_shift(self.uv_grids[-1], xy, z)

    def forward(self, image, threshold, focal=None):
        # Resize input to internal resolution
        image = image.float()
        if self.internal_size != self.output_image_size:
            image = torch.nn.functional.interpolate(
                image, self.internal_size, mode="bilinear",
                align_corners=False, antialias=False
            )

        # Normalize image
        x = image * self.inv_image_std_255 - self.image_mean_inv_std

        # Patch embedding
        x = self.model.encoder.backbone.patch_embed.proj(x)
        x = self.model.encoder.backbone.patch_embed.norm(x.flatten(2).transpose(1, 2))
        x = torch.cat([self.model.encoder.backbone.cls_token, x], dim=1) + self.pos_embed.data

        # Encoder forward pass
        outputs = []
        for i, blk in enumerate(self.model.encoder.backbone.blocks):
            x = blk(x)
            if i in self.model.encoder.intermediate_layers:
                outputs.append(x)

        outputs = [self.model.encoder.backbone.norm(out) for out in outputs]
        num_reg_tokens = self.model.encoder.backbone.num_register_tokens
        features = [(out[:, num_reg_tokens:], out[:, 0]) for out in outputs]

        # Feature projection
        x = sum(
            proj(feat.permute(0, 2, 1).unflatten(2, self.base_hw).contiguous())
            for proj, (feat, _) in zip(self.model.encoder.output_projections, features)
        )

        # Multi-scale features with UV
        features_with_uv = [torch.cat([x, self.uv_grids[0]], dim=1)] + self.uv_grids[1:5]
        features_with_uv = self.model.neck(features_with_uv)
        points = self.model.points_head(features_with_uv)[-1].permute(0, 2, 3, 1)
        points = self.model._remap_points(points)

        # Recover metric scale and shift
        cls_token = features[-1][1]
        metric_scale = self.model.scale_head(cls_token).exp()
        shift = self.recover_focal_shift(points, focal=focal)

        # Compute depth
        depth = (points[..., 2] + shift) * metric_scale

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


# ==============================================================================
# EXPORT TO ONNX
# ==============================================================================

def export_model_to_onnx():
    """Load model, export to ONNX format."""
    model = MoGeModel.from_pretrained(MODEL_PATH).to('cpu').eval().float()
    model = MoGeV2(
        model, 
        INPUT_IMAGE_SIZE, 
        NUM_TOKENS, 
        sobel_kernel_size=SOBEL_KERNEL_SIZE,
        bev_roi_start_ratio=BEV_ROI_START_RATIO
    )

    image = torch.ones([1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]], dtype=torch.uint8)
    threshold_tensor = torch.tensor([DEFAULT_GRAD_THRESHOLD], dtype=torch.float32)

    roi_start_pixel = int(INPUT_IMAGE_SIZE[0] * BEV_ROI_START_RATIO)
    
    print(f'\nModel Configuration:')
    print(f'  Sobel kernel: {SOBEL_KERNEL_SIZE}x{SOBEL_KERNEL_SIZE}')
    print(f'  Internal resolution: {model.internal_size}')
    print(f'  Output resolution: {INPUT_IMAGE_SIZE}')
    print(f'  BEV ROI start ratio: {BEV_ROI_START_RATIO} (pixel row {roi_start_pixel}/{INPUT_IMAGE_SIZE[0]})')
    print(f'  BEV range: {BEV_WIDTH_METERS}m (width) × {BEV_DEPTH_METERS}m (depth)')
    print('\nExporting to ONNX...')

    with torch.inference_mode():
        if FOCAL:
            focal = torch.tensor([FOCAL], dtype=torch.float32)
            inputs = (image, threshold_tensor, focal)
            input_names = ['image', 'threshold', 'focal']
        else:
            inputs = (image, threshold_tensor)
            input_names = ['image', 'threshold']

        if OUTPUT_BEV:
            output_names = ['depth_map', 'bev_map']
        else:
            output_names = ['depth_map']

        torch.onnx.export(
            model, inputs, ONNX_MODEL_PATH,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            opset_version=OP_SET,
            dynamo=False
        )

    print('✅ Export complete.\n')

    # Cleanup
    del model, inputs, image
    if PROJECT_PATH in sys.path:
        sys.path.remove(PROJECT_PATH)


# ==============================================================================
# ONNX INFERENCE
# ==============================================================================

def run_onnx_inference():
    """Run inference using exported ONNX model."""
    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 4
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_session = onnxruntime.InferenceSession(
        ONNX_MODEL_PATH,
        sess_options=session_opts,
        providers=['CPUExecutionProvider']
    )

    print(f"ONNX Runtime Providers: {ort_session.get_providers()}")

    # Prepare input
    raw_img = cv2.imread(TEST_IMAGE_PATH)
    if raw_img is None:
        print(f"❌ Error: Could not read image at {TEST_IMAGE_PATH}")
        sys.exit(1)

    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, (INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
    image_tensor = np.expand_dims(resized_img, axis=0).transpose(0, 3, 1, 2)

    # Build input dictionary
    input_names = [inp.name for inp in ort_session.get_inputs()]
    output_names = [out.name for out in ort_session.get_outputs()]

    input_feed = {input_names[0]: image_tensor}
    if 'threshold' in input_names:
        input_feed['threshold'] = np.array([DEFAULT_GRAD_THRESHOLD], dtype=np.float32)
    if 'focal' in input_names and FOCAL:
        input_feed['focal'] = np.array([FOCAL], dtype=np.float32)

    # Run inference
    print(f"\nRunning inference on {TEST_IMAGE_PATH}...")
    start = time.time()
    results = ort_session.run(output_names, input_feed)
    elapsed = time.time() - start
    print(f'⏱️  Inference time: {elapsed:.3f} seconds')

    return resized_img, results[0][0], results[1] if len(output_names) > 1 else None


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_results(input_image, depth_map, bev_map):
    """Display input, depth, and BEV maps with consistent sizing."""
    print("\n📊 Generating visualization...")

    roi_start_pixel = int(INPUT_IMAGE_SIZE[0] * BEV_ROI_START_RATIO)
    
    # Create figure with fixed aspect ratio subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Input Image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(input_image)
    ax1.axhline(y=roi_start_pixel, color='r', linestyle='--', linewidth=2)
    ax1.text(10, roi_start_pixel - 10, 'IGNORED REGION', color='red', fontweight='bold', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax1.set_title(f'Input Image ({INPUT_IMAGE_SIZE[1]}×{INPUT_IMAGE_SIZE[0]})')
    ax1.axis('off')

    # Depth Map
    ax2 = plt.subplot(1, 3, 2)
    im = ax2.imshow(depth_map, cmap='turbo')
    ax2.axhline(y=roi_start_pixel, color='w', linestyle='--', linewidth=2)
    ax2.set_title(f'Depth Map (ROI starts at y={roi_start_pixel})')
    plt.colorbar(im, ax=ax2, label='Depth (m)', fraction=0.046, pad=0.04)
    ax2.axis('off')

    if bev_map is not None:
        # BEV Occupancy
        ax3 = plt.subplot(1, 3, 3)
        extent = [-BEV_WIDTH_METERS / 2, BEV_WIDTH_METERS / 2, 0, BEV_DEPTH_METERS]
        ax3.imshow(bev_map, cmap='Greys', interpolation='bilinear', extent=extent, aspect='auto')
        ax3.plot(0, 0, 'ro', markersize=8, label='Camera Position')
        ax3.set_title(f'BEV Occupancy Map ({INPUT_IMAGE_SIZE[1]}×{INPUT_IMAGE_SIZE[0]})')
        ax3.set_xlabel('Lateral Distance (m)')
        ax3.set_ylabel('Longitudinal Distance (m)')
        ax3.legend()
        ax3.grid(True, color='green', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("✅ Visualization complete.")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Export model
    export_model_to_onnx()

    # Run inference
    input_img, depth, bev = run_onnx_inference()

    # Visualize results
    visualize_results(input_img, depth, bev)
