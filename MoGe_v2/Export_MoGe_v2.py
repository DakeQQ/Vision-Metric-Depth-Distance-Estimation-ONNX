import sys
import cv2
import time
import torch
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================


PROJECT_PATH = '/home/iamj/Downloads/MoGe-main'                   # https://github.com/microsoft/MoGe
MODEL_PATH = '/home/iamj/Downloads/moge-2-vits-normal/model.pt'   # Only support MoGe_V2
ONNX_MODEL_PATH = '/home/iamj/Downloads/MoGe_ONNX/MoGe.onnx'
TEST_IMAGE_PATH = "./test.jpg"

INPUT_IMAGE_SIZE = [720, 1280]   # Input image shape [Height, Width].
NUM_TOKENS = 3600                # Larger is finer but slower.
FOCAL = None                     # Set None for auto else fixed.

OUTPUT_BEV = True                # True for output the bird eye view occupancy map.
SOBEL_KERNEL_SIZE = 3            # [3, 5] set for gradient map.
DEFAULT_GRAD_THRESHOLD = 0.001   # Set a appropriate value for detected object.
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
    for _ in range(num_iters):
        focal_xy_z_shifted = focal_xy / z_shifted
        residual = focal_xy_z_shifted - uv_flat
        jacobian = focal_xy_z_shifted / z_shifted
        delta_shift = (-jacobian * residual).sum() / ((jacobian ** 2).sum() + 1e-6)
        shift -= delta_shift
        z_shifted = z + shift
    return shift


def get_sobel_kernels(kernel_size):
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
        raise ValueError(f"Unsupported kernel size: {kernel_size}")
    sobel_x, sobel_y = kernels[kernel_size]
    return sobel_x.view(1, 1, kernel_size, kernel_size), sobel_y.view(1, 1, kernel_size, kernel_size)


# ==============================================================================
# MODEL WRAPPER
# ==============================================================================
class MoGeV2(torch.nn.Module):
    def __init__(self, model, output_image_size, num_tokens, sobel_kernel_size=3, bev_roi_start_ratio=0.5):
        super().__init__()
        self.model = model
        self._replace_gelu_with_tanh_approximation(self.model)
        self.output_image_size = output_image_size
        self.aspect_ratio = output_image_size[1] / output_image_size[0]
        self.bev_roi_start_ratio = bev_roi_start_ratio

        # Internal resolution setup
        sqrt_tokens_aspect = num_tokens ** 0.5
        self.base_h = int(sqrt_tokens_aspect / self.aspect_ratio ** 0.5)
        self.base_w = int(sqrt_tokens_aspect * self.aspect_ratio ** 0.5)
        self.base_hw = (self.base_h, self.base_w)
        patch_size = model.encoder.backbone.patch_embed.patch_size
        self.internal_size = [self.base_h * patch_size[0], self.base_w * patch_size[1]]

        # Normalization
        self.inv_image_std_255 = 1.0 / (255.0 * model.encoder.image_std)
        self.image_mean_inv_std = model.encoder.image_mean / model.encoder.image_std

        self._setup_pos_embeddings()
        self._setup_uv_grids()
        
        # Sobel Setup
        self.sobel_x, self.sobel_y = get_sobel_kernels(sobel_kernel_size)
        self.sobel_padding = sobel_kernel_size // 2

        # BEV Setup
        self._setup_bev_parameters()

        self.zeros_h = torch.zeros([1, 1, int(output_image_size[0] * (1 - BEV_ROI_START_RATIO)) - self.sobel_padding, self.sobel_padding], dtype=torch.float32)
        self.zeros_h_plus_1 = torch.zeros([1, 1, int(output_image_size[0] * (1 - BEV_ROI_START_RATIO)) - self.sobel_padding + 1, self.sobel_padding], dtype=torch.float32)
        self.zeros_h_minus_1 = torch.zeros([1, 1, int(output_image_size[0] * (1 - BEV_ROI_START_RATIO)) - self.sobel_padding - 1, self.sobel_padding], dtype=torch.float32)
        self.zeros_w = torch.zeros([1, 1, self.sobel_padding, output_image_size[1] - ((sobel_kernel_size - 1) * 2)], dtype=torch.float32)

    def _replace_gelu_with_tanh_approximation(self, module):
        """
        Recursively replace all GELU(approximate='none' or None)
        with GELU(approximate='tanh') in the module tree
        """
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                # Replace with tanh approximation version
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                # Recursively apply to child modules
                self._replace_gelu_with_tanh_approximation(child)

    def _setup_pos_embeddings(self):
        pos_embed = self.model.encoder.backbone.pos_embed
        class_pos_embed = pos_embed[:, :1, :]
        patch_pos_embed = pos_embed[:, 1:, :]
        M = int((pos_embed.shape[1] - 1) ** 0.5)
        interpolate_offset = self.model.encoder.backbone.interpolate_offset
        scale_factor = ((self.base_h + interpolate_offset) / M, (self.base_w + interpolate_offset) / M)
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, -1).permute(0, 3, 1, 2),
            mode="bicubic", antialias=False, align_corners=False, scale_factor=scale_factor
        ).permute(0, 2, 3, 1).flatten(1, 2)
        self.pos_embed = torch.cat([class_pos_embed, patch_pos_embed], dim=1)
        self.model.encoder.backbone.num_register_tokens += 1

    def _setup_uv_grids(self):
        self.uv_grids = []
        for level in range(5):
            scale = 2 ** level
            uv = create_normalized_uv_grid(self.base_w * scale, self.base_h * scale, self.aspect_ratio)
            self.uv_grids.append(uv.permute(2, 0, 1).unsqueeze(0))

        uv = create_normalized_uv_grid(self.base_w, self.base_h, self.aspect_ratio)
        uv_lr = torch.nn.functional.interpolate(
            uv.permute(2, 0, 1).unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False
        )
        self.uv_grids.append(uv_lr.permute(0, 2, 3, 1).reshape(-1, 2))

        # Generate full UV grid for output resolution (Both U and V channels)
        # Shape: (1, 2, H, W)
        full_uv = create_normalized_uv_grid(self.output_image_size[1], self.output_image_size[0], self.aspect_ratio)
        self.full_projection_uv = full_uv.permute(2, 0, 1).unsqueeze(0)

    def _setup_bev_parameters(self):
        bev_h, bev_w = self.output_image_size
        
        self.register_buffer('bev_w', torch.tensor([bev_w], dtype=torch.int32))
        self.register_buffer('bev_h', torch.tensor([bev_h], dtype=torch.int64))
        self.register_buffer('bev_scale_x', torch.tensor([bev_w / BEV_WIDTH_METERS], dtype=torch.float32))
        self.register_buffer('bev_offset_x', torch.tensor([bev_w / 2.0], dtype=torch.float32))
        self.register_buffer('bev_scale_z', torch.tensor([bev_h / BEV_DEPTH_METERS], dtype=torch.float32))

        ratio = max(0.0, min(1.0, self.bev_roi_start_ratio))
        self.h_start = int(bev_h * ratio)
        self.h_end = bev_h - self.sobel_padding
        self.w_start = self.sobel_padding
        self.w_end = bev_w - self.sobel_padding

        # Slice UV grids for ROI
        # U-coordinate for BEV projection: (1, ROI_H, ROI_W) -> Flatten
        self.uv_roi_u_flat = self.full_projection_uv[:, 0, self.h_start:self.h_end, self.w_start:self.w_end].reshape(-1)

        # V-coordinate for Height calculation: (1, 1, ROI_H, ROI_W) - Keep spatial dims for element-wise mul
        self.uv_roi_v = self.full_projection_uv[:, 1:2, self.h_start:self.h_end, self.w_start:self.w_end]

        self.register_buffer('bev_flat_buffer', torch.zeros(bev_h * bev_w, dtype=torch.uint8))

    def recover_focal_shift(self, points, focal=None, downsample_size=(64, 64)):
        points_lr = torch.nn.functional.interpolate(
            points.permute(0, 3, 1, 2), size=downsample_size, mode='bilinear', align_corners=False
        ).permute(0, 2, 3, 1).reshape(-1, 3)
        xy, z = torch.split(points_lr, [2, 1], dim=-1)
        if focal:
            return solve_optimal_shift(self.uv_grids[-1], xy, z, focal)
        return solve_optimal_focal_shift(self.uv_grids[-1], xy, z)

    def forward(self, image, threshold, focal=None):
        # 1. Preprocessing
        image = image.float()
        if self.internal_size != self.output_image_size:
            image = torch.nn.functional.interpolate(image, self.internal_size, mode="bilinear", align_corners=False)
        x = image * self.inv_image_std_255 - self.image_mean_inv_std

        # 2. Encoder
        x = self.model.encoder.backbone.patch_embed.proj(x)
        x = self.model.encoder.backbone.patch_embed.norm(x.flatten(2).transpose(1, 2))
        x = torch.cat([self.model.encoder.backbone.cls_token, x], dim=1) + self.pos_embed.data

        outputs = []
        for i, blk in enumerate(self.model.encoder.backbone.blocks):
            x = blk(x)
            if i in self.model.encoder.intermediate_layers:
                outputs.append(x)

        outputs = [self.model.encoder.backbone.norm(out) for out in outputs]
        num_reg_tokens = self.model.encoder.backbone.num_register_tokens
        features = [(out[:, num_reg_tokens:], out[:, 0]) for out in outputs]

        # 3. Decoder
        x = sum(proj(feat.permute(0, 2, 1).unflatten(2, self.base_hw).contiguous())
                for proj, (feat, _) in zip(self.model.encoder.output_projections, features))

        features_with_uv = [torch.cat([x, self.uv_grids[0]], dim=1)] + self.uv_grids[1:5]
        features_with_uv = self.model.neck(features_with_uv)
        points = self.model.points_head(features_with_uv)[-1].permute(0, 2, 3, 1)
        points = self.model._remap_points(points)

        cls_token = features[-1][1]
        metric_scale = self.model.scale_head(cls_token).exp()
        shift = self.recover_focal_shift(points, focal=focal)

        depth = ((points[..., 2] + shift) * metric_scale).unsqueeze(0)

        if list(depth.shape[-2:]) != self.output_image_size:
            depth = torch.nn.functional.interpolate(depth, self.output_image_size, mode="bilinear", align_corners=False)

        if OUTPUT_BEV:
            # Extract ROI Depth
            depth_roi = depth[..., self.h_start:self.h_end, self.w_start:self.w_end]

            # Calculate Height Proxy (H = Z * v)
            # Ground plane (flat) has approx constant H. Obstacles have varying H.
            # This removes the "gradient due to perspective" issue on the road.
            height_map_roi = depth_roi * self.uv_roi_v

            # Compute Gradients on HEIGHT, not DEPTH
            dx = torch.nn.functional.conv2d(height_map_roi, self.sobel_x, padding=0)
            dy = torch.nn.functional.conv2d(height_map_roi, self.sobel_y, padding=0)
            gradient_map = dx ** 2 + dy ** 2

            gradient_map = torch.cat([self.zeros_w, gradient_map, self.zeros_w], dim=-2)
            if self.zeros_h_minus_1.shape[-2] == gradient_map.shape[-2]:
                gradient_map = torch.cat([self.zeros_h_minus_1, gradient_map, self.zeros_h_minus_1], dim=-1)
            elif self.zeros_h_plus_1.shape[-2] == gradient_map.shape[-2]:
                gradient_map = torch.cat([self.zeros_h_plus_1, gradient_map, self.zeros_h_plus_1], dim=-1)
            else:
                gradient_map = torch.cat([self.zeros_h, gradient_map, self.zeros_h], dim=-1)

            # Flatten
            depth_roi_flat = depth_roi.reshape(-1)
            grad_roi_flat = gradient_map.reshape(-1)

            # Create Mask based on Height Gradient
            mask_flat = (grad_roi_flat > threshold).to(torch.uint8)

            # BEV Projection (Using U coordinate and Depth)
            x_idx = (self.uv_roi_u_flat * depth_roi_flat * self.bev_scale_x + self.bev_offset_x).int()
            z_idx = (depth_roi_flat * self.bev_scale_z).int()

            # Scatter
            linear_idx = z_idx * self.bev_w + x_idx
            self.bev_flat_buffer.scatter_add_(0, linear_idx, mask_flat)

            # Output, No need to reshape back for ONNX Runtime C-API.
            # bev_map = self.bev_flat_buffer.view(self.bev_h, self.bev_w)

            return depth.squeeze(), self.bev_flat_buffer

        return depth.squeeze()


# ==============================================================================
# EXPORT TO ONNX
# ==============================================================================
def export_model_to_onnx():
    print("Loading model for export...")
    model = MoGeModel.from_pretrained(MODEL_PATH).to('cpu').eval().float()
    model = MoGeV2(model, INPUT_IMAGE_SIZE, NUM_TOKENS,
                   sobel_kernel_size=SOBEL_KERNEL_SIZE,
                   bev_roi_start_ratio=BEV_ROI_START_RATIO)

    image = torch.ones([1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]], dtype=torch.uint8)
    threshold_tensor = torch.tensor([DEFAULT_GRAD_THRESHOLD], dtype=torch.float32)

    print(f'Exporting to ONNX (Opset {OP_SET})...')
    with torch.inference_mode():
        inputs = (image, threshold_tensor) if not FOCAL else (image, threshold_tensor, torch.tensor([FOCAL]))
        input_names = ['image', 'threshold'] if not FOCAL else ['image', 'threshold', 'focal']
        output_names = ['depth_map', 'bev_map'] if OUTPUT_BEV else ['depth_map']

        torch.onnx.export(
            model, inputs, ONNX_MODEL_PATH,
            input_names=input_names, output_names=output_names,
            do_constant_folding=True, opset_version=OP_SET, dynamo=False
        )
    print('✅ Export complete.')


# ==============================================================================
# ONNX INFERENCE & VISUALIZATION
# ==============================================================================
def run_onnx_inference():
    session_opts = onnxruntime.SessionOptions()
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, sess_options=session_opts, providers=['CPUExecutionProvider'])

    raw_img = cv2.imread(TEST_IMAGE_PATH)
    if raw_img is None:
        sys.exit("Error reading image")
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, (INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
    image_tensor = np.expand_dims(resized_img, axis=0).transpose(0, 3, 1, 2)

    input_feed = {ort_session.get_inputs()[0].name: image_tensor}
    if len(ort_session.get_inputs()) > 1:
        input_feed['threshold'] = np.array([DEFAULT_GRAD_THRESHOLD], dtype=np.float32)

    print("Running inference...")
    
    # --- START TIMING ---
    start_time = time.time()
    
    results = ort_session.run(None, input_feed)
    
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000
    print(f"⏱️ Inference time: {inference_time:.2f} ms")
    # --- END TIMING ---

    return resized_img, results[0], results[1] if len(results) > 1 else None


# ==============================================================================
# VISUALIZATION (Restored to Original Quality)
# ==============================================================================
def visualize_results(input_image, depth_map, bev_map):
    """Display input, depth, and BEV maps with consistent sizing and professional formatting."""
    print("\n📊 Generating visualization...")

    roi_start_pixel = int(INPUT_IMAGE_SIZE[0] * BEV_ROI_START_RATIO)

    # Create figure with fixed aspect ratio subplots
    fig = plt.figure(figsize=(20, 7))

    # 1. Input Image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(input_image)
    # Draw the ROI line
    ax1.axhline(y=roi_start_pixel, color='r', linestyle='--', linewidth=2)
    # Add text label for the ignored region
    ax1.text(INPUT_IMAGE_SIZE[1] // 2, roi_start_pixel // 2,
             'BEV IGNORED REGION\n',
             color='red', fontweight='bold', fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.set_title(f'Input Image\n({INPUT_IMAGE_SIZE[1]}x{INPUT_IMAGE_SIZE[0]})', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Depth Map
    ax2 = plt.subplot(1, 3, 2)
    im = ax2.imshow(depth_map, cmap='turbo')
    ax2.axhline(y=roi_start_pixel, color='white', linestyle='--', linewidth=2)
    ax2.set_title(f'Depth Map\n(ROI starts at y={roi_start_pixel})', fontsize=12, fontweight='bold')
    # Add a nice colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (meters)', rotation=270, labelpad=15)
    ax2.axis('off')

    # 3. BEV Occupancy
    if bev_map is not None:
        ax3 = plt.subplot(1, 3, 3)
        # origin='lower' ensures 0m is at the bottom
        ax3.imshow((bev_map * 255).reshape(INPUT_IMAGE_SIZE), cmap='Greys', extent=[-BEV_WIDTH_METERS / 2, BEV_WIDTH_METERS / 2, 0, BEV_DEPTH_METERS], origin='lower')
        ax3.set_title("BEV Occupancy")

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


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # 1. Export model
    export_model_to_onnx()

    # 2. Run inference
    input_img, depth, bev = run_onnx_inference()

    # 3. Visualize results
    visualize_results(input_img, depth, bev)
    
