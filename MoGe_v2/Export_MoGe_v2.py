import sys
import cv2
import torch
import time
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt


project_path = '/home/DakeQQ/Downloads/MoGe-main'                     # The project folder path. https://github.com/microsoft/MoGe
model_path = '/home/DakeQQ/Downloads/moge-2-vits-normal/model.pt'     # The target depth model. Only support the v2 series.
onnx_model_A = '/home/DakeQQ/Downloads/MoGe_ONNX/MoGe_v2.onnx'        # The exported onnx model path.
test_image_path = './test.jpg'                                        # The test input after the export process.


INPUT_IMAGE_SIZE = [720, 1280]          # Input image shape [Height, Width]. Should be a multiple of GPU group (e.g., 16) for optimal efficiency.
NUM_TOKENS = 3600                       # Larger is finer but slower.
FOCAL = None                            # Set None for auto else fixed. FOCAL here is the focal length relative to half the image diagonal.
EDGE_DETECTION = False                  # True for convert the depth map into edge map.
SOBEL_KERNEL_SIZE = 3                   # Sobel kernel size: 3, 5, or 7. Larger kernels are less sensitive to noise but may miss fine details.

"""
fov_x: The horizontal camera FoV in degrees. Smartphone ultra-wide: ~110-120°

aspect_ratio = INPUT_IMAGE_SIZE[1] / INPUT_IMAGE_SIZE[0]

FOCAL = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(fov_x * 0.5))

        Camera
          📷
         /│\
        / │ \
       /  │  \
      /   │   \
     /    │    \
    /     │     \
   /      │      \
  /       │       \
 /_______FoV_______\
    Scene width
"""


if project_path not in sys.path:
    sys.path.append(project_path)

from moge.model.v2 import MoGeModel


def normalized_view_plane_uv(width, height, aspect_ratio) -> torch.Tensor:
    # Simplify normalization computation
    denom = (1 + aspect_ratio ** 2) ** 0.5

    # Factor out common (size - 1) / size computation
    scale_factor = lambda size: (size - 1) / size
    span_x = aspect_ratio / denom * scale_factor(width)
    span_y = scale_factor(height) / denom

    u = torch.linspace(-span_x, span_x, width, dtype=torch.float32)
    v = torch.linspace(-span_y, span_y, height, dtype=torch.float32)

    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
    return torch.stack([u_grid, v_grid], dim=-1)  # (H, W, 2)


def solve_optimal_focal_shift(uv_flat, xy, z, num_iters=15):
    shift = 0.0
    # Gauss-Newton iterations
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
    shift = 0.0
    focal_xy = focal * xy
    # Gauss-Newton iterations
    for _ in range(num_iters):
        z_shifted = z + shift
        focal_xy_z_shifted = focal_xy / z_shifted
        residual = focal_xy_z_shifted - uv_flat
        jacobian = focal_xy_z_shifted / z_shifted

        delta_shift = (-jacobian * residual).sum() / ((jacobian ** 2).sum() + 1e-6)
        shift -= delta_shift
    return shift


def get_sobel_kernels(kernel_size):
    """
    Generate Sobel kernels of specified size (3, 5, or 7).
    Returns (sobel_x, sobel_y) as PyTorch tensors.
    """
    if kernel_size == 3:
        # Standard 3x3 Sobel kernels
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
    elif kernel_size == 5:
        # 5x5 Sobel kernels (smoother, less noise-sensitive)
        sobel_x = torch.tensor([
            [-1, -2,  0,  2,  1],
            [-2, -3,  0,  3,  2],
            [-3, -5,  0,  5,  3],
            [-2, -3,  0,  3,  2],
            [-1, -2,  0,  2,  1]
        ], dtype=torch.float32).view(1, 1, 5, 5)
        
        sobel_y = torch.tensor([
            [-1, -2, -3, -2, -1],
            [-2, -3, -5, -3, -2],
            [ 0,  0,  0,  0,  0],
            [ 2,  3,  5,  3,  2],
            [ 1,  2,  3,  2,  1]
        ], dtype=torch.float32).view(1, 1, 5, 5)
        
    elif kernel_size == 7:
        # 7x7 Sobel kernels (even smoother, good for noisy images)
        sobel_x = torch.tensor([
            [-1, -2, -3,  0,  3,  2,  1],
            [-2, -3, -5,  0,  5,  3,  2],
            [-3, -5, -7,  0,  7,  5,  3],
            [-4, -6, -8,  0,  8,  6,  4],
            [-3, -5, -7,  0,  7,  5,  3],
            [-2, -3, -5,  0,  5,  3,  2],
            [-1, -2, -3,  0,  3,  2,  1]
        ], dtype=torch.float32).view(1, 1, 7, 7)
        
        sobel_y = torch.tensor([
            [-1, -2, -3, -4, -3, -2, -1],
            [-2, -3, -5, -6, -5, -3, -2],
            [-3, -5, -7, -8, -7, -5, -3],
            [ 0,  0,  0,  0,  0,  0,  0],
            [ 3,  5,  7,  8,  7,  5,  3],
            [ 2,  3,  5,  6,  5,  3,  2],
            [ 1,  2,  3,  4,  3,  2,  1]
        ], dtype=torch.float32).view(1, 1, 7, 7)
    else:
        raise ValueError(f"Unsupported kernel size: {kernel_size}. Choose 3, 5, or 7.")
    
    return sobel_x, sobel_y


class MoGeV2(torch.nn.Module):
    def __init__(self, model, input_image_size, num_tokens, sobel_kernel_size=3):
        super(MoGeV2, self).__init__()
        self.model = model
        self.input_image_size = input_image_size
        self.aspect_ratio = input_image_size[1] / input_image_size[0]
        self.sobel_kernel_size = sobel_kernel_size

        # Compute base dimensions
        sqrt_tokens_aspect = num_tokens ** 0.5
        self.base_h = int(sqrt_tokens_aspect / self.aspect_ratio ** 0.5)
        self.base_w = int(sqrt_tokens_aspect * self.aspect_ratio ** 0.5)
        self.base_hw = (self.base_h, self.base_w)

        # Precompute image processing constants
        patch_size = self.model.encoder.backbone.patch_embed.patch_size
        self.image_resize = [self.base_h * patch_size[0], self.base_w * patch_size[1]]
        self.inv_image_std_255 = 1.0 / (255.0 * self.model.encoder.image_std)
        self.image_mean_inv_std = self.model.encoder.image_mean / self.model.encoder.image_std

        # Precompute positional embeddings
        pos_embed = self.model.encoder.backbone.pos_embed
        class_pos_embed = pos_embed[:, 0:1, :]
        patch_pos_embed = pos_embed[:, 1:, :]

        M = int((pos_embed.shape[1] - 1) ** 0.5)
        interpolate_offset = self.model.encoder.backbone.interpolate_offset
        scale_factor = (
            (self.base_h + interpolate_offset) / M,
            (self.base_w + interpolate_offset) / M
        )

        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, -1).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=False,
            align_corners=False,
            scale_factor=scale_factor
        ).permute(0, 2, 3, 1).flatten(1, 2)

        self.pos_embed = torch.cat([class_pos_embed, patch_pos_embed], dim=1)
        self.model.encoder.backbone.num_register_tokens += 1

        # Precompute UV grids
        self.save_uv = []
        for level in range(5):
            scale = 2 ** level
            uv = normalized_view_plane_uv(
                self.base_w * scale,
                self.base_h * scale,
                self.aspect_ratio
            )
            self.save_uv.append(uv.permute(2, 0, 1).unsqueeze(0))

        # Low-resolution UV for optimization
        uv = normalized_view_plane_uv(self.base_w, self.base_h, self.aspect_ratio)
        uv_lr = torch.nn.functional.interpolate(
            uv.permute(2, 0, 1).unsqueeze(0),
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        )
        self.save_uv.append(uv_lr.permute(0, 2, 3, 1).reshape(-1, 2))

        # Initialize Sobel kernels with specified size
        self.sobel_x, self.sobel_y = get_sobel_kernels(sobel_kernel_size)
        self.sobel_padding = sobel_kernel_size // 2

    def recover_focal_shift(self, points, focal=None, downsample_size=(64, 64)):
        points_lr = torch.nn.functional.interpolate(
            points.permute(0, 3, 1, 2),
            size=downsample_size,
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1).reshape(-1, 3)

        xy, z = torch.split(points_lr, [2, 1], dim=-1)

        if focal:
            return solve_optimal_shift(self.save_uv[-1], xy, z, focal)
        else:
            return solve_optimal_focal_shift(self.save_uv[-1], xy, z)

    def forward(self, image, focal=None):
        # Preprocess image
        image = image.float()
        if self.image_resize != INPUT_IMAGE_SIZE:
            image = torch.nn.functional.interpolate(
                image,
                self.image_resize,
                mode="bilinear",
                align_corners=False,
                antialias=False
            )
        x = image * self.inv_image_std_255 - self.image_mean_inv_std

        # Patch embedding
        x = self.model.encoder.backbone.patch_embed.proj(x)
        x = self.model.encoder.backbone.patch_embed.norm(x.flatten(2).transpose(1, 2))
        x = torch.cat([self.model.encoder.backbone.cls_token, x], dim=1) + self.pos_embed.data

        # Process through blocks and collect intermediate outputs
        outputs = []
        for i, blk in enumerate(self.model.encoder.backbone.blocks):
            x = blk(x)
            if i in self.model.encoder.intermediate_layers:
                outputs.append(x)

        # Normalize and separate class tokens
        outputs = [self.model.encoder.backbone.norm(out) for out in outputs]
        num_reg_tokens = self.model.encoder.backbone.num_register_tokens
        features = [(out[:, num_reg_tokens:], out[:, 0]) for out in outputs]

        # Project and combine features
        x = sum(
            proj(feat.permute(0, 2, 1).unflatten(2, self.base_hw).contiguous())
            for proj, (feat, _) in zip(self.model.encoder.output_projections, features)
        )

        # Concatenate UV grids with features
        features_with_uv = [
            torch.cat([x, self.save_uv[0]], dim=1),
            self.save_uv[1],
            self.save_uv[2],
            self.save_uv[3],
            self.save_uv[4]
        ]

        # Decode to points
        features_with_uv = self.model.neck(features_with_uv)
        points = self.model.points_head(features_with_uv)[-1].permute(0, 2, 3, 1)
        points = self.model._remap_points(points)

        # Compute metric depth
        cls_token = features[-1][1]
        metric_scale = self.model.scale_head(cls_token).exp()
        shift = self.recover_focal_shift(points, focal=focal)

        depth = (points[..., 2] + shift) * metric_scale
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0),
            self.input_image_size,
            mode='bilinear',
            align_corners=False,
            antialias=False
        )

        if EDGE_DETECTION:
            # Apply Sobel filters with the specified kernel size
            dx = torch.nn.functional.conv2d(depth, self.sobel_x, padding=self.sobel_padding)
            dy = torch.nn.functional.conv2d(depth, self.sobel_y, padding=self.sobel_padding)
            gradient_map = torch.pow(dx ** 2 + dy ** 2, 0.4)  # Use 0.4 for amplify the small gradient values.
            min_val, max_val = torch.aminmax(gradient_map)    
            gradient_map = (gradient_map - min_val) / (max_val - min_val)  # Normalize to 0 ~ 1
            return gradient_map

        return depth


model = MoGeModel.from_pretrained(model_path).to('cpu').eval().float()
model = MoGeV2(model, INPUT_IMAGE_SIZE, NUM_TOKENS, sobel_kernel_size=SOBEL_KERNEL_SIZE)
image = torch.ones([1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]], dtype=torch.uint8)

print(f'\nUsing Sobel kernel size: {SOBEL_KERNEL_SIZE}x{SOBEL_KERNEL_SIZE}')
print('\nExport Start.')
with torch.inference_mode():
    if FOCAL:
        focal = torch.tensor([FOCAL], dtype=torch.float32)
        in_feed = (image, focal)
        input_names = ['image', 'focal']
    else:
        in_feed = (image,)
        input_names = ['image']

    torch.onnx.export(
            model,
            in_feed,
            onnx_model_A,
            input_names=input_names,
            output_names=['gradient_map'] if EDGE_DETECTION else ['depth_in_meters'],
            do_constant_folding=True,
            opset_version=17,
            dynamo=False
        )
    del model
    del in_feed
    del input_names
    del image

    if project_path in sys.path:
        sys.path.remove(project_path)

    print('\nExport Done.')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4              # Error level, it an adjustable value.
session_opts.inter_op_num_threads = 0            # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0            # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True         # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A = [i.name for i in in_name_A]
out_name_A = [out_name_A[0].name]


# Load the raw image using OpenCV
raw_img = cv2.imread(test_image_path)
if raw_img is None:
    print(f"Error: Could not read image at {test_image_path}")
    sys.exit()

# The model expects RGB, but OpenCV loads images in BGR format, so we convert.
rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

# Resize the image to the exact size the model expects.
# cv2.resize expects (width, height).
resized_img = cv2.resize(rgb_img, (ort_session_A._inputs_meta[0].shape[-1], ort_session_A._inputs_meta[0].shape[-2]))

# The model's ONNX graph expects a uint8 tensor with the shape (N, C, H, W).
# 1. Add a batch dimension: (H, W, C) -> (1, H, W, C)
# 2. Change layout to (1, C, H, W)
image_tensor = np.expand_dims(resized_img, axis=0).transpose(0, 3, 1, 2)

print(f"\nInput tensor prepared with shape: {image_tensor.shape} and dtype: {image_tensor.dtype}")

# --- Run Inference ---
input_feed_A = {
    in_name_A[0]: onnxruntime.OrtValue.ortvalue_from_numpy(image_tensor, 'cpu', 0)
}

if len(in_name_A) > 1:
    model_A_dtype = ort_session_A._inputs_meta[1].type
    if 'float16' in model_A_dtype:
        model_A_dtype = np.float16
    else:
        model_A_dtype = np.float32
    input_feed_A[in_name_A[1]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([FOCAL], dtype=model_A_dtype), 'cpu', 0)

print(f"\nRunning inference on ONNX model...")
start = time.time()
onnx_result = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)
print(f'\nTime cost: {time.time() - start:.3f} seconds')

# The output is a list containing one array. We extract the array.
depth_map_onnx = onnx_result[0].numpy()

# The output depth map has a shape of (1, H, W), so we remove the batch dimension.
depth_map_onnx = np.squeeze(depth_map_onnx)
print(f"\n✅ ONNX inference complete! Output depth map shape: {depth_map_onnx.shape}")


# =================================================================================
# 3. VISUALIZE THE RESULT (Method 2: Matplotlib)
# =================================================================================

print("\nVisualizing results using Matplotlib...")

plt.figure(figsize=(14, 7))

# --- Display Original Image ---
plt.subplot(1, 2, 1)
# Use the RGB image we converted earlier for correct color display in Matplotlib
plt.imshow(rgb_img)
plt.title('Original Image')
plt.axis('off')

# --- Display Depth Heatmap from ONNX Inference ---
plt.subplot(1, 2, 2)
# imshow can directly handle the floating-point depth array.
plt.imshow(depth_map_onnx, cmap='turbo')
title_suffix = f' ({SOBEL_KERNEL_SIZE}x{SOBEL_KERNEL_SIZE} Sobel)' if EDGE_DETECTION else ''
plt.title(f'{"Edge Heatmap" if EDGE_DETECTION else "Depth Heatmap"} (from ONNX model){title_suffix}')
# Add a color bar to show the mapping of colors to depth values.
plt.colorbar(label='Normalized Gradient' if EDGE_DETECTION else 'Depth Metric')
plt.axis('off')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

print("\n✅ Visualization complete.")
