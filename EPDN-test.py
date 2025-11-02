# #!/usr/bin/env python3
# """
# inference_epdn.py
#
# 遍历 noisy 文件夹中的相位图进行去噪，并与 clean 文件夹中的对应文件进行 PSNR 评估。
# 去噪结果保存到 denoisy 文件夹，文件名与输入对应。
# 适配 EPDN_NEW_UNET_DATASET.py 的 Generator 模型。
# """
# import argparse
# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.metrics import peak_signal_noise_ratio as ski_psnr
# from scipy.ndimage import zoom
# # 导入与训练一致的 Generator
# from EPDN30000 import Generator
# # 解决 matplotlib 负号和中文显示问题
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示（如 Windows）
#
# def denoise_and_evaluate_epdn(noisy_folder: str,
#                               clean_folder: str,
#                               denoisy_folder: str,
#                               model_path: str):
#     """
#     遍历 noisy 文件夹中的相位图进行去噪，与 clean 文件夹中的对应文件进行 PSNR 评估，
#     并将去噪结果保存到 denoisy 文件夹。
#
#     Args:
#         noisy_folder: 包含噪声相位图的文件夹路径 (noisy0.npy - noisy29.npy)
#         clean_folder: 包含干净相位图的文件夹路径 (clean0.npy - clean29.npy)
#         denoisy_folder: 保存去噪结果的文件夹路径 (denoisy0.npy - denoisy29.npy)
#         model_path: 训练好的 Generator 模型 .pth 权重文件路径
#     """
#     # 1) 设备设置
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"使用设备: {device}")
#
#     # 2) 加载模型
#     if not os.path.isfile(model_path):
#         raise FileNotFoundError(f"未找到模型文件: {model_path}")
#     model = Generator(base_ch=128).to(device)  # 与训练代码一致，base_ch=128
#     try:
#         state_dict = torch.load(model_path, map_location=device, weights_only=True)
#         model.load_state_dict(state_dict)
#     except Exception as e:
#         raise RuntimeError(f"加载模型权重失败: {str(e)}")
#     model.eval()
#     print(f"已加载模型: {model_path}")
#
#     # 3) 确保输出文件夹存在
#     denoisy_folder = os.path.normpath(denoisy_folder)  # 规范化路径
#     os.makedirs(denoisy_folder, exist_ok=True)
#
#     # 4) 遍历 noisy0.npy 到 noisy29.npy
#     total_psnr = 0.0
#     valid_files = 0
#     for i in range(30):
#         noisy_file = os.path.join(os.path.normpath(noisy_folder), f'noisy{i}.npy')
#         clean_file = os.path.join(os.path.normpath(clean_folder), f'clean{i}.npy')
#         denoisy_file = os.path.join(denoisy_folder, f'denoisy{i}.npy')
#
#         # 检查 noisy 文件是否存在
#         if not os.path.isfile(noisy_file):
#             print(f"警告: 未找到 {noisy_file}，跳过处理。")
#             continue
#
#         # 加载噪声相位
#         try:
#             noisy_phase = np.load(noisy_file)
#             if noisy_phase.ndim != 2:
#                 raise ValueError(f"噪声相位图 {noisy_file} 必须为2D数组，但得到 ndim={noisy_phase.ndim}")
#             print(f"加载噪声相位: {noisy_file}, shape={noisy_phase.shape}")
#         except Exception as e:
#             print(f"错误: 加载 {noisy_file} 失败: {str(e)}")
#             continue
#
#         # 调整输入尺寸以匹配训练 (256x256)
#         h, w = noisy_phase.shape
#         target_size = 256
#         if h != target_size or w != target_size:
#             noisy_phase = zoom(noisy_phase, (target_size / h, target_size / w), order=1)
#
#         # 构建双通道输入 (cos, sin)
#         noisy_cos = np.cos(noisy_phase)
#         noisy_sin = np.sin(noisy_phase)
#         inp = np.stack([noisy_cos, noisy_sin], axis=0).astype(np.float32)
#         inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(device)  # [1, 2, 256, 256]
#
#         # 推理
#         with torch.no_grad():
#             out = model(inp_tensor)  # [1, 2, 256, 256]
#         cos_out, sin_out = out[0, 0], out[0, 1]
#         denoisy = torch.atan2(sin_out, cos_out)
#         # wrap 到 [-π, π]
#         denoisy = (denoisy + np.pi) % (2 * np.pi) - np.pi
#         denoisy_np = denoisy.cpu().numpy()
#
#         # 保存去噪结果
#         try:
#             np.save(denoisy_file, denoisy_np)
#             print(f"去噪结果已保存: {denoisy_file}")
#         except Exception as e:
#             print(f"错误: 保存 {denoisy_file} 失败: {str(e)}")
#             continue
#
#         # PSNR 评估（如果 clean 文件存在）
#         clean_np = None
#         if os.path.isfile(clean_file):
#             try:
#                 clean_np = np.load(clean_file)
#                 if clean_np.ndim != 2:
#                     raise ValueError(f"干净相位图 {clean_file} 必须为2D数组，但得到 ndim={clean_np.ndim}")
#                 # 调整干净相位图尺寸
#                 if clean_np.shape != (target_size, target_size):
#                     clean_np = zoom(clean_np, (target_size / h, target_size / w), order=1)
#                 psnr_val = ski_psnr(clean_np.astype(np.float32),
#                                     denoisy_np.astype(np.float32),
#                                     data_range=2 * np.pi)
#                 print(f"PSNR for {denoisy_file}: {psnr_val:.2f} dB")
#                 total_psnr += psnr_val
#                 valid_files += 1
#             except Exception as e:
#                 print(f"错误: 加载或处理 {clean_file} 失败: {str(e)}")
#         else:
#             print(f"警告: 未找到 {clean_file}，跳过 PSNR 计算。")
#
#         # 可视化对比
#         cols = 2 + (1 if clean_np is not None else 0)
#         fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))
#         axes = axes.flatten()
#
#         im0 = axes[0].imshow(noisy_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
#         axes[0].set_title(f'噪声相位图 {i}')
#         fig.colorbar(im0, ax=axes[0], fraction=0.046)
#
#         im1 = axes[1].imshow(denoisy_np, cmap='twilight', vmin=-np.pi, vmax=np.pi)
#         axes[1].set_title(f'去噪相位图 {i}')
#         fig.colorbar(im1, ax=axes[1], fraction=0.046)
#
#         if clean_np is not None:
#             im2 = axes[2].imshow(clean_np, cmap='twilight', vmin=-np.pi, vmax=np.pi)
#             axes[2].set_title(f'干净相位图 {i}')
#             fig.colorbar(im2, ax=axes[2], fraction=0.046)
#
#         plt.tight_layout()
#         plt.show()
#
#         # 清理 GPU 内存
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#     # 输出平均 PSNR
#     if valid_files > 0:
#         avg_psnr = total_psnr / valid_files
#         print(f"平均 PSNR: {avg_psnr:.2f} dB ({valid_files} 文件)")
#     else:
#         print("未计算 PSNR（无有效 clean 文件）")
#
# if __name__ == '__main__':
#     # 直接调用示例（可通过命令行修改）
#     denoise_and_evaluate_epdn(
#         noisy_folder='noisy',
#         clean_folder='clean',
#         denoisy_folder='denoisy1',
#         model_path='best_generator30000.pth'
#     )

# !/usr/bin/env python3
"""
process_single_phase_image_epdn.py

直接对指定的单张含噪相位图进行去噪，并与对应的干净相位图进行 PSNR 评估。
去噪结果将保存到指定的输出文件路径。同时在控制台输出 PSNR 值并进行可视化。
适配 EPDN30000.py 的 Generator 模型。
"""
# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.metrics import peak_signal_noise_ratio as ski_psnr
# from scipy.ndimage import zoom
#
# # 导入与训练一致的 Generator
# # 确保 EPDN30000.py 在同一目录下或 Python 路径中
# from EPDN30000 import Generator
#
# # 解决 matplotlib 负号和中文显示问题
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示（如 Windows）
#
#
# def denoise_and_evaluate_epdn_direct(noisy_file_path: str,
#                                      clean_file_path: str,
#                                      output_denoisy_file_path: str,  # 新增参数：去噪结果保存路径
#                                      model_path: str):
#     """
#     直接对指定的单张含噪相位图进行去噪，与对应的干净相位图进行 PSNR 评估，
#     并将去噪结果保存到指定路径。同时在控制台输出 PSNR 并进行可视化。
#     适配 EPDN Generator 模型，输入为 2 通道 (cos/sin)。
#
#     Args:
#         noisy_file_path: 含噪相位图的完整路径 (e.g., 'path/to/your/noisy_image.npy')
#         clean_file_path: 干净相位图的完整路径 (e.g., 'path/to/your/clean_image.npy')
#         output_denoisy_file_path: 去噪结果保存的完整路径 (e.g., 'path/to/save/denoised_image.npy')
#         model_path: 训练好的 Generator 模型 .pth 权重文件路径
#     """
#     # 1) 设备设置
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"使用设备: {device}")
#
#     # 2) 加载模型
#     if not os.path.isfile(model_path):
#         raise FileNotFoundError(f"未找到模型文件: {model_path}")
#     model = Generator(base_ch=128).to(device)  # 与训练代码一致，base_ch=128
#     try:
#         state_dict = torch.load(model_path, map_location=device, weights_only=True)
#         model.load_state_dict(state_dict)
#     except Exception as e:
#         raise RuntimeError(f"加载模型权重失败: {str(e)}")
#     model.eval()
#     print(f"已加载模型: {model_path}")
#
#     # 3) 加载噪声相位
#     if not os.path.isfile(noisy_file_path):
#         raise FileNotFoundError(f"未找到含噪文件: {noisy_file_path}")
#     try:
#         noisy_phase = np.load(noisy_file_path)
#         if noisy_phase.ndim != 2:
#             raise ValueError(f"噪声相位图 {noisy_file_path} 必须为2D数组，但得到 ndim={noisy_phase.ndim}")
#         print(f"加载含噪相位: {noisy_file_path}, shape={noisy_phase.shape}")
#     except Exception as e:
#         raise RuntimeError(f"错误: 加载 {noisy_file_path} 失败: {str(e)}")
#
#     # 调整输入尺寸以匹配训练 (256x256)
#     original_h, original_w = noisy_phase.shape
#     target_size = 256
#     if original_h != target_size or original_w != target_size:
#         print(f"警告: 输入图像尺寸 ({original_h}x{original_w}) 与训练尺寸 ({target_size}x{target_size}) 不符，正在进行缩放...")
#         noisy_phase_resized = zoom(noisy_phase, (target_size / original_h, target_size / original_w), order=1)
#     else:
#         noisy_phase_resized = noisy_phase
#
#     # 构建双通道输入 (cos, sin)
#     noisy_cos = np.cos(noisy_phase_resized)
#     noisy_sin = np.sin(noisy_phase_resized)
#     inp = np.stack([noisy_cos, noisy_sin], axis=0).astype(np.float32)
#     inp_tensor = torch.from_numpy(inp).unsqueeze(0).to(device)  # [1, 2, 256, 256]
#
#     # 推理
#     with torch.no_grad():
#         out = model(inp_tensor)  # [1, 2, 256, 256]
#     cos_out, sin_out = out[0, 0], out[0, 1]
#     denoised_phase = torch.atan2(sin_out, cos_out)
#
#     # wrap 到 [-π, π]
#     denoised_phase = (denoised_phase + np.pi) % (2 * np.pi) - np.pi
#     denoised_phase_np = denoised_phase.cpu().numpy()
#
#     # 如果原始图像尺寸不同，将去噪结果重新缩放回原始尺寸
#     if original_h != target_size or original_w != target_size:
#         denoised_phase_np = zoom(denoised_phase_np, (original_h / target_size, original_w / target_size), order=1)
#         print(f"去噪结果已缩放回原始尺寸: {denoised_phase_np.shape}")
#
#     # --- 保存去噪结果 ---
#     try:
#         # 确保输出目录存在
#         output_dir = os.path.dirname(output_denoisy_file_path)
#         if output_dir and not os.path.exists(output_dir):
#             os.makedirs(output_dir, exist_ok=True)
#             print(f"已创建输出文件夹: {output_dir}")
#
#         np.save(output_denoisy_file_path, denoised_phase_np)
#         print(f"去噪结果已保存到: {output_denoisy_file_path}")
#     except Exception as e:
#         print(f"错误: 保存去噪结果到 {output_denoisy_file_path} 失败: {str(e)}")
#
#     # PSNR 评估
#     clean_phase_np = None
#     psnr_val = None
#     if os.path.isfile(clean_file_path):
#         try:
#             clean_phase_np = np.load(clean_file_path)
#             if clean_phase_np.ndim != 2:
#                 raise ValueError(f"干净相位图 {clean_file_path} 必须为2D数组，但得到 ndim={clean_file_path.ndim}")
#
#             # 确保干净相位图也调整到原始尺寸进行PSNR计算
#             if clean_phase_np.shape != (original_h, original_w):
#                 print(f"警告: 干净图像尺寸 ({clean_phase_np.shape}) 与原始含噪图像尺寸 ({original_h}x{original_w}) 不符。")
#                 print("PSNR 计算将基于去噪结果缩放回原始尺寸后进行。")
#
#             psnr_val = ski_psnr(clean_phase_np.astype(np.float32),
#                                 denoised_phase_np.astype(np.float32),
#                                 data_range=2 * np.pi)  # 相位范围通常是 2*pi
#             print(f"\n处理文件: {os.path.basename(noisy_file_path)}")
#             print(f"PSNR: {psnr_val:.2f} dB")
#         except Exception as e:
#             print(f"错误: 加载或处理 {clean_file_path} 失败: {str(e)}")
#     else:
#         print(f"警告: 未找到干净文件: {clean_file_path}，跳过 PSNR 计算。")
#
#     # 可视化对比
#     cols = 2 + (1 if clean_phase_np is not None else 0)
#     fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))
#     axes = axes.flatten()
#
#     im0 = axes[0].imshow(noisy_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
#     axes[0].set_title(f'原始含噪相位图\n({os.path.basename(noisy_file_path)})')
#     fig.colorbar(im0, ax=axes[0], fraction=0.046)
#
#     im1 = axes[1].imshow(denoised_phase_np, cmap='twilight', vmin=-np.pi, vmax=np.pi)
#     title_text = '去噪相位图'
#     if psnr_val is not None:
#         title_text += f'\n(PSNR: {psnr_val:.2f} dB)'
#     axes[1].set_title(title_text)
#     fig.colorbar(im1, ax=axes[1], fraction=0.046)
#
#     if clean_phase_np is not None:
#         im2 = axes[2].imshow(clean_phase_np, cmap='twilight', vmin=-np.pi, vmax=np.pi)
#         axes[2].set_title(f'干净相位图\n({os.path.basename(clean_file_path)})')
#         fig.colorbar(im2, ax=axes[2], fraction=0.046)
#
#     plt.tight_layout()
#     plt.show()
#
#     # 清理 GPU 内存
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#
#
# if __name__ == '__main__':
#     # >>> 请在这里修改为你实际的文件路径 <<<
#     # 示例路径 - 你需要根据你的实际文件位置进行调整。
#     # 例如，如果你的含噪和干净文件在 'data' 文件夹下
#     noisy_image_path = 'noisy13.npy'  # 例如，如果 noisy13.npy 在当前目录下的 'noisy' 文件夹里
#     clean_image_path = 'clean13.npy'  # 例如，如果 clean13.npy 在当前目录下的 'clean' 文件夹里
#
#     # 去噪结果将保存到的路径和文件名。你可以指定一个不同的文件夹，例如 'output_epdn_images/'
#     # 脚本会自动创建指定的输出文件夹（如果它不存在）
#     output_denoised_path = 'CGAN13.npy'
#
#     # 你的模型路径，同样需要根据实际情况修改
#     model_path = 'best_generator30000.pth'
#
#     # 调用函数处理指定的单张数据
#     try:
#         denoise_and_evaluate_epdn_direct(
#             noisy_file_path=noisy_image_path,
#             clean_file_path=clean_image_path,
#             output_denoisy_file_path=output_denoised_path,
#             model_path=model_path
#         )
#     except Exception as e:
#         print(f"执行去噪和评估时发生错误: {e}")


# !/usr/bin/env python3


# !/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3


# !/usr/bin/env python3
#


"""
inference_nlm.py

对大尺寸相位图进行分块去噪处理，采用重叠分块和加权融合避免拼接痕迹。
去噪结果保存为指定文件，不依赖干净图像文件进行PSNR评估。
适配 EPDN30000.py 的 Generator 模型，输入为 2 通道（cos/sin）以匹配提供的模型权重。
【最终统一修正】：采用基于 overlap/2 的填充量，使用高斯窗融合，精确裁剪。
【关键修复】：强制裁剪模型的输出到 256x256，以解决 320x320 和 256x256 的广播冲突。
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom  # 虽然不需要，但保留导入
from skimage.restoration import denoise_nl_means # 虽然不需要，但保留导入
from tqdm import tqdm
from scipy.signal.windows import gaussian  # 引入高斯窗函数

# 导入与训练一致的 Generator
# 确保 EPDN30000.py 在同一目录下或 Python 路径中
from EPDN30000 import Generator

# 解决 matplotlib 负号和中文显示问题
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']


def denoise_large_image_patchwise(
        noisy_image_path: str,
        model_path: str,
        output_denoised_path: str,  # 去噪后文件保存路径
        patch_out_size: int = 256,  # 模型每次有效输出的中心块大小 (O=256)
        input_context: int = 32,  # 为模型提供额外上下文的边界大小 (C=32)
        overlap: int = 64  # 输出块之间的重叠大小 (Overlap)
):
    """
    对大尺寸相位图进行分块去噪处理，采用重叠分块和高斯加权融合，并精确裁剪。
    """
    # 1. 设备设置和模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    G = Generator(base_ch=128).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        G.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"加载模型权重失败: {str(e)}")
    G.eval()  # 设置为评估模式
    print(f"已加载模型: {model_path}")

    # 2. 加载原始图像 (numpy.ndarray)
    if not os.path.isfile(noisy_image_path):
        raise FileNotFoundError(f"未找到噪声图像文件: {noisy_image_path}")
    try:
        noisy_phase_np = np.load(noisy_image_path).astype(np.float32)
        if noisy_phase_np.ndim != 2:
            raise ValueError(f"噪声相位图 {noisy_image_path} 必须为2D数组，但得到 ndim={noisy_phase_np.ndim}")
    except Exception as e:
        raise RuntimeError(f"错误: 加载 {noisy_image_path} 失败: {str(e)}")

    original_h, original_w = noisy_phase_np.shape
    print(f"原始图像尺寸: {original_h}x{original_w}")

    # 计算模型实际输入时的块大小 (包含上下文)
    model_input_size = patch_out_size + 2 * input_context  # 320x320
    # 计算每次移动的步长
    stride = patch_out_size - overlap  # 256 - 64 = 192
    if stride <= 0:
        raise ValueError("步长必须大于 0。请调整 patch_out_size 或 overlap。")


    # 3. 对输入图像进行上下文填充
    # 填充量使用 max(input_context, overlap // 2)
    pad_amount = max(input_context, overlap // 2)

    padded_noisy_phase_np = np.pad(noisy_phase_np, pad_amount, mode='reflect')
    padded_h, padded_w = padded_noisy_phase_np.shape
    print(f"边界填充 ({pad_amount}) 后用于分块处理的图像尺寸: {padded_h}x{padded_w}")

    # 初始化用于累加去噪结果的总和图像和权重计数器图像
    denoised_full_image_sum = np.zeros_like(padded_noisy_phase_np, dtype=np.float32)
    denoised_full_image_count = np.zeros_like(padded_noisy_phase_np, dtype=np.float32)

    # 创建一个2D高斯混合权重掩码 (256x256)
    sigma = patch_out_size / 6
    win_1d = gaussian(patch_out_size, std=sigma)
    weight_mask = np.outer(win_1d, win_1d).astype(np.float32)
    weight_mask /= weight_mask.max()

    # 4. 分块处理 (【统一修正分块逻辑】)

    # 计算【原始图像】上有效输出区域的起始点坐标
    h_coords = list(range(0, original_h - patch_out_size + 1, stride))
    w_coords = list(range(0, original_w - patch_out_size + 1, stride))

    # 确保最右边和最底部边缘也被覆盖（贴合边缘）
    if (original_h - patch_out_size) % stride != 0:
        h_coords.append(original_h - patch_out_size)
    if (original_w - patch_out_size) % stride != 0:
        w_coords.append(original_w - patch_out_size)

    h_coords = sorted(list(set(h_coords)))
    w_coords = sorted(list(set(w_coords)))

    num_patches_h = len(h_coords)
    num_patches_w = len(w_coords)

    print(f"即将处理 {num_patches_h}x{num_patches_w} = {num_patches_h * num_patches_w} 个块...")

    for r_idx in tqdm(range(num_patches_h), desc="去噪处理进度"):
        for c_idx in range(num_patches_w):

            # --- 当前有效输出块在【原始图像】上的起始坐标 ---
            original_row_start = h_coords[r_idx]
            original_col_start = w_coords[c_idx]

            # --- 提取模型输入块 (大小为 320x320) ---
            # 提取区域在【填充图像】上的起始点：
            # 提取起始点 = original_start + pad_amount - input_context
            row_start_padded = original_row_start + pad_amount - input_context
            col_start_padded = original_col_start + pad_amount - input_context

            input_patch = padded_noisy_phase_np[
                          row_start_padded: row_start_padded + model_input_size,
                          col_start_padded: col_start_padded + model_input_size
                          ]

            # 核心：移除缩放，模型输入块就是 input_patch (320x320)
            input_patch_for_model = input_patch

            if input_patch_for_model.shape != (model_input_size, model_input_size):
                 print(f"警告: 提取的输入块尺寸为 {input_patch_for_model.shape}，跳过。")
                 continue

            # 归一化输入 [-pi, pi] -> [-1, 1]
            input_tensor_normalized = torch.from_numpy(input_patch_for_model).float().unsqueeze(0).unsqueeze(0).to(
                device) / np.pi

            # 构建 2 通道输入 (cos, sin)
            gen_input = torch.cat((
                torch.cos(input_tensor_normalized),
                torch.sin(input_tensor_normalized),
            ), 1)

            with torch.no_grad():
                denoised_output_raw = G(gen_input)  # 假设输出是 [1, 2, H, W]

            # --------------------- 关键修复：强制裁剪模型输出 ---------------------
            h_out_raw, w_out_raw = denoised_output_raw.shape[-2:]
            denoised_output = denoised_output_raw

            if h_out_raw != patch_out_size or w_out_raw != patch_out_size:
                # 如果模型输出不是 256x256，我们裁剪中心区域
                if h_out_raw >= model_input_size and w_out_raw >= model_input_size:
                    # 假设模型输出是 320x320 或更大，裁剪中心 256x256 区域
                    # 裁剪起始点 = (原始输出尺寸 - 期望输出尺寸) // 2
                    crop_h_start = (h_out_raw - patch_out_size) // 2
                    crop_w_start = (w_out_raw - patch_out_size) // 2
                    crop_h_end = crop_h_start + patch_out_size
                    crop_w_end = crop_w_start + patch_out_size

                    denoised_output = denoised_output_raw[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]

                    if denoised_output.shape[-2:] != (patch_out_size, patch_out_size):
                         raise RuntimeError(f"裁剪失败: 预期 {patch_out_size}x{patch_out_size}，实际 {denoised_output.shape[-2:]}")
                else:
                    raise ValueError(f"模型输出尺寸异常: {denoised_output_raw.shape[-2:]}。期望 256x256 或 320x320。")
            # -------------------------------------------------------------

            # 处理模型输出
            if denoised_output.shape[1] == 2:
                # 假设为 cos 和 sin，使用 atan2重构相位 (此时 denoised_output 保证是 256x256)
                cos_denoised = denoised_output.squeeze(0)[0]
                sin_denoised = denoised_output.squeeze(0)[1]
                denoised_patch_np = torch.atan2(sin_denoised, cos_denoised).cpu().numpy() # 256x256
            else:
                raise ValueError(f"无法处理的模型输出通道数: {denoised_output.shape[1]}。EPDN/CGAN 模型通常输出 2 通道。")

            # 应用权重掩码 (权重掩码是 256x256)
            denoised_patch_blended = denoised_patch_np * weight_mask

            # 确定当前块在【填充后完整图像】中的有效输出区域的坐标
            # 有效输出区域的起始点是 original_start + pad_amount
            output_row_start = original_row_start + pad_amount
            output_col_start = original_col_start + pad_amount
            output_row_end = output_row_start + patch_out_size
            output_col_end = output_col_start + patch_out_size

            # 此时 denoised_patch_blended 保证是 256x256，累加操作应该成功。

            # 累加
            denoised_full_image_sum[
            output_row_start:output_row_end,
            output_col_start:output_col_end
            ] += denoised_patch_blended

            denoised_full_image_count[
            output_row_start:output_row_end,
            output_col_start:output_col_end
            ] += weight_mask

    # 5. 合并分块结果
    denoised_full_image_count[denoised_full_image_count == 0] = 1
    final_denoised_padded = denoised_full_image_sum / denoised_full_image_count

    # 6. 裁剪回原始图像尺寸 (【精确裁剪掉填充区域】)
    # 裁剪掉 pad_amount 的边界
    denoised_phase_np = final_denoised_padded[
                        pad_amount: pad_amount + original_h,
                        pad_amount: pad_amount + original_w
                        ]

    # 7. 保存去噪后的文件 (保持不变)
    try:
        output_dir = os.path.dirname(output_denoised_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"已创建输出文件夹: {output_dir}")

        np.save(output_denoised_path, denoised_phase_np)
        print(f"去噪后的相位图已保存到: {output_denoised_path}")
    except Exception as e:
        print(f"错误: 保存去噪结果到 {output_denoised_path} 失败: {str(e)}")

    # 8. 可视化结果 (保持不变)
    plot_results(noisy_phase_np, denoised_phase_np, noisy_file_path=noisy_image_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def plot_results(noisy_phase_np, denoised_phase_np, noisy_file_path=""):
    """
    绘制原始噪声图和去噪图。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    vmin = -np.pi
    vmax = np.pi

    im0 = axes[0].imshow(noisy_phase_np, cmap='twilight', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'噪声相位图\n({os.path.basename(noisy_file_path)})')
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(denoised_phase_np, cmap='twilight', vmin=vmin, vmax=vmax)
    axes[1].set_title('去噪相位图')
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # >>> 请在这里修改为你实际的文件路径和参数 <<<
    noisy_image_path = 'EPDN4.npy'
    model_weights_path = 'best_generator30000.pth'
    output_denoised_path = 'CGAN.npy'

    # **分块处理参数：**
    PATCH_OUT_SIZE = 256  # 模型有效输出尺寸
    INPUT_CONTEXT = 32  # 模型期望的上下文边界 (输入块 = 320x320)
    OVERLAP = 64  # 建议 OVERLAP 至少取 64，如果仍有分割线，请增大到 128 或 192。

    # 检查文件路径是否存在，如果不存在则生成模拟数据用于测试
    if not os.path.exists(noisy_image_path):
        print(f"警告: 找不到噪声图像文件 '{noisy_image_path}'。正在生成一个2448x2048的模拟噪声图像...")
        dummy_noisy_phase = np.random.uniform(-np.pi, np.pi, (2448, 2048)).astype(np.float32)
        np.save(noisy_image_path, dummy_noisy_phase)
        print(f"已生成模拟噪声图像到 '{noisy_image_path}'")

    if not os.path.exists(model_weights_path):
        print(f"错误: 找不到模型权重文件 '{model_weights_path}'。请确保模型已训练并保存。")
        exit()

    # 调用分块去噪函数
    try:
        denoise_large_image_patchwise(
            noisy_image_path=noisy_image_path,
            model_path=model_weights_path,
            output_denoised_path=output_denoised_path,
            patch_out_size=PATCH_OUT_SIZE,
            input_context=INPUT_CONTEXT,
            overlap=OVERLAP
        )
    except Exception as e:
        print(f"执行去噪时发生错误: {e}")