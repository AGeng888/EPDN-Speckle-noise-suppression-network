import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from pytorch_msssim import ssim as ssim_loss_fn
from scipy.ndimage import gaussian_filter
import os
from tqdm import tqdm  
from skimage.restoration import denoise_nl_means  


try:
    from wafer import generate_random_pattern
except ImportError:
    print("Warning: 'wafer' module not found. Using random matrix as fallback.")


    def generate_random_pattern(size):
        return np.random.rand(size, size)



def set_seed(seed: int):
    """固定随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
set_seed(SEED)



class MixedNoisyDataset(Dataset):
    def __init__(self, total: int, size: int, lambda_: float, pixel: float):
        self.total = total
        self.size = size
        self.lambda_ = lambda_
        self.pixel = pixel
        self.alpha_range = (np.pi / 9, np.pi / 7)
        self.std_range = (0.009, 0.011)
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, scale=(0.9, 1.1)),
            transforms.GaussianBlur(3, sigma=(0.1, 0.5))
        ])

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        s = self.size
        rel = generate_random_pattern(s)
        sub, et = np.random.uniform(0.5, 1.0), np.random.uniform(0.2, 0.5)
        h_map = torch.from_numpy(((rel <= 0.5) * sub + (rel > 0.5) * (sub + et))).float()
        phi_clean_cont = (4 * np.pi / self.lambda_) * h_map
        phi_clean = torch.angle(torch.exp(1j * phi_clean_cont))
        bg_mask = torch.from_numpy((rel <= 0.5).astype(np.float32))

        coords = torch.linspace(-s / 2 * self.pixel, s / 2 * self.pixel, s)
        X, Y = torch.meshgrid(coords, coords, indexing='ij')
        sigma_beam = s * self.pixel / 4
        gauss_env = torch.exp(-(X ** 2 + Y ** 2) / (2 * sigma_beam ** 2))

        z = np.random.uniform(1.0, 60.0) * 1e-3
        pr = np.random.uniform(3.0, 15.0) * 1e-3
        fx = torch.fft.fftfreq(s, d=self.pixel)
        fy = torch.fft.fftfreq(s, d=self.pixel)
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        rsq = FX ** 2 + FY ** 2
        fcut = pr / (self.lambda_ * z)
        n = 4
        sg = torch.exp(-(rsq / fcut ** 2) ** n)
        H = sg * torch.exp(-1j * np.pi * self.lambda_ * z * rsq)

        obj = gauss_env * torch.exp(1j * phi_clean_cont)
        U1 = torch.fft.fft2(obj)
        u_out = torch.fft.ifft2(U1 * H)

        amp = u_out.abs().cpu().numpy()
        amp_blur = gaussian_filter(amp, sigma=1)
        amp_blur = torch.from_numpy(amp_blur)
        u_out = amp_blur * torch.exp(1j * torch.angle(u_out))

        phi_speckle = torch.angle(u_out)

        intensity = u_out.abs() ** 2
        imin, imax = intensity.min(), intensity.max()
        intensity_norm = (intensity - imin) / ((imax - imin) + 1e-8)
        weight = 1 - intensity_norm

        alpha = np.random.uniform(*self.alpha_range) * np.random.uniform(0.8, 1.2)
        std = np.random.uniform(*self.std_range) * np.random.uniform(0.8, 1.2)
        phi_coh = (torch.rand(s, s) * 2 - 1) * alpha * weight
        phi_gau = torch.randn(s, s) * std * weight

        speckle = (phi_speckle - phi_clean) * weight
        total_noise = speckle + phi_coh + phi_gau

        noisy_phase = torch.angle(torch.exp(1j * (phi_clean + total_noise)))
        original_noisy = torch.stack([torch.cos(noisy_phase), torch.sin(noisy_phase)], 0)
        clean = torch.stack([torch.cos(phi_clean), torch.sin(phi_clean)],
                            0)  # Changed clean76 back to clean for consistency

        noisy_cos_np = ((original_noisy[0].cpu().numpy() + 1) / 2).astype(np.float32)
        noisy_sin_np = ((original_noisy[1].cpu().numpy() + 1) / 2).astype(np.float32)

        nlm_cos_denoised_np = denoise_nl_means(noisy_cos_np, h=0.1, patch_size=7, patch_distance=11, fast_mode=True)
        nlm_sin_denoised_np = denoise_nl_means(noisy_sin_np, h=0.1, patch_size=7, patch_distance=11, fast_mode=True)

        nlm_cos_denoised = torch.from_numpy(nlm_cos_denoised_np * 2 - 1).float()
        nlm_sin_denoised = torch.from_numpy(nlm_sin_denoised_np * 2 - 1).float()
        nlm_denoised = torch.stack([nlm_cos_denoised, nlm_sin_denoised], 0)

        if torch.rand(1).item() > 0.5:
            seed = torch.seed()
            torch.manual_seed(seed)
            original_noisy = self.augment(original_noisy)
            torch.manual_seed(seed)
            nlm_denoised = self.augment(nlm_denoised)
            torch.manual_seed(seed)
            clean = self.augment(clean)  # Changed clean76 back to clean

        original_noisy += torch.randn_like(original_noisy) * 0.01
        combined_noisy_input = torch.cat([original_noisy, nlm_denoised], dim=0)

        return combined_noisy_input, clean, phi_clean, noisy_phase, bg_mask


class PreGeneratedDataset(Dataset):
    def __init__(self, data_dir, total=30000):
        self.data_dir = os.path.normpath(data_dir)
        self.total = total
        self.data_list = [f"sample_{i}.pt" for i in range(self.total)]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        sample_path = os.path.join(self.data_dir, self.data_list[idx])
        try:
            sample = torch.load(sample_path, weights_only=False)
            return sample
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file {sample_path} not found.")


def pregenerate_dataset(total, size, lambda_, pixel, data_dir):
    """预生成数据集并保存"""
    data_dir = os.path.normpath(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    ds = MixedNoisyDataset(total, size, lambda_, pixel)
    print(f"Starting dataset pregeneration for {total} samples in {data_dir}...")
    for i in tqdm(range(total), desc=f"Generating dataset ({total} samples)"):
        # 为每个样本设置独立种子，确保预生成数据一致性
        np.random.seed(SEED + i)
        torch.manual_seed(SEED + i)
        sample = ds[i]
        torch.save(sample, os.path.join(data_dir, f"sample_{i}.pt"))
    print(f"Dataset pregeneration complete. {total} files saved in {data_dir}")


def check_dataset_channels(data_dir, num_samples_to_check=1):
    print(f"\nChecking channel count for first {num_samples_to_check} samples in {data_dir}...")
    data_dir = os.path.normpath(data_dir)
    if not os.path.exists(data_dir):
        print(f"  Error: Dataset directory '{data_dir}' not found.")
        return False

   
    
    global total_samples  # 访问全局变量 total_samples
    actual_samples_to_check = min(num_samples_to_check, total_samples)  # 确保不越界

    for i in range(actual_samples_to_check):
        sample_path = os.path.join(data_dir, f"sample_{i}.pt")
        if not os.path.exists(sample_path):
            print(f"  Sample file not found: {sample_path}.")
            return False
        try:
            loaded_data = torch.load(sample_path, weights_only=False)
            combined_noisy_input = loaded_data[0]
            channels = combined_noisy_input.shape[0]
            print(f"  Sample {i}: Channels: {channels}")
            if channels != 4:
                print(f"  Mismatch: Expected 4 channels, got {channels}.")
                return False
        except Exception as e:
            print(f"  Error loading sample {i}: {e}")
            return False
    print(f"  All {actual_samples_to_check} samples have 4 channels.")
    return True


# --- 3. 网络架构定义 ---
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class Generator(nn.Module):
    def __init__(self, base_ch=128):
        super().__init__()
        self.conv_in = nn.Conv2d(4, base_ch, 3, padding=1)
        self.rb1, self.pool1 = ResBlock(base_ch), nn.MaxPool2d(2)
        self.rb2, self.pool2 = ResBlock(base_ch), nn.MaxPool2d(2)
        self.rb3, self.pool3 = ResBlock(base_ch), nn.MaxPool2d(2)
        self.rb_bot1, self.rb_bot2 = ResBlock(base_ch), ResBlock(base_ch)
        self.up3 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.rb4, self.rb4b = ResBlock(base_ch), ResBlock(base_ch)
        self.up2 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.rb5, self.rb5b = ResBlock(base_ch), ResBlock(base_ch)
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.rb6, self.rb6b = ResBlock(base_ch), ResBlock(base_ch)
        self.conv_out = nn.Conv2d(base_ch, 2, 1)

    def forward(self, x):
        e1 = F.relu(self.conv_in(x))
        e1 = self.rb1(e1)
        e2 = self.pool1(e1)
        e2 = self.rb2(e2)
        e3 = self.pool2(e2)
        e3 = self.rb3(e3)
        b = self.pool3(e3)
        b = self.rb_bot1(b)
        b = self.rb_bot2(b)
        d3 = self.up3(b)
        d3 = self.rb4(d3 + e3)
        d3 = self.rb4b(d3)
        d2 = self.up2(d3)
        d2 = self.rb5(d2 + e2)
        d2 = self.rb5b(d2)
        d1 = self.up1(d2)
        d1 = self.rb6(d1 + e1)
        d1 = self.rb6b(d1)
        return torch.tanh(self.conv_out(d1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 128, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 3, 2, 0),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], 1))


# --- 损失函数和辅助函数 ---
def gradient_loss(phi_pred, phi_gt):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=phi_pred.device).unsqueeze(
        0).unsqueeze(0)
    sobel_y = sobel_x.transpose(2, 3)
    gx_p = F.conv2d(phi_pred.unsqueeze(1), sobel_x, padding=1)
    gy_p = F.conv2d(phi_pred.unsqueeze(1), sobel_y, padding=1)
    gx_g = F.conv2d(phi_gt.unsqueeze(1), sobel_x, padding=1)
    gy_g = F.conv2d(phi_gt.unsqueeze(1), sobel_y, padding=1)
    return F.l1_loss(gx_p, gx_g) + F.l1_loss(gy_p, gy_g)


def calculate_gradient_penalty(discriminator, real_images, fake_images, noisy_inputs, device, lambda_gp):
    alpha = torch.rand(real_images.size(0), 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_images)
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)
    disc_interpolates = discriminator(noisy_inputs, interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp
    return gradient_penalty


# --- 4. 训练函数 ---
def train_model(G, D, train_loader, val_loader, num_epochs=60, device='cuda'):
    G.to(device)
    D.to(device)

    initial_lr = 0.0002
    g_opt = optim.Adam(G.parameters(), lr=initial_lr)
    d_opt = optim.Adam(D.parameters(), lr=initial_lr)

    λ_l1, λ_ssim, λ_gan_base, λ_grad, λ_phase, lambda_gp = 5.0, 3.0, 0.2, 1.2, 0.3, 5
    best_metric = -float('inf')

    print("\nStarting training...")
    for ep in range(num_epochs):
        λ_gan = λ_gan_base if ep < 30 else λ_gan_base + (0.5 - λ_gan_base) * (ep - 30) / 30

        G.train()
        D.train()
        sumG = sumD = 0.0
        acc_l = acc_s = acc_g = acc_grad = acc_phase = 0.0
        cnt = dcnt = 0

        for i, (combined_noisy_input, clean, phi_gt, noisy_phase, bg_mask) in enumerate(train_loader):
            combined_noisy_input, clean, phi_gt, bg_mask = combined_noisy_input.to(device), clean.to(device), phi_gt.to(
                device), bg_mask.to(device).unsqueeze(1)
            fg_mask = 1.0 - bg_mask

            # --- 更新判别器 ---
            d_opt.zero_grad()
            pred_real = D(combined_noisy_input, clean)
            lossD_real = -torch.mean(pred_real)

            with torch.no_grad():
                out_fake_det = G(combined_noisy_input).detach()
            pred_fake = D(combined_noisy_input, out_fake_det)
            lossD_fake = torch.mean(pred_fake)

            gp = calculate_gradient_penalty(D, clean, out_fake_det, combined_noisy_input, device, lambda_gp)
            lossD = lossD_real + lossD_fake + gp
            lossD.backward()
            d_opt.step()

            sumD += lossD.item()
            dcnt += 1

            # --- 更新生成器 ---
            g_opt.zero_grad()
            out = G(combined_noisy_input)
            phi_pred = torch.atan2(out[:, 1, :, :], out[:, 0, :, :])

            l1_map = torch.abs(out - clean)
            l1_bg = (l1_map * bg_mask).mean()
            l1_fg = (l1_map * fg_mask).mean()
            l1_total = 2.0 * l1_bg + 1.0 * l1_fg

            ls = 1 - ssim_loss_fn(out, clean, data_range=2, size_average=True)

            pred_fake_g = D(combined_noisy_input, out)
            lg = -torch.mean(pred_fake_g)

            lgrad = gradient_loss(phi_pred, phi_gt)
            mse_phase = F.mse_loss(phi_pred, phi_gt)

            lossG = λ_l1 * l1_total + λ_ssim * ls + λ_gan * lg + λ_grad * lgrad + λ_phase * mse_phase
            lossG.backward()
            g_opt.step()

            sumG += lossG.item()
            acc_l += l1_total.item()
            acc_s += ls.item()
            acc_g += lg.item()
            acc_grad += lgrad.item()
            acc_phase += mse_phase.item()
            cnt += 1

        # --- 验证阶段 ---
        G.eval()
        D.eval()
        val_loss = sumQ = sumSigma = 0.0
        with torch.no_grad():
            for combined_noisy_input_v, clean_v, phi_gt_v, _, bg_mask_v in val_loader:
                combined_noisy_input_v, clean_v, phi_gt_v, bg_mask_v = combined_noisy_input_v.to(device), clean_v.to(
                    device), phi_gt_v.to(device), bg_mask_v.to(device).unsqueeze(1)
                fg_mask_v = 1.0 - bg_mask_v

                outv = G(combined_noisy_input_v)
                phi_pred_v = torch.atan2(outv[:, 1, :, :], outv[:, 0, :, :])
                dphi = (phi_pred_v - phi_gt_v + np.pi) % (2 * np.pi) - np.pi

                cov = ((phi_pred_v - phi_pred_v.mean()) * (phi_gt_v - phi_gt_v.mean())).mean()
                stdp, stdg = phi_pred_v.std(), phi_gt_v.std()
                Q = cov / (stdp * stdg + 1e-8)
                sigma = dphi.std()
                sumQ += Q.item()
                sumSigma += sigma.item()

                l1_map_v = torch.abs(outv - clean_v)
                l1_bg_v = (l1_map_v * bg_mask_v).mean()
                l1_fg_v = (l1_map_v * fg_mask_v).mean()
                l1_total_v = 2.0 * l1_bg_v + 1.0 * l1_fg_v

                ls_v = 1 - ssim_loss_fn(outv, clean_v, data_range=2, size_average=True)
                lgrad_v = gradient_loss(phi_pred_v, phi_gt_v)
                mse_phase_v = F.mse_loss(phi_pred_v, phi_gt_v)

                lv = λ_l1 * l1_total_v + λ_ssim * ls_v + λ_grad * lgrad_v + λ_phase * mse_phase_v
                pred_f = D(combined_noisy_input_v, outv)
                lv += λ_gan * (-torch.mean(pred_f))

                val_loss += lv.item()

        nval = len(val_loader)
        val_loss /= nval
        avgQ = sumQ / nval
        avgSigma = sumSigma / nval

        # 保存最佳模型 (基于 Q - Sigma_phi 指标)
        metric = avgQ - avgSigma
        if metric > best_metric:
            best_metric = metric
            torch.save(G.state_dict(), 'best_generator.pth')
            print(f"  New best model saved! Metric: {best_metric:.4f}")

        # 恢复更详细的打印格式
        print(
            f"Epoch {ep + 1}/{num_epochs} │ "
            f"G_loss={sumG / cnt:.4f} │ "
            f"L1={acc_l / cnt:.4f} │ "  # 重新添加 L1
            f"SSIM={acc_s / cnt:.4f} │ "  # 重新添加 SSIM
            f"PhaseMSE={acc_phase / cnt:.4f} │ "  # 重新添加 PhaseMSE
            f"Grad={acc_grad / cnt:.4f} │ "  # 重新添加 Grad
            f"GAN={acc_g / cnt:.4f} │ "  # GAN 保持
            f"D_loss={sumD / dcnt:.4f} │ "
            f"Val={val_loss:.4f} │ "
            f"Q={avgQ:.4f} │ "
            f"σφ={avgSigma:.4f}"
        )

    torch.save(G.state_dict(), 'final_generator.pth')
    print("Training complete.")


# --- 5. 主执行逻辑 ---
if __name__ == '__main__':
    try:
        import skimage.restoration
        # 再次确认 tqdm 导入方式
        from tqdm import tqdm
    except ImportError:
        print("Please install required packages: pip install scikit-image tqdm")
        exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 注意：这里是您设置的 total_samples
    total_samples = 30000  # 调试时可设置为较小的值如 1000，正式训练建议 30000
    data_dir = os.path.normpath("pregenerated_data_with_nlm")

    dataset_needs_regeneration = True
    if os.path.exists(data_dir):
        existing_files = [f for f in os.listdir(data_dir) if f.startswith("sample_") and f.endswith(".pt")]
        # 检查文件数量是否完整且通道数是否正确
        if len(existing_files) == total_samples:
            print(f"Found existing dataset directory '{data_dir}' with {total_samples} files.")
            if check_dataset_channels(data_dir, num_samples_to_check=min(5, total_samples)):  # 检查前5个样本或所有样本（如果少于5个）
                dataset_needs_regeneration = False
                print("Using existing dataset.")
            else:
                print("Dataset channel check failed for existing data. Regenerating.")
        else:
            print(f"Existing dataset incomplete ({len(existing_files)}/{total_samples} files). Regenerating.")
    else:
        print(f"Dataset directory '{data_dir}' not found. Regenerating.")

    if dataset_needs_regeneration:
        pregenerate_dataset(total_samples, 256, 0.532e-3, 3.45e-3, data_dir)

    dataset = PreGeneratedDataset(data_dir, total=total_samples)
    tlen = int(0.8 * len(dataset))
    vlen = len(dataset) - tlen
    train_ds, val_ds = random_split(
        dataset, [tlen, vlen],
        generator=torch.Generator().manual_seed(SEED)  # 确保 DataLoader 的数据顺序可复现
    )
    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True,
        pin_memory=True, num_workers=4,
        generator=torch.Generator().manual_seed(SEED)  # 确保 DataLoader 的数据顺序可复现
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False,
        pin_memory=True, num_workers=4
    )

    G = Generator(base_ch=128)
    D = Discriminator()
    print("Generator and Discriminator models initialized.")
    print(f"Generator first conv input channels: {G.conv_in.in_channels}")
    print(f"Discriminator first conv input channels: {D.net[0].in_channels}")

    train_model(G, D, train_loader, val_loader, num_epochs=60, device=device)
