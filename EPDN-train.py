import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from pytorch_msssim import ssim as ssim_loss_fn
from wafer import generate_random_pattern
from scipy.ndimage import gaussian_filter
import os
from tqdm import tqdm

# -------------------- fix random seed --------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

# -------------------- dataset --------------------
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
        # 为每个样本设置独立种子，确保预生成数据一致
        np.random.seed(SEED + idx)
        torch.manual_seed(SEED + idx)
        s = self.size

        # ——— height map & clean phase ———
        rel = generate_random_pattern(s)
        sub, et = np.random.uniform(0.5, 1.0), np.random.uniform(0.2, 0.5)
        h_map = torch.from_numpy(((rel <= 0.5) * sub + (rel > 0.5) * (sub + et))).float()
        phi_clean_cont = (4 * np.pi / self.lambda_) * h_map
        phi_clean = torch.angle(torch.exp(1j * phi_clean_cont))

        # ——— background mask ———
        bg_mask = torch.from_numpy((rel <= 0.5).astype(np.float32))

        # ——— Gaussian beam & propagation ———
        coords = torch.linspace(-s / 2 * self.pixel, s / 2 * self.pixel, s)
        X, Y = torch.meshgrid(coords, coords, indexing='ij')
        sigma_beam = s * self.pixel / 4
        gauss_env = torch.exp(- (X ** 2 + Y ** 2) / (2 * sigma_beam ** 2))

        z = np.random.uniform(1.0, 60.0) * 1e-3
        pr = np.random.uniform(3.0, 15.0) * 1e-3
        fx = torch.fft.fftfreq(s, d=self.pixel)
        fy = torch.fft.fftfreq(s, d=self.pixel)
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        rsq = FX ** 2 + FY ** 2
        fcut = pr / (self.lambda_ * z)
        n = 4
        sg = torch.exp(- (rsq / fcut ** 2) ** n)
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
        imin = intensity.min()
        imax = intensity.max()
        intensity_norm = (intensity - imin) / ((imax - imin) + 1e-8)
        weight = 1 - intensity_norm

        alpha = np.random.uniform(*self.alpha_range) * np.random.uniform(0.8, 1.2)
        std = np.random.uniform(*self.std_range) * np.random.uniform(0.8, 1.2)
        phi_coh = (torch.rand(s, s) * 2 - 1) * alpha * weight
        phi_gau = torch.randn(s, s) * std * weight

        speckle = (phi_speckle - phi_clean) * weight
        total_noise = speckle + phi_coh + phi_gau

        noisy_phase = torch.angle(torch.exp(1j * (phi_clean + total_noise)))
        noisy = torch.stack([torch.cos(noisy_phase), torch.sin(noisy_phase)], 0)
        clean = torch.stack([torch.cos(phi_clean), torch.sin(phi_clean)], 0)

        # data augmentation
        if torch.rand(1).item() > 0.5:
            noisy = self.augment(noisy)
            clean = self.augment(clean)
        noisy += torch.randn_like(noisy) * 0.01

        return noisy, clean, phi_clean, noisy_phase, bg_mask

class PreGeneratedDataset(Dataset):
    def __init__(self, data_dir, total=30000):  # 添加total参数，默认30000
        self.data_dir = os.path.normpath(data_dir)  # 规范化路径
        self.total = total
        self.data_list = [f"sample_{i}.pt" for i in range(self.total)]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        sample_path = os.path.join(self.data_dir, self.data_list[idx])
        try:
            # 显式设置 weights_only=False，因为加载的是可信的预生成数据集
            sample = torch.load(sample_path, weights_only=False)
            return sample
        except FileNotFoundError:
            missing_files = [f for f in self.data_list if not os.path.exists(os.path.join(self.data_dir, f))]
            raise FileNotFoundError(
                f"Dataset file {sample_path} not found. Missing {len(missing_files)} files: "
                f"{missing_files[:5]} (and {len(missing_files)-5} more if >5). "
                f"Please ensure the dataset is fully pregenerated with {self.total} samples "
                f"using pregenerate_dataset() and check the data_dir path: {self.data_dir}"
            )

def pregenerate_dataset(total, size, lambda_, pixel, data_dir):
    data_dir = os.path.normpath(data_dir)  # 规范化路径
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    ds = MixedNoisyDataset(total, size, lambda_, pixel)
    for i in tqdm(range(total), desc=f"Generating dataset ({total} samples)"):
        np.random.seed(SEED + i)  # 确保预生成数据一致
        torch.manual_seed(SEED + i)
        sample = ds[i]
        torch.save(sample, os.path.join(data_dir, f"sample_{i}.pt"))
    print(f"Dataset pregeneration complete. {total} files saved in {data_dir}")

# -------------------- network --------------------
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
    def __init__(self, base_ch=128):  # 修改为128
        super().__init__()
        self.conv_in = nn.Conv2d(2, base_ch, 3, padding=1)
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
            nn.Conv2d(4, 128, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 3, 2, 0),  # 修改为2x2输出
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], 1))

def gradient_loss(phi_pred, phi_gt):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                           dtype=torch.float32, device=phi_pred.device).unsqueeze(0).unsqueeze(0)
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

# -------------------- train --------------------
def train_model(G, D, train_loader, val_loader, num_epochs=60, device='cuda'):
    G.to(device)
    D.to(device)

    initial_lr = 0.0002  # 修改学习率为0.0002
    g_opt = optim.Adam(G.parameters(), lr=initial_lr)
    d_opt = optim.Adam(D.parameters(), lr=initial_lr)

    λ_l1 = 5.0
    λ_ssim = 3.0
    λ_gan_base = 0.2  # 基础λ_gan
    λ_grad = 1.2
    λ_phase = 0.3
    lambda_gp = 5

    best_metric = -float('inf')

    for ep in range(num_epochs):
        # 动态调整λ_gan
        if ep < 30:
            λ_gan = λ_gan_base
        else:
            λ_gan = λ_gan_base + (0.5 - λ_gan_base) * (ep - 30) / 30  # 线性增加到0.5

        G.train()
        D.train()
        sumG = sumD = 0.0
        acc_l = acc_s = acc_g = acc_grad = acc_phase = 0.0
        cnt = dcnt = 0

        for i, (noisy, clean, phi_gt, noisy_phase, bg_mask) in enumerate(train_loader):
            noisy, clean = noisy.to(device), clean.to(device)
            phi_gt = phi_gt.to(device)
            bg_mask = bg_mask.to(device).unsqueeze(1)
            fg_mask = 1.0 - bg_mask

            # 更新判别器
            d_opt.zero_grad()
            pred_real = D(noisy, clean)
            lossD_real = -torch.mean(pred_real)

            with torch.no_grad():
                out_fake_det = G(noisy).detach()
            pred_fake = D(noisy, out_fake_det)
            lossD_fake = torch.mean(pred_fake)

            gp = calculate_gradient_penalty(D, clean, out_fake_det, noisy, device, lambda_gp)
            lossD = lossD_real + lossD_fake + gp
            lossD.backward()
            d_opt.step()

            sumD += lossD.item()
            dcnt += 1

            # 更新生成器
            g_opt.zero_grad()
            out = G(noisy)
            phi_pred = torch.atan2(out[:, 1, :, :], out[:, 0, :, :])

            l1_map = torch.abs(out - clean)
            l1_bg = (l1_map * bg_mask).mean()
            l1_fg = (l1_map * fg_mask).mean()
            l1_total = 2.0 * l1_bg + 1.0 * l1_fg

            ls = 1 - ssim_loss_fn(out, clean, data_range=2, size_average=True)
            pred_fake_g = D(noisy, out)
            lg = -torch.mean(pred_fake_g)

            lgrad = gradient_loss(phi_pred, phi_gt)
            mse_phase = F.mse_loss(phi_pred, phi_gt)

            lossG = (
                λ_l1 * l1_total +
                λ_ssim * ls +
                λ_gan * lg +
                λ_grad * lgrad +
                λ_phase * mse_phase
            )
            lossG.backward()
            g_opt.step()

            sumG += lossG.item()
            acc_l += l1_total.item()
            acc_s += ls.item()
            acc_g += lg.item()
            acc_grad += lgrad.item()
            acc_phase += mse_phase.item()
            cnt += 1

        # 验证
        G.eval()
        D.eval()
        val_loss = sumQ = sumSigma = 0.0
        with torch.no_grad():
            for noisy_v, clean_v, phi_gt_v, _, bg_mask_v in val_loader:
                noisy_v, clean_v = noisy_v.to(device), clean_v.to(device)
                phi_gt_v = phi_gt_v.to(device)
                bg_mask_v = bg_mask_v.to(device).unsqueeze(1)
                fg_mask_v = 1.0 - bg_mask_v

                outv = G(noisy_v)
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

                lv = (
                    λ_l1 * l1_total_v +
                    λ_ssim * ls_v +
                    λ_grad * lgrad_v +
                    λ_phase * mse_phase_v
                )
                pred_f = D(noisy_v, outv)
                lv += λ_gan * (-torch.mean(pred_f))

                val_loss += lv.item()

        nval = len(val_loader)
        val_loss /= nval
        avgQ = sumQ / nval
        avgSigma = sumSigma / nval

        # 保存最佳模型
        metric = avgQ - avgSigma
        if metric > best_metric:
            best_metric = metric
            torch.save(G.state_dict(), 'best_generator.pth')

        print(
            f"Epoch {ep+1}/{num_epochs} │ "
            f"G_loss={sumG/cnt:.4f} │ "
            f"L1={acc_l/cnt:.4f} │ "
            f"SSIM={acc_s/cnt:.4f} │ "
            f"PhaseMSE={acc_phase/cnt:.4f} │ "
            f"Grad={acc_grad/cnt:.4f} │ "
            f"GAN={acc_g/cnt:.4f} │ "
            f"D_loss={sumD/dcnt:.4f} │ "
            f"Val={val_loss:.4f} │ "
            f"Q={avgQ:.4f} │ "
            f"σφ={avgSigma:.4f} │ "
            f"LR_G={g_opt.param_groups[0]['lr']:.2e} │ "
            f"LR_D={d_opt.param_groups[0]['lr']:.2e}"
        )

    torch.save(G.state_dict(), 'final_generator.pth')
    print("Training complete.")

# -------------------- main --------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 设置数据集大小（调试用300，正式训练用30000）
    total_samples = 30000  # 修改为300用于调试，正式训练可改为30000
    data_dir = os.path.normpath("pregenerated_data")  # 规范化路径

    # 检查数据集完整性
    expected_files = [f"sample_{i}.pt" for i in range(total_samples)]
    missing_files = [f for f in expected_files if not os.path.exists(os.path.join(data_dir, f))]
    if missing_files:
        print(f"Dataset incomplete. Missing {len(missing_files)} files: {missing_files[:5]} (and {len(missing_files)-5} more if >5).")
        print(f"Generating dataset with {total_samples} samples in {data_dir}...")
        pregenerate_dataset(total_samples, 256, 0.532e-3, 3.45e-3, data_dir)
    else:
        print(f"Dataset found in {data_dir}. All {len(expected_files)} files present. Skipping pregeneration.")

    # 数据加载
    dataset = PreGeneratedDataset(data_dir, total=total_samples)  # 传递total_samples
    tlen = int(0.8 * len(dataset))
    vlen = len(dataset) - tlen
    train_ds, val_ds = random_split(
        dataset, [tlen, vlen],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True,
        pin_memory=True, num_workers=4,
        generator=torch.Generator().manual_seed(SEED)
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False,
        pin_memory=True, num_workers=4
    )

    # 初始化模型
    G = Generator(base_ch=128)  # 基础通道数128
    D = Discriminator()  # PatchGAN输出2x2
    train_model(G, D, train_loader, val_loader, num_epochs=60, device=device)