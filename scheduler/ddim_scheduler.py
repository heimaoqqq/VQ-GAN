import torch
import math
import numpy as np


class DDIMScheduler:
    """
    DDIM调度器 - 实现去噪扩散隐式模型的采样
    相比DDPM可以使用更少的采样步骤，大大提高采样速度
    """
    
    def __init__(self, num_timesteps=1000, beta_start=0.0015, beta_end=0.0195, sampling_steps=100):
        """
        初始化DDIM调度器
        
        参数:
            num_timesteps: 总的时间步数（训练时使用的步数）
            beta_start: 噪声调度beta的起始值
            beta_end: 噪声调度beta的结束值
            sampling_steps: 实际采样时使用的步数（远小于num_timesteps）
        """
        self.num_timesteps = num_timesteps
        self.sampling_steps = sampling_steps
        
        # 线性噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 计算采样时间步索引
        self.timesteps = np.arange(0, num_timesteps, num_timesteps // sampling_steps)
        if len(self.timesteps) > sampling_steps:
            self.timesteps = self.timesteps[:sampling_steps]
        
        print(f"DDIM调度器: 训练步数={num_timesteps}, 采样步数={len(self.timesteps)}")
    
    def add_noise(self, x_0, noise, t):
        """
        将噪声添加到干净的图像x_0，得到噪声图像x_t
        
        参数:
            x_0: 干净的图像
            noise: 要添加的噪声
            t: 时间步
        
        返回:
            x_t: 添加了噪声的图像
        """
        # 确保t在正确的设备上
        if isinstance(t, torch.Tensor):
            t = t.to(self.alphas_cumprod.device)
        
        alpha_cumprod_t = self.alphas_cumprod[t].to(x_0.device).view(-1, 1, 1, 1)
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
        
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t
    
    def sample_prev_timestep(self, x_t, predicted_noise, t, eta=0.0):
        """
        使用DDIM采样从x_t到x_{t-1}
        
        参数:
            x_t: 当前时间步t的噪声图像
            predicted_noise: 模型预测的噪声
            t: 当前时间步索引
            eta: DDIM中的随机性控制参数（0表示确定性采样，1表示随机采样，即DDPM）
        
        返回:
            x_prev: 前一个时间步的图像
            pred_x0: 模型预测的原始图像x_0
        """
        # 将标量t转换为张量并确保其在正确的设备上
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.long, device=self.alphas_cumprod.device)
        else:
            t = t.to(self.alphas_cumprod.device)
        
        # 确保t是合适的维度
        t = t.view(-1)
        
        # 获取t时刻的alpha_cumprod
        alpha_cumprod_t = self.alphas_cumprod[t].to(x_t.device)
        alpha_cumprod_t = alpha_cumprod_t.view(-1, 1, 1, 1)
        
        # 预测x0
        sqrt_recip_alpha_cumprod_t = torch.sqrt(1 / alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
        
        pred_x0 = sqrt_recip_alpha_cumprod_t * x_t - sqrt_one_minus_alpha_cumprod_t / \
                 torch.sqrt(alpha_cumprod_t) * predicted_noise
        
        # 确定上一个时间步
        if t[0] == 0:
            return pred_x0, pred_x0
        
        prev_t = t - 1
        # 确保prev_t也在正确的设备上
        prev_t = prev_t.to(self.alphas_cumprod.device)
        
        alpha_cumprod_prev_t = self.alphas_cumprod[prev_t].to(x_t.device)
        alpha_cumprod_prev_t = alpha_cumprod_prev_t.view(-1, 1, 1, 1)
        
        # DDIM公式
        # 方差项
        variance = eta * torch.sqrt((1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)) * \
                  torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_prev_t)
        
        # 确定性项
        deterministic_component = torch.sqrt(alpha_cumprod_prev_t) * \
                                 (1 - alpha_cumprod_t/alpha_cumprod_prev_t) / \
                                 torch.sqrt(1 - alpha_cumprod_t) * predicted_noise
        
        # 如果eta > 0，添加随机性
        noise = torch.randn_like(x_t)
        random_component = variance * noise
        
        # 计算x_{t-1}
        x_prev = torch.sqrt(alpha_cumprod_prev_t) * pred_x0 + \
                deterministic_component + random_component
        
        return x_prev, pred_x0
    
    def get_timesteps(self):
        """
        返回用于采样的时间步列表
        """
        return self.timesteps[::-1]  # 反向，从T到0 