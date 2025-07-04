import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
import argparse
import yaml
import json
import gc
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from scheduler.ddim_scheduler import DDIMScheduler
from utils.config_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample_batch(model, scheduler, diffusion_model_config,
              autoencoder_model_config, diffusion_config, dataset_config, vae, class_ids, batch_idx, batch_size=10, eta=0.0):
    r"""
    从类别条件模型批量采样多张微多普勒时频图
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    
    ########### 随机噪声作为起点 ##########
    # 批量生成图片
    xt = torch.randn((batch_size,
                      autoencoder_model_config['z_channels'],
                      im_size,
                      im_size)).to(device)
    ###############################################
    
    ############# 验证配置 #################
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    assert condition_config is not None, "采样脚本需要条件配置，但未找到"
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'class' in condition_types, "采样脚本需要类别条件，但未在配置中找到"
    validate_class_config(condition_config)
    ###############################################
    
    ############ 创建条件输入 ###############
    num_classes = condition_config['class_condition_config']['num_classes']
    
    # 使用指定的类别ID数组
    sample_classes = torch.tensor(class_ids).to(device)
    
    cond_input = {
        'class': torch.nn.functional.one_hot(sample_classes, num_classes).float().to(device)
    }
    # 无条件输入用于分类器引导
    uncond_input = {
        'class': cond_input['class'] * 0
    }
    ###############################################
    
    # 分类器引导尺度
    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)
        
    # 获取采样步骤
    if isinstance(scheduler, DDIMScheduler):
        # 如果使用DDIM，使用其特定的timesteps
        timesteps = scheduler.get_timesteps()
        print(f"使用DDIM采样，步数: {len(timesteps)}")
    else:
        # 传统DDPM，使用所有timesteps
        steps = diffusion_config['num_timesteps']
        timesteps = list(reversed(range(steps)))
        print(f"使用DDPM采样，步数: {len(timesteps)}")
    
    # 创建采样进度条
    pbar = tqdm(timesteps)
    for i in pbar:
        # 更新进度条描述
        pbar.set_description(f"批次 {batch_idx+1}，采样步骤 {i}")
        
        # 获取噪声预测
        t = torch.tensor([i] * batch_size).long().to(device)
        noise_pred_cond = model(xt, t, cond_input)
        
        # 应用分类器引导（如果启用）
        if cf_guidance_scale > 1:
            noise_pred_uncond = model(xt, t, uncond_input)
            noise_pred = noise_pred_uncond + cf_guidance_scale*(noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond
        
        # 使用调度器获取x0和xt-1（对于DDIM，传递eta参数）
        if isinstance(scheduler, DDIMScheduler):
            # 确保i是CPU上的标量，让调度器自己处理设备转换
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, int(i), eta=eta)
        else:
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
    
    # 解码最终结果
    with torch.no_grad():
        ims = vae.decode(xt)
    
    # 后处理图像
    ims = torch.clamp(ims, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    
    # 批量处理为PIL图像列表
    images = []
    for idx in range(batch_size):
        img = torchvision.transforms.ToPILImage()(ims[idx])
        images.append(img)
    
    # 清理GPU内存
    del xt, noise_pred, noise_pred_cond, ims
    if cf_guidance_scale > 1:
        del noise_pred_uncond
    torch.cuda.empty_cache()
    gc.collect()
    
    return images


def sample(model, scheduler, train_config, diffusion_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, class_info=None, batch_size=10, eta=0.0):
    r"""
    从类别条件模型采样微多普勒时频图
    """
    # 总共生成50张图片
    num_samples = 50
    
    ################# 创建保存目录 ########################
    save_dir = os.path.join(train_config['task_name'], 'cond_class_samples')
    os.makedirs(save_dir, exist_ok=True)
    
    # 如果指定了特定类别，为其创建子目录
    if class_info is not None and 'id_name' in class_info:
        save_dir = os.path.join(save_dir, class_info['id_name'])
        os.makedirs(save_dir, exist_ok=True)
    
    # 确定使用哪个类别ID
    if class_info is not None and 'class_id' in class_info:
        class_id = class_info['class_id']
        print(f'为用户 {class_info.get("id_name", "未知")} (类别ID: {class_id}) 生成图像')
        # 全部使用同一个类别ID
        all_class_ids = [class_id] * num_samples
    else:
        # 为每张图片随机选择一个类别ID
        num_classes = diffusion_model_config['condition_config']['class_condition_config']['num_classes']
        all_class_ids = torch.randint(0, num_classes, (num_samples,)).tolist()
        print(f'随机生成类别图像，类别IDs将在生成过程中随机选择')
    
    # 计算需要的批次数
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # 进度计数器
    total_saved = 0
    
    # 按批次生成图片
    for batch_idx in range(num_batches):
        # 计算本批次大小
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        
        # 获取当前批次的类别ID
        batch_class_ids = all_class_ids[batch_idx * batch_size:batch_idx * batch_size + current_batch_size]
        
        # 生成当前批次的图片
        print(f"生成批次 {batch_idx+1}/{num_batches}，类别IDs: {batch_class_ids}")
        images = sample_batch(model, scheduler, diffusion_model_config, 
                         autoencoder_model_config, diffusion_config, dataset_config, vae, 
                            batch_class_ids, batch_idx, current_batch_size, eta)
        
        # 保存本批次的图片
        for i, img in enumerate(images):
            img_idx = batch_idx * batch_size + i
            img.save(os.path.join(save_dir, f'sample_{img_idx:03d}.png'))
            total_saved += 1
            if (i + 1) % 5 == 0 or i == len(images) - 1:
                print(f'已保存 {total_saved}/{num_samples} 张图片')
    
    print(f'采样完成！生成的{total_saved}张图像已保存到 {save_dir}')


def infer(args):
    # 读取配置文件
    with open(args.config_path, 'r', encoding='utf-8') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    # 解析配置
    global train_config  # 让sample_batch函数可以访问
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # 创建噪声调度器
    if args.sampler == "ddim":
        scheduler = DDIMScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                 beta_start=diffusion_config['beta_start'],
                                 beta_end=diffusion_config['beta_end'],
                                 sampling_steps=args.steps)
    else:
        scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    # 加载Unet模型
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()
    model_path = os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])
    if os.path.exists(model_path):
        print(f'加载模型检查点: {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise Exception(f'模型检查点未找到: {model_path}')
    
    # 创建输出目录
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # 加载VQVAE模型
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    
    vae_path = os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'])
    if os.path.exists(vae_path):
        print(f'加载VAE检查点: {vae_path}')
        vae.load_state_dict(torch.load(vae_path, map_location=device), strict=True)
    else:
        raise Exception(f'VAE检查点未找到: {vae_path}')
    
    # 检查潜在空间通道数
    z_channels = autoencoder_model_config['z_channels']
    if z_channels > 4:
        print(f"注意: 潜在空间有{z_channels}个通道，在可视化时将截取前3个通道，但完整通道将用于生成")
    
    # 获取类别映射信息
    class_info = None
    if args.id_name:
        class_mapping_path = os.path.join('./metadata', 'class_mapping.json')
        try:
            with open(class_mapping_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            id_to_class = mapping_data.get('id_to_class', {})
            
            # 检查指定的ID是否存在
            if args.id_name in id_to_class:
                class_info = {
                    'id_name': args.id_name,
                    'class_id': id_to_class[args.id_name]
                }
                
                # 输出详细的映射信息
                id_name_to_number = mapping_data.get('id_name_to_number', {})
                if args.id_name in id_name_to_number:
                    id_number = id_name_to_number[args.id_name]
                    print(f'找到用户ID映射: {args.id_name} (ID号: {id_number}) -> 类别索引 {class_info["class_id"]}')
                else:
                    print(f'找到用户ID映射: {args.id_name} -> 类别索引 {class_info["class_id"]}')
            else:
                print(f'警告: 未找到用户ID {args.id_name} 的映射，将随机选择类别')
        except Exception as e:
            print(f'无法加载类别映射: {e}')
    
    # 获取批量大小，默认为10
    batch_size = args.batch_size
    print(f"采用批量生成模式，批次大小: {batch_size}张/批")
    
    # 开始采样
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, class_info, batch_size, args.eta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从类别条件模型生成微多普勒时频图')
    parser.add_argument('--config', dest='config_path',
                        default='config/doppler.yaml', type=str,
                        help='配置文件路径')
    parser.add_argument('--id', dest='id_name', default=None, type=str,
                        help='要生成的用户ID (例如: ID_1, ID_2, ...), 如果未指定将随机选择')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='批量生成的批次大小，默认为10')
    parser.add_argument('--sampler', type=str, default='ddpm', choices=['ddpm', 'ddim'],
                        help='采样器类型: ddpm (默认) 或 ddim (更快)')
    parser.add_argument('--steps', type=int, default=100,
                        help='DDIM采样步数，默认为100，更小的值会更快但可能降低质量')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM随机性参数: 0=确定性, 1=DDPM等效，默认为0')
    args = parser.parse_args()
    infer(args) 
