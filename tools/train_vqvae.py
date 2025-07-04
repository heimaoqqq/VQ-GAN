import os
import sys
import numpy as np
import torch
import random
import torchvision
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.vqvae import VQVAE
from models.lpips import LPIPS
from models.discriminator import Discriminator
from torchvision.utils import make_grid
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from dataset.doppler_dataset import DopplerDataset
from tqdm import tqdm
from utils.config_utils import read_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 保留save_latents函数但不在train函数中调用，用户可以单独使用infer_vqvae.py
def save_latents(model, dataset_config, train_config, device):
    """保存所有训练和验证数据的潜在表示"""
    # 创建保存目录
    latent_dir = os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name'])
    if not os.path.exists(latent_dir):
        os.makedirs(latent_dir)
        
    # 加载训练数据集
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'doppler': DopplerDataset,
    }.get(dataset_config['name'])
    
    for split in ['train', 'val']:
        print(f"处理{split}数据集的潜在表示...")
        im_dataset = im_dataset_cls(split=split,
                                    im_path=dataset_config['im_path'],
                                    im_size=dataset_config['im_size'],
                                    im_channels=dataset_config['im_channels'])
        
        data_loader = DataLoader(im_dataset,
                                batch_size=16,  # 使用较大的批量加速处理
                                shuffle=False)  # 保持顺序
        
        all_latents = []
        all_indices = []
        
        with torch.no_grad():
            for batch_idx, im in enumerate(tqdm(data_loader)):
                im = im.float().to(device)
                # 获取潜在表示
                _, z, _ = model(im)
                z = z.cpu().numpy()
                all_latents.append(z)
                
                # 生成索引
                batch_size = im.shape[0]
                start_idx = batch_idx * data_loader.batch_size
                indices = list(range(start_idx, start_idx + batch_size))
                all_indices.extend(indices)
        
        # 合并所有批次的潜在表示
        all_latents = np.concatenate(all_latents, axis=0)
        
        # 保存潜在表示
        for idx, latent in zip(all_indices, all_latents):
            latent_path = os.path.join(latent_dir, f"{split}_{idx}.npy")
            np.save(latent_path, latent)
    
    print(f"潜在表示已保存到 {latent_dir}")

def train(args):
    # Read the config file #
    try:
        config = read_config(args.config_path)
    except Exception as exc:
        if hasattr(exc, 'message'):
            print(exc.message)
        else:
            print(exc)
    print(config)
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # 设置use_amp为False，表明不使用混合精度
    use_amp = False
    print("未启用混合精度训练")
    
    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################
    
    # Create the model and dataset #
    model = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)
    # Create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'doppler': DopplerDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=True)
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        
    num_epochs = train_config['autoencoder_epochs']

    # 设置感知损失控制参数
    perceptual_start_epoch = 10  # 从第10轮开始使用感知损失
    use_perceptual_loss = False  # 初始不使用感知损失
    print(f"将在第 {perceptual_start_epoch} 轮后开始使用感知损失")

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # Disc Loss can even be BCEWithLogits
    disc_criterion = torch.nn.MSELoss()
    
    # No need to freeze lpips as lpips.py takes care of that
    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)
    
    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    
    disc_step_start = train_config['disc_start']
    step_count = 0
    
    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0
    
    for epoch_idx in range(num_epochs):
        # 检查是否开始使用感知损失
        if epoch_idx == perceptual_start_epoch:
            use_perceptual_loss = True
            print(f'开始使用感知损失，从第 {epoch_idx+1} 轮开始')
            
        recon_losses = []
        codebook_losses = []
        #commitment_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []
        
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        
        for im in tqdm(data_loader):
            step_count += 1
            im = im.float().to(device)
            
            # Fetch autoencoders output(reconstructions)
            model_output = model(im)
            output, z, quantize_losses = model_output
            
            # Image Saving Logic
            if step_count % image_save_steps == 0 or step_count == 1:
                sample_size = min(8, im.shape[0])
                save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                save_output = ((save_output + 1) / 2)
                save_input = ((im[:sample_size] + 1) / 2).detach().cpu()
                
                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid)
                if not os.path.exists(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples')):
                    os.mkdir(os.path.join(train_config['task_name'], 'vqvae_autoencoder_samples'))
                img.save(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples',
                                      'current_autoencoder_sample_{}.png'.format(img_save_count)))
                img_save_count += 1
                img.close()
            
            ######### Optimize Generator ##########
            # L2 Loss
            recon_loss = recon_criterion(output, im) 
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            g_loss = (recon_loss +
                      (train_config['codebook_weight'] * quantize_losses['codebook_loss'] / acc_steps) +
                      (train_config['commitment_beta'] * quantize_losses['commitment_loss'] / acc_steps))
            codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())
            
            # Adversarial loss only if disc_step_start steps passed
            if step_count > disc_step_start:
                disc_fake_pred = discriminator(model_output[0])
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.ones(disc_fake_pred.shape,
                                                          device=disc_fake_pred.device))
                gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
                g_loss += train_config['disc_weight'] * disc_fake_loss / acc_steps
            
            # 只在指定轮次后添加感知损失
            if use_perceptual_loss:
                lpips_loss = torch.mean(lpips_model(output, im))
                perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
                g_loss += train_config['perceptual_weight'] * lpips_loss / acc_steps
            else:
                # 不使用感知损失时，记录0以便打印
                perceptual_losses.append(0)
                
            losses.append(g_loss.item())
            
            # 直接进行反向传播，不使用scaler
            g_loss.backward()
            
            # Optimize Discriminator（删除autocast上下文）
            if step_count > disc_step_start:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(im)
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.zeros(disc_fake_pred.shape,
                                                            device=disc_fake_pred.device))
                disc_real_loss = disc_criterion(disc_real_pred,
                                                torch.ones(disc_real_pred.shape,
                                                          device=disc_real_pred.device))
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps
            
                # 直接进行反向传播，不使用scaler
                disc_loss.backward()
                
                if step_count % acc_steps == 0:
                    # 更新判别器参数，直接使用optimizer.step()
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            
            if step_count % acc_steps == 0:
                # 更新生成器参数，直接使用optimizer.step()
                optimizer_g.step()
                optimizer_g.zero_grad()
                
        # 确保最后一批次的优化步骤完成
        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()
        
        # 输出当前是否使用感知损失
        perceptual_status = "启用" if use_perceptual_loss else "禁用"
        
        if len(disc_losses) > 0:
            print(
                f'完成轮次: {epoch_idx + 1} | 感知损失状态: {perceptual_status} | 重建损失: {np.mean(recon_losses):.4f} | '
                f'感知损失: {np.mean(perceptual_losses):.4f} | 码书损失: {np.mean(codebook_losses):.4f} | '
                f'生成器损失: {np.mean(gen_losses):.4f} | 判别器损失: {np.mean(disc_losses):.4f}')
        else:
            print(
                f'完成轮次: {epoch_idx + 1} | 感知损失状态: {perceptual_status} | 重建损失: {np.mean(recon_losses):.4f} | '
                f'感知损失: {np.mean(perceptual_losses):.4f} | 码书损失: {np.mean(codebook_losses):.4f}')

        # Save checkpoint after every epoch
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                     train_config['vqvae_autoencoder_ckpt_name']))
        torch.save(discriminator.state_dict(), os.path.join(train_config['task_name'],
                                                             train_config['vqvae_discriminator_ckpt_name']))

    # 移除自动保存潜在表示的代码，用户应该使用infer_vqvae.py生成
    print('训练完成...')
    print('请使用 python -m tools.infer_vqvae --config config/doppler.yaml 生成潜在表示')
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for vqvae training')
    parser.add_argument('--config', dest='config_path', type=str)
    args = parser.parse_args()
    train(args)
