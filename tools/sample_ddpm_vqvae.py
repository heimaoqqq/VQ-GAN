import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
from models.unet_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae):
    r"""
    Sample stepwise by going backward one timestep at a time.
    Only save the final decoded images as separate files
    """
    # 修改为生成50张图像
    num_samples = 50
    im_size = dataset_config['im_size'] // 2**sum(autoencoder_model_config['down_sample'])
    xt = torch.randn((num_samples,
                      autoencoder_model_config['z_channels'],
                      im_size,
                      im_size)).to(device)

    # 逐步去噪，但不保存中间结果
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # 只在最后一步保存图像
        if i == 0:
            # Decode the final images
            ims = vae.decode(xt)
            ims = torch.clamp(ims, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2
            
            # 创建输出目录
            if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
                os.mkdir(os.path.join(train_config['task_name'], 'samples'))
            
            # 保存单独的图像文件
            for idx in range(num_samples):
                img = torchvision.transforms.ToPILImage()(ims[idx])
                img.save(os.path.join(train_config['task_name'], 'samples', f'sample_{idx}.png'))
                img.close()


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ldm_ckpt_name'])):
        print('Loaded unet checkpoint')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ldm_ckpt_name']),
                                         map_location=device))
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    
    # Load vae if found
    if os.path.exists(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name'])):
        print('Loaded vae checkpoint')
        vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']),
                                       map_location=device), strict=True)
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    infer(args)
