import glob
import os
import random
import torchvision
from PIL import Image
from tqdm import tqdm
import re
from utils.diffusion_utils import load_latents
from torch.utils.data.dataset import Dataset

class DopplerDataset(Dataset):
    def __init__(self, split, im_path, im_size, im_channels,
                 use_latents=False, latent_path=None, condition_config=None,
                 train_ratio=0.8, seed=42):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.latent_maps = None
        self.use_latents = False
        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        self.image_to_class = {}
        
        # 加载所有图像路径
        all_images = self.load_nested_images(im_path)
        
        # 划分数据集(8:2)
        self.images = self.split_dataset(all_images, train_ratio, seed)
        
        # 加载潜在表示
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) > 0:
                self.use_latents = True
                self.latent_maps = latent_maps
                print(f'找到{len(self.latent_maps)}个潜在表示')
            else:
                print('未找到潜在表示')
    
    def load_nested_images(self, dataset_root):
        assert os.path.exists(dataset_root), f"数据集目录{dataset_root}不存在"
        all_images = []
        
        # 获取所有ID子目录
        id_folders = [f for f in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, f)) and f.startswith('ID_')]
        
        # 按ID的数值大小排序文件夹，而不是字符串排序
        def extract_id_number(folder_name):
            # 从"ID_X"中提取X，并转换为整数
            match = re.search(r'ID_(\d+)', folder_name)
            if match:
                return int(match.group(1))
            return 0  # 默认值，如果无法提取数字
        
        id_folders.sort(key=extract_id_number)
        
        print("文件夹按数值排序:")
        for i, folder in enumerate(id_folders):
            print(f"{folder} -> 类别索引 {i}")
        
        # 构建ID到类别索引的映射
        id_to_class = {folder: idx for idx, folder in enumerate(id_folders)}
        
        # 构建ID名称到对应数字的映射，用于在采样时验证
        id_name_to_number = {folder: extract_id_number(folder) for folder in id_folders}
        
        for id_folder in tqdm(id_folders):
            folder_path = os.path.join(dataset_root, id_folder)
            
            # 支持多种图像格式
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_paths = glob.glob(os.path.join(folder_path, ext))
                
                for img_path in image_paths:
                    self.image_to_class[img_path] = id_to_class[id_folder]
                    all_images.append(img_path)
        
        print(f'共找到{len(all_images)}张图像，来自{len(id_folders)}个用户ID')
        
        # 保存类别映射信息
        os.makedirs(os.path.join('./metadata'), exist_ok=True)
        import json
        with open(os.path.join('./metadata', 'class_mapping.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'id_to_class': id_to_class,
                'id_name_to_number': id_name_to_number,
                'num_classes': len(id_folders)
            }, f, ensure_ascii=False)
        
        return all_images
    
    def split_dataset(self, all_images, train_ratio, seed):
        random.seed(seed)
        random.shuffle(all_images)
        
        train_size = int(len(all_images) * train_ratio)
        
        if self.split == 'train':
            images = all_images[:train_size]
        else:  # val
            images = all_images[train_size:]
            
        print(f'{self.split}集包含{len(images)}张图像')
        
        if self.split == 'train':
            split_info = {
                'train': all_images[:train_size],
                'val': all_images[train_size:]
            }
            
            # 保存分割信息
            os.makedirs(os.path.join('./metadata'), exist_ok=True)
            import pickle
            with open(os.path.join('./metadata', 'split_info.pkl'), 'wb') as f:
                pickle.dump(split_info, f)
            
            print(f"数据集划分完成: 训练集={len(split_info['train'])}张, 验证集={len(split_info['val'])}张")
            
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = self.images[index]
        
        # 准备条件输入
        cond_input = {}
        
        # 如果需要类别条件，添加到条件输入中
        if 'class' in self.condition_types:
            cond_input['class'] = self.image_to_class[image_path]
        
        # 获取图像或潜在表示
        if self.use_latents:
            latent = self.latent_maps[image_path]
            # 根据是否需要条件输入返回不同格式的数据
            if len(self.condition_types) > 0:
                return latent, cond_input
            else:
                return latent
        else:
            im = Image.open(image_path)
            im = im.resize((self.im_size, self.im_size))
            im_tensor = torchvision.transforms.ToTensor()(im)
            
            # 转换为-1到1范围
            im_tensor = (2 * im_tensor) - 1
            
            # 根据是否需要条件输入返回不同格式的数据
            if len(self.condition_types) > 0:
                return im_tensor, cond_input
            else:
                return im_tensor 