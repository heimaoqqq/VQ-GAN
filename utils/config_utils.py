import yaml

def read_config(config_path):
    """
    读取YAML配置文件并返回配置字典
    
    参数:
    config_path: YAML配置文件的路径
    
    返回:
    配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            raise exc

def validate_class_config(condition_config):
    assert 'class_condition_config' in condition_config, \
        "Class conditioning desired but class condition config missing"
    assert 'num_classes' in condition_config['class_condition_config'], \
        "num_class missing in class condition config"


def validate_text_config(condition_config):
    assert 'text_condition_config' in condition_config, \
        "Text conditioning desired but text condition config missing"
    assert 'text_embed_dim' in condition_config['text_condition_config'], \
        "text_embed_dim missing in text condition config"
    

def validate_image_config(condition_config):
    assert 'image_condition_config' in condition_config, \
        "Image conditioning desired but image condition config missing"
    assert 'image_condition_input_channels' in condition_config['image_condition_config'], \
        "image_condition_input_channels missing in image condition config"
    assert 'image_condition_output_channels' in condition_config['image_condition_config'], \
        "image_condition_output_channels missing in image condition config"
    

def validate_image_conditional_input(cond_input, x):
    assert 'image' in cond_input, \
        "Model initialized with image conditioning but cond_input has no image information"
    assert cond_input['image'].shape[0] == x.shape[0], \
        "Batch size mismatch of image condition and input"
    assert cond_input['image'].shape[2] % x.shape[2] == 0, \
        "Height/Width of image condition must be divisible by latent input"


def validate_class_conditional_input(cond_input, x, num_classes):
    assert 'class' in cond_input, \
        "Model initialized with class conditioning but cond_input has no class information"
    assert cond_input['class'].shape == (x.shape[0], num_classes), \
        "Shape of class condition input must match (Batch Size, )"
def get_config_value(config, key, default_value):
    return config[key] if key in config else default_value