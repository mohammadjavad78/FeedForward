
import yaml

# Load config file
def Getyaml():
    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config_dict={}

    # Access hyperparameters
    config_dict['learning_rate'] = config['learning_rate']
    config_dict['batch_size'] = config['batch_size']
    config_dict['num_epochs'] = config['num_epochs']
    config_dict['dataset_path'] = config['dataset']['path']
    config_dict['dataset_num_classes'] = config['dataset']['num_classes']
    config_dict['img_height'] = config['dataset']['img_height']
    config_dict['img_width'] = config['dataset']['img_width']
    config_dict['train_split'] = config['dataset']['train_split']
    config_dict['model_name'] = config['model']['name']
    config_dict['pretrained'] = config['model']['pretrained']
    config_dict['model_num_classes'] = config['model']['num_classes']
    config_dict['optimizer_name'] = config['optimizer']['name']
    config_dict['weight_decay'] = config['optimizer']['weight_decay']
    config_dict['scheduler_name'] = config['scheduler']['name']
    config_dict['step_size'] = config['scheduler']['step_size']
    config_dict['gamma'] = config['scheduler']['gamma']

    return config_dict
    