import os
import torch
import constants

def get_device(idle = False, config=None):
    if idle:
        return "cpu" if config is None else config["idle_device"]

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return torch.device(device)

def build_model_id(model_config, model_name):
    for family, members in model_config.items():
        if model_name in members:
            return family + "/" + model_name
    return None

def build_base_model_path(model_name):
    return os.path.join(constants.MODEL_PATH_BASE, model_name)

def build_fine_tuned_model_path(model_name, dataset_name, method, framework):
    return os.path.join(constants.MODEL_PATH_FINE_TUNED, 
                        model_name,
                        dataset_name,
                        method,
                        framework)

def build_dataset_id_and_path(dataset_config, dataset_name):
    for family, members in dataset_config.items():
        for member, dataset in members.items():
            if dataset_name in dataset:
                data_id     = family + "/" + member
                data_path   = os.path.join(constants.DATASETS_PATH_MASTER, 
                                           member, 
                                           dataset_name)
                return data_id, data_path 
    return None