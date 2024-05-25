r"""Kernel pipeline: main pipeline, initialization, task loading, etc.
"""
import os
import time
from typing import Tuple, Union
import pickle
import torch.nn
from torch.utils.data import DataLoader

from GOOD import config_summoner
from GOOD.data import load_dataset, create_dataloader
from GOOD.kernel.pipeline_manager import load_pipeline
from GOOD.networks.model_manager import load_model
from GOOD.ood_algorithms.ood_manager import load_ood_alg
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.logger import load_logger
from GOOD.definitions import OOM_CODE
from torch_geometric.data import Data, Dataset, InMemoryDataset
from typing import Callable, Optional
import matplotlib.pyplot as plt
import torch
import numpy as np




def initialize_model_dataset(config: Union[CommonArgs, Munch]) -> Tuple[torch.nn.Module, Union[dict, DataLoader]]:
    r"""
    Fix random seeds and initialize a GNN and a dataset. (For project use only)

    Returns:
        A GNN and a data loader.
    """
    # Initial
    reset_random_seed(config)

    print(f'#IN#\n-----------------------------------\n    Task: {config.task}\n'
          f'{time.asctime(time.localtime(time.time()))}')
    # Load dataset
    print(f'#IN#Load Dataset {config.dataset.dataset_name}')
    dataset = load_dataset(config.dataset.dataset_name, config)
    print(f"#D#Dataset: {dataset}")
    print('#D#', dataset['train'][0] if type(dataset) is dict else dataset[0])
    
    train_dataset = InverseDataset(datalist= [0], data_name='cmnist', data_split='train')
    id_val_dataset = InverseDataset(datalist= [0], data_name='cmnist', data_split='id_val')
    test_dataset = InverseDataset(datalist= [0], data_name='cmnist', data_split='test')
    dataset['train'] = train_dataset 
    dataset['id_val'] = id_val_dataset 
    dataset['test'] = test_dataset 
    loader = create_dataloader(dataset, config)

    # Load model
    print('#IN#Loading model...')
    model = load_model(config.model.model_name, config)

    return model, loader




def main():
    args = args_parser()
    config = config_summoner(args)
    load_logger(config)

    model, loader = initialize_model_dataset(config)
    ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

    pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
    pipeline.load_task()

    if config.task == 'train':
        pipeline.task = 'test'
        pipeline.load_task()


def goodtg():
    try:
        main()
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'#E#{e}')
            exit(OOM_CODE)
        else:
            raise e


if __name__ == '__main__':
    main()


class InverseDataset(InMemoryDataset):
    def __init__(self, datalist, data_name, data_split,
                    transform: Optional[Callable] = None,
                    pre_transform: Optional[Callable] = None,
                    pre_filter: Optional[Callable] = None,       
                    ):
        self.data_name = data_name + '_'
        self.data_split = data_split
        self.datalist = datalist
        super(InverseDataset, self).__init__(transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def processed_file_names(self) -> str:
        return self.data_name+ self.data_split + '.pt'

    def process(self):
        data_list = []
        for data in self.datalist:
            data_list.append(data)
        
        torch.save(self.collate(data_list), self.processed_paths[0])


def print_points(data, figure_name):
    x_y = data.pos
    colors = data.x  # RGB colors for each point

    # Convert PyTorch tensors to NumPy arrays
    x_y_np = x_y.cpu().numpy()
    colors_np = colors.cpu().numpy()

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_y_np[:, 0], x_y_np[:, 1], c=colors_np)
    plt.title(figure_name)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.savefig(figure_name + '.png')