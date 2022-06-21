import os
from pathlib import Path

import gdown
import pytest
from torch_geometric.data import extract_zip

from GOOD import config_summoner, args_parser
from GOOD.definitions import ROOT_DIR, STORAGE_DIR
from GOOD.kernel.pipeline import initialize_model_dataset, load_ood_alg, load_logger, config_model
from GOOD.kernel.evaluation import evaluate


class Reproducer(object):
    def __init__(self, config_path):
        self.args = args_parser(['--config_path', config_path,
                                 '--ckpt_root', os.path.join(STORAGE_DIR, 'reproduce_ckpts'),
                                 '--exp_round', '1'])
        self.config = config_summoner(self.args)
        self.download()

    def download(self):
        if os.path.exists(self.config.ckpt_root):
            return
        os.makedirs(self.config.ckpt_root)
        path = gdown.download("https://drive.google.com/file/d/17FfHYCP0-wwUILPD-PczwjjrYQHKxU-l/view?usp=sharing",
                              output=os.path.join(self.config.ckpt_root, 'round1.zip'), fuzzy=True)
        extract_zip(path, self.config.ckpt_root)
        os.unlink(path)

    def __call__(self, *args, **kwargs):
        config = self.config
        load_logger(config, sub_print=False)
        model, loader = initialize_model_dataset(config)
        ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

        test_score, test_loss = config_model(model, 'test', config, load_param=True)
        test_stat = evaluate(model, loader, ood_algorithm, 'test', config)
        return test_score, test_loss.cpu().numpy(), test_stat['score'], test_stat['loss'].cpu().numpy()


config_paths = []
config_root = Path(ROOT_DIR, 'configs', 'GOOD_configs')
for dataset_path in config_root.iterdir():
    if not dataset_path.is_dir():
        continue
    for domain_path in dataset_path.iterdir():
        if not domain_path.is_dir():
            continue
        for shift_path in domain_path.iterdir():
            if not shift_path.is_dir():
                continue
            for ood_config_path in shift_path.iterdir():
                if 'base' in ood_config_path.name:
                    continue
                config_paths.append(str(ood_config_path))


@pytest.mark.parametrize("config_path", config_paths)
def test_reproduce(config_path):
    reproducer = Reproducer(config_path)
    saved_score, saved_loss, run_score, run_loss = reproducer()
    assert run_score == pytest.approx(saved_score, 1e-3) or run_loss == pytest.approx(saved_loss, 1e-3)
