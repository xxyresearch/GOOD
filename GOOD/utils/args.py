r"""An important module that is used to define all arguments for both argument container and configuration container.
"""
import os
from os.path import join as opj
from typing import List, Union

from tap import Tap
from typing_extensions import Literal

from GOOD.definitions import ROOT_DIR


class TrainArgs(Tap):
    r"""
    Correspond to ``train`` configs in config files.
    """
    tr_ctn: bool = None  #: Flag for training continue.
    ctn_epoch: int = None  #: Start epoch for continue training.
    max_epoch: int = None  #: Max epochs for training stop.
    save_gap: int = None  #: Hard checkpoint saving gap.

    train_bs: int = None  #: Batch size for training.
    val_bs: int = None  #: Batch size for validation.
    test_bs: int = None  #: Batch size for test.
    num_steps: int = None  #: Number of steps in each epoch for node classifications.

    lr: float = None  #: Learning rate.
    mile_stones: List[int] = None  #: Milestones for a scheduler to decrease learning rate: 0.1
    weight_decay: float = None  #: Weight decay.

    alpha = None  #: A parameter for DANN.


class DatasetArgs(Tap):
    r"""
    Correspond to ``dataset`` configs in config files.
    """
    dataset_name: str = None  #: Name of the chosen dataset.
    shift_type: Literal['no_shift', 'covariate', 'concept'] = None  #: The shift type of the chosen dataset.
    domain: str = None  #: Domain selection.
    generate: bool = None  #: The flag for generating GOOD datasets from scratch instead of downloading
    dataset_root: str = None  #: Dataset storage root. Default STORAGE_ROOT/datasets
    dataset_type: str = None  #: Dataset type: molecule, real-world, synthetic, etc. For special usages.

    dim_node: int = None  #: Dimension of node
    dim_edge: int = None  #: Dimension of edge
    num_classes: int = None  #: Number of labels for multi-label classifications.
    num_envs: int = None  #: Number of environments in training set.


class ModelArgs(Tap):
    r"""
    Correspond to ``model`` configs in config files.
    """
    model_name: str = None  #: Name of the chosen GNN.
    model_layer: int = None  #: Number of the GNN layer.
    model_level: Literal['node', 'link', 'graph'] = 'graph'  #: What is the model use for? Node, link, or graph predictions.

    dim_hidden: int = None  #: Node hidden feature's dimension.
    dim_ffn: int = None  #: Final linear layer dimension.
    global_pool: str = None  #: Readout pooling layer type. Currently allowed: 'max', 'mean'.
    dropout_rate: float = None  #: Dropout rate.


class OODArgs(Tap):
    r"""
    Correspond to ``ood`` configs in config files.
    """
    ood_alg: str = None  #: Name of the chosen OOD algorithm.
    ood_param: Union[float, list, dict] = None  #: OOD algorithms' hyperparameter(s). Currently, most of algorithms use it as a float value.


class Auto(Tap):
    allow_datasets: List[str] = None  #: Allow dataset in list to run.
    allow_devices: List[int] = None  #: Devices allowed to run.
    round: int = None  #: Number of experiment rounds.


class CommonArgs(Tap):
    r"""
    Correspond to general configs in config files.
    """
    config_path: str = None  #: (Required) The path for the config file.

    task: Literal['train', 'test'] = None  #: Running mode. Allowed: 'train' and 'test'.
    random_seed: int = None  #: Fixed random seed for reproducibility.
    exp_round: int = None  #: Current experiment round.
    pytest: bool = None

    ckpt_root: str = None  #: Checkpoint root for saving checkpoint files, where inner structure is automatically generated
    ckpt_dir: str = None  #: The direct directory for saving ckpt files
    test_ckpt: str = None  #: Path of the model general test or out-of-domain test checkpoint
    id_test_ckpt: str = None  #: Path of the model in-domain checkpoint
    save_tag: str = None  #: Special save tag for distinguishing special training checkpoints.

    gpu_idx: int = None  #: GPU index.
    device = None  #: Automatically generated by choosing gpu_idx.

    log_file: str = None  #: Log file name.
    log_path: str = None  #: Log file path.

    tensorboard_logdir: str = None  #: Tensorboard logging place.

    # For code auto-complete
    train: TrainArgs = None  #: For code auto-complete
    model: ModelArgs = None  #: For code auto-complete
    dataset: DatasetArgs = None  #: For code auto-complete
    ood: OODArgs = None  #: For code auto-complete

    def __init__(self, argv):
        super(CommonArgs, self).__init__()
        self.argv = argv

        from GOOD.utils.metric import Metric
        from GOOD.utils.train import TrainHelper
        self.metric: Metric = None
        self.train_helper: TrainHelper = None

    def process_args(self) -> None:
        super().process_args()
        if self.config_path is None:
            raise AttributeError('Please provide command argument --config_path.')
        if not os.path.isabs(self.config_path):
            self.config_path = opj(ROOT_DIR, 'configs', self.config_path)

        self.dataset = DatasetArgs().parse_args(args=self.argv, known_only=True)
        self.train = TrainArgs().parse_args(args=self.argv, known_only=True)
        self.model = ModelArgs().parse_args(args=self.argv, known_only=True)
        self.ood = OODArgs().parse_args(args=self.argv, known_only=True)


def args_parser(argv: list=None):
    r"""
    Arguments parser.

    Args:
        argv: Input arguments. *e.g.*, ['--config_path', config_path,
            '--ckpt_root', os.path.join(STORAGE_DIR, 'reproduce_ckpts'),
            '--exp_round', '1']

    Returns:
        General arguments

    """
    common_args = CommonArgs(argv=argv).parse_args(args=argv, known_only=True)
    return common_args
