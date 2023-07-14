import sys
from argparse import ArgumentParser
import logging
import pytorch_lightning
import torch

workspace_path = '/mnt/netcache/bodyct/experiments/airway_labeling_t7828/bronchial'
print("We add {} into python path for module lookup.".format(workspace_path))
sys.path.append(workspace_path)

from pytorch_lightning.strategies import DDPStrategy
from models import SubtypeDataModule, ScanCLSLightningModule, ScanRegLightningModule
from pathlib import Path
from utils import get_loggers, load_state_dict_greedy



def run_testing_job():
    parser = ArgumentParser()
    parser.add_argument("--model_arch", default="med3d", type=str)
    parser.add_argument("--ngpus", default=2, type=int)
    parser.add_argument("--ckp", type=str)
    # Data parameters.
    parser.add_argument("--data_path", default="/share/cached/COPDGene/", type=str)
    parser.add_argument("--train_csv",
                        default="/mnt/netcache/bodyct/experiments/emphysema_subtyping_t8610/steve/tr.csv", type=str)
    parser.add_argument("--valid_csv",
                        default="/mnt/netcache/bodyct/experiments/emphysema_subtyping_t8610/steve/val.csv", type=str)
    parser.add_argument("--test_csv",
                        default="/mnt/netcache/bodyct/experiments/emphysema_subtyping_t8610/steve/te_duplicated_lastone.csv", type=str)
    parser.add_argument("--model_path",
                        default="/mnt/netcache/bodyct/experiments/emphysema_subtyping_t8610/lightning_models/",
                        type=str)
    parser.add_argument("--target_size", default=(128, 224, 288), type=tuple)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)

    parser.set_defaults(
        replace_sampler_ddp=False,
        accelerator="gpu",
    )
    args = parser.parse_args()
    args.exp_name = f"subtyping_{args.model_arch}"
    exp_path = Path(args.model_path + f"/{args.exp_name}/")
    ckp_path = exp_path / "checkpoints" / f"epoch={args.ckp}.ckpt"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{exp_path}/debug.log"),
            logging.StreamHandler()
        ]
    )

    pytorch_lightning.trainer.seed_everything()
    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=False)

    if 'dram' in args.model_arch:
        module = ScanRegLightningModule(args)
    else:
        module = ScanCLSLightningModule(args)
    checkpoint = torch.load(ckp_path, map_location='cpu')
    if 'epoch' in checkpoint.keys():
        module.epoch_number = checkpoint['epoch']
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']

    load_state_dict_greedy(module, checkpoint)
    data_module = SubtypeDataModule(args)

    trainer = pytorch_lightning.Trainer.from_argparse_args(args, strategy=ddp,
                                                           sync_batchnorm=True,
                                                           resume_from_checkpoint=None,
                                                           logger=get_loggers(exp_path),
                                                           devices=args.ngpus,
                                                           default_root_dir=exp_path)

    trainer.test(module, data_module)


if __name__ == "__main__":
    print("Running training job.")
    run_testing_job()
