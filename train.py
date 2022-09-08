import sys
import os
from argparse import ArgumentParser
import logging
import pytorch_lightning
import torch

workspace_path = '/mnt/netcache/bodyct/experiments/airway_labeling_t7828/bronchial'
print("We add {} into python path for module lookup.".format(workspace_path))
sys.path.append(workspace_path)
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from models import SubtypeDataModule, ScanCLSLightningModule, ScanRegLightningModule
from pathlib import Path
from utils import get_loggers, load_state_dict_greedy
import glob


def run_training_job():
    parser = ArgumentParser()

    parser.add_argument("--model_arch", default="med3ddram", type=str)
    parser.add_argument("--lr", "--learning-rate", default=0.00005, type=float)
    parser.add_argument("--ngpus", default=2, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--reload_only_weights", default=1, type=int)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--ckp", type=str, default=None)
    # Data parameters.
    parser.add_argument("--data_path", default="/share/cached/COPDGene/", type=str)
    parser.add_argument("--train_csv",
                        default="/mnt/netcache/bodyct/experiments/emphysema_subtyping_t8610/steve/tr.csv", type=str)
    parser.add_argument("--valid_csv",
                        default="/mnt/netcache/bodyct/experiments/emphysema_subtyping_t8610/steve/val.csv", type=str)
    parser.add_argument("--test_csv",
                        default="/mnt/netcache/bodyct/experiments/emphysema_subtyping_t8610/steve/te.csv", type=str)
    parser.add_argument("--model_path",
                        default="/mnt/netcache/bodyct/experiments/emphysema_subtyping_t8610/lightning_models/",
                        type=str)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_samples", default=128, type=int)
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)

    parser.set_defaults(
        max_epochs=300,
        replace_sampler_ddp=False,
        accelerator="gpu",
        log_every_n_steps=5
    )
    args = parser.parse_args()
    args.exp_name = f"subtyping_{args.model_arch}"
    exp_path = Path(args.model_path + f"/{args.exp_name}/")
    exp_path.mkdir(exist_ok=True, parents=True)
    ckp_path = exp_path / "checkpoints"
    ckp_path.mkdir(exist_ok=True, parents=True)
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
    data_module = SubtypeDataModule(args)
    list_of_files = list(glob.glob(ckp_path.as_posix() + '/*.ckpt')) + list(glob.glob(ckp_path.as_posix() + '/*.pth'))
    if len(list_of_files) != 0:
        if args.ckp is not None:
            resume_from_checkpoint = exp_path / "checkpoints" / args.ckp
        else:
            resume_from_checkpoint = max(list_of_files, key=os.path.getctime)
        if args.reload_only_weights:
            checkpoint = torch.load(resume_from_checkpoint, map_location='cpu')
            if 'state_dict' in checkpoint.keys():
                checkpoint = checkpoint['state_dict']

            load_state_dict_greedy(module, checkpoint)
            resume_from_checkpoint = None
    else:
        resume_from_checkpoint = None
    callbacks = [ModelCheckpoint(
        save_on_train_epoch_end=True,
        save_top_k=-1,
        every_n_epochs=1,
        monitor='train_loss',
        dirpath=ckp_path,
        filename='{epoch:02d}'
    )]
    trainer = pytorch_lightning.Trainer.from_argparse_args(args, strategy=ddp,
                                                           sync_batchnorm=True,
                                                           callbacks=callbacks,
                                                           devices=args.ngpus,
                                                           resume_from_checkpoint=resume_from_checkpoint,
                                                           logger=get_loggers(exp_path),
                                                           default_root_dir=exp_path)
    trainer.fit(module, data_module)
    trainer.test(model=module, datamodule=data_module, ckpt_path='best')


if __name__ == "__main__":
    print("Running training job.")
    run_training_job()
