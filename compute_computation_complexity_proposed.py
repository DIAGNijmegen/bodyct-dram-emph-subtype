import sys
from argparse import ArgumentParser
import logging
import pytorch_lightning

from pytorch_lightning.strategies import DDPStrategy
from models import SubtypeDataModule, ScanCLSLightningModule, ScanRegLightningModule



def run_measurement_job():
    parser = ArgumentParser()
    parser.add_argument("--model_arch", default="med3ddram18", type=str)
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--ckp", type=str)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--lr", "--learning-rate", default=0.0001, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--num_samples", default=32, type=int)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--target_size", default=(128, 224, 288), type=tuple)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")
    parser.add_argument("--data_path", default=r"D:\workspace\datasets\COPDGene_cache/", type=str)
    parser.add_argument("--train_csv",
                        default=r"D:\workspace\datasets\COPDGene_cache/merged.csv", type=str)
    parser.add_argument("--valid_csv",
                        default=r"D:\workspace\datasets\COPDGene_cache/merged.csv", type=str)
    parser.add_argument("--test_csv",
                        default=r"D:\workspace\datasets\COPDGene_cache/merged.csv", type=str)
    parser.add_argument("--model_path",
                        default=r"D:\workspace\models/",
                        type=str)
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)

    parser.set_defaults(
        replace_sampler_ddp=False,
        accelerator="gpu",
    )
    args = parser.parse_args()
    module = ScanRegLightningModule(args)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    pytorch_lightning.trainer.seed_everything()
    ddp = DDPStrategy(process_group_backend="gloo", find_unused_parameters=False)
    data_module = SubtypeDataModule(args)

    trainer = pytorch_lightning.Trainer.from_argparse_args(args, strategy=ddp,
                                                           sync_batchnorm=True,
                                                           resume_from_checkpoint=None,
                                                           devices=args.ngpus)

    trainer.fit(module, data_module)


if __name__ == "__main__":
    run_measurement_job()
