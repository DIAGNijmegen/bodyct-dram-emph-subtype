import sys
import os
import logging
import matplotlib
from argparse import ArgumentParser
import pytorch_lightning
import torch
from utils import load_state_dict_greedy
from pytorch_lightning.strategies import DDPStrategy
import numpy as np
from functools import reduce
import json

matplotlib.use('Agg')
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("/output/process.log"),
#         logging.StreamHandler()
#     ]
# )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("D:/workspace/process.log"),
        logging.StreamHandler()
    ]
)

from models import ScanRegLightningModule, SubtypeDataModule
from dataset import COPDGeneSubtyping
from utils import write_array_to_mha_itk, windowing, draw_mask_tile_singleview_heatmap
from pytorch_lightning.trainer.states import RunningStage


def ratio_to_label(ratio, ratio_mapping):
    inv_ratio_map = {v: k for k, v in ratio_mapping.items()}
    label = [inv_ratio_map[k] for k in inv_ratio_map.keys()
             if k[0] <= ratio and ratio < k[1]][0]
    return label


def run_testing_job():


    input_image_path = r'D:\workspace\datasets\COPDGene\images/'
    input_lobe_path = r'D:\workspace\datasets\COPDGene\lobes/'
    output_path = r'D:\workspace\datasets\COPDGene\outputs/'

    ckp_path = r'D:\newckp.ckpt'

    # input_image_path = '/input/images/ct/'
    # input_lobe_path = '/input/images/pulmonary-lobes/'
    # output_path = '/output/images/'
    # ckp_path = 'newckp.ckpt'
    parser = ArgumentParser()
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--model_arch", default="med3ddram", type=str)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--target_size", default=(128, 224, 288), type=tuple)
    parser.add_argument("--scan_path", default=input_image_path, type=str)
    parser.add_argument("--lobe_path", default=input_lobe_path, type=str)
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)

    parser.set_defaults(
        replace_sampler_ddp=False,
        accelerator="gpu",
    )
    args = parser.parse_args()

    module = ScanRegLightningModule(args)

    checkpoint = torch.load(ckp_path, map_location='cpu')


    load_state_dict_greedy(module, checkpoint['state_dict'])
    data_module = SubtypeDataModule(args)
    ddp = DDPStrategy(process_group_backend="gloo", find_unused_parameters=False)
    trainer = pytorch_lightning.Trainer.from_argparse_args(args, strategy=ddp,
                                                           sync_batchnorm=True,
                                                           resume_from_checkpoint=None,
                                                           devices=args.ngpus,
                                                           default_root_dir=output_path)
    logging.info("starting the inference.")
    predictions = trainer.predict(module, data_module)

    # build the output
    logging.info("building the output.")
    # merge batches
    cle_dense_outs = torch.cat([out['cle_dense_outs'] for out in predictions])
    pse_dense_outs = torch.cat([out['pse_dense_outs'] for out in predictions])
    cle_precentages = torch.cat([out['cle_precentages'] for out in predictions])
    pse_precentages = torch.cat([out['pse_precentages'] for out in predictions])
    crop_slices = torch.cat([out['crop_slices'] for out in predictions])
    original_sizes = torch.cat([out['original_size'] for out in predictions])
    uids = reduce(lambda x, y: x + y, [out['uids'] for out in predictions])
    results = []

    for cle_dense_out, pse_dense_out, cle_precentage, pse_precentage, crop_slice, original_size, uid, \
            in zip(cle_dense_outs, pse_dense_outs, cle_precentages, pse_precentages, crop_slices, original_sizes, uids):
        error_messages = []
        metrics = {}
        recon_size = tuple(s[1].item() - s[0].item() for s in crop_slice)
        original_size = tuple(original_size.cpu().numpy())
        cle_dense_out = torch.nn.functional.interpolate(cle_dense_out.unsqueeze(0),
                                                        size=recon_size, mode='trilinear', align_corners=True)
        cle_dense_out_np = cle_dense_out.squeeze(0).squeeze(0).cpu().numpy()
        full_cle = np.zeros(original_size)
        full_cle[tuple([slice(s[0].item(), s[1].item()) for s in crop_slice])] = cle_dense_out_np

        pse_dense_out = torch.nn.functional.interpolate(pse_dense_out.unsqueeze(0),
                                                        size=recon_size, mode='trilinear', align_corners=True)
        pse_dense_out_np = pse_dense_out.squeeze(0).squeeze(0).cpu().numpy()
        full_pse = np.zeros(original_size)
        full_pse[tuple([slice(s[0].item(), s[1].item()) for s in crop_slice])] = pse_dense_out_np
        metrics['runtime_seconds'] = 1.0
        metrics['cle_severity_score'] = ratio_to_label(cle_precentage.item(), COPDGeneSubtyping.cle_ratio_map)
        metrics['cle_lesion_percentage_per_lung'] = cle_precentage.item()

        metrics['pse_severity_score'] = ratio_to_label(pse_precentage.item(), COPDGeneSubtyping.pse_ratio_map)
        metrics['pse_lesion_percentage_per_lung'] = pse_precentage.item()

        results.append({
            'entity': uid,
            'metrics': metrics,
            'error_messages': error_messages
        })
        full_cle_w = windowing(full_cle, from_span=(0, 1)).astype(np.uint8)
        scan_meta = data_module.datasets[RunningStage.PREDICTING].scan_meta_cache[uid]

        write_array_to_mha_itk(output_path, [full_cle_w],
                               [uid], type=np.uint8,
                               origin=scan_meta["origin"][::-1],
                               direction=np.asarray(scan_meta["direction"]).reshape(3, 3)[
                                         ::-1].flatten().tolist(),
                               spacing=scan_meta["spacing"][::-1])
        full_pse_w = windowing(full_pse, from_span=(0, 1)).astype(np.uint8)
        write_array_to_mha_itk(output_path, [full_pse_w],
                               [uid + '_pse'], type=np.uint8,
                               origin=scan_meta["origin"][::-1],
                               direction=np.asarray(scan_meta["direction"]).reshape(3, 3)[
                                         ::-1].flatten().tolist(),
                               spacing=scan_meta["spacing"][::-1])


    json_path = os.path.join(output_path, 'results.json')
    with open(json_path, 'w') as f:
        print('results:', results)
        j = json.dumps(results)
        f.write(j)


if __name__ == "__main__":
    print("Docker start running testing job.")
    run_testing_job()