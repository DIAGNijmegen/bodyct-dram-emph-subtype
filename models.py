import logging
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from torchvision.transforms import (
    Compose,
)
from pytorch_lightning.loggers import TensorBoardLogger
from utils import get_model_by_name
import torch.nn.functional as F
from dataset import COPDGeneSubtyping, SubtypingInference
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DistributedSampler
from metrics import BinaryDice, BinaryCrossEntropy
from utils import extract_logger, plot_to_numpy_array, save_image, cat_all_gather
from confusion_matrix import plot_confusion_matrix_from_data
from sampler import DistributedSamplerWrapper
from data_sampler import SubtypingStratifiedSampler
from utils import windowing, draw_mask_tile_singleview_heatmap
from spatial_transforms import CropAndResize, Flip, Interpolate
from base import NumpyToTensor
from intensity_transforms import IntensityWindow, Standardize, GaussianAddictive, BoxMaskOut
import torch.distributed as dist
from pathlib import Path
from sklearn.metrics import confusion_matrix
from pytorch_lightning.trainer.states import RunningStage
from ptflops import get_model_complexity_info

TRAIN_PHASE = RunningStage.TRAINING
VALID_PHASE = RunningStage.VALIDATING
TEST_PHASE = RunningStage.TESTING
PREDICT_PHASE = RunningStage.PREDICTING

_DATASET_CLASS = COPDGeneSubtyping  # for development


class SubtypeDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.datasets = {
            TRAIN_PHASE: TRAIN_PHASE,
            VALID_PHASE: VALID_PHASE,
            TEST_PHASE: TEST_PHASE,
            PREDICT_PHASE: PREDICT_PHASE
        }

    def _make_transforms(self, mode: str):
        transform = [
            self._transform(mode),
        ]

        return Compose(transform)

    def _transform(self, mode: str):
        args = self.args
        return Compose(
            [
                NumpyToTensor(),
                IntensityWindow(from_span=(-1150, -300), to_span=(0, 1), output_dtype=torch.float32),
                Standardize(),
                Interpolate(args.target_size, None, align_corners=True),
            ]
            + (
                [
                    GaussianAddictive(p=0.5, always_apply=False),
                    BoxMaskOut(p=0.5, always_apply=False, n_masks=(1, 10)),
                    # GaussianSmooth(p=0.5, always_apply=False, sigma=(0.3, 0.6)),
                    Flip(0.5, False, dim=(1, 3)),
                    CropAndResize(0.5, False,
                                  (0.45, 0.55),
                                  (0.95, 1.0),
                                  align_corners=True
                                  )
                ]
                if mode == TRAIN_PHASE
                else [
                ]
            )
        )

    def predict_dataloader(self):
        train_transform = self._make_transforms(mode=TEST_PHASE)
        self.datasets[PREDICT_PHASE] = SubtypingInference(
            scan_path=self.args.scan_path,
            lobe_path=self.args.lobe_path,
            transforms=train_transform,
        )

        return torch.utils.data.DataLoader(
            self.datasets[PREDICT_PHASE],
            sampler=DistributedSampler(self.datasets[PREDICT_PHASE], shuffle=False),
            drop_last=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )


    def train_dataloader(self):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer trains/tests with.
        """
        train_transform = self._make_transforms(mode=TRAIN_PHASE)
        self.datasets[TRAIN_PHASE] = _DATASET_CLASS(
            archive_path=self.args.data_path,
            series_uids=_DATASET_CLASS.get_series_uids(self.args.train_csv),
            transforms=train_transform,
        )
        train_sampler = SubtypingStratifiedSampler(self.datasets[TRAIN_PHASE], self.args.num_samples)
        self.datasets[TRAIN_PHASE].cle_statistics = train_sampler.cle_statistics
        self.datasets[TRAIN_PHASE].cle_class_weights = train_sampler.cle_class_weights

        self.datasets[TRAIN_PHASE].pse_statistics = train_sampler.pse_statistics
        self.datasets[TRAIN_PHASE].pse_class_weights = train_sampler.pse_class_weights

        return torch.utils.data.DataLoader(
            self.datasets[TRAIN_PHASE],
            sampler=DistributedSamplerWrapper(train_sampler, shuffle=True),
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            # pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        val_transform = self._make_transforms(mode=VALID_PHASE)
        self.datasets[VALID_PHASE] = _DATASET_CLASS(
            archive_path=self.args.data_path,
            series_uids=_DATASET_CLASS.get_series_uids(self.args.valid_csv),
            transforms=val_transform,
        )

        return torch.utils.data.DataLoader(
            self.datasets[VALID_PHASE],
            sampler=DistributedSampler(self.datasets[VALID_PHASE], shuffle=False),
            drop_last=False,
            batch_size=self.args.batch_size,
            # pin_memory=True,
            num_workers=self.args.workers,
        )

    def test_dataloader(self):
        val_transform = self._make_transforms(mode=TEST_PHASE)
        self.datasets[TEST_PHASE] = _DATASET_CLASS(
            archive_path=self.args.data_path,
            series_uids=_DATASET_CLASS.get_series_uids(self.args.test_csv),
            transforms=val_transform,
        )
        logging.info(f"dataset has {len(self.datasets[TEST_PHASE])} items.")

        return torch.utils.data.DataLoader(
            self.datasets[TEST_PHASE],
            sampler=DistributedSampler(self.datasets[TEST_PHASE], shuffle=False),
            drop_last=False,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )


class ScanCLSLightningModule(pl.LightningModule):

    def __init__(self, args):
        """
        This LightningModule implementation constructs a PyTorchVideo ResNet,
        defines the train and val loss to be trained with (cross_entropy), and
        configures the optimizer.
        """
        self.args = args
        super().__init__()

        self.model = get_model_by_name(args.model_arch)

        self.save_hyperparameters()
        self.trace = True

        macs, params = get_model_complexity_info(self.model, (1, ) + args.target_size, as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        logging.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        logging.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

    @property
    def tb_logger(self):
        return extract_logger(self.trainer.loggers, TensorBoardLogger)

    def forward(self, x, lungs):
        return self.model(x, lungs)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, TRAIN_PHASE)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, VALID_PHASE)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, TEST_PHASE)

    def _draw_predictions(self, scans, lungs, ems, dense_outs, cls_labels, pse_labels,
                          pred_cls_labels, pred_pse_labels, indices, stage):
        if stage == TEST_PHASE:
            epoch = self.epoch_number if hasattr(self, 'epoch_number') else self.trainer.current_epoch
        else:
            epoch = self.trainer.current_epoch
        input_cpath = Path(self.trainer.default_root_dir) / 'debug_input_data' / str(
            epoch) / self.trainer.state.stage
        input_cpath.mkdir(exist_ok=True, parents=True)
        cls_dense_predictions = F.interpolate(dense_outs[0], size=lungs.shape[-3:], mode='trilinear').cpu()
        pse_dense_predictions = F.interpolate(dense_outs[1], size=lungs.shape[-3:], mode='trilinear').cpu()
        for scan, lung, em, cls_dense_prediction, pse_dense_prediction, \
            cls_label, pse_label, pred_cls_label, pred_pse_label, index in zip(scans,
                                                                               lungs,
                                                                               ems,
                                                                               cls_dense_predictions,
                                                                               pse_dense_predictions,
                                                                               cls_labels,
                                                                               pse_labels,
                                                                               pred_cls_labels,
                                                                               pred_pse_labels,
                                                                               indices):
            scan_np = scan.cpu().squeeze(0).numpy()
            lung_np = lung.cpu().squeeze(0).numpy()
            em_np = em.cpu().squeeze(0).numpy()
            dp_cls_np = F.relu(cls_dense_prediction[1:, ]).numpy().sum(0)
            dp_pse_np = F.relu(pse_dense_prediction[1:, ]).numpy().sum(0)
            # dp_cls_np = cls_dense_prediction[pred_cls_label.item()].numpy()
            # dp_pse_np = pse_dense_prediction[pred_pse_label.item()].numpy()
            dp_cls_np = dp_cls_np / (dp_cls_np.max() + 1e-7)
            dp_pse_np = dp_pse_np / (dp_pse_np.max() + 1e-7)
            uid = self.trainer.datamodule.datasets[stage].series_uids[index]
            debug_path = input_cpath / f"{uid}_label_{cls_label.item()}_{pred_cls_label.item()}" \
                                       f"_{pse_label.item()}_{pred_pse_label.item()}"
            draw_mask_tile_singleview_heatmap(windowing(scan_np, from_span=None).astype(np.uint8),
                                              [[(lung_np * 255).astype(np.uint8)],
                                               [windowing(dp_cls_np * lung_np, from_span=(0, 1)).astype(np.uint8)],
                                               [windowing(dp_pse_np * lung_np, from_span=(0, 1)).astype(np.uint8)],
                                               [(em_np * 255).astype(np.uint8)]
                                               ],
                                              lung_np > 0, 5,
                                              debug_path, coord_axis=0,
                                              titles=["lung", "heatmap (cle)", "heatmap (pse)", "LAA950"])

    def shared_step(self, batch, batch_idx, stage):
        with torch.set_grad_enabled(stage == TRAIN_PHASE):
            scans = batch["image"].unsqueeze(1)
            lungs = batch["lung_mask"].unsqueeze(1).float()
            ems = batch["em_mask"].unsqueeze(1).float()
            cle_labels = batch["cls_label"]
            pse_labels = batch["pse_label"]

            indices = batch["index"].squeeze(-1)
            dense_outs, cls_outs = self.forward(scans, lungs)
            pred_cle_labels = cls_outs[0].detach().argmax(-1)
            pred_pse_labels = cls_outs[1].detach().argmax(-1)
            if stage == TRAIN_PHASE:
                cle_class_weights = torch.Tensor(
                    self.trainer.datamodule.datasets[TRAIN_PHASE].cle_class_weights).float().cuda()
                pse_class_weights = torch.Tensor(
                    self.trainer.datamodule.datasets[TRAIN_PHASE].pse_class_weights).float().cuda()
                loss_cle = F.cross_entropy(cls_outs[0], cle_labels,
                                           weight=cle_class_weights)

                loss_pse = F.cross_entropy(cls_outs[1], pse_labels,
                                           weight=pse_class_weights)
                loss = loss_cle + loss_pse
                self.log(f"{TRAIN_PHASE}_loss_cle", loss_cle, on_step=True, on_epoch=True, prog_bar=True)
                self.log(f"{TRAIN_PHASE}_loss_pse", loss_pse, on_step=True, on_epoch=True, prog_bar=True)
                self.log(f"{TRAIN_PHASE}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
                return {'loss': loss, "pred_cle_labels": pred_cle_labels.detach(),
                        "pred_pse_labels": pred_pse_labels.detach(),
                        "cle_labels": cle_labels.detach(), "pse_labels": pse_labels.detach(), "index": indices}
            else:
                if self.trainer.is_global_zero:
                    logging.info(f"input spatial shape: {scans.shape[-3:]}")
                    if batch_idx < 50:
                        with torch.no_grad():
                            self._draw_predictions(scans, lungs, ems, dense_outs,
                                                   cle_labels, pse_labels, pred_cle_labels, pred_pse_labels, indices,
                                                   stage)

                return {"pred_cle_labels": pred_cle_labels.detach(),
                        "pred_pse_labels": pred_pse_labels.detach(),
                        "cle_labels": cle_labels.detach(), "pse_labels": pse_labels.detach(), "index": indices}

    def training_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, TRAIN_PHASE)

    def validation_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, VALID_PHASE)

    def test_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, TEST_PHASE)

    def shared_epoch_end(self, step_outputs, phase):
        with torch.no_grad():
            pred_cle_labels = torch.cat([out['pred_cle_labels'] for out in step_outputs])
            cle_labels = torch.cat([out['cle_labels'] for out in step_outputs])
            pred_pse_labels = torch.cat([out['pred_pse_labels'] for out in step_outputs])
            pse_labels = torch.cat([out['pse_labels'] for out in step_outputs])
            indices = torch.cat([out['index'] for out in step_outputs])
            pred_cle_labels = cat_all_gather(pred_cle_labels)
            cle_labels = cat_all_gather(cle_labels)
            pred_pse_labels = cat_all_gather(pred_pse_labels)
            pse_labels = cat_all_gather(pse_labels)
            indices = cat_all_gather(indices)

            if dist.get_rank() == 0:
                acc_cle = (pred_cle_labels == cle_labels).float().mean()
                acc_pse = (pred_pse_labels == pse_labels).float().mean()
                pred_cle_labels, pred_pse_labels, cle_labels, pse_labels, indices = pred_cle_labels.cpu().numpy(), \
                                                                                    pred_pse_labels.cpu().numpy(), cle_labels.cpu().numpy(), \
                                                                                    pse_labels.cpu().numpy(), indices.cpu().numpy()
                # remove the duplicated indices
                indices, unique_indices_ids = np.unique(indices, return_index=True)
                pred_cle_labels = pred_cle_labels[unique_indices_ids]
                pred_pse_labels = pred_pse_labels[unique_indices_ids]
                cle_labels = cle_labels[unique_indices_ids]
                pse_labels = pse_labels[unique_indices_ids]
                self._log_confusion_matrix(pred_cle_labels, cle_labels, phase, 'cle', 6)
                self._log_confusion_matrix(pred_pse_labels, pse_labels, phase, 'pse', 3)
                self._log_csv(pred_cle_labels, pred_pse_labels, cle_labels, pse_labels, indices, phase)
                self.log(f"epoch_{phase}_acc_cle", acc_cle, on_step=False, on_epoch=True)
                self.log(f"epoch_{phase}_acc_pse", acc_pse, on_step=False, on_epoch=True)
                logging.debug(f"rank {dist.get_rank()} {phase} log confusion and record csvs!")

    def _log_csv(self, y_preds_cle, y_preds_pse, y_cle, y_pse, indices, phase) -> None:
        y_preds_cle = y_preds_cle
        y_cle = y_cle
        y_preds_pse = y_preds_pse
        y_pse = y_pse
        indices = indices
        uids = [self.trainer.datamodule.datasets[phase].series_uids[i] for i in indices]
        val_records = {
            'uid': uids,
            'y_preds_cle': y_preds_cle,
            'y_preds_pse': y_preds_pse,
            'y_cle': y_cle,
            'y_pse': y_pse,
        }

        df = pd.DataFrame().from_dict(val_records)
        predicts_log_path = Path(self.trainer.default_root_dir) / 'predicts' / phase
        predicts_log_path.mkdir(exist_ok=True, parents=True)
        if phase == TEST_PHASE:
            epoch = self.epoch_number if hasattr(self, 'epoch_number') else self.trainer.current_epoch
        else:
            epoch = self.trainer.current_epoch
        df.to_csv(predicts_log_path / f'{epoch}_predicts.csv', index=False)

    def _log_confusion_matrix(self, y_preds, y, phase, name, n_classes) -> None:
        if phase == TEST_PHASE:
            epoch = self.epoch_number if hasattr(self, 'epoch_number') else self.trainer.current_epoch
        else:
            epoch = self.trainer.current_epoch
        y_pred = y_preds
        y_true = y
        plt_obj = plot_confusion_matrix_from_data(
            y_true, y_pred, list(range(n_classes)), line_width=0.5, fig_size=10, font_size=11
        )
        image_array = plot_to_numpy_array(plt_obj)
        ml_log_path = Path(self.trainer.default_root_dir) / 'confusion_matrices' / phase
        ml_log_path.mkdir(exist_ok=True, parents=True)
        save_image(
            ml_log_path / f'{phase}_epoch_{epoch}_cm_{name}.png',
            image_array,
        )

        self.tb_logger.experiment.add_image(
            tag=f'{phase}_confusion_matrix_{name}',
            img_tensor=image_array,
            global_step=self.trainer.current_epoch,
            dataformats='HWC',
        )

        # update class weights based on training per-class metrics
        if phase == TRAIN_PHASE:
            matrix = confusion_matrix(y_true, y_pred)
            per_class_acc = matrix.diagonal() / matrix.sum(axis=1)

            if hasattr(self.trainer.datamodule.datasets[TRAIN_PHASE], f'{name}_class_weights'):
                current_class_weights = getattr(self.trainer.datamodule.datasets[TRAIN_PHASE], f'{name}_class_weights')
                class_weights = current_class_weights * (1.0 - per_class_acc)
                class_weights /= class_weights.sum()  # normalize
                setattr(self.trainer.datamodule.datasets[TRAIN_PHASE], f'{name}_class_weights', class_weights)
                logging.info(f"reset class weights: from {current_class_weights}"
                             f" to {class_weights} using per-class acc {per_class_acc}")

    def configure_optimizers(self):
        """
        We use the SGD optimizer with per step cosine annealing scheduler.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr,
            # momentum=self.args.momentum,
            # weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95, last_epoch=-1
        )
        return [optimizer], [scheduler]


class ScanRegLightningModule(pl.LightningModule):

    def __init__(self, args):
        """
        This LightningModule implementation constructs a PyTorchVideo ResNet,
        defines the train and val loss to be trained with (cross_entropy), and
        configures the optimizer.
        """
        self.args = args
        super().__init__()

        self.model = get_model_by_name(args.model_arch)

        self.save_hyperparameters()
        self.trace = True
        self.dice_score = BinaryDice(1e-7)
        self.bce = BinaryCrossEntropy()
        self.beta = 0.7338
        self.gamma = 0.2578
        macs, params = get_model_complexity_info(self.model, (1, ) + args.target_size, as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    @property
    def tb_logger(self):
        return extract_logger(self.trainer.loggers, TensorBoardLogger)

    def forward(self, x, lungs):
        return self.model(x, lungs)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, TRAIN_PHASE)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, VALID_PHASE)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        with torch.no_grad():
            scans = batch["image"].unsqueeze(1)
            lungs = batch["lung_mask"].unsqueeze(1).float()
            ess = batch["ess_mask"].unsqueeze(1).float()
            crop_slices = batch["crop_slice"]
            original_size = batch["original_size"]
            dense_outs, reg_outs = self.forward(scans, lungs)
            cle_dense_outs = F.interpolate(dense_outs[0], size=scans.shape[-3:], mode='trilinear', align_corners=True) * ess
            pse_dense_outs = F.interpolate(dense_outs[1], size=scans.shape[-3:], mode='trilinear', align_corners=True) * ess
            cle_precentages = cle_dense_outs.view(cle_dense_outs.shape[0], -1).sum(-1) / lungs.sum()
            pse_precentages = pse_dense_outs.view(pse_dense_outs.shape[0], -1).sum(-1) / lungs.sum()
            return {
                "cle_dense_outs": cle_dense_outs,
                "pse_dense_outs": pse_dense_outs,
                "cle_precentages": cle_precentages,
                "pse_precentages": pse_precentages,
                "crop_slices": crop_slices,
                "original_size": original_size,
                "uids": batch["uid"]
            }

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, TEST_PHASE)

    def _draw_predictions(self, scans, lungs, ems, dense_outs, cls_labels, pse_labels,
                          pred_cls_labels, pred_pse_labels, indices, stage):
        if stage == TEST_PHASE:
            epoch = self.epoch_number if hasattr(self, 'epoch_number') else self.trainer.current_epoch
        else:
            epoch = self.trainer.current_epoch
        input_cpath = Path(self.trainer.default_root_dir) / 'debug_input_data' / str(
            epoch) / self.trainer.state.stage
        input_cpath.mkdir(exist_ok=True, parents=True)
        cls_dense_predictions = F.interpolate(dense_outs[0], size=lungs.shape[-3:], mode='trilinear').cpu()
        pse_dense_predictions = F.interpolate(dense_outs[1], size=lungs.shape[-3:], mode='trilinear').cpu()
        for scan, lung, em, cls_dense_prediction, pse_dense_prediction, \
            cls_label, pse_label, pred_cls_label, pred_pse_label, index in zip(scans, lungs,
                                                                               ems,
                                                                               cls_dense_predictions,
                                                                               pse_dense_predictions,
                                                                               cls_labels,
                                                                               pse_labels,
                                                                               pred_cls_labels,
                                                                               pred_pse_labels,
                                                                               indices):

            scan_np = scan.cpu().squeeze(0).numpy()
            lung_np = lung.cpu().squeeze(0).numpy()
            em_np = em.cpu().squeeze(0).numpy()
            dp_cls_np = cls_dense_prediction.squeeze(0).numpy()
            dp_pse_np = pse_dense_prediction.squeeze(0).numpy()
            uid = self.trainer.datamodule.datasets[stage].series_uids[index]
            debug_path = input_cpath / f"{uid}_label_{cls_label.item()}_{pred_cls_label.item()}" \
                                       f"_{pse_label.item()}_{pred_pse_label.item()}"
            draw_mask_tile_singleview_heatmap(windowing(scan_np, from_span=None).astype(np.uint8),
                                              [[(lung_np * 255).astype(np.uint8)],
                                               [windowing(dp_cls_np * lung_np, from_span=(0, 1)).astype(np.uint8)],
                                               [windowing(dp_pse_np * lung_np, from_span=(0, 1)).astype(np.uint8)],
                                               [(em_np * 255).astype(np.uint8)]
                                               ],
                                              lung_np > 0, 5,
                                              debug_path, coord_axis=0,
                                              titles=["lung", "heatmap (cle)", "heatmap (pse)", "LAA950"])

    def _generate_regression_labels(self, cls_targets, ratio_mapping, tightness=1.0):
        labels = []
        for ctss in cls_targets:
            ctss_lb, ctss_ub = ratio_mapping[int(ctss)]
            if ctss_lb < 1e-7:
                # correction for score 0
                label_band = (0.0, 0.0)
            else:
                m = (ctss_lb + ctss_ub) / 2.0
                span = (ctss_ub - ctss_lb) * tightness / 2.0
                label_band = (m - span, m + span)
                assert label_band[0] < label_band[1]
            labels.append(label_band)

        labels = torch.FloatTensor(labels).cuda()
        return labels

    def _interval_regression_loss(self, outs, reg_targets, weight_factors):
        _n_data = torch.cat([outs.unsqueeze(1), reg_targets], dim=1)
        _n_data = self.beta * _n_data ** self.gamma  # power correction
        K = (0.5 * (_n_data[:, 2] - _n_data[:, 1])) ** 2
        loss_unhinge = (_n_data[:, 0] - (_n_data[:, 2] + _n_data[:, 1]) / 2.0) ** 2 - K
        loss_unweight = F.leaky_relu(loss_unhinge, negative_slope=0.0)
        loss_lesion = 10.0 * loss_unweight * weight_factors

        loss = loss_lesion.sum()
        return loss

    def _segmentation_loss(self, dense_cle_predictions, dense_pse_predictions, ems, lungs):
        # multual exclusive loss
        mul_loss = self.dice_score(dense_cle_predictions * lungs, dense_pse_predictions * lungs)

        # both should cover ems
        dnese_p = torch.clamp(dense_cle_predictions + dense_pse_predictions, min=0.0, max=1.0)
        seg_loss = self.bce(ems, dnese_p, lungs, smoothness=0.85)

        return mul_loss, seg_loss

    def _ratio_to_label(self, ratios, ratio_mapping):
        inv_ratio_map = {v: k for k, v in ratio_mapping.items()}
        labels = [[inv_ratio_map[k] for k in inv_ratio_map.keys()
                   if k[0] <= ratio.item() and ratio.item() < k[1]][0] for ratio in ratios]
        return torch.as_tensor(labels).long().to(ratios.device)

    def shared_step(self, batch, batch_idx, stage):
        with torch.set_grad_enabled(stage == TRAIN_PHASE):
            scans = batch["image"].unsqueeze(1)
            lungs = batch["lung_mask"].unsqueeze(1).float()
            B = scans.shape[0]
            if batch_idx == 0 and self.trainer.is_global_zero:
                logging.info(f"input spatial shapes: {scans.shape[-3:]}")
            ems = batch["em_mask"].unsqueeze(1).float()
            cle_labels = batch["cls_label"]
            pse_labels = batch["pse_label"]
            cle_reg_labels = self._generate_regression_labels(cle_labels, _DATASET_CLASS.cle_ratio_map)
            pse_reg_labels = self._generate_regression_labels(pse_labels, _DATASET_CLASS.pse_ratio_map)
            indices = batch["index"].squeeze(-1)
            dense_outs, reg_outs = self.forward(scans, lungs)
            pred_cle_labels = self._ratio_to_label(reg_outs[0], _DATASET_CLASS.cle_ratio_map)
            pred_pse_labels = self._ratio_to_label(reg_outs[1], _DATASET_CLASS.pse_ratio_map)
            if stage == TRAIN_PHASE:
                cle_class_weights = torch.Tensor(
                    [self.trainer.datamodule.datasets[TRAIN_PHASE].cle_class_weights[int(ctss.item())]
                     for ctss in cle_labels]).float().cuda()
                pse_class_weights = torch.Tensor(
                    [self.trainer.datamodule.datasets[TRAIN_PHASE].pse_class_weights[int(ctss.item())]
                     for ctss in pse_labels]).float().cuda()

                loss_cle = self._interval_regression_loss(reg_outs[0], cle_reg_labels, cle_class_weights)
                loss_pse = self._interval_regression_loss(reg_outs[1], pse_reg_labels, pse_class_weights)

                # segmentation loss terms
                binary_labels = torch.logical_or(cle_labels > 0, pse_labels > 0).long()
                seg_labels = F.interpolate(ems * binary_labels.float().view(B, 1, 1, 1, 1),
                                           dense_outs[0].shape[-3:], mode='nearest').detach()
                lung_labels = F.interpolate(lungs, size=dense_outs[0].shape[-3:], mode='nearest')
                mul_loss, seg_loss = self._segmentation_loss(dense_outs[0], dense_outs[1],
                                                             seg_labels,
                                                             lung_labels)
                loss = loss_cle + loss_pse + 2.0 * mul_loss + seg_loss
                self.log(f"{TRAIN_PHASE}_loss_cle", loss_cle, on_step=True, on_epoch=True, prog_bar=True)
                self.log(f"{TRAIN_PHASE}_loss_pse", loss_pse, on_step=True, on_epoch=True, prog_bar=True)
                self.log(f"{TRAIN_PHASE}_mul_loss", mul_loss, on_step=True, on_epoch=True, prog_bar=True)
                self.log(f"{TRAIN_PHASE}_seg_loss", seg_loss, on_step=True, on_epoch=True, prog_bar=True)
                self.log(f"{TRAIN_PHASE}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
                return {'loss': loss, "pred_cle_labels": pred_cle_labels.detach(),
                        "pred_pse_labels": pred_pse_labels.detach(),
                        "cle_labels": cle_labels.detach(), "pse_labels": pse_labels.detach(), "index": indices}
            else:
                if self.trainer.is_global_zero:
                    with torch.no_grad():
                        self._draw_predictions(scans, lungs, ems, dense_outs,
                                                   cle_labels, pse_labels, pred_cle_labels, pred_pse_labels, indices,
                                                   stage)

                return {"pred_cle_labels": pred_cle_labels.detach(),
                        "pred_pse_labels": pred_pse_labels.detach(),
                        "cle_labels": cle_labels.detach(), "pse_labels": pse_labels.detach(), "index": indices}

    def training_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, TRAIN_PHASE)

    def validation_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, VALID_PHASE)

    def test_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, TEST_PHASE)

    def shared_epoch_end(self, step_outputs, phase):
        with torch.no_grad():
            pred_cle_labels = torch.cat([out['pred_cle_labels'] for out in step_outputs])
            cle_labels = torch.cat([out['cle_labels'] for out in step_outputs])
            pred_pse_labels = torch.cat([out['pred_pse_labels'] for out in step_outputs])
            pse_labels = torch.cat([out['pse_labels'] for out in step_outputs])
            indices = torch.cat([out['index'] for out in step_outputs])
            pred_cle_labels = cat_all_gather(pred_cle_labels)
            cle_labels = cat_all_gather(cle_labels)
            pred_pse_labels = cat_all_gather(pred_pse_labels)
            pse_labels = cat_all_gather(pse_labels)
            indices = cat_all_gather(indices)

            if dist.get_rank() == 0:
                acc_cle = (pred_cle_labels == cle_labels).float().mean()
                acc_pse = (pred_pse_labels == pse_labels).float().mean()
                pred_cle_labels, pred_pse_labels, cle_labels, pse_labels, indices = pred_cle_labels.cpu().numpy(), \
                                                                                    pred_pse_labels.cpu().numpy(), cle_labels.cpu().numpy(), \
                                                                                    pse_labels.cpu().numpy(), indices.cpu().numpy()
                # remove the duplicated indices
                indices, unique_indices_ids = np.unique(indices, return_index=True)
                pred_cle_labels = pred_cle_labels[unique_indices_ids]
                pred_pse_labels = pred_pse_labels[unique_indices_ids]
                cle_labels = cle_labels[unique_indices_ids]
                pse_labels = pse_labels[unique_indices_ids]
                self._log_confusion_matrix(pred_cle_labels, cle_labels, phase, 'cle', 6)
                self._log_confusion_matrix(pred_pse_labels, pse_labels, phase, 'pse', 3)
                self._log_csv(pred_cle_labels, pred_pse_labels, cle_labels, pse_labels, indices, phase)
                self.log(f"epoch_{phase}_acc_cle", acc_cle, on_step=False, on_epoch=True)
                self.log(f"epoch_{phase}_acc_pse", acc_pse, on_step=False, on_epoch=True)
                logging.debug(f"rank {dist.get_rank()} {phase} log confusion and record csvs!")

    def _log_csv(self, y_preds_cle, y_preds_pse, y_cle, y_pse, indices, phase) -> None:
        y_preds_cle = y_preds_cle
        y_cle = y_cle
        y_preds_pse = y_preds_pse
        y_pse = y_pse
        indices = indices
        uids = [self.trainer.datamodule.datasets[phase].series_uids[i] for i in indices]
        val_records = {
            'uid': uids,
            'y_preds_cle': y_preds_cle,
            'y_preds_pse': y_preds_pse,
            'y_cle': y_cle,
            'y_pse': y_pse,
        }

        df = pd.DataFrame().from_dict(val_records)
        predicts_log_path = Path(self.trainer.default_root_dir) / 'predicts' / phase
        predicts_log_path.mkdir(exist_ok=True, parents=True)
        if phase == TEST_PHASE:
            epoch = self.epoch_number if hasattr(self, 'epoch_number') else self.trainer.current_epoch
        else:
            epoch = self.trainer.current_epoch
        df.to_csv(predicts_log_path / f'{epoch}_predicts.csv', index=False)

    def _log_confusion_matrix(self, y_preds, y, phase, name, n_classes) -> None:
        if phase == TEST_PHASE:
            epoch = self.epoch_number if hasattr(self, 'epoch_number') else self.trainer.current_epoch
        else:
            epoch = self.trainer.current_epoch
        y_pred = y_preds
        y_true = y
        plt_obj = plot_confusion_matrix_from_data(
            y_true, y_pred, list(range(n_classes)), line_width=0.5, fig_size=10, font_size=11
        )
        image_array = plot_to_numpy_array(plt_obj)
        ml_log_path = Path(self.trainer.default_root_dir) / 'confusion_matrices' / phase
        ml_log_path.mkdir(exist_ok=True, parents=True)
        save_image(
            ml_log_path / f'{phase}_epoch_{epoch}_cm_{name}.png',
            image_array,
        )

        self.tb_logger.experiment.add_image(
            tag=f'{phase}_confusion_matrix_{name}',
            img_tensor=image_array,
            global_step=self.trainer.current_epoch,
            dataformats='HWC',
        )


    def configure_optimizers(self):
        """
        We use the SGD optimizer with per step cosine annealing scheduler.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr,
            # momentum=self.args.momentum,
            # weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95, last_epoch=-1
        )
        return [optimizer], [scheduler]
