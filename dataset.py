import numpy as np

from utils import read_csv_in_dict, find_crops
from torch.utils.data import Dataset
import glob
import SimpleITK as sitk
from scipy import ndimage
from pathlib import Path
import torch


class SubtypingInference(Dataset):
    label_to_cle = {
        0: "absent",
        1: "trace",
        2: "mild",
        3: "moderate",
        4: "confluence",
        5: "destructive",
    }

    label_to_pse = {
        0: "absent",
        1: "mild",
        2: "substantial",
    }

    def __init__(self, scan_path, lobe_path, transforms=None,
                 keep_sorted=True, crop_border=5):
        super(SubtypingInference, self).__init__()
        self.scan_path = scan_path
        self.lobe_path = lobe_path
        self.keep_sorted = keep_sorted
        self.transforms = transforms
        self.crop_border = crop_border
        all_scans = glob.glob(self.scan_path + '/*.mha', recursive=False)
        assert len(all_scans) == 1 # for grand-challenge

        self.scan_file = all_scans[0]
        self.lobe_file = glob.glob(self.lobe_path + '/*.mha', recursive=False)[0]
        self.scan_meta_cache = {}

    def __len__(self):
        return 1

    def __getitem__(self, index):
        d = self.get_data(index)
        return d

    def read_image(self, path):
        sitk_image = sitk.ReadImage(path)
        spacing = sitk_image.GetSpacing()[::-1]
        origin = sitk_image.GetOrigin()[::-1]
        direction = np.asarray(sitk_image.GetDirection()).reshape(3, 3)[::-1].flatten().tolist()
        scan = sitk.GetArrayFromImage(sitk_image)
        return scan, origin, spacing, direction

    def get_data(self, index):
        series_uid = Path(self.scan_file).stem # dummy name, should be ignored.
        scan, origin, spacing, direction = self.read_image(self.scan_file)
        original_size = scan.shape
        lobe, origin_, spacing_, direction_ = self.read_image(self.lobe_file)
        assert lobe.shape == scan.shape
        lung = lobe > 0
        dlung = ndimage.binary_dilation(lung, ndimage.generate_binary_structure(3, 3), iterations=2)
        scan[dlung < 1e-7] = -2048
        slices = find_crops(lung, spacing, self.crop_border)
        scan = scan[slices]
        lung = lung[slices]
        ret = {
            "image": scan.astype(np.int16),
            "lung_mask": lung.astype(np.bool),
            "crop_slice": np.asarray([(ss.start, ss.stop) for ss in slices]),
            "original_size": np.asarray(original_size),
            "uid": series_uid
        }

        self.scan_meta_cache[series_uid] = {
            "spacing": spacing,
            "origin": origin,
            "direction": direction
        }
        if self.transforms:
            ret = self.transforms(ret)
        return ret


# dataset for model development
class COPDGeneSubtyping(Dataset):
    ON_PREMISE_ROOT = None

    cle_ratio_map = {
        0: (0.0, 0.01),
        1: (0.01, 0.05),
        2: (0.05, 0.1),
        3: (0.1, 0.2),
        4: (0.2, 0.3),
        5: (0.3, 1.0001)
    }

    pse_ratio_map = {
        0: (0.0, 0.01),
        1: (0.01, 0.05),
        2: (0.05, 1.0001),
    }

    @classmethod
    def get_series_uids(cls, csv_file):
        scan_selected, _ = read_csv_in_dict(csv_file, 'SeriesInstanceUID')
        return sorted(list(scan_selected.keys()))

    def __init__(self, archive_path, series_uids, transforms=None,
                 keep_sorted=True):
        super(COPDGeneSubtyping, self).__init__()
        self.archive_path = archive_path
        self.keep_sorted = keep_sorted
        self.transforms = transforms
        self.series_uids = series_uids
        self.meta, _ = read_csv_in_dict(archive_path + '/merged.csv', "SeriesInstanceUID")
        self.subtyping_labels = {}

        for series_uid in series_uids:
            cle = int(float(self.meta[series_uid]['CT_Visual_Emph_Severity_P1']))
            pse = int(float(self.meta[series_uid]['CT_Visual_Emph_Paraseptal_P1']))

            self.subtyping_labels[series_uid] = {
                "cle": cle,
                "pse": pse
            }

    def __len__(self):
        return len(self.series_uids)

    def __getitem__(self, index):
        uid = self.series_uids[index]
        d = self.get_data(uid)
        d['index'] = torch.LongTensor([index])
        return d

    def get_data(self, series_uid):
        data = torch.load(self.archive_path + f"/{series_uid}.pth")
        em_mask = torch.logical_and(data['image'] < -950, data['lung_mask'] > 0)
        data['em_mask'] = em_mask
        assert data['cls_label'] == self.subtyping_labels[series_uid]['cle']
        assert data['pse_label'] == self.subtyping_labels[series_uid]['pse']
        if self.transforms:
            data = self.transforms(data)
        return data
