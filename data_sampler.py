from torch.utils.data.sampler import Sampler
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import time
import logging

class SubtypingStratifiedSampler(Sampler):
    def __init__(self, data_source, balance_label_count):
        super(SubtypingStratifiedSampler, self).__init__(data_source)
        self.data_source = data_source
        self.balance_label_count = balance_label_count

        uid_scores_tuple = [(uid, int(float(data_source.subtyping_labels[uid]['cle']))
                             , int(float(data_source.subtyping_labels[uid]['pse'])))
                            for uid in data_source.series_uids]
        uids, cle_scores, pse_scores = zip(*uid_scores_tuple)
        unique_cle_labels, unique_cle_counts = np.unique(cle_scores, return_counts=True)
        self.cle_class_weights = compute_class_weight(class_weight='balanced',
                                                 classes=unique_cle_labels,
                                                 y=cle_scores)
        self.cle_class_weights = list(np.clip(self.cle_class_weights /
                                              np.sum(self.cle_class_weights), a_min=0.2, a_max=0.8))
        self.cle_statistics = {ucl: ucc / np.sum(unique_cle_counts) for ucl, ucc in
                               zip(unique_cle_labels, unique_cle_counts)}
        for ctss_type in range(0, 6):
            if ctss_type not in unique_cle_labels:
                self.cle_class_weights.insert(ctss_type, max(self.cle_class_weights))
                self.cle_statistics[ctss_type] = 1e-5

        unique_pse_labels, unique_pse_counts = np.unique(pse_scores, return_counts=True)
        self.pse_class_weights = compute_class_weight(class_weight='balanced',
                                                 classes=unique_pse_labels,
                                                   y=pse_scores)
        self.pse_class_weights = list(np.clip(self.pse_class_weights /
                                              np.sum(self.pse_class_weights), a_min=0.2, a_max=0.8))
        self.pse_statistics = {ucl: ucc / np.sum(unique_pse_counts) for ucl, ucc in
                               zip(unique_pse_labels, unique_pse_counts)}
        for ctss_type in range(0, 3):
            if ctss_type not in unique_pse_labels:
                self.pse_class_weights.insert(ctss_type, max(self.pse_class_weights))
                self.pse_statistics[ctss_type] = 1e-5
        logging.info(f"cle label weights: {self.cle_class_weights}")
        logging.info(f"pse label weights: {self.pse_class_weights}")
        logging.info(f"cle label statistics: {self.cle_statistics}")
        logging.info(f"pse label statistics: {self.pse_statistics}")
        assert set(uids) == set(data_source.series_uids)

        self.cle_label_groups = {l: np.where(cle_scores == l)[0] for l in unique_cle_labels}
        self.pse_label_groups = {l: np.where(pse_scores == l)[0] for l in unique_pse_labels}
        logging.info(f"head of cle sampled indices: {self.cle_label_groups[0][:20]}.")
        logging.info(f"head of pse sampled indices: {self.pse_label_groups[0][:20]}.")
        self.num_samples = len(unique_cle_labels) * self.balance_label_count

    def get_indices(self):
        indices = []
        for n in range(self.num_samples):
            sl = np.random.choice(list(self.cle_label_groups.keys()))
            index = np.random.choice(self.cle_label_groups[sl])
            indices.append(index)
        return indices

    def __iter__(self):
        np.random.seed(int(time.time()))
        indices = self.get_indices()
        return iter(indices)

    def __len__(self):
        return self.num_samples
