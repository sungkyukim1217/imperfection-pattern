"""Optimized version of data/preprocess.py for NAP / OP (and ERTP/CRTP) tasks.

Drop-in replacement: change `from data.preprocess import SETGENERATOR`
to                  `from data.preprocess_optimized import SETGENERATOR`
in main.py. The returned (train_loader, test_loader, meta) tuple has
identical shapes and semantics to the original.

Key optimizations vs. data/preprocess.py
----------------------------------------
1. Training data: the original built all `n` prefixes of every trace via
   `_zero_pad` and then discarded all but the last one. That is O(n^2)
   work per case for nothing. Here we skip prefix generation entirely
   for training and emit a single padded sample per case.

2. Testing data: instead of materializing `n * max_length * dim` copies
   per case (each prefix as its own padded tensor), we keep ONE padded
   trace per case and let a custom `PrefixDataset` produce prefix samples
   by index. Because the LSTM uses `pack_padded_sequence` with `enforce_sorted=True`,
   only the first `length` positions are ever read — padding beyond that
   never reaches the network. Memory drops by roughly `max_length / 2`x.

3. Per-case Python loop replaced by vectorized scatter: a single advanced-
   indexing assignment fills the (num_cases, max_length, attr_size) tensor
   for the entire dataset. One-hot encoding for resources is done via
   scatter rather than `F.one_hot` per case.

4. Injection ratio: instead of recomputing a `Counter` for every prefix
   length k = 1..n (O(n^2) per case), we compute a per-event indicator
   `is_injected` once and take its cumulative sum. The ratio at prefix k
   is then a single tensor lookup.

5. Dead code removed (an unused `df_train[column].unique()` call) and
   redundant unique computations merged.

Sorting behaviour is preserved: both train and test samples are emitted
in descending prefix-length order using a stable sort, so DataLoader
batches still satisfy `pack_padded_sequence(enforce_sorted=True)`.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils.config import COL, INJM, LOGTYPE, META, TASK

random.seed(42)


# -----------------------------------------------------------------------------
# Custom Dataset: lazy prefix views
# -----------------------------------------------------------------------------
class PrefixDataset(Dataset):
    """Stores one padded trace per case; emits prefix samples by index.

    Why this is safe: the model wraps its input in `pack_padded_sequence`
    with `enforce_sorted=True`. That packing only reads the first `length`
    positions of each sequence, so padding values beyond `length` do not
    affect the LSTM output. We therefore reuse the same padded trace for
    every prefix length of that case and only vary `length` between
    samples. The original code instead created a fresh zero-padded copy
    for every prefix length, which was wasteful.

    Args:
        padded_act:    (num_cases, max_length) int64. Activity ids; 0 = pad.
        padded_attr:   (num_cases, max_length, attr_size) float.
        padded_label:  (num_cases, max_length) int64 or float depending on task.
        sample_index:  list of (case_idx, prefix_length). Should be pre-sorted
                       in descending prefix_length order so the DataLoader
                       produces batches that satisfy `enforce_sorted=True`.
        cumulative_inj: optional (num_cases, max_length) int64. Cumulative
                       count of injected activities along positions; the
                       injection ratio for prefix length k of case c is
                       cumulative_inj[c, k - 1] / k.
    """

    def __init__(
        self,
        padded_act: torch.Tensor,
        padded_attr: torch.Tensor,
        padded_label: torch.Tensor,
        sample_index: List[Tuple[int, int]],
        cumulative_inj: Optional[torch.Tensor] = None,
    ):
        self.padded_act = padded_act
        self.padded_attr = padded_attr
        self.padded_label = padded_label
        self.sample_index = sample_index
        self.cumulative_inj = cumulative_inj

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        c_idx, k = self.sample_index[idx]
        act = self.padded_act[c_idx]
        attr = self.padded_attr[c_idx]
        label = self.padded_label[c_idx].clone()
        label[k:] = 0  # ← prefix k 이후는 0으로 마스킹
        length = torch.tensor(k, dtype=torch.int64)

        if self.cumulative_inj is not None:
            inj_count = int(self.cumulative_inj[c_idx, k - 1].item())
            ratio_val = inj_count / k if inj_count > 0 else 0.0
            ratio = torch.tensor(ratio_val, dtype=torch.float)
            return act, attr, label, length, ratio
        return act, attr, label, length


# -----------------------------------------------------------------------------
# SETGENERATOR (optimized)
# -----------------------------------------------------------------------------
class SETGENERATOR:
    """Optimized drop-in replacement for data.preprocess.SETGENERATOR."""

    def __init__(self, train_csv, test_csv, inj_type_train, inj_type_test, task, batchsize):
        self._train_csv = train_csv
        self._test_csv = test_csv
        self._task = task
        self._batchsize = batchsize
        self._inj_type_train = inj_type_train
        self._inj_type_test = inj_type_test
        self._logtype_train = LOGTYPE(self._inj_type_train, self._task)
        self._logtype_test = LOGTYPE(self._inj_type_test, self._task)

    # ----- CSV reader -----
    def _log_reader(self):
        df_train = pd.read_csv(
            self._train_csv,
            usecols=list(self._logtype_train.keys()),
            dtype={k: v for k, v in self._logtype_train.items() if k != COL.TIME.value},
            parse_dates=[COL.TIME.value],
        )
        df_test = pd.read_csv(
            self._test_csv,
            usecols=list(self._logtype_test.keys()),
            dtype={k: v for k, v in self._logtype_test.items() if k != COL.TIME.value},
            parse_dates=[COL.TIME.value],
        )
        return df_train, df_test

    # ----- Categorical -> integer tokens -----
    def _tokenization(self, df_train, df_test, column):
        # Mirror the original's set().union() ordering exactly so token
        # assignments match (otherwise verify_preprocess.py would see every
        # batch differ in token values).
        unique_vals = list(set(df_train[column].unique()).union(set(df_test[column].unique())))
        str_to_idx = {v: i + 1 for i, v in enumerate(unique_vals)}
        df_train[column] = df_train[column].map(str_to_idx)
        df_test[column] = df_test[column].map(str_to_idx)
        return df_train, df_test

    # ----- Time features -----
    def _time_feature(self, df):
        time_col = COL.TIME.value
        case_col = COL.CASE.value
        case_group = df.groupby(case_col)[time_col]
        df[COL.TSP.value] = (df[time_col] - case_group.shift(1)).dt.total_seconds().fillna(0)
        df[COL.TSSC.value] = (df[time_col] - case_group.transform("min")).dt.total_seconds()
        return df

    # ----- Task-specific labels -----
    def _nap_label(self, df_train, df_test):
        df_train[COL.LABEL.value] = df_train.groupby(COL.CASE.value)[COL.ACT.value].shift(-1)
        df_test[COL.LABEL.value] = df_test.groupby(COL.CASE.value)[COL.ACT.value].shift(-1)
        max_idx = max(df_train[COL.LABEL.value].max(), df_test[COL.LABEL.value].max())
        df_train[COL.LABEL.value] = df_train[COL.LABEL.value].fillna(max_idx + 1).astype(int)
        df_test[COL.LABEL.value] = df_test[COL.LABEL.value].fillna(max_idx + 1).astype(int)
        unique_label = pd.unique(
            pd.concat([df_train[COL.LABEL.value], df_test[COL.LABEL.value]], ignore_index=True)
        )
        lab_to_idx = {v: i + 1 for i, v in enumerate(unique_label)}
        df_train[COL.LABEL.value] = df_train[COL.LABEL.value].map(lab_to_idx)
        df_test[COL.LABEL.value] = df_test[COL.LABEL.value].map(lab_to_idx)
        output_dim = len(unique_label)
        return df_train, df_test, output_dim

    def _op_label(self, df_train, df_test):
        mapping = {"deviant": 0, "regular": 1}
        df_train[COL.LABEL.value] = df_train[COL.OC.value].map(mapping)
        df_test[COL.LABEL.value] = df_test[COL.OC.value].map(mapping)
        return df_train, df_test

    def _ertp_label(self, df_train, df_test):
        for df in (df_train, df_test):
            shifted = df.groupby(COL.CASE.value)[COL.TIME.value].shift(-1)
            df[COL.LABEL.value] = (shifted - df[COL.TIME.value]).dt.total_seconds().fillna(0).astype(float)
        return df_train, df_test

    def _crtp_label(self, df_train, df_test):
        for df in (df_train, df_test):
            max_t = df.groupby(COL.CASE.value)[COL.TIME.value].transform("max")
            df[COL.LABEL.value] = (max_t - df[COL.TIME.value]).dt.total_seconds()
        return df_train, df_test

    def _scaler(self, df_train, df_test, column):
        scaler = StandardScaler()
        df_train[column] = scaler.fit_transform(df_train[[column]])
        df_test[column] = scaler.transform(df_test[[column]])
        return df_train, df_test, scaler

    # ----- Metadata -----
    def _meta_scrap(self, df_train, df_test):
        # After tokenization, activity / resource ids are 1..N so max() gives N.
        num_act = int(max(df_train[COL.ACT.value].max(), df_test[COL.ACT.value].max()))
        num_res = int(max(df_train[COL.RES.value].max(), df_test[COL.RES.value].max()))
        max_length = int(
            max(
                df_train.groupby(COL.CASE.value).size().max(),
                df_test.groupby(COL.CASE.value).size().max(),
            )
        )
        if COL.INJ.value in df_test.columns:
            inj_acts = df_test.loc[df_test[COL.INJ.value].notna(), COL.ACT.value]
            inject_act_list = inj_acts.unique().tolist()
        else:
            inject_act_list = None
        return num_act, num_res, max_length, inject_act_list

    # ----- Vectorized padded tensor builder -----
    def _build_padded_per_case(
        self,
        df: pd.DataFrame,
        num_res: int,
        max_length: int,
        classification: bool,
        inject_act_set: Optional[set] = None,
    ):
        """Scatter every event in `df` into (num_cases, max_length, ...) tensors.

        Replaces the per-case Python loop in the original `_prfx_bucket`.
        Returns padded act/attr/label tensors, case lengths, and (optionally)
        a cumulative-injection-count tensor used to derive prefix ratios.
        """
        # Stable sort by case id so events of each case become contiguous while
        # preserving their original within-case order.
        df_sorted = df.sort_values(COL.CASE.value, kind="stable").reset_index(drop=True)

        case_groups = df_sorted.groupby(COL.CASE.value, sort=False)
        case_lengths_np = case_groups.size().values.astype(np.int64)
        case_lengths = torch.from_numpy(case_lengths_np)
        num_cases = int(case_lengths.shape[0])
        total_events = int(case_lengths.sum().item())

        # For each row in the flat dataframe: which case it belongs to and its
        # position within that case.
        case_idx_per_row = torch.repeat_interleave(torch.arange(num_cases), case_lengths)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), case_lengths.cumsum(0)])
        pos_per_row = torch.arange(total_events) - offsets[case_idx_per_row]

        # Flat per-event arrays.
        act_flat = torch.from_numpy(df_sorted[COL.ACT.value].values.astype(np.int64))
        res_flat = torch.from_numpy(df_sorted[COL.RES.value].values.astype(np.int64)) - 1
        tsp_flat = torch.from_numpy(df_sorted[COL.TSP.value].values.astype(np.float32))
        tssc_flat = torch.from_numpy(df_sorted[COL.TSSC.value].values.astype(np.float32))
        if classification:
            label_flat = torch.from_numpy(df_sorted[COL.LABEL.value].values.astype(np.int64))
            label_dtype = torch.int64
        else:
            label_flat = torch.from_numpy(df_sorted[COL.LABEL.value].values.astype(np.float32))
            label_dtype = torch.float

        # Output tensors.
        padded_act = torch.zeros((num_cases, max_length), dtype=torch.int64)
        padded_attr = torch.zeros((num_cases, max_length, num_res + 2), dtype=torch.float)
        padded_label = torch.zeros((num_cases, max_length), dtype=label_dtype)

        # Scatter — one assignment fills the whole dataset.
        padded_act[case_idx_per_row, pos_per_row] = act_flat
        # One-hot for resource via scatter rather than F.one_hot per case.
        padded_attr[case_idx_per_row, pos_per_row, res_flat] = 1.0
        padded_attr[case_idx_per_row, pos_per_row, num_res] = tsp_flat
        padded_attr[case_idx_per_row, pos_per_row, num_res + 1] = tssc_flat
        padded_label[case_idx_per_row, pos_per_row] = label_flat

        # Cumulative injection counts per position. ratio at prefix k = cum[c, k-1] / k.
        cumulative_inj: Optional[torch.Tensor] = None
        if inject_act_set:
            inj_tensor = torch.tensor(list(inject_act_set), dtype=torch.int64)
            is_inj_flat = torch.isin(act_flat, inj_tensor).to(torch.int64)
            scatter_inj = torch.zeros((num_cases, max_length), dtype=torch.int64)
            scatter_inj[case_idx_per_row, pos_per_row] = is_inj_flat
            cumulative_inj = scatter_inj.cumsum(dim=1)

        return padded_act, padded_attr, padded_label, case_lengths, cumulative_inj

    # ----- Main entry point -----
    def SetGenerator(self):
        df_train, df_test = self._log_reader()
        df_train, df_test = self._tokenization(df_train, df_test, COL.ACT.value)
        df_train, df_test = self._tokenization(df_train, df_test, COL.RES.value)
        df_train = self._time_feature(df_train)
        df_test = self._time_feature(df_test)
        df_train, df_test, _ = self._scaler(df_train, df_test, COL.TSP.value)
        df_train, df_test, _ = self._scaler(df_train, df_test, COL.TSSC.value)

        lab_scaler = None
        if self._task == TASK.NAP.value:
            df_train, df_test, output_dim = self._nap_label(df_train, df_test)
            classification = True
        elif self._task == TASK.OP.value:
            df_train, df_test = self._op_label(df_train, df_test)
            output_dim = 1
            classification = True
        elif self._task == TASK.ERTP.value:
            df_train, df_test = self._ertp_label(df_train, df_test)
            output_dim = 1
            df_train, df_test, lab_scaler = self._scaler(df_train, df_test, COL.LABEL.value)
            classification = False
        elif self._task == TASK.CRTP.value:
            df_train, df_test = self._crtp_label(df_train, df_test)
            output_dim = 1
            df_train, df_test, lab_scaler = self._scaler(df_train, df_test, COL.LABEL.value)
            classification = False
        else:
            raise ValueError(f"Unknown task: {self._task}")

        num_act, num_res, max_length, inject_act_list = self._meta_scrap(df_train, df_test)
        inject_act_set = set(inject_act_list) if inject_act_list else None

        # ---- TRAIN: ONE sample per case (full padded trace). No prefix copies. ----
        train_act, train_attr, train_label, train_lens, _ = self._build_padded_per_case(
            df_train, num_res, max_length, classification, inject_act_set=None
        )
        train_sorted = torch.argsort(train_lens, descending=True, stable=True)
        train_sample_index = [
            (int(train_sorted[i].item()), int(train_lens[train_sorted[i]].item()))
            for i in range(int(train_sorted.shape[0]))
        ]
        train_dataset = PrefixDataset(
            train_act, train_attr, train_label, train_sample_index, cumulative_inj=None
        )

        # ---- TEST: all prefixes per case, lazily materialised via PrefixDataset ----
        test_act, test_attr, test_label, test_lens, cumulative_inj = self._build_padded_per_case(
            df_test, num_res, max_length, classification, inject_act_set=inject_act_set
        )
        test_sample_index = [
            (c, k) for c, n in enumerate(test_lens.tolist()) for k in range(1, n + 1)
        ]
        # Stable sort by prefix length descending; preserves case order within
        # equal-length samples for reproducibility.
        test_sample_index.sort(key=lambda p: -p[1])
        test_dataset = PrefixDataset(
            test_act, test_attr, test_label, test_sample_index, cumulative_inj=cumulative_inj
        )

        train_loader = DataLoader(train_dataset, batch_size=self._batchsize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self._batchsize, shuffle=False)

        # Dimensionality bumps to match the model's expectation (zero padding +
        # time features).
        num_res += 2
        num_act += 1
        if self._task == TASK.NAP.value:
            output_dim += 1

        meta = {
            META.OUTDIM.value: output_dim,
            META.SCALER.value: lab_scaler,
            META.NUMACT.value: num_act,
            META.ATTRSZ.value: num_res,
            META.MAXLEN.value: max_length,
        }
        return train_loader, test_loader, meta
