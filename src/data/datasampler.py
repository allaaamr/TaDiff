import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import math
from src.data.data_loader import val_transforms, non_load_val_transforms

# --- Configuration for sampling (paper-inspired) ---
# overall probability for target being in (past, middle, future)
P_TARGET_BUCKET = {
    'past': 0.2,
    'middle': 0.3,
    'future': 0.5
}
#  k (number of inputs) distribution - by default uniform over {1,2,3}
P_K = None  # None -> uniform. Or set e.g. {1:0.2,2:0.3,3:0.5}
npz_keys = ['image', 'label', 'days', 'treatment', 'geno']


class PatientSamplingDataset(Dataset):
    """
    Dataset that implements the paper-style sampling across a patient's full timeline:
      - Choose a target session among all sessions (biased to produce past/middle/future ratios)
      - Choose k in {1,2,3} input sessions from remaining sessions WITH replacement
      - Pad inputs to exactly 3 and put target last → [in1,in2,in3,target]
      - Collapse session+modalities into channels, apply non_load_val_transforms (MONAI),
        then reshape back to (S=4, C_use, H, W, D)
    Returns a dict with tensors:
      'image': (S=4, C, H, W, D)
      'label': (S=4, H, W, D)
      'days': (S=4,)
      'treatment': (S=4,)
      'geno': (G,) not time variant (1 per patient not per session)
      'sample_info': metadata for debugging (dict)
    Args:
      file_dicts: list of dicts with keys npz_keys (paths to .npy)
      transform: non_load_val_transforms (val_transforms with LoadImaged removed)
      samples_per_patient: how many samples to draw per patient per epoch (len = n_patients * samples_per_patient)
      rng_seed: optional seed for reproducibility
    """
    def __init__(self, file_dicts: List[Dict], transform=None, samples_per_patient: int = 100, geno_mean=None, geno_std=None, rng_seed: Optional[int]=None):
        self.file_dicts = file_dicts[:]
        self.transform = transform
        self.samples_per_patient = int(samples_per_patient)
        self.rng = np.random.default_rng(rng_seed)
        self.geno_mean = geno_mean
        self.geno_std = geno_std

        # precompute patient session counts to quickly check valid patients
        self.patient_session_counts = []
        for fd in self.file_dicts:
            lbl = np.load(fd['label'])
            self.patient_session_counts.append(lbl.shape[0])

    def __len__(self):
        return max(1, len(self.file_dicts)) * self.samples_per_patient

    def _choose_target_session(self, S_all: int) -> int:
        """
        Choose a target session index in 0..S_all-1 with bucketed probabilities.
        We divide the session indices into 3 buckets (past/middle/future):
           past = first ceil(S_all/3) indices
           middle = next ceil(S_all/3)
           future = remaining indices
        We assign each bucket the total probability P_TARGET_BUCKET[*] and distribute uniformly
        inside the bucket (i.e., per-session weight = bucket_prob / bucket_size).
        This yields the overall mass near the requested bucket probs.
        """
        # bucket boundaries (rough thirds)
        s = S_all
        b1 = math.ceil(s / 3)                 # size of past bucket
        b2 = math.ceil(s / 3)                 # size of middle bucket (may overlap but it's fine)
        # compute index ranges
        past_idx = list(range(0, min(b1, s)))
        middle_idx = list(range(min(b1, s), min(b1 + b2, s)))
        future_idx = list(range(min(b1 + b2, s), s))

        # handle edge cases to ensure every session has at least some bucket
        buckets = []
        if len(past_idx) > 0:
            buckets.append(('past', past_idx))
        if len(middle_idx) > 0:
            buckets.append(('middle', middle_idx))
        if len(future_idx) > 0:
            buckets.append(('future', future_idx))

        # build per-session weights
        weights = np.zeros(s, dtype=float)
        for name, idxs in buckets:
            if len(idxs) == 0:
                continue
            total_prob = P_TARGET_BUCKET[name]
            per_session = total_prob / len(idxs)
            for ii in idxs:
                weights[ii] = per_session

        # normalize (numerical safety)
        if weights.sum() <= 0:
            weights[:] = 1.0 / s
        else:
            weights = weights / weights.sum()

        # sample according to weights
        chosen = int(self.rng.choice(s, p=weights))
        return chosen

    def _choose_k(self) -> int:
        if P_K is None:
            return int(self.rng.integers(1, 4))  # 1..3
        # else sample from distribution P_K dictionary
        ks = sorted(P_K.keys())
        probs = np.array([P_K[k] for k in ks], dtype=float)
        probs = probs / probs.sum()
        return int(self.rng.choice(ks, p=probs))

    def __getitem__(self, idx):
        """
        idx ranges up to len(self) = N_patients * samples_per_patient.
        We map idx -> patient_idx and sample one training example from that patient.
        """
        patient_idx = (idx // self.samples_per_patient) % len(self.file_dicts)
        file_dict = self.file_dicts[patient_idx]
        print("idx ", idx)
        print("patient_idx ", patient_idx)


        # Load arrays
        data = {k: np.load(file_dict[k]) for k in npz_keys}
        img_full = data['image']    # (C_times_S, D, H, W)
        lbl_full = data['label']    # (S_all, D, H, W)
        days_full = data['days']    # (S_all,)
        treat_full = data['treatment']  # (S_all,)
        geno = data["geno"]

        if self.geno_mean is not None and self.geno_std is not None:
            geno = (geno - self.geno_mean) / self.geno_std

        if geno is None:
            print("GENO IS NONE FOR PT ", patient_idx)



        S_all = int(lbl_full.shape[0])
        C_times_S, D, H, W = img_full.shape
        C_full = C_times_S // S_all
        assert C_full * S_all == C_times_S, f"image channels ({C_times_S}) not divisible by sessions ({S_all})"

        S_treat = treat_full.shape[0]
        if S_all != S_treat:
            treat_full = treat_full[: S_all - 1]
            days_full = days_full[: S_all - 1]

        # print("Patient : ", patient_idx)
        # print("img_full : ", img_full.shape)
        # print("lbl_full : ", lbl_full.shape)
        # print("days_full : ", days_full.shape)
        # print("treat_full : ", treat_full.shape)
        if treat_full[0] == -1 :
            print("WARNING TREATMENT -1?? IDK WHY!! for patient ", patient_idx )
        S = S_all
        CS = img_full.shape[0]           
        C = CS // S                      # number of modalities (should be 4)
        D, H, W = img_full.shape[1:]

        img_full = img_full.reshape(C, S, H, W, D)  # (C, S, H, W, D)
        img_full = np.moveaxis(img_full, 0, 1)  # -> (S, C, H, W, D)

        # Reshape images to (C_full, S_all, H, W, D)
        # img_full = img_full.reshape(C_full, S_all, D, H, W)   
        # img_full = np.transpose(img_full, (0, 1, 3, 4, 2))    # (C, S, H, W, D)
        # img_full = np.moveaxis(img_full, 0, 1)  # -> (S, C, H, W, D)

        S, D, H, W = lbl_full.shape
        lbl_full = lbl_full.reshape(S, H, W, D)

        print("img_full : ", img_full.shape)

        # keep first 3 modalities (if T2 was removed in data creation)
        # img_full = img_full[:3, ...]   # ( S_all, C_use, H, W, D)
        S_treat = treat_full.shape[0]
        img_full = img_full[:, :3, ...]

        # Select target session biased by bucket probs
        target_idx = self._choose_target_session(S_treat)

        # Choose k inputs and sample WITH replacement from remaining sessions
        remaining = [i for i in range(S_treat) if i != target_idx]
        if len(remaining) == 0:
            # degenerate case: only one session -> use it as both input and target (repeat)
            chosen_inputs = [target_idx] * 3
            chosen_sorted = chosen_inputs.copy()
        else:
            k = self._choose_k()  # 1..3
            # draw k items from remaining with replacement
            chosen = [int(self.rng.choice(remaining)) for _ in range(k)]
            # sort for temporal order
            chosen_sorted = sorted(chosen)
            # pad to length 3 by repeating last chosen
            while len(chosen_sorted) < 3:
                chosen_sorted.append(chosen_sorted[-1])

        # final order (inputs then target)
        seq_order = chosen_sorted + [target_idx]   # length == 4

        # --- Build the 4-session blocks from full patient arrays ---
        # img_full: (C_use, S_all, H, W, D) -> select sessions
        # print(img_full.shape)
        # print(seq_order)
        # print("treat_full", treat_full)
        # print("days_full", days_full)

        img_sel = img_full[ seq_order,:, :, :, :]  
        lbl_sel = lbl_full[seq_order, ...]          # shape: (4, D, H, W)
        days_sel = np.asarray(days_full)[seq_order]    # (4,)
        treat_sel = np.asarray(treat_full)[seq_order]  # (4,)
        # print(days_sel)
        # print(treat_sel)
        # days_int, treat_int = build_days_and_treat_intervals(seq_order, days_sel, treat_sel)
        # print(days_int)
        # print(treat_int)
        # # --- Prepare for MONAI transforms: collapse session+modalities into channels ---
        # # target transform shape: image_for_t (C_use*4, D, H, W)
        # img_for_t = np.transpose(img_sel, (0, 1, 4, 2, 3))  # (C_use, 4, D, H, W)
        # img_for_t = img_for_t.reshape(C_use * 4, D, H, W)   # (C_use*4, D, H, W)
        # lbl_for_t = lbl_sel  # (4, D, H, W)

        # transform_input = {
        #     'image': img_for_t,
        #     'label': lbl_for_t,
        #     'days': days_sel,
        #     'treatment': treat_sel,
        #     'geno': geno
        # }

        # # Apply provided transform (MUST be non-loading) — safe if user passed non_load_val_transforms
        # if self.transform:
        #     transformed = self.transform(transform_input)
        # else:
        #     transformed = transform_input

        # img_t = transformed['image']
        # lbl_t = transformed['label']
        # days_t = transformed.get('days', days_sel)
        # treat_t = transformed.get('treatment', treat_sel)

        # # Convert tensors -> numpy for reshape if needed
        # if isinstance(img_t, torch.Tensor):
        #     img_t = img_t.detach().cpu().numpy()
        # if isinstance(lbl_t, torch.Tensor):
        #     lbl_t = lbl_t.detach().cpu().numpy()

        # # Validate expected shapes
        # assert img_t.ndim == 4, f"Transformed image must be 4D (C, D, H, W). got {img_t.shape}"
        # CxS, Dn, Hn, Wn = img_t.shape
        # assert CxS == C_use * 4, f"unexpected channel count {CxS} != {C_use}*4"
        # assert lbl_t.ndim == 4 and lbl_t.shape[0] == 4, f"unexpected label shape {lbl_t.shape}"

        # # reshape back => (4, C_use, Hn, Wn, Dn)
        # img_back = img_t.reshape(C_use, 4, Dn, Hn, Wn)
        # img_back = np.transpose(img_back, (1, 0, 3, 4, 2))  # sessions first: (4, C_use, Hn, Wn, Dn)

        # # labels: (4, Dn, Hn, Wn) -> (4, Hn, Wn, Dn)
        # lbl_back = np.transpose(lbl_t, (0, 2, 3, 1))

        # convert to torch tensors
        image_tensor = torch.from_numpy(img_sel).float()    # (4, C, H, W, D)
        label_tensor = torch.from_numpy(lbl_sel).float()    # (4, H, W, D)
        days_tensor = torch.from_numpy(np.asarray(days_sel)).long()
        treat_tensor = torch.from_numpy(np.asarray(treat_sel)).long()

        sample_info = {
            'patient_idx': patient_idx,
            'patient_id_path': file_dict,
            'seq_order': seq_order,            # indices used (inputs then target)
            'chosen_inputs': chosen_sorted,    # pre-padded inputs (for debug)
            'target_original_idx': int(target_idx)
        }

        # print("Returneddd  by sampler for loader : ")
        # print("image_tensor : ", image_tensor.shape)
        # print("label_tensor : ", label_tensor.shape)
        # print("days_tensor : ", days_tensor.shape)
        # print("treat_tensor : ", treat_tensor.shape)
        # also save which sessions were sampled
        # with open("debug_patient_sampler/sample_info.txt", "w") as f:
        #     f.write(f"seq_order: {seq_order}\n")
        #     f.write(f"chosen_inputs: {chosen_sorted}\n")
        #     f.write(f"target_idx: {target_idx}\n")
        #     f.write(f"img_back shape: {img_back.shape}\n")
        #     f.write(f"lbl_back shape: {lbl_back.shape}\n")

        import os
        import matplotlib.pyplot as plt
        os.makedirs("debug_patient_sampler", exist_ok=True)

        # choose mid-slice in depth
        midz = img_sel.shape[-1] // 2  # Dn
        # save session 0, modality 0
        plt.imsave(
            "debug_patient_sampler/imgsel_s0_m0_midz.png",
            img_sel[0, 0, :, :, midz],
            cmap="gray"
        )

        # save matching label slice (session 0)
        # lbl_back is (4, H, W, D)
        plt.imsave(
            "debug_patient_sampler/lblsel_s0_midz.png",
            lbl_sel[0, :, :, midz].astype("int16"),
            cmap="tab10",
            vmin=0,
            vmax=int(lbl_sel.max()) if lbl_sel.size > 0 else 1
        )


        return {
            'image': image_tensor,      # (4, C, H, W, D)
            'label': label_tensor,      # (4, H, W, D)
            'days': days_tensor,        # (4,)
            'treatment': treat_tensor,  # (4,)
            'geno': geno,
            'sample_info': sample_info
        }

import numpy as np

def build_days_and_treat_intervals(seq_order, days, treat_sel, fill_days=0, fill_treat=0):
    """
    seq_order: list/array of selected session indices (ideally length 4)
    days:      (S_all-1,) interval days: days[i] corresponds to session i -> i+1
    treat_sel: (S_all-1,) interval treatment: treat_sel[i] corresponds to session i -> i+1

    Returns:
      days_int  (3,) where days_int[i]  = sum(days between seq_order[i] and seq_order[i+1])
      treat_int (3,) where treat_int[i] = treat_sel[ seq_order[i+1] - 1 ]  (interval right before next session)
    """
    seq = list(map(int, seq_order))
    days = np.asarray(days)
    treat_sel = np.asarray(treat_sel)

    n_edges_days = len(days)         # should be S_all-1
    n_edges_treat = len(treat_sel)   # should be S_all-1

    days_int = np.full((3,), fill_days, dtype=days.dtype if n_edges_days else np.int64)
    treat_int = np.full((3,), fill_treat, dtype=treat_sel.dtype if n_edges_treat else np.int64)

    for i in range(3):
        if i + 1 >= len(seq):
            break

        a = seq[i]
        b = seq[i + 1]

        # ---- days interval: computed from consecutive selected sessions ----
        if n_edges_days == 0:
            days_int[i] = fill_days
        elif a == b:
            days_int[i] = 0
        else:
            lo, hi = (a, b) if a < b else (b, a)
            # clamp to [0, n_edges_days]
            lo = max(0, min(lo, n_edges_days))
            hi = max(0, min(hi, n_edges_days))
            days_int[i] = days[lo:hi].sum() if hi > lo else 0

        # ---- treatment interval: "next element's interval" => treat_sel[b-1] ----
        t_idx = b - 1
        if 0 <= t_idx < n_edges_treat:
            treat_int[i] = treat_sel[t_idx]
        else:
            treat_int[i] = fill_treat

    return days_int, treat_int
