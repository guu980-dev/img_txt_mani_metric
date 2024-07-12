import torch
import numpy as np
import argparse
import json
from tqdm import tqdm
from PIL import Image

from pathlib import Path
from eval_base import BaseScore

import torch.nn.functional as F
import numpy as np

class AugCLIP(BaseScore):
    def __init__(
            self, 
            device, 
            model_name,
            cov_type='empirical',
            ):
        
        super().__init__(device, model_name)
        # self.clustering = KMeans(n_clusters=n_cluster, n_init='auto')
        self.cs = lambda x, y: torch.einsum("ij, kj -> ik", x, y)
        self.src_thres = 0.2
        self.n_bin = 20

    def z_dist(self, V, img_set, src_img_feat, tgt_img_feat):
        X_distn = self.cs(V, img_set)
        sd = X_distn.std(1)
        sd_inv = (sd ** (-1))[:, np.newaxis]
        src, tgt = self.cs(V, src_img_feat), self.cs(V, tgt_img_feat)
        z_dist = (tgt - src) * sd_inv
        return z_dist
        
    def compute(self, src_img_feat, tgt_img_feat, src_text_feat, tgt_text_feat, src_desc_feat, tgt_desc_feat, src_set_feat, tgt_set_feat, thres):
        #! Filter out visual properties
        src_sims = self.cs(src_desc_feat, src_img_feat)
        in_source = (src_sims > self.src_thres).flatten() # check if the source image has the given source property
        src_desc_feat = src_desc_feat[in_source]
        src_desc_feat = torch.cat([src_text_feat, src_desc_feat]) # merge text prompt and descriptions
        tgt_desc_feat = torch.cat([tgt_text_feat, tgt_desc_feat])
        
        manip_score = None
        presv_score = None

        for name, desc, img_set in [("s", src_desc_feat, src_set_feat), ("t", tgt_desc_feat, tgt_set_feat)]:
            #! calculate overlap between source and target samples
            dist1 = self.cs(desc, src_set_feat)
            dist2 = self.cs(desc, tgt_set_feat)

            n_feat, n1 = dist1.shape
            _, n2 = dist2.shape
            hist1, hist2 = [], []
            for idx in range(n_feat):
                hist1.append(torch.histc(dist1[idx].float(), bins=self.n_bin, min=0, max=1) / n1)
                hist2.append(torch.histc(dist2[idx].float(), bins=self.n_bin, min=0, max=1) / n2)
            hist1, hist2 = torch.stack(hist1), torch.stack(hist2)
            o_v = (hist1 * hist2).sum(dim=-1)
            mask_p = o_v > thres
            mask_m = o_v <= thres

            if torch.any(mask_p):
                V = desc[mask_p]
                z_dist = torch.abs(self.z_dist(V, img_set, src_img_feat, tgt_img_feat)).detach().cpu().numpy()
                if presv_score is not None:
                    presv_score = np.concatenate((presv_score, z_dist), axis=0)
                else:
                    presv_score = z_dist

            if torch.any(mask_m):
                V = desc[mask_m]
                z_dist = self.z_dist(V, img_set, src_img_feat, tgt_img_feat).detach().cpu().numpy()
                if manip_score is not None:
                    manip_score = np.concatenate((manip_score, z_dist), axis=0)
                else:
                    manip_score = z_dist

        manip_score = manip_score.mean() if manip_score is not None else 0
        presv_score = presv_score.mean() if presv_score is not None else 0

        augclip = manip_score - presv_score
        return augclip