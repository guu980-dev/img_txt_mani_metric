import os 
# import sys
# sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import argparse
from PIL import Image
from tqdm import tqdm
import torch
from torch.nn import functional as F
torch.set_grad_enabled(False)

from eval_class import *
from augclip import AugCLIP
from utils import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vl_model", type=str, default="ViT-B/16")
    parser.add_argument("--dataset", type=str, default="editval")
    parser.add_argument("--data_dir", type=str, default="../../hdd/soohyun_workspace/data/")
    parser.add_argument("--desc", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--result_dir", type=str, default='./scores/')
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
   
    augclip = AugCLIP(args.device, args.vl_model)

    ASSETS = read_json(f"{args.data_dir}{args.dataset}/edit_prompt.json")
    if args.dataset == 'editval':
        ASSETS = [item for item in ASSETS if item['attribute'] not in ['object_addition', 'positional_addition', 'position_replacement']]
    if args.desc:
        descs = read_json(args.desc)
        for item in ASSETS:
            item['target_desc'] = descs[item['output_name'].split('/')[-1]]
        
    MODELS = MODEL_DICT[args.dataset]
        
    scores = {}
    failures = {}
    for case in tqdm(ASSETS):
        per_case = {}
        per_failure = []
        
        img_path = case['img_name'].replace('ROOT_DIR/', args.data_dir)
        manip_path = case['output_name'].replace('ROOT_DIR/', args.data_dir)
        
        src_image = Image.open(img_path).convert("RGB").resize((512, 512))
        
        src_text, tgt_text = case["prompt"]
        src_object, tgt_object = case["object"]
        src_desc, tgt_desc = case["source_desc"], case["target_desc"]
        src_set = [f"{src_text} with {desc}".replace(".", "") for desc in src_desc]
        tgt_set = [f"{tgt_text} with {desc}".replace(".", "") for desc in tgt_desc]
        
        src_img_feat = augclip.get_image_embeddings([src_image])
        src_text_feat, tgt_text_feat = augclip.get_text_embeddings([src_text]), augclip.get_text_embeddings([tgt_text])
        src_desc_feat, tgt_desc_feat = augclip.get_text_embeddings(src_desc), augclip.get_text_embeddings(tgt_desc)
        src_set_feat, tgt_set_feat = augclip.get_text_embeddings(src_set), augclip.get_text_embeddings(tgt_set)
        delta_T = F.normalize(tgt_text_feat - src_text_feat, p=2, dim=-1)
        
        for model in MODELS:
            manip_model_path = str(manip_path).replace("MODEL_NAME", model)
            try:
                manip_image = Image.open(manip_model_path).convert("RGB").resize((512, 512))
            except:
                per_failure.append(model)
                continue
                
            tgt_img_feat = augclip.get_image_embeddings([manip_image])
            delta_I = F.normalize(tgt_img_feat - src_img_feat, p=2, dim=-1)
        
            clips = (delta_T @ delta_I.T).detach().cpu().numpy().tolist()[0]
            clips_tgt = (tgt_text_feat @ delta_I.T).detach().cpu().numpy().tolist()[0]
        
            input_arguments = {
                "src_img_feat": src_img_feat,
                "tgt_img_feat": tgt_img_feat,
                "src_text_feat": src_text_feat,
                "tgt_text_feat": tgt_text_feat,
                "src_desc_feat": src_desc_feat,
                "tgt_desc_feat": tgt_desc_feat,
                "src_set_feat": src_desc_feat,
                "tgt_set_feat": tgt_desc_feat,
                "thres": 0 # probability of target + src attr happening
            }

            try:
                augclips = augclip.compute(**input_arguments)
            except:
                per_failure.append(model + '_cal')
                continue
        
            per_case[model] = clips + clips_tgt + [float(augclips)]
            
        scores[manip_path.split('/')[-1]] = [img_path.split('/')[-1], per_case]
        if len(per_failure) > 0:
            failures[manip_path.split('/')[-1]] = [img_path.split('/')[-1], per_failure]
        
    write_json(f'{args.result_dir}{args.dataset}.json', scores)
    if len(failures) > 0:
        write_json(f'{args.result_dir}{args.dataset}_failures.json', failures)