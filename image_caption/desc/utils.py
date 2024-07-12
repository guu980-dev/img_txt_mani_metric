import os
import json


MODEL_DICT = {
    'cat2dog': ['diffedit', 'instructpix2pix'],
    'dog2cat': ['diffedit', 'instructpix2pix'],
    'horse2zebra': ['diffedit', 'instructpix2pix'],
    'zebra2horse': ['diffedit', 'instructpix2pix'],
    'CelebA': ['asyrp', 'diffusionclip', 'multi2one', 'styleclip'],
    'dreambooth': ['DDS', 'diffedit', 'instructpix2pix', 'prompt-to-prompt'],
    'editval': ['DDS', 'diffedit', 'instructpix2pix', 'prompt-to-prompt'],
    'tedbench': ['DDS', 'diffedit', 'instructpix2pix', 'prompt-to-prompt', 'imagic', 'retrieval', 'gt', 'generation', 'noisy'],
    'MagicBrush': ['DDS', 'diffedit', 'instructpix2pix', 'prompt-to-prompt', 'gt', 'generation', 'noisy'],
}


def read_json_or_dict(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            result = json.load(f)
    else:
        result = {}
    return result

def read_json(path):
    with open(path, 'r') as f:
        result = json.load(f)
    return result

def write_json(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(content, f, indent='\t')

def process_numbered_list(response):
    items = response.split('\n')
    items = [item.split('. ')[-1] for item in items]
    return items

def process_multiline_list(response):
    items = response.split('\"')
    items = [item for i, item in enumerate(items) if i%2==1]
    return items