import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import PIL
from pathlib import Path
import argparse
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from openai import OpenAI
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# FILL API KEY HERE
client = OpenAI()

from utils import *


def generated_captions(model, processor, image, prompt):
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0")
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    caption, entities = processor.post_process_generation(generated_text)
    return caption, entities

def generate_descs_source(source_caption, source_text, target_text, source_object, target_object, args):
    if args.prompt_option==0:
        #! 30 visual characteristics from image caption
        messages=[
            {
                "role": "system", 
                "content": f"Give me a python list of 30 visual characteristics \
                    describes {source_caption}."
            },
            {
                "role": "user", 
                "content": "Return python list of that 30 text \
                    descriptions that depict a photo of a fluffy brown coated dog"
            },
            {"role": "assistant", "content": 
                        """
                        [
                            "Furry coat",
                            "Four legs",
                            "Tail wagging",
                            "Barking",
                            "Playful behavior",
                            "Snout",
                            "Collar",
                            "Leash",
                            "Walking on all fours",
                            "Wagging tail",
                            ]
                        """ 
            }
            ,
            {
                "role": "user", 
                "content": f"Give me a python list of 30 visual characteristics \
                    describes {source_caption}."}
        ]
    
    elif args.prompt_option==1:
        #! Get visual characteristics in source image caption compared to the target text
        
        messages=[
            {
                "role": "user", 
                "content": "Return python list of that 30 text \
                    descriptions that depict a photo of a fluffy brown coated dog"
            },
            {"role": "assistant", "content": 
                        """
                        [
                            "Furry coat",
                            "Four legs",
                            "Tail wagging",
                            "Barking",
                            "Playful behavior",
                            "Snout",
                            "Collar",
                            "Leash",
                            "Walking on all fours",
                            "Wagging tail",
                            ]
                        """ 
            }
            ,
            {
                "role": "user", 
                "content": f"Give me a python list of 30 visual characteristics \
                    describes {source_caption}."}
        ]

        question = f"""
            
            For example, 
            1. Source: "a photo of a dog", Target: "a photo of a cat", answer: ["Slanted almond-shaped eyes", "Soft and fluffy fur coat", "Long and elegant whiskers", "Pointed ears with tufts of fur", "Graceful and agile movements", ...]
            2. Source: "a photo of a horse", Target: "A photo of a breakdancing horse", answer: ["Windmill", "Twisting body", "Fur and ears flowing", ...]

            Source: {source_text}
            Target: {target_text}
            The answer is:
            """
        messages = [{"role": "user", "content": question}]

    elif args.prompt_option==2:
        messages = [
            {
                "role": "system", 
                "content": f"""
                            Extract all descriptions on the given entity in the sentence.
                            """
            },
            {
                "role": "user", 
                "content": f"""
                            Given a sentence 
                            'red haired girl in the image has visual features like curly hair, wearing a hat and blue pajamas and holding a cup of coffee, 
                            while a black haired woman is wearing black suit and writing down notes. They are both in a cafe, sitting around a round table'
                            Extract all descriptions on the entity 'red haired girl'.
                            Then given the descsriptions, augment the given descriptions into a sub-categories of visual characteristics.
                            """
            },
            {"role": "assistant", 
            "content": 
                        """
                        descs: [
                            "Red haired girl has curly hair",
                            "Red haired girl is wearing a hat", 
                            "Red haired girl is wearing blue pajamas", 
                            "Red haired girl is holding a cup of coffee",
                            "Red haired girl is in a cafe",
                            "Red haired girl is sitting around a round table"
                        ]
                        """ 
            },
            {
                "role": "user", 
                "content": f"""
                            Given a sentence 
                            '{source_caption}'
                            Extract all descriptions on the entity '{source_object}'.

                            """
            }
        ]
    elif args.prompt_option==3:
        messages = [
            {
                "role": "system", 
                "content": f"""
                            Extract all descriptions on the given entity in the sentence.
                            """
            },
            {
                "role": "user", 
                "content": f"""
                            Given a sentence 
                            'An image of red haired girl in the image has visual features like 
                            curly hair, wearing a hat and blue pajamas and holding a cup of coffee, 
                            while a black haired woman is wearing black suit and writing down notes. 
                            They are both in a cafe, sitting around a round table':

                            1. Parse the given sentence and extract all descriptions about the entity 'red haired girl'.
                            2. Generate extra visual characteristics of red-haired girl.
                            """
            },
            {"role": "assistant", 
            "content": 
                        """
                            [
                            "caption: 
                            "Parsed: Red haired girl has curly hair",
                            "Parsed: Red haired girl is wearing a hat", 
                            "Parsed: Red haired girl is wearing blue pajamas", 
                            "Parsed: Red haired girl is holding a cup of coffee",
                            "Parsed: Red haired girl is in a cafe",
                            "Parsed: Red haired girl is sitting around a round table",
                            "Generated: Girl has long hairs", 
                            "Generated: Girl has feminine features",
                            "Generated: Girl holds a coffee and drinks",
                            "Generated: Girl sitting on the chair in a cafe",
                            ]
                        """ 
            },
            {
                "role": "user", 
                "content": f"""
                            Given a sentence '{source_caption}':

                            1. Parse the given sentence and extract all descriptions about the entity {source_text}.
                            2. Generate extra visual characteristics of {source_text}.

                            """
            }
        ]

    completion = client.chat.completions.create(
        model='gpt-3.5-turbo', 
        messages=messages,
        temperature=args.temp,
        frequency_penalty=args.freq_pen,
    )
    response = completion.choices[0].message.content
    x = str(response)
    s_idx, e_idx = x.find("["), x.find("]")
    desc_list = x[s_idx+1: e_idx].split(",")
    desc_list = [desc.strip()[1:-1] for desc in desc_list if len(desc.strip()[1:-1])]
    processed_desc_list = [desc.split(":")[-1].strip() for desc in desc_list]
    return processed_desc_list

def generate_descs_target(source_caption, source_text, target_text, source_object, target_object, args):
    if args.prompt_option==0:
        messages=[
            {
                "role": "system", 
                "content": f"Give me a python list of 30 visual characteristics \
                    describes {target_text}."
            },
            {
                "role": "user", 
                "content": "Return python list of that 30 text \
                    descriptions that depict a photo of a fluffy brown coated dog"
            },
            {"role": "assistant", "content": 
                        """
                        [
                            "Furry coat",
                            "Four legs",
                            "Tail wagging",
                            "Barking",
                            "Playful behavior",
                            "Snout",
                            "Collar",
                            "Leash",
                            "Walking on all fours",
                            "Wagging tail",
                            ]
                        """ 
            }
            ,
            {
                "role": "user", 
                "content": f"Give me a python list of 30 visual characteristics \
                    describes {target_text}."}
        ]
    
    elif args.prompt_option==1:
        question = f"""
            Given the source text and target text, give me a python list of 30 visual characteristics that a target might have. 
            Note that you should extract the characteristics of only the target compared to the source.
            The visual features must be easy to portray: color, texture, shape, material, objects that are seen together, usage and etc.
            For example, 
            1. Source: "a photo of a dog", Target: "a photo of a cat", answer: ["Slanted almond-shaped eyes", "Soft and fluffy fur coat", "Long and elegant whiskers", "Pointed ears with tufts of fur", "Graceful and agile movements", ...]
            2. Source: "a photo of a horse", Target: "A photo of a breakdancing horse", answer: ["Windmill", "Twisting body", "Fur and ears flowing", ...]

            Source: {source_text}
            Target: {target_text}
            The answer is:
            """
        messages = [{"role": "user", "content": question}]
        
    elif args.prompt_option==2:
        question = f"""
            Given the source text and target text, give me a python list of 30 visual characteristics that describes the target. 
            Note that it is important to extract visual features that well represent the characteristics of the target in comparison to the source.
            For example, 
            1. Source: "a photo of a dog", Target: "a photo of a cat", answer: ["Slanted almond-shaped eyes", "Fluffy fur coat", "Long and elegant whiskers", "Pointed ears", "Graceful and agile movements", ...]
            2. Source: "a photo of a horse", Target: "A photo of a breakdancing horse", answer: ["Twisting body", "Bending knees", "Fur in motion", "Head tilted", ...]

            Source: {source_text}
            Target: {target_text}
            The answer is:
            """
        messages = [{"role": "user", "content": question}]
   
    elif args.prompt_option==3:
        
        messages = [
            {
                "role": "system", 
                "content": f"""
                            Provide image level characteristics 
                            such as color, texture, object category information, 
                            context of appearance, background 
                            of a given text that represents the image.
                            """
            },
            {
                "role": "user", 
                "content": f"""
                            Given a sentence, analyze the 
                            visual characteristics of the image where 'A dog is breakdancing',
                            that is not shown in the image of 'A dog is sitting'.
                            For example, focus on the how breakdancing is different from sitting only and do not mention dog's appearance.
                            Rearrange into a python list format.
                            """
            },
            {"role": "assistant", 
            "content": 
                        """
                        [
                            "motion: A dog shows dynamic movement",
                            "motion: A dog is spinning on its head", 
                            "environment: A dog is dancing on dance floor",
                            "environment: A dog is dancing in a hip hop scene",
                            "appearance: A dog shows stylish dancer outfit", 
                            "appearance: A dog has its legs up in the air",
                        ]
                        """ 
            },
            {
                "role": "user", 
                "content": f"""
                Given a sentence, analyze the 
                visual characteristics of the image where {target_text},
                that is not shown in the image of {source_text}.
                Rearrange into a python list format.
                """
            }
        ]
    elif args.prompt_option==4:
        
        messages = [
            {
                "role": "system", 
                "content": f"""
                            Provide image level characteristics 
                            such as color, texture, object category information, 
                            context of appearance, background 
                            of a given text that represents the image.
                            """
            },
            {
                "role": "user", 
                "content": f"""
                            Given a sentence, analyze the 
                            visual characteristics of the image where 'A dog is breakdancing',
                            that is not shown in the image of 'A dog is sitting'.
                            For example, focus on the how breakdancing is different from sitting only and do not mention dog's appearance.
                            Rearrange into a python list format.
                            """
            },
            {"role": "assistant", 
            "content": 
                        """
                        [
                            "motion: A dog shows dynamic movement",
                            "motion: A dog is spinning on its head", 
                            "environment: A dog is dancing on dance floor",
                            "environment: A dog is dancing in a hip hop scene",
                            "appearance: A dog shows stylish dancer outfit", 
                            "appearance: A dog has its legs up in the air",
                        ]
                        """ 
            },
            {
                "role": "user", 
                "content": f"""
                Given a sentence, analyze the 
                visual characteristics of the image where {target_text},
                that is not shown in the image of {source_text}.
                Rearrange into a python list format.
                """
            }
        ]
    
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo', 
        messages=messages,
        temperature=args.temp,
        frequency_penalty=args.freq_pen,
    )
    x = str(completion.choices[0].message.content)
    s_idx, e_idx = x.find("["), x.find("]")
    desc_list = x[s_idx+1: e_idx].split(",")
    desc_list = [desc.strip()[1:-1] for desc in desc_list if len(desc.strip()[1:-1])]
    processed_desc_list = [desc.split(":")[-1].strip() for desc in desc_list]
    return processed_desc_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sameswap", choices=["sameswap", "CelebA", "tedbench"])
    parser.add_argument("--data_dir", type=str, default="/home/server08/yoonjeon_workspace/augclip_data/")
    parser.add_argument("--freq_pen", type=float, default=0.)
    parser.add_argument("--prompt_option", type=int, default=2)
    parser.add_argument("--temp", type=float, default=0.)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    args.trial = f"trial{args.prompt_option}_fpen{args.freq_pen}_temp{args.temp}"
    os.makedirs(f"{args.trial}/", exist_ok=True)
    model = "kosmos"

    edit_prompt_path = str(Path(args.data_dir) / f"{args.dataset}/edit_prompt_new.json")
    edit_data= read_json(edit_prompt_path)
    if args.dataset == "dreambooth":
        all_images = set([(item['img_name'], (item['prompt'][0], '')) for item in edit_data])
        edit_data = [{'img_name': path[0], 'prompt': path[1], 'output_name': path[0]} for path in all_images]

    desc_src_path = f"{args.trial}/{args.dataset}_src.json"
    desc_tgt_path = f"{args.trial}/{args.dataset}_tgt.json"
    descs_src = read_json_or_dict(desc_src_path)
    descs_tgt = read_json_or_dict(desc_tgt_path)

    #! Select captioning model: BLIP or Kosmos
    if model=="blip":
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(args.device)
    elif model=="kosmos":
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224").to(args.device)

    prompt_dict = {
        "CelebA": (lambda src_obj, src_prompt: f"<grounding>An image has facial features like {src_prompt}, "),
        "sameswap": (lambda src_obj, src_prompt: f"<grounding>An image of {src_obj} "),
        "tedbench": (lambda src_obj, src_prompt: f"<grounding>An image of {src_obj} "),
    }

    
    for item in tqdm(edit_data):
        output_name = item["output_name"].split('/')[-1]
        src_prompt, tgt_prompt = item["prompt"]
        src_obj, tgt_obj = item["object"]
        img_path = item["img_name"].replace('ROOT_DIR/', args.data_dir)
        image = PIL.Image.open(img_path)
        

        prompt = prompt_dict[args.dataset](src_obj, src_prompt)
        if "caption" in item:
            caption = item["caption"]
            entities = item["entities"]
        else:
            caption, entities = generated_captions(model, processor, image, prompt)
            item["caption"] = caption
            item["entities"] = entities

        # if output_name not in descs_src:
        #     desc = generate_descs_source(caption, src_prompt, tgt_prompt, src_obj, tgt_obj, args)
        #     descs_src[output_name] = desc
        
        # if output_name not in descs_tgt:
        #     desc = generate_descs_target(caption, src_prompt, tgt_prompt, src_obj, tgt_obj, args)
        #     descs_tgt[output_name] = desc
        
        
    write_json(desc_src_path, descs_src)
    write_json(desc_tgt_path, descs_tgt)
    write_json(str(Path(args.data_dir) / f"{args.dataset}/edit_prompt.json").replace(".json", "_new.json"), edit_data)