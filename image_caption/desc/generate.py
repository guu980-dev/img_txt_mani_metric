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
from dotenv import load_dotenv
import datetime
import torch
import clip

# FILL API KEY HERE
load_dotenv()
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

def get_source_desc_prompt(prompt_option, source_caption, source_text, target_text, source_object):
    if prompt_option==0:
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
    
    elif prompt_option==1:
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

    elif prompt_option==2:
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
    elif prompt_option==3:
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

    # 4: Visual 요소에 집중해라. Compare 을 위함이라는 목적을 제시. (차이 미비)
    elif prompt_option==4:
        messages = [
            {
                "role": "system", 
                "content": f"""
                            Extract all visual descriptions on the given entity in the sentence.
                            It will be used to compare visual characteristics changes of main visual part.
                            Please find important features of both similar part and different part from given two entities.
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
                            Extract all visual descriptions on the entity '{source_object}'.
                            It will be used to compare visual characteristics changes of {source_object} from {source_text} to {target_text}.
                            Please find important features of both similar part and different part from {source_text}, {target_text}.

                            """
            }
        ]

    # 5: 카테고리를 나눠 뽑아라
    elif prompt_option==5:
        messages = [
            {
                "role": "system", 
                "content": f"""
                            Extract all descriptions of the sentence.
                            Main entity of the sentence should be analyzed in detail.
                            Descriptions should be analyzed in six categories.
                            1. Landscape and Background: Natural Elements, Man-made Structures, Weather and Time
                            2. Subjects: People, Animals, Objects
                            3. Appearance: Physical Characteristics, Actions and Poses, Clothing and Accessories
                            4. Emotion and Atmosphere: Expressions and Emotions, Mood and Tone
                            5. Colors and Lighting: Colors, Lighting
                            6. Composition: Arrangement, Perspective
                            Important component should be highlighted and described in detail.
                            """
            },
            {
                "role": "user", 
                "content": f"""
                            Given a sentence 
                            'red haired girl in the image has visual features like curly hair, wearing a hat and blue pajamas and holding a cup of coffee, 
                            while a black haired woman is wearing black suit and writing down notes. They are both in a cafe, sitting around a round table'
                            Extract all descriptions in six categories and pay more attention on the main entity 'red haired girl'.
                            Then given the descsriptions, augment the given descriptions into a sub-categories of visual characteristics.
                            """
            },
            {"role": "assistant", 
            "content": 
                        """
                        descs: [
                            "Cafe with round table",
                            "Red haired girl has curly hair",
                            "Red haired girl is wearing a hat", 
                            "Red haired girl is wearing blue pajamas", 
                            "Red haired girl is holding a cup of coffee",
                            "Red haired girl is in a cafe",
                            "Red haired girl is sitting around a round table"
                            "Black haired woman has black hair",
                            "Black haired woman is wearing black suit",
                            "Black hiared woman is writing down notes",
                            "Black haired woman is in a cafe",
                            "Black haired woman is sitting around a round table",
                            "Black haired woman is in calm mood",
                            "Cup of coffee is on the table",
                            "The images of the two women seen from a distance"
                        ]
                        """ 
            },
            {
                "role": "user", 
                "content": f"""
                            Given a sentence 
                            '{source_caption}'
                            
                            Extract all descriptions of the sentence.
                            Main entity '{source_object}' should be analyzed in detail.
                            Descriptions should be analyzed in six categories.
                            1. Landscape and Background: Natural Elements, Man-made Structures, Weather and Time
                            2. Subjects: People, Animals, Objects
                            3. Appearance: Physical Characteristics, Actions and Poses, Clothing and Accessories
                            4. Emotion and Atmosphere: Expressions and Emotions, Mood and Tone
                            5. Colors and Lighting: Colors, Lighting
                            6. Composition: Arrangement, Perspective
                            Important part '{source_text}' should be highlighted and described in detail.

                            """
            }
        ]
    
    # 6: 예시 수정
    elif prompt_option==6:
        messages = [
            {
                "role": "system", 
                "content": f"""
                            Extract all descriptions of the sentence.
                            Main entity of the sentence should be analyzed in detail.
                            Descriptions should be analyzed in six categories.
                            1. Landscape and Background: Natural Elements, Man-made Structures, Weather and Time
                            2. Subjects: People, Animals, Objects
                            3. Appearance: Physical Characteristics, Actions and Poses, Clothing and Accessories
                            4. Emotion and Atmosphere: Expressions and Emotions, Mood and Tone
                            5. Colors and Lighting: Colors, Lighting
                            6. Composition: Arrangement, Perspective
                            Important component should be highlighted and described in detail.
                            """
            },
            {
                "role": "user", 
                "content": f"""
                            Given a sentence 
                            'red haired girl in the image has visual features like curly hair, wearing a hat and blue pajamas and holding a cup of coffee, 
                            while a black haired woman is wearing black suit and writing down notes. They are both in a cafe, sitting around a round table'
                            Extract all descriptions in six categories and pay more attention on the main entity 'red haired girl'.
                            Then given the descsriptions, augment the given descriptions into a sub-categories of visual characteristics.
                            """
            },
            {"role": "assistant", 
            "content": 
                        """
                        descs: [
                            "Red-haired girl has curly hair.",
                            "Red-haired girl is wearing a hat.",
                            "Red-haired girl is wearing blue pajamas",
                            "Red-haired girl is holding a cup of coffee.",
                            "Black-haired woman has black hair.",
                            "Black-haired woman is wearing a black suit.",
                            "Black-haired woman is writing down notes.",
                            "Red-haired girl appears relaxed as she holds a cup of coffee.",
                            "Black-haired woman seems focused as she writes notes.",
                            "Red-haired girl has red hair.",
                            "Red-haired girl and black-haired woman are sitting around a round table.",
                            "Red-haired girl and black-haired woman are both in a cafe.",
                        ]
                        """ 
            },
            {
                "role": "user", 
                "content": f"""
                            Given a sentence 
                            '{source_caption}'
                            
                            Extract all descriptions of the sentence.
                            Main entity '{source_object}' should be analyzed in detail.
                            Descriptions should be analyzed in six categories.
                            1. Landscape and Background: Natural Elements, Man-made Structures, Weather and Time
                            2. Subjects: People, Animals, Objects
                            3. Appearance: Physical Characteristics, Actions and Poses, Clothing and Accessories
                            4. Emotion and Atmosphere: Expressions and Emotions, Mood and Tone
                            5. Colors and Lighting: Colors, Lighting
                            6. Composition: Arrangement, Perspective
                            Important part '{source_text}' should be highlighted and described in detail.

                            """
            }
        ]
    
    # 6+Captioning 을 좀더 자세하게?
    
    # 7: 일단 카테고리에 따라 주루룩 나열하고(as detail as possible) 중요도에 따라 선별해라? (소거법)
    
    
    return messages


def generate_descs_source(source_caption, source_text, target_text, source_object, source_image, args):
    # if args.use_image_for_desc:
    #     messages = get_image_desc_prompt(args.prompt_option, source_caption, source_text, target_text, source_object, source_image)
    # else:
    #     messages = get_non_image_desc_prompt(args.prompt_option, source_caption, source_text, target_text, source_object)
    messages = get_source_desc_prompt(args.prompt_option, source_caption, source_text, target_text, source_object)


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

# def generate_descs_target(source_caption, source_text, target_text, source_object, target_object, args):
#     if args.prompt_option==0:
#         messages=[
#             {
#                 "role": "system", 
#                 "content": f"Give me a python list of 30 visual characteristics \
#                     describes {target_text}."
#             },
#             {
#                 "role": "user", 
#                 "content": "Return python list of that 30 text \
#                     descriptions that depict a photo of a fluffy brown coated dog"
#             },
#             {"role": "assistant", "content": 
#                         """
#                         [
#                             "Furry coat",
#                             "Four legs",
#                             "Tail wagging",
#                             "Barking",
#                             "Playful behavior",
#                             "Snout",
#                             "Collar",
#                             "Leash",
#                             "Walking on all fours",
#                             "Wagging tail",
#                             ]
#                         """ 
#             }
#             ,
#             {
#                 "role": "user", 
#                 "content": f"Give me a python list of 30 visual characteristics \
#                     describes {target_text}."}
#         ]
    
#     elif args.prompt_option==1:
#         question = f"""
#             Given the source text and target text, give me a python list of 30 visual characteristics that a target might have. 
#             Note that you should extract the characteristics of only the target compared to the source.
#             The visual features must be easy to portray: color, texture, shape, material, objects that are seen together, usage and etc.
#             For example, 
#             1. Source: "a photo of a dog", Target: "a photo of a cat", answer: ["Slanted almond-shaped eyes", "Soft and fluffy fur coat", "Long and elegant whiskers", "Pointed ears with tufts of fur", "Graceful and agile movements", ...]
#             2. Source: "a photo of a horse", Target: "A photo of a breakdancing horse", answer: ["Windmill", "Twisting body", "Fur and ears flowing", ...]

#             Source: {source_text}
#             Target: {target_text}
#             The answer is:
#             """
#         messages = [{"role": "user", "content": question}]
        
#     elif args.prompt_option==2:
#         question = f"""
#             Given the source text and target text, give me a python list of 30 visual characteristics that describes the target. 
#             Note that it is important to extract visual features that well represent the characteristics of the target in comparison to the source.
#             For example, 
#             1. Source: "a photo of a dog", Target: "a photo of a cat", answer: ["Slanted almond-shaped eyes", "Fluffy fur coat", "Long and elegant whiskers", "Pointed ears", "Graceful and agile movements", ...]
#             2. Source: "a photo of a horse", Target: "A photo of a breakdancing horse", answer: ["Twisting body", "Bending knees", "Fur in motion", "Head tilted", ...]

#             Source: {source_text}
#             Target: {target_text}
#             The answer is:
#             """
#         messages = [{"role": "user", "content": question}]
   
#     elif args.prompt_option==3:
        
#         messages = [
#             {
#                 "role": "system", 
#                 "content": f"""
#                             Provide image level characteristics 
#                             such as color, texture, object category information, 
#                             context of appearance, background 
#                             of a given text that represents the image.
#                             """
#             },
#             {
#                 "role": "user", 
#                 "content": f"""
#                             Given a sentence, analyze the 
#                             visual characteristics of the image where 'A dog is breakdancing',
#                             that is not shown in the image of 'A dog is sitting'.
#                             For example, focus on the how breakdancing is different from sitting only and do not mention dog's appearance.
#                             Rearrange into a python list format.
#                             """
#             },
#             {"role": "assistant", 
#             "content": 
#                         """
#                         [
#                             "motion: A dog shows dynamic movement",
#                             "motion: A dog is spinning on its head", 
#                             "environment: A dog is dancing on dance floor",
#                             "environment: A dog is dancing in a hip hop scene",
#                             "appearance: A dog shows stylish dancer outfit", 
#                             "appearance: A dog has its legs up in the air",
#                         ]
#                         """ 
#             },
#             {
#                 "role": "user", 
#                 "content": f"""
#                 Given a sentence, analyze the 
#                 visual characteristics of the image where {target_text},
#                 that is not shown in the image of {source_text}.
#                 Rearrange into a python list format.
#                 """
#             }
#         ]
#     elif args.prompt_option==4:
        
#         messages = [
#             {
#                 "role": "system", 
#                 "content": f"""
#                             Provide image level visual characteristics 
#                             such as color, texture, object category information, 
#                             context of appearance, background 
#                             of a given text that represents the image.
#                             """
#             },
#             {
#                 "role": "user", 
#                 "content": f"""
#                             Given a sentence, analyze the 
#                             visual characteristics of the image where 'A dog is breakdancing',
#                             that is not shown in the image of 'A dog is sitting'.
#                             For example, focus on the how breakdancing is different from sitting only and do not mention dog's appearance.
#                             Rearrange into a python list format.
#                             """
#             },
#             {"role": "assistant", 
#             "content": 
#                         """
#                         [
#                             "motion: A dog shows dynamic movement",
#                             "motion: A dog is spinning on its head", 
#                             "environment: A dog is dancing on dance floor",
#                             "environment: A dog is dancing in a hip hop scene",
#                             "appearance: A dog shows stylish dancer outfit", 
#                             "appearance: A dog has its legs up in the air",
#                         ]
#                         """ 
#             },
#             {
#                 "role": "user", 
#                 "content": f"""
#                 Given a sentence, analyze the 
#                 visual characteristics of the image where {target_text},
#                 that is not shown in the image of {source_text}.
#                 Rearrange into a python list format.
#                 """
#             }
#         ]
    
#     completion = client.chat.completions.create(
#         model='gpt-3.5-turbo', 
#         messages=messages,
#         temperature=args.temp,
#         frequency_penalty=args.freq_pen,
#     )
#     x = str(completion.choices[0].message.content)
#     s_idx, e_idx = x.find("["), x.find("]")
#     desc_list = x[s_idx+1: e_idx].split(",")
#     desc_list = [desc.strip()[1:-1] for desc in desc_list if len(desc.strip()[1:-1])]
#     processed_desc_list = [desc.split(":")[-1].strip() for desc in desc_list]
#     return processed_desc_list


def calculate_text_sim(model, preprocess, image_path, text, args):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(args.device)
    text = clip.tokenize([text]).to(args.device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).item()
    return similarity

# # with mean text embedding
# def calculate_descs_sim(model, preprocess, image_path, _descs, args):
#     image = preprocess(Image.open(image_path)).unsqueeze(0).to(args.device)
#     # descs = clip.tokenize(_descs).to(args.device)
#     descs = clip.tokenize([f"Image with {', '.join(_descs)}"]).to(args.device)
#     with torch.no_grad():
#         image_features = model.encode_image(image)
#         descs_features = model.encode_text(descs)

#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     descs_features /= descs_features.norm(dim=-1, keepdim=True)
    
#     similarities = (image_features @ descs_features.T).squeeze(0)
#     average_similarity = similarities.mean().item()
#     return average_similarity


# with mean text embedding
def calculate_descs_sim(model, preprocess, image_path, _descs, args):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(args.device)
    descs = clip.tokenize(_descs).to(args.device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        descs_features = model.encode_text(descs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    descs_features /= descs_features.norm(dim=-1, keepdim=True)
    mean_descs_features = descs_features.mean(dim=0, keepdim=True)
    mean_descs_features /= mean_descs_features.norm(dim=-1, keepdim=True)
    
    similarities = (image_features @ mean_descs_features.T).squeeze(0)
    average_similarity = similarities.mean().item()
    return average_similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sameswap", choices=["sameswap", "CelebA", "tedbench"])
    # parser.add_argument("--data_dir", type=str, default="/home/server08/yoonjeon_workspace/augclip_data/")
    parser.add_argument("--data_dir", type=str, default="/home/server21/hdd/hyunkoo_workspace/data/")
    parser.add_argument("--freq_pen", type=float, default=0.)
    parser.add_argument("--prompt_option", type=int, default=2)
    parser.add_argument("--temp", type=float, default=0.)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--crop_image", type=bool, default=False)
    args = parser.parse_args()
    
    args.trial = f"trial{args.prompt_option}_fpen{args.freq_pen}_temp{args.temp}"
    os.makedirs(f"{args.trial}/", exist_ok=True)
    model = "kosmos"

    # if args.dataset == "sameswap":
    #     args.use_image_for_desc = True

    edit_prompt_path = str(Path(args.data_dir) / f"{args.dataset}/edit_prompt_new.json")
    edit_data= read_json(edit_prompt_path)
    if args.dataset == "dreambooth":
        all_images = set([(item['img_name'], (item['prompt'][0], '')) for item in edit_data])
        edit_data = [{'img_name': path[0], 'prompt': path[1], 'output_name': path[0]} for path in all_images]

    desc_src_path = f"result/{args.trial}/{args.dataset}_src.json"
    desc_tgt_path = f"result/{args.trial}/{args.dataset}_tgt.json"
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

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=args.device)
    desc_sims = []
    text_sims = []
    desc_win_rate = 0
    for item in tqdm(edit_data):
        output_name = item["output_name"].split('/')[-1]
        src_prompt, tgt_prompt = item["prompt"]
        src_obj, tgt_obj = item["object"]
        img_path = item["img_name"].replace('ROOT_DIR/', args.data_dir)
        
        if args.crop_image:
            bbox_o = item["bbox_xywh"] # 순서대료 x, y, w, h 값임
            bbox_o = [bbox_o[0], bbox_o[1], bbox_o[0]+bbox_o[2], bbox_o[1]+bbox_o[3]]
            org_img = PIL.Image.open(img_path).resize((512,512)).crop(bbox_o)
        else:
            image = PIL.Image.open(img_path)
        

        prompt = prompt_dict[args.dataset](src_obj, src_prompt)
        if "caption" in item:
            caption = item["caption"]
            entities = item["entities"]
        else:
            caption, entities = generated_captions(model, processor, image, prompt)
            item["caption"] = caption
            item["entities"] = entities

        if output_name not in descs_src:
            desc = generate_descs_source(caption, src_prompt, tgt_prompt, src_obj, image, args)
            descs_src[output_name] = desc
        
        text_sim = calculate_text_sim(clip_model, clip_preprocess, img_path, caption, args)
        descs_sim = calculate_descs_sim(clip_model, clip_preprocess, img_path, desc, args)
        item["text_sim"] = text_sim
        text_sims.append(text_sim)
        item["descs_sim"] = descs_sim
        desc_sims.append(descs_sim)
        if descs_sim > text_sim:
            desc_win_rate += 1

        # if output_name not in descs_tgt:
        #     desc = generate_descs_target(caption, src_prompt, tgt_prompt, src_obj, tgt_obj, args)
        #     descs_tgt[output_name] = desc
        
    now = datetime.datetime.now()
    descs_src["E_cs_img_descs"] = sum(desc_sims) / len(desc_sims)
    descs_src["E_cs_img_text"] = sum(text_sims) / len(text_sims)
    descs_src["desc_txt_win_rate"] = desc_win_rate / len(desc_sims)
    write_json(desc_src_path, descs_src)
    
    # write_json(desc_tgt_path, descs_tgt)
    write_json(str(Path(args.data_dir) / f"{args.dataset}/edit_prompt.json").replace(".json", f"_new_{args.trial}_{now}.json"), edit_data)