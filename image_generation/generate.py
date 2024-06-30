import os, json, argparse
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv
import torch

def create_by_sd(prompt, image_path):
  load_dotenv()
  print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))

  model_id = "stabilityai/stable-diffusion-2-1"
  device = "cuda" if torch.cuda.is_available() else "cpu"

  pipe = StableDiffusionPipeline.from_pretrained(model_id)
  pipe = pipe.to(device)

  with torch.autocast("cuda"):
    image = pipe(prompt).images[0]

  image.save(image_path)

def generate_image(data_path, save_base_dir, data_name):
  skipped_log_path = os.path.join(save_base_dir, f'{data_name}_skipped_log.txt')
  error_log_path = os.path.join(save_base_dir, f'{data_name}_error_log.txt')
  skipped_images = []

  with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

    for image_name, prompt in data.items():
      image_path = os.path.join(save_base_dir, image_name)
      # Check image file already exists
      if os.path.exists(image_path):
        skipped_images.append(image_path)
        continue

      # Create directory if not exist
      os.makedirs(os.path.dirname(image_path), exist_ok=True)

      try:
        print(f"generate image: {image_name}")
        create_by_sd(prompt, image_path)
      except RuntimeError as e:
        with open(error_log_path, 'a', encoding='utf-8') as error_file:
          error_file.write(f"Failed on image {image_name}: {str(e)}\n")

  # Record skipped images
  with open(skipped_log_path, 'w', encoding='utf-8') as log_file:
    for skipped_image in skipped_images:
      log_file.write(skipped_image + '\n')

def main():
  # Get save base dir path
  parser = argparse.ArgumentParser(description='Retrieve Image using Google Custom Search API')
  parser.add_argument('--save_base_path', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result'), help='Base directory path to save images')
  args = parser.parse_args()

  # Prepare (Load data, create save directory)
  data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
  data_files = os.listdir(data_dir) # should be json

  for data_file in data_files:
    save_base_dir = args.save_base_path
    generate_image(os.path.join(data_dir, data_file), save_base_dir, data_file[:-5])

if __name__ == "__main__":
  main()