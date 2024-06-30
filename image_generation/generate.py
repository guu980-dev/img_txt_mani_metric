import os, json, argparse
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from dotenv import load_dotenv
import torch
from huggingface_hub import login, hf_hub_download

def create_by_sd(prompt, image_path):
  load_dotenv()
  print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))

  # Hugging Face Hub 토큰으로 로그인
  hf_token = os.getenv("HUGGINGFACE_TOKEN")
  if hf_token:
      login(token=hf_token)

  base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
  refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # 로컬 경로에서 모델 로드
  base_pipe = DiffusionPipeline.from_pretrained(
    base_model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
  )
  base_pipe.to(device)

  refiner_pipe = DiffusionPipeline.from_pretrained(
    refiner_model_id,
    text_encoder_2=base_pipe.text_encoder_2,
    vae=base_pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
  )
  refiner_pipe.to(device)
  # base_pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32)
  # base_pipe = base_pipe.to(device)

  # refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(refiner_model_id, torch_dtype=torch.float32)
  # refiner_pipe = refiner_pipe.to(device)

  # 초기 이미지 생성
# image = base(
#     prompt=prompt,
#     num_inference_steps=40,
#     denoising_end=0.8,
#     output_type="latent",
# ).images
# image = refiner(
#     prompt=prompt,
#     num_inference_steps=40,
#     denoising_start=0.8,
#     image=image,
# ).images[0]
# image

  base_image = base_pipe(prompt).images[0]

  torch.cuda.empty_cache()  # 메모리 비우기

  # Refiner를 사용하여 이미지 개선
  refined_image = refiner_pipe(prompt=prompt, image=base_image).images[0]

  refined_image.save(image_path)



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
        # else:
        #   raise RuntimeError("Failed to generate image (Image None)")
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