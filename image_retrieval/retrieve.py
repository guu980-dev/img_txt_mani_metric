from selenium import webdriver
from selenium.webdriver.common.by import By
import os, json, argparse, time, base64
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def find_first_element(driver):
  selectors = ['img.YQ4gaf', '.rg_i']
  for selector in selectors:
    try:
      first_image_element = None
      image_elements = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
      )

      for image_element in image_elements:
        # Filter 46x46 case since it's usually related keyword with image
        width = image_element.get_attribute("width")
        height = image_element.get_attribute("height")
        if width and height:
          width = int(width)
          height = int(height)
          if width > 46 and height > 46:
            first_image_element = image_element
            break

      if not first_image_element:
        print("Can't find first_image_element")
        driver.quit()
        return

      else:
        print(f'selector: {selector} succeed')
        return first_image_element

    except:
      print(f'selector: {selector} failed')


def find_large_element(driver):
  selectors = ['img.sFlh5c.pT0Scc.iPVvYb', 'img.sFlh5c.pT0Scc', 'img.YsLeY']
  for selector in selectors:
    try:
      large_element = driver.find_element(By.CSS_SELECTOR, selector)
      print(f'selector: {selector} succeed')
      return large_element
    except:
      print(f'selector: {selector} failed')


def search_and_download_image(query, headless):
  options = webdriver.ChromeOptions()
  if headless:
    options.add_argument('headless')
  driver = webdriver.Chrome(options=options)

  search_url = f'https://www.google.com/imghp'
  driver.get(search_url)
  search_bar = driver.find_element(By.NAME,"q")
  search_bar.send_keys(query)
  search_bar.submit()

  # Find first searched image element
  time.sleep(2)  # wait for page loading
  first_image_element = find_first_element(driver)
  
  if not first_image_element:
    no_results_element = WebDriverWait(driver, 10).until(
      EC.presence_of_element_located((By.XPATH, "//div[contains(text(), \"이 주제에 관해 '이미지' 필터가 적용된 검색 결과가 없는 것 같습니다\")]"))
    )
    if no_results_element:
      cleaned_query = query.replace("\"", "")
      driver.get(search_url)
      search_bar = driver.find_element(By.NAME,"q")
      search_bar.send_keys(cleaned_query)
      search_bar.submit()
      first_image_element = find_first_element(driver)

  # Click thumbnail to move to real image
  first_image_element.click()
  time.sleep(2.5) # wait until large image is loaded

  # Load large image url
  large_image_element = find_large_element(driver)
  first_image_url = large_image_element.get_attribute('src')

  # Download image
  try:
    # Base64 case
    if first_image_url.startswith('data:image'):
      header, encoded = first_image_url.split(",", 1)
      image_data = base64.b64decode(encoded)
      image = Image.open(BytesIO(image_data))
    # Image url case
    else:
      headers = {
        # "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.google.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
      }
      image_response = requests.get(first_image_url, headers=headers)
      image_response.raise_for_status()
      image = Image.open(BytesIO(image_response.content))
  except Exception as e:
    print(f"Error on downloading image: {e}")
    raise e

  driver.quit()

  return image


def retrieve_image(data_path, save_base_dir, data_name, headless):
  skipped_log_path = os.path.join(save_base_dir, f'{data_name}_skipped_log.txt')
  error_log_path = os.path.join(save_base_dir, f'{data_name}_error_log.txt')
  skipped_images = []

  with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)['data']
    for each_image_data in data:
      image_name = each_image_data['img_name'].split('/')[-1]
      image_format = image_name.split('.')[1]
      image_path = os.path.join(save_base_dir, f'{data_name}/retrieval/{image_name}')
      # Check image file already exists
      if os.path.exists(image_path):
        skipped_images.append(image_path)
        continue

      # Create directory if not exist
      os.makedirs(os.path.dirname(image_path), exist_ok=True)

      try:
        print(f"download image: {image_name}")
        image = search_and_download_image(each_image_data['target_text'], headless)
        if image:
          if image.mode == 'RGBA' and image_format.lower() == 'jpeg':
            image = image.convert('RGB')
          image.save(image_path)
      except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP Request Exception on search: {e}")
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
  parser.add_argument('--headless', type=bool, default=False, help='Turn off GUI browser if this option is on. Default is false')
  args = parser.parse_args()

  # Prepare (Load data, create save directory)
  data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
  data_files = os.listdir(data_dir) # should be json

  for data_file in data_files:
    save_base_dir = args.save_base_path
    retrieve_image(os.path.join(data_dir, data_file), save_base_dir, data_file[:-5], args.headless)


if __name__ == '__main__':
  main()