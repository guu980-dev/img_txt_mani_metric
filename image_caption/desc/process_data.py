from pathlib import Path
from utils import *   

def main():
  edit_prompt_path = str(Path("/home/server21/hdd/hyunkoo_workspace/data/sameswap/") / f"{'/home/server21/hdd/hyunkoo_workspace/data/sameswap/'}/edit_prompt_new_orig.json")
  new_edit_prompt_path = str(Path("/home/server21/hdd/hyunkoo_workspace/data/sameswap/") / f"{'/home/server21/hdd/hyunkoo_workspace/data/sameswap/'}/edit_prompt_new_new.json")
  edit_data= read_json(edit_prompt_path)
  new_edit_data = []
  for key, value in edit_data.items():
    # Your code here
    new_edit_data.append(value)
    pass
  write_json(new_edit_prompt_path, new_edit_data)
  

if __name__ == "__main__":
  main()