import json
import os
import lmdb
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.vis_utils import save_video
from tqdm import tqdm

def find_json_files(parent_folder):
    json_files = []
    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def process_json_files(parent_folder):
    json_file_paths = find_json_files(parent_folder)
    
    for _, json_file_path in enumerate(json_file_paths):
        # Read and parse the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        first_50_items = data[:50]
        
        # Iterate through the list in the JSON file
        for idx, element in tqdm(enumerate(first_50_items)):
            video_name = element.get('vid')
            start = element.get('start')
            end = element.get('end')
            path = f"{os.path.dirname(json_file_path)}/{idx}.mp4"
            
            # Call the function with extracted values
            save_video(video_name, start, end, path, rgb_lmdb_env)
        
        with open(json_file_path, 'w') as file:
            json.dump(first_50_items, file, indent=4)

# Example usage

parent_folder = 'logs/llama_sub1_s_prev1/runs/2024-08-06_00-01-06/vis/'
rgb_lmdb_env = lmdb.open(
                "../cslr2_t/bobsl/lmdbs/lmdb-rgb_anon-public_1962/", readonly=True, lock=False, max_readers=512
            )
process_json_files(parent_folder)