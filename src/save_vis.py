import json
import os
import lmdb
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.vis_utils import save_video
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import pickle


def find_json_files(parent_folder):
    json_files = []
    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.endswith('info.json'):
                if file.split('_')[0] != 'cap':
                    json_files.append(os.path.join(root, file))
    return json_files

def process_json_files(parent_folder):
    json_file_paths = find_json_files(parent_folder)
    
    for _, json_file_path in enumerate(json_file_paths):
        # Read and parse the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        first_50_items = data[:100]
        
        # Iterate through the list in the JSON file
        for idx, element in tqdm(enumerate(first_50_items)):
           
            video_name = element.get('vid')
            if video_name is None:
                video_name = element.get('episode_name') + ".mp4"
            start = element.get('start')
            end = element.get('end')
            if end - start < 0.5:
                start = max(0, end - 1.0)

            path = f"{os.path.dirname(json_file_path)}/{idx}.mp4"
            
            # Call the function with extracted values
            save_video(video_name, start, end, path, rgb_lmdb_env)
        
        with open(json_file_path, 'w') as file:
            json.dump(first_50_items, file, indent=4)

# Example usage

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:

    # json_path= "/lustre/fswork/projects/rech/vvh/upk96qz/hrn/vgg_slt/test_analysis.json"
    parent_folder = "/lustre/fswork/projects/rech/vvh/upk96qz/hrn/vgg_slt/teaser_stuff"
    # if not os.path.exists(parent_folder):
    #     os.makedirs(parent_folder)
    # vid_folder = f"{parent_folder}/videos"
    # if not os.path.exists(vid_folder):
    #     os.makedirs(vid_folder)
    
    # save_path = f"{parent_folder}/teaser_stuff.json"

    rgb_lmdb_env = lmdb.open(
                    "../cslr2_t/bobsl/lmdbs/lmdb-rgb_anon-public_1962/", readonly=True, lock=False, max_readers=512
                )
    # process_json_files(parent_folder)

    # sub_path = "../cslr2_t/bobsl/cslr2_pls_v3.pkl"
    # bg_path = ""


    # data = json.load(open(json_path, 'r'))

    # data = pickle.load(open("../cslr2_t/bobsl/cslr2_pls_v3.pkl", 'rb'))

    # dataset = hydra.utils.instantiate(
    #             cfg.data,
    #         )

    # dataset.setup()

    # pbar = tqdm(list(range(dataset.data_train.dataset.__len__())), total=dataset.data_train.dataset.__len__())
    # pl_avg = 0
    # count = 0
    # for idx in pbar:
    #     count += 1
    #     sample = dataset.data_train.dataset.__getitem__(idx)

    #     pl_avg = pl_avg * (count - 1) + len(sample["pls"])
    #     pl_avg /= count
    #     pbar.set_description(f"PL avg: {pl_avg:.2f}")

    
    # for idx, d in tqdm(enumerate(data), total=len(data)):
    #     vid = d["episode_name"]
    #     start = d["start"]
    #     end = d["end"]
    #     path = f"{parent_folder}/videos/{idx}.mp4"
    #     save_video(vid, start, end, path, rgb_lmdb_env)

    #     pls, probs = dataset.data_train.dataset.get_pls(vid, start, end)

    #     d["pls"] = pls
    #     d["probs"] = probs
    
    # with open(save_path, 'w') as file:
    #     json.dump(data, file, indent=4)


    # teaser
    save_vid = "6182391104004934170"
    start = 731.035
    end = 736.717

    # start_1 = 485.24
    # end_1 = 488.6


    path = f"{parent_folder}/teaser_roman.mp4"
    save_video(save_vid, start, end, path, rgb_lmdb_env)
    # path = f"{parent_folder}/teaser_1.mp4"
    # save_video(save_vid, start_1, end_1, path, rgb_lmdb_env)
    # pls, _ = dataset.data_train.dataset.get_pls(save_vid, start, end)
    # print(pls)
    # print(len(pls))


if __name__ == "__main__":
    main()

