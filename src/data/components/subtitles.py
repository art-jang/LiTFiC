"""
Generic dataset to load subtitles from data files
"""
import json
import random
import pickle
import numpy as np
from typing import Optional
from torch.utils.data import Dataset
import math
from src.utils.data_utils import unique_ordered_list, sample_sub_prev

from einops import rearrange
from operator import itemgetter

import os
import random
import re
import torch
from tqdm import tqdm

def remove_punctuation(text):
    punctuation_pattern = r'[^\w\s]'  # Matches any character that is not a word character or whitespace
    text_without_punctuation = re.sub(punctuation_pattern, '', text)

    return text_without_punctuation


class Subtitles(Dataset):
    """Generic dataset to load subtitles from data files"""
    def __init__(
        self,
        subset2episode: str,
        setname: str,
        subtitles_path: str,
        subtitles_temporal_shift: float,
        subtitles_max_duration: float,
        subtitles_min_duration: float,
        temporal_pad: float,
        info_pkl: str,
        max_previous_sentences: int = 0,
        question_pool: Optional[str] = None,
        filter_stop_words: bool = False,
        subtitles_random_offset: Optional[float] = None,
        text_augmentations: Optional[object] = None,
        fps: int = 25,
        verbose: bool = False,
        blip_cap_path: Optional[str] = None,
        filter_blip: bool = False,
        aug_prev: bool = False,
        aug_prev_pct: float = 0.5,
        aug_prev_shuffle: bool = False,
        man_gloss_path=None,
        use_man_gloss=False,
        cslr2_pl_type=None,
        sent_ret_n = 20,
        train_sent_ret_path = None,
        val_sent_ret_path = None,
        filter_based_on_pls = False,
        pl_configs: Optional[dict] = None,
        pl_filter: Optional[float] = None,
        pl_min_count: Optional[int] = None,
        pl_synonym_grouping: Optional[bool] = None,
        synonyms_pkl: Optional[str] = None,
        vocab_pkl: Optional[str] = None,
        aug_prev_neg: Optional[bool] = False,
        aug_prev_neg_prob: Optional[float] = 0.0,
        train_cap_path: Optional[str] = None,
        train_cap_prob: Optional[float] = 0.0,
        aligned_subtitles_path: Optional[str] = None,
    ):
        """
        Args:
            subset2episode (str): path to the json file containing the mapping
                between subset and episode.
            setname (str): name of the subset to load.
            subtitles_path (str): path to the subtitles pickle file.
            subtitles_temporal_shift (float): temporal shift to apply to the
                subtitles.
            subtitles_max_duration (float): maximum duration of the subtitles.
            subtitles_min_duration (float): minimum duration of the subtitles.
            temporal_pad (float): temporal padding to apply to the subtitles.
            info_pkl (str): path to the info pickle file.
            filter_stop_words (bool, optional): whether to filter stop words
            subtitles_random_offset (float, optional): randomly add an offset to
                the subtitles.
            text_augmentations (object, optional): text augmentations to apply
            fps (int): fps of the videos associated to the subtitles.
            verbose: (bool, optional): verbosity.
        """
        self.blip_cap_data = None
        if blip_cap_path is not None:
            with open(blip_cap_path, 'rb') as f:
                self.blip_cap_data = pickle.load(f)
        self.verbose = verbose
        with open(subset2episode, "rb") as json_f:
            subset2episode = json.load(json_f)
        
        if not use_man_gloss:
            if setname != 'train':
                subtitles_max_duration = 1000000
                subtitles_min_duration = 0
                subtitles_random_offset = 0.0
                
                if aligned_subtitles_path is not None:
                    subtitles_path = aligned_subtitles_path

            with open(subtitles_path, "rb") as pickle_f:
                self.subtitles = pickle.load(pickle_f)
            if self.verbose:
                print(f"Loaded {len(self.subtitles['episode_name'])} subtitles.")
        else:
            with open(os.path.join(man_gloss_path, f"{setname}.pkl"), "rb") as pickle_f:
                self.subtitles = pickle.load(pickle_f)
        
        self.setname = setname
        self.setname_episode = subset2episode[self.setname] # list of all episodes in a particular set
        del subset2episode
        self.text_augmentations = text_augmentations
        if self.verbose:
            print(f"Loading {self.setname} subtitles.")
        
        self.use_man_gloss = use_man_gloss
            
        self.subtitles_temporal_shift = subtitles_temporal_shift
        self.subtitles_random_offset = subtitles_random_offset
        self.subtitles_temporal_pad = temporal_pad
        self.fps = fps
        for key, val in self.subtitles.items():
            if key in ["start", "end"]:
                self.subtitles[key] = np.array(
                    [
                        self.convert_strtime_to_seconds(
                            time=x,
                            temporal_shift=self.subtitles_temporal_shift,
                        ) for x in val
                    ]
                )
            # else:
            elif key in ["class_label", "class_prob", "nn_label", "nn_prob"]:
                self.subtitles[key] = np.array(val, dtype=object)
            else:
                self.subtitles[key] = np.array(val)
        # filter by episodes
        if self.verbose:
            print(
                f"Filtering to {self.setname} subtitles.",
            )
        
        if not use_man_gloss:
            filtered_indices = np.where(
                np.isin(self.subtitles["episode_name"], self.setname_episode),
            )[0]
            self.filter_subtitles(filtered_indices)

        # if filter_blip:
        #     filtered_indices = np.where(
        #         np.isin(self.subtitles["episode_name"], self.blip_cap_data["video"]),
        #     )[0]
        #     self.filter_subtitles(filtered_indices)
        # filter by duration
        if self.verbose:
            print(
                "Filtering to subtitles with duration in" +
                f" [{subtitles_min_duration}, {subtitles_max_duration}].",
            )
        filtered_indices = np.where(
            self.subtitles["duration"] <= subtitles_max_duration,
        )[0]
        filtered_indices = np.intersect1d(
            filtered_indices,
            np.where(
                self.subtitles["duration"] >= subtitles_min_duration,
            )[0],
        )
        self.filter_subtitles(filtered_indices)

        if setname != 'train':
            if self.verbose:
                print(
                    f"Filtering to {self.setname} subtitles with Error cases.",
                )            
            filtered_indices = []
            for i in range(len(self.subtitles["subtitle"])):
                pattern = r"\[([A-Z\-]+)\]"
                match = re.search(pattern, self.subtitles["subtitle"][i])
                if match is None:
                    filtered_indices.append(i)
                
            if len(filtered_indices) > 0:
                filtered_indices = np.array(filtered_indices)
                self.filter_subtitles(filtered_indices)

        if train_cap_path is not None:
            self.train_cap = json.load(open(train_cap_path, "rb"))
            self.train_cap_prob = train_cap_prob
        else:
            self.train_cap = None
            self.train_cap_prob = None

        if filter_based_on_pls and setname == "train":
            filtered_indices = np.load(f"filtered_indices_pls_{setname}.npy")
            self.filter_subtitles(filtered_indices)
            print(f"Filtered based on pseudo-labels. {len(filtered_indices)} subtitles left.")

        self.synonyms_dict = pickle.load(open(synonyms_pkl, "rb"))

        sent_ret_path = train_sent_ret_path if setname == "train" else val_sent_ret_path
        if sent_ret_path is not None:
            with open(sent_ret_path, "rb") as f:
                self.sent_ret = pickle.load(f)
        else:
            self.sent_ret = None
        
        # if "id" in self.subtitles.keys():
        # # filtering sent_ret
        #     filtered_indices = np.where(
        #         np.isin(self.subtitles["id"], list(self.sent_ret.keys())),
        #     )[0]

        #     self.filter_subtitles(filtered_indices)

        self.aug_prev = aug_prev
        self.aug_prev_pct = aug_prev_pct
        self.aug_prev_shuffle = aug_prev_shuffle
        self.aug_prev_neg = aug_prev_neg
        self.aug_prev_neg_prob = aug_prev_neg_prob

        self.info_file_idx = {}
        with open(info_pkl, "rb") as pickle_f:
            info_file = pickle.load(pickle_f)["videos"]
        self.length = info_file["videos"]["T"]
        for vid_idx, vid_name in enumerate(info_file["name"]):
            if vid_name.split(".")[0] in self.setname_episode:
                self.info_file_idx[vid_name] = vid_idx
        del info_file

        self.nltk_stop_words = None
        if filter_stop_words:
            self.nltk_stop_words = {
                'ourselves', 'hers', 'between', 'yourself', 'but', 'again',
                'there', 'about', 'once', 'during', 'out', 'very', 'having',
                'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off',
                'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
                'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his',
                'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this',
                'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours',
                'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and',
                'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that',
                'over', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those',
                'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a',
                'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than',
            }  # removed 'what', 'why' and 'because'
        self.max_previous_sentences = max_previous_sentences
        if question_pool is not None:
            with open(question_pool, "r") as f:
                self.question_pool = [line.strip() for line in f.readlines()]
        else:
            self.question_pool = None
        
        self.cslr2_pl_type = cslr2_pl_type

        self.sent_ret_n = sent_ret_n
        
        
    def fix_synonyms_dict(self) -> None:
        """
        Make sure that the synonyms dictionary satisfies the following:
            - if a is a synonym of b, then b is a synonym of a
            - a is a synonym of a
        """
        for word, syns in self.synonyms_dict.items():
            if word not in syns:
                syns.append(word)
                # need to check that for each synonym in the list
                # the word is in the list of synonyms for that synonym
            for syn in syns:
                if word not in self.synonyms_dict[syn]:
                    self.synonyms_dict[syn].append(word)

    def synonym_combine(self, labels: np.ndarray, probs: torch.Tensor):
        """
        Function to aggregate probs of synonyms
        """
        change = False
        new_probs = []
        for anchor_idx, anchor in enumerate(labels):
            try:
                anchor = anchor.replace("-", " ")
                syns = self.synonyms_dict[anchor]
                anchor_new_prob = 0
                for checked_idx, checked_label in enumerate(labels):
                    checked_label = checked_label.replace("-", " ")
                    if checked_label in syns:
                        anchor_new_prob += probs[checked_idx]
                    if checked_idx != anchor_idx:
                        change = True
                new_probs.append(anchor_new_prob)
            except KeyError:
                # prediction not in the synonym list
                new_probs.append(probs[anchor_idx])
        if change:
            # need to sort
            sorted_indices = torch.argsort(- 1 * torch.tensor(new_probs))
            new_probs = torch.tensor(new_probs)[sorted_indices]
            labels = np.array(labels)[sorted_indices]
        else:
            new_probs = torch.tensor(new_probs)
            labels = np.array(labels)
        return new_probs, labels

        
    @staticmethod
    def convert_strtime_to_seconds(
        time: str, temporal_shift: float,
    ) -> float:
        """
        Convert a string time in the format HH:MM:SS.SSS to seconds
        with a potential additional temporal shift.

        Args:
            time (str): time in the format HH:MM:SS.SSS
            temporal_shift (float): additional temporal shift in seconds
        """
        if isinstance(time, float):
            return time + temporal_shift
        else:
            assert isinstance(time, str)
            time = time.split(":")
            time = [float(x) for x in time]
            time = sum([x * y for x, y in zip(time, [3600, 60, 1, 1e-3])])
            time += temporal_shift
            return time

    def filter_subtitles(
        self, filtered_indices: np.ndarray,
    ) -> None:
        """
        Filter self.subtitles wrt. filtered_indices.
        Args:
            filtered_indices (np.ndarray): indices to keep
        """
        previous_length = len(self.subtitles["episode_name"])
        # filtering each key
        for key, val in self.subtitles.items():
            self.subtitles[key] = val[filtered_indices]
        if self.verbose:
            print(
                f"\tFrom {previous_length} subtitles," +
                f" {len(filtered_indices)} are kept.",
            )

    def shuffle(self) -> None:
        """Shuffle all subtitles."""
        shuffled_indices = np.arange(len(self.subtitles["episode_name"]))
        for key, val in self.subtitles.items():
            self.subtitles[key] = val[shuffled_indices]

    def __len__(self) -> int:
        return len(self.subtitles["episode_name"])

    def __getitem__(self, idx: int) -> dict:
        """Loads subtitles[idx]"""
        video_name = self.subtitles["episode_name"][idx] + ".mp4"
        if self.subtitles_random_offset is not None and self.subtitles_random_offset > 0:
            # adds a random offset to the subtitles
            # in (- self.subtitles_random_offset, self.subtitles_random_offset)
            sub_start = self.subtitles["start"][idx] - \
                self.subtitles_temporal_pad
            sub_end = self.subtitles["end"][idx] + self.subtitles_temporal_pad
            random_start = np.random.uniform(
                sub_start - self.subtitles_random_offset,
                min(sub_start + self.subtitles_random_offset, sub_end - 1.0),
            )  # ensure that the random start is at least 1 second before the end
            random_end = np.random.uniform(
                max(random_start + 1.0, sub_end - self.subtitles_random_offset),
                sub_end + self.subtitles_random_offset,
            )
            sub_start = max(0, random_start)
            # 0.32 is 8 frames at 25 fps (16f windows)
            sub_end = min(
                random_end, self.length[self.info_file_idx[video_name]] / self.fps - 0.32)
        else:
            sub_start = max(
                0, self.subtitles["start"][idx] - self.subtitles_temporal_pad
            )
            sub_end = min(
                self.subtitles["end"][idx] + self.subtitles_temporal_pad,
                self.length[self.info_file_idx[video_name]] / self.fps - 0.32,
            )
        subtitle = self.subtitles["subtitle"][idx]

        if self.nltk_stop_words is not None:
            subtitle = " ".join(
                [word for word in subtitle.split() if word not in self.nltk_stop_words]
            )
        if self.text_augmentations is not None:
            subtitle = self.text_augmentations(subtitle)

        if sub_end - sub_start <= 0.5:
            # change the start of the subtitle so that it is at least 1 second long
            sub_start = max(0, sub_end - 1.0)
        subtitles, sub_starts, sub_ends, video_names = subtitle, sub_start, sub_end, video_name

        previous_context = None
        prev_start = []
        prev_end = []
        if self.max_previous_sentences > 0:
            num_sentences = self.max_previous_sentences
            for i in range(num_sentences):
                if idx - i - 1 < 0:
                    break
                if self.subtitles["episode_name"][idx - i - 1] != self.subtitles["episode_name"][idx]:
                    continue

                if self.train_cap is not None:
                    if self.setname == "train" and random.random() < self.train_cap_prob:
                        prev_text = self.train_cap[str(self.subtitles["id"][idx - i - 1])]["pred"]

                        if previous_context is None:
                            previous_context = prev_text
                        else:
                            previous_context = prev_text + ' ' + previous_context
                        
                        continue
                
                prev_text = self.subtitles["subtitle"][idx - i - 1]
                if self.aug_prev_neg and  random.random() < random.random() * self.aug_prev_neg_prob and self.setname == "train":
                    prev_text = random.sample(self.subtitles["subtitle"].tolist(), 1)[0]
                
                if self.aug_prev and self.setname == "train":
                    prev_text = sample_sub_prev(prev_text, pct=self.aug_prev_pct, shuffle=self.aug_prev_shuffle)

                prev_start.append(self.subtitles["start"][idx - i - 1])
                prev_end.append(self.subtitles["end"][idx - i - 1])

                if previous_context is None:
                    previous_context = prev_text
                else:
                    previous_context = prev_text + ' ' + previous_context
        
            if previous_context is None:
                previous_context = ""
        
        for i in range(len(prev_start)):
            if prev_end[i] - prev_start[i] <= 0.5:
                prev_start[i] = max(0, prev_end[i] - 1.0)

        prev_start = []
        prev_end = []
        
        man_gloss = None
        if self.use_man_gloss:
            man_gloss = self.subtitles["man_gloss"][idx].strip().split()
        question = None
        if self.question_pool is not None:
            if self.setname == "train":
                question = random.choice(self.question_pool)
            else:
                question = self.question_pool[0]
        
        start_second = math.floor(sub_starts)
        end_second = math.ceil(sub_ends)

        if self.blip_cap_data is not None:
            try:
                bg_description = self.blip_cap_data["captions"][self.blip_cap_data["video"].index(video_name.split(".")[0])][start_second:end_second+1]
                bg_description = unique_ordered_list(bg_description)
                bg_description = [x for x in bg_description if x != '']
                if len(bg_description) == 0:
                    bg_description = ['No description available']
                bg_description = '. '.join(bg_description) + '.'
            except:
                bg_description = 'No description available.'
        
        else:
            bg_description = 'No description available.'
        

        cslr2_labels = None
        cslr2_probs = None
        
        if "nn_label" in self.subtitles.keys():
            if self.cslr2_pl_type == "nn":
                label_key = "nn_label"
                prob_key = "nn_prob"
            else:
                label_key = "class_label"
                prob_key = "class_prob"
            
            cslr2_labels = self.subtitles[label_key][idx][0]
            cslr2_probs = self.subtitles[prob_key][idx][0]
        
        ret_sent = ""
        # if "id" in self.subtitles.keys():
        #     if self.setname == "train":
        #         ret_sent = self.sent_ret[self.subtitles["id"][idx]]
        #         # select an index at random from ret_sent in the top self.sent_ret_n
        #         if len(ret_sent) > 0:
        #             ret_sent = random.choice(ret_sent[:self.sent_ret_n])
        #         else:
        #             ret_sent = ""
            
        #     else:
        #         ret_sent = self.sent_ret[self.subtitles["id"][idx]]
        #         # select the first index from ret_sent in the top self.sent_ret_n
        #         if len(ret_sent) > 0:
        #             ret_sent = ret_sent[0]
        #         else:
        #             ret_sent = ""

        
        return {
            "subtitle": subtitles,
            "sub_start": sub_starts,
            "sub_end": sub_ends,
            "video_name": video_names,
            "previous_context": previous_context,
            "question": question,
            "bg_description": bg_description,   
            "man_gloss": man_gloss,
            "cslr2_labels": cslr2_labels,
            "cslr2_probs": cslr2_probs,
            "ret_sent": ret_sent,
            "id": self.subtitles["id"][idx] if "id" in self.subtitles.keys() else None,
            "prev_start": prev_start,
            "prev_end": prev_end,
        }
