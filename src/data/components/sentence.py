"""
Generic dataset class to load sentences from data files
Refactored from sentence_features.py
"""
import torch
import pickle
import numpy as np

from einops import rearrange
from operator import itemgetter
from multiprocessing import Value
from typing import List, Optional
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from src.data.components.subtitles import Subtitles
from src.data.components.lmdb_loader import LMDBLoader
from src.utils.data_utils import sample_sub
from src.utils.cslr_tools import compress_and_average

class Sentences(Dataset):
    """General dataset class to load sentences from data files"""
    def __init__(
        self,
        subset2episode: str,
        setname: str,
        subtitles_path :str,
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
        load_features: bool = False,
        feats_lmdb: Optional[str] = None,
        feats_load_stride: int = 1,
        feats_load_float16: bool = False,
        feats_lmdb_window_size: int = 16,
        feats_lmdb_stride: int = 2,
        feats_dim: int = 768,
        video_augmentations: Optional[object] = None,
        load_pl: bool = False,
        pl_lmdb: Optional[str] = None,
        pl_load_stride: int = 1,
        pl_load_float16: bool = False,
        pl_lmdb_window_size: int = 16,
        pl_lmdb_stride: int = 2,
        pl_filter: float = 0.6,
        pl_min_count: int = 1,
        pl_synonym_grouping: bool = False,
        synonyms_pkl: Optional[str] = None,
        vocab_pkl: Optional[str] = None,
        load_word_embds: bool = False,
        word_embds_pkl: Optional[str] = None,
        verbose: bool = False,
        sub_sample_shuffle: Optional[str] = True,
        sub_sample_pct: Optional[float] = 1.0,
    ):
        """
        Args:
            subset2episode: path to json file mapping subset to episode
            setname (str): split name
            subtitles_path (str): path to subtitles
            subtitles_temporal_shift (float): temporal shift between subtitles and video
            subtitles_max_duration (float): maximum duration of subtitles
            subtitles_min_duration (float): minimum duration of subtitles
            temporal_pad (float): temporal padding for subtitles
            info_pkl (str): path to info pickle file
            filter_stop_words (bool): whether to filter stop words
            subtitles_random_offset (float, optional): random offset for subtitles
            text_augmentations (object, optional): text augmentation object
            fps (int, optional): video fps

            # options for features loading
            load_features (bool): whether to load features
            feats_lmdb (str, optional): path to lmdb features
            feats_load_stride (int): stride for loading features
            feats_load_float16 (bool): whether to load features in float16
            feats_lmdb_window_size (int): window size for lmdb features saving
            feats_lmdb_stride (int): stride for lmdb features saving
            feats_dim (int): dimension of features
            video_augmentations (object, optional): video augmentation object

            # options for pl loading
            load_pl (bool): whether to load pl
            pl_lmdb (str, optional): path to lmdb pl
            pl_load_stride (int): stride for loading pl
            pl_load_float16 (bool): whether to load pl in float16
            pl_lmdb_window_size (int): window size for lmdb pl saving
            pl_lmdb_stride (int): stride for lmdb pl saving
            pl_filter (float): filtering threshold for pl
            pl_min_count (int): minimum count for pl
            pl_synonym_grouping (bool): whether to group synonyms
            synonyms_pkl (str, optional): path to synonyms pickle file
            vocab_pkl (str, optional): path to vocab pickle file

            # word embeddings
            load_word_embds (bool): whether to load word embeddings
            word_embds_pkl (str, optional): path to word embeddings pickle file

            # other
            verbose (bool): whether to print debug info
        """
        self.skip_mode = Value("i", False)  # shared variable to skip loading
        # subtitles
        self.subtitles = Subtitles(
            subset2episode=subset2episode,
            setname=setname,
            subtitles_path=subtitles_path,
            subtitles_temporal_shift=subtitles_temporal_shift,
            subtitles_max_duration=subtitles_max_duration,
            subtitles_min_duration=subtitles_min_duration,
            temporal_pad=temporal_pad,
            info_pkl=info_pkl,
            max_previous_sentences=max_previous_sentences,
            question_pool=question_pool,
            filter_stop_words=filter_stop_words,
            subtitles_random_offset=subtitles_random_offset,
            text_augmentations=text_augmentations,
            fps=fps,
            verbose=verbose,
        )
        
        # features
        self.features, self.video_augmentations = None, None
        if load_features:
            self.features = LMDBLoader(
                lmdb_path=feats_lmdb,
                load_stride=feats_load_stride,
                load_float16=feats_load_float16,
                load_type="feats",
                verbose=verbose,
                lmdb_window_size=feats_lmdb_window_size,
                lmdb_stride=feats_lmdb_stride,
                feat_dim=feats_dim,
            )
            self.video_augmentations = video_augmentations

        # pseudo-labels
        self.pseudo_label, self.vocab, self.synonym_grouping = None, None, None
        self.pl_filter, self.pl_min_count = None, None
        if load_pl:
            self.pseudo_label = LMDBLoader(
                lmdb_path=pl_lmdb,
                load_stride=pl_load_stride,
                load_float16=pl_load_float16,
                load_type="pseudo-labels",
                verbose=verbose,
                lmdb_window_size=pl_lmdb_window_size,
                lmdb_stride=pl_lmdb_stride,
            )
            msg = "vocab_pkl must be provided if load_pl is True"
            assert vocab_pkl is not None, msg
            self.vocab = pickle.load(open(vocab_pkl, "rb"))
            if "words_to_id" in self.vocab.keys():
                self.vocab = self.vocab["words_to_id"]
            self.vocab_size = len(self.vocab)
            self.vocab["bos"] = self.vocab_size
            self.vocab["eos"] = self.vocab_size + 1
            self.vocab["<pad>"] = self.vocab_size + 2  # <pad> as 'pad' in vocab
            self.vocab["no annotation"] = -1  # dummy class for no annotation
            self.inverted_vocab = {v: k for k, v in self.vocab.items()}
            self.synonym_grouping = pl_synonym_grouping
            if self.synonym_grouping:
                msg = "synonyms_pkl must be provided if synonym_grouping is True"
                assert synonyms_pkl is not None, msg
                self.synonyms_dict = pickle.load(open(synonyms_pkl, "rb"))
                self.fix_synonyms_dict()
            self.pl_filter = pl_filter
            self.pl_min_count = pl_min_count

        # word embeddings
        self.word_embds = None
        if load_word_embds:
            msg = "word_embds_pkl must be provided if load_word_embds is True"
            assert word_embds_pkl is not None, msg
            self.word_embds = pickle.load(open(word_embds_pkl, "rb"))
        
        self.sub_sample_shuffle = True if setname == "train" else False
        self.sub_sample_pct = sub_sample_pct


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

    def synonym_combine(self, labels: np.ndarray, probs: torch.Tensor) -> List:
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

    def change_skip_mode(self) -> None:
        """
        Changes skip mode
        Beware: does not work as expected when dataloader.persistent_workers=True
        """
        print(f"Changing skip mode from {self.skip_mode.value} to {not self.skip_mode.value}")
        self.skip_mode.value = not self.skip_mode.value
        print(f"Current skip mode: {self.skip_mode.value}")

    def __len__(self) -> int:
        return len(self.subtitles)

    def get_single_item(
        self,
        subtitle: str,
        video_name: str,
        sub_start: float,
        sub_end: float,
        previous_context: Optional[str] = None,
        question: Optional[str] = None,
    ) -> dict:
        """Loads single item based on subtitle and video name"""
        if self.skip_mode.value:
            return {
                "subtitle": subtitle,
                "features": torch.zeros((1, 1)) if self.features is not None else None,
                "target_indices": torch.zeros((1, 1)) if self.pseudo_label is not None else None,
                "target_labels": torch.zeros((1, 1)) if self.pseudo_label is not None else None,
                "annotation_dict": {} if self.pseudo_label is not None else None,
                "target_word_embds": torch.zeros((1, 1)) if self.word_embds is not None else None,
                "video_name": video_name,
                "sub_start": sub_start,
                "sub_end": sub_end,
                "previous_context": previous_context,
                "question": question,
            }
        feats = None
        if self.features is not None:
            # load features corresponding to subtitle
            feats = self.features.load_sequence(
                episode_name=video_name,
                begin_frame=int(sub_start * self.subtitles.fps),
                end_frame=int(sub_end * self.subtitles.fps),
            )
        probs, labels, min_count_indices = None, None, None
        target_word_embds = None
        if self.pseudo_label is not None:
            labels, probs = self.pseudo_label.load_sequence(
                episode_name=video_name,
                begin_frame=int(sub_start * self.subtitles.fps),
                end_frame=int(sub_end * self.subtitles.fps),
            )
            if self.synonym_grouping:
                words = itemgetter(
                    *rearrange(labels.numpy(), "t k -> (t k)")
                )(self.inverted_vocab)
                words = rearrange(
                    np.array(words), "(t k) -> t k", k=5,
                )
                new_words, new_probs = [], []
                for word, prob in zip(words, probs):
                    new_prob, new_word = self.synonym_combine(word, prob)
                    new_words.append(new_word)
                    new_probs.append(new_prob)
                new_words = rearrange(np.array(new_words), "t k -> (t k)")
                labels = itemgetter(*new_words)(self.vocab)
                labels = rearrange(torch.tensor(labels), "(t k) -> t k", k=5)
                probs = torch.stack(new_probs)
            # filter
            # get indices of annots occuring at least pl_min_count times
            if self.pl_min_count > 1:
                # torch-based method
                _, counts = torch.unique_consecutive(labels[:, 0], return_counts=True)
                repeated_counts = torch.repeat_interleave(counts, counts)
                min_count_indices = torch.where(repeated_counts >= self.pl_min_count)[0]
            else:
                min_count_indices = torch.arange(start=0, end=len(labels))

            # get dictionnary of annotation
            if self.word_embds is not None:
                word_embds_dict = {}
            annotation_dict, target_dict = {}, {}
            for annotation_idx, (prob, label) in enumerate(zip(probs, labels)):
                # with enumerate, we save tensor(idx) as key
                # using range, we can directly have idx as key
                prob = prob[0].item()
                label = label[0].item()
                # convert annotation_idx to feature_idx
                correct_annotation_idx = int(
                    annotation_idx * self.pseudo_label.lmdb_stride / \
                        (self.features.lmdb_stride * self.pseudo_label.load_stride)
                )
                if prob >= self.pl_filter and annotation_idx in min_count_indices:
                    # only keep annotations with
                    # 1) prob >= pl_filter
                    # 2) occuring at least pl_min_count times
                    try:
                        target_dict[correct_annotation_idx].append(label)
                    except KeyError:
                        target_dict[correct_annotation_idx] = [label]
                    annotation_list = [
                        int(int(sub_start * self.subtitles.fps) + correct_annotation_idx),
                        "pseudo-label", prob,
                    ]
                    try:
                        annotation_dict[label].append(annotation_list)
                    except KeyError:
                        annotation_dict[label] = [annotation_list]
                    if self.word_embds is not None:
                        word_embds_dict[correct_annotation_idx] = self.word_embds[label]
            # video augmentation
            probs = [p[0].item() for p in probs]
            pls = [self.inverted_vocab[x[0]] for x in list(target_dict.values())]

            pls, probs = compress_and_average(pls, probs)
            
            if self.video_augmentations is not None:
                feats, kept_indices = self.video_augmentations(feats)
                if self.pseudo_label is not None:
                    # need to map the keys of target_dict to new indices
                    mapping_dict = {
                        old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)
                    }
                    target_dict = {
                        mapping_dict[k]: v for k, v in target_dict.items() if k in kept_indices
                    }
                    annotation_dict = {
                        mapping_dict[k]: v for k, v in annotation_dict.items() if k in kept_indices
                    }
                    if self.word_embds is not None:
                        word_embds_dict = {
                            mapping_dict[k]: v for k, v in word_embds_dict.items() \
                                if k in kept_indices
                        }

            target_indices = torch.tensor(list(target_dict.keys()), dtype=torch.long)
            target_labels = torch.tensor(list(target_dict.values())).squeeze(-1)
            assert len(target_indices) == len(target_labels)
            if self.word_embds is not None:
                # replace the word_embds_dict by two lists: (i) indices, (ii) word embds
                try:
                    target_word_embds = torch.stack(list(word_embds_dict.values()))
                    assert len(target_indices) == len(target_word_embds)
                except RuntimeError:
                    target_word_embds = torch.tensor(list(word_embds_dict.values()))  # empty
                    assert len(target_indices) == len(target_word_embds)
        return {
            "subtitle": subtitle,
            "features": feats if self.features is not None else None,
            "target_indices": target_indices if self.pseudo_label is not None else None,
            "target_labels": target_labels if self.pseudo_label is not None else None,
            "annotation_dict": annotation_dict if self.pseudo_label is not None else None,
            "target_word_embds": target_word_embds if self.word_embds is not None else None,
            "video_name": video_name,
            "sub_start": sub_start,
            "sub_end": sub_end,
            "previous_context": previous_context,
            "question": question,
            "pls": pls if self.pseudo_label is not None else None,
            "sub_gt": sample_sub(subtitle, self.sub_sample_shuffle, self.sub_sample_pct),
            "probs": probs if self.pseudo_label is not None else None,
        }

        '''
        subtitle -> The full subtitle sentence (string)
        features -> video swin features (feat_num x 768)
        target_indices ->  indices that have associated pseaudo labels with them (filted_feat_num,)
        target_labels -> pseudo labels (filted_feat_num,)
        annotation_dict -> ... hold on
        target_word_embeds -> T5 embed layer feats for the pseudo_labels (filted_feat_num x 1024)
        video_name -> name of the video file
        sub_start, sub_end -> starting time (sec), and ending time of the subtitle with the random_offset and stuff
        '''

    def __getitem__(self, idx: int) -> dict:
        """Loads evrything related to self.subtitles[idx]"""
        
        subtitles = self.subtitles[idx]
        video_names = subtitles["video_name"]
        sub_starts, sub_ends = subtitles["sub_start"], subtitles["sub_end"]
        return self.get_single_item(subtitle=subtitles["subtitle"],
                                    video_name=video_names,
                                    sub_start=sub_starts,
                                    sub_end=sub_ends,
                                    previous_context=subtitles["previous_context"],
                                    question=subtitles["question"],)


def pad_tensors_and_create_attention_masks(tensor_list, padding_side='right'):
    # Find the maximum length in the list
    max_length = max(tensor.size(0) for tensor in tensor_list)
    
    # Initialize lists for padded tensors and attention masks
    padded_tensors = []
    attention_masks = []
    
    for tensor in tensor_list:
        # Calculate the padding length
        padding_length = max_length - tensor.size(0)
        
        if padding_side == 'right':
            # Pad the tensor with zeros (for embeddings: (0, 0) for last dimension, (0, padding_length) for second last)
            padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padding_length), "constant", 0)
            
            # Create the attention mask (1 for actual values, 0 for padded values)
            attention_mask = torch.cat([torch.ones(tensor.size(0)), torch.zeros(padding_length)])
        elif padding_side == 'left':
            # Pad the tensor with zeros (for embeddings: (0, 0) for last dimension, (padding_length, 0) for second last)
            padded_tensor = torch.nn.functional.pad(tensor, (0, 0, padding_length, 0), "constant", 0)
            
            # Create the attention mask (1 for actual values, 0 for padded values)
            attention_mask = torch.cat([torch.zeros(padding_length), torch.ones(tensor.size(0))])
        
        padded_tensors.append(padded_tensor)
        attention_masks.append(attention_mask)
    
    # Stack the padded tensors and attention masks into 3D and 2D tensors respectively
    padded_tensors = torch.stack(padded_tensors)
    attention_masks = torch.stack(attention_masks)
    
    return padded_tensors, attention_masks


def collate_fn_padd_t(batch: List):
    subtitles = [item["subtitle"] for item in batch]
    features = [item["features"] for item in batch]
    questions = [item["question"] for item in batch]
    previous_contexts = [item["previous_context"] for item in batch]
    pls = [item["pls"] for item in batch]
    target_indices = [item["target_indices"] for item in batch]
    target_labels = [item["target_labels"] for item in batch]
    start = [item["sub_start"] for item in batch]
    end = [item["sub_end"] for item in batch]
    video_names = [item["video_name"] for item in batch]
    sub_gt = [item["sub_gt"] for item in batch]
    probs = [item["probs"] for item in batch]

    padded_features, attn_masks = pad_tensors_and_create_attention_masks(features, padding_side='right')

    return {
        "features": padded_features,
        "attn_masks": attn_masks,
        "subtitles": subtitles,
        "questions": questions,
        "previous_contexts": previous_contexts,
        "pls": pls,
        "target_indices": target_indices,
        "target_labels": target_labels,
        "start": start,
        "end": end,
        "video_names": video_names,
        "sub_gt": sub_gt,
        "probs": probs
    }


def collate_fn_padd(batch: List) -> List:
    """Padds batches of variable length"""
    hn_mining = isinstance(batch[0]["subtitle"], List)
    if not hn_mining:
        annotation_dict = [item["annotation_dict"] for item in batch]
        target_indices = [item["target_indices"] for item in batch]
        target_labels = [item["target_labels"] for item in batch]
        subtitles = [item["subtitle"] for item in batch]
        features = [item["features"] for item in batch]
        if features[0] is not None:
            features = pad_sequence(features, batch_first=True, padding_value=0)
        target_word_embds = [item["target_word_embds"] for item in batch]
        video_name = [item["video_name"] for item in batch]
        sub_start = [item["sub_start"] for item in batch]
        sub_end = [item["sub_end"] for item in batch]
    else:
        # unpack the batch
        annotation_dict = [item for items in batch for item in items["annotation_dict"]]
        target_indices = [item for items in batch for item in items["target_indices"]]
        target_labels = [item for items in batch for item in items["target_labels"]]
        subtitles = [item for items in batch for item in items["subtitle"]]
        features = [item for items in batch for item in items["features"]]
        if features[0] is not None:
            features = pad_sequence(features, batch_first=True, padding_value=0)
        target_word_embds = [item for items in batch for item in items["target_word_embds"]]
        video_name = [item for items in batch for item in items["video_name"]]
        sub_start = [item for items in batch for item in items["sub_start"]]
        sub_end = [item for items in batch for item in items["sub_end"]]
    return subtitles, features, target_indices, target_labels, target_word_embds, \
        annotation_dict, video_name, sub_start, sub_end
