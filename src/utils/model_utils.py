import torch

import numpy as np

def calculate_overlap_metrics(gt, pred):
    iou_list = []
    precision_list = []
    recall_list = []
    
    for gt_sent, pred_sent in zip(gt, pred):
        # Split sentences into words
        gt_words = set(gt_sent.split())
        pred_words = set(pred_sent.split())

        # remove punctuation from each of the words (iuncludes question and exclaimation marks)
        gt_words = set([word.strip('.,!?') for word in gt_words])
        pred_words = set([word.strip('.,!?') for word in pred_words])

        
        # Calculate intersection and union
        intersection = gt_words.intersection(pred_words)
        union = gt_words.union(pred_words)
        
        # Calculate metrics
        iou = len(intersection) / len(union) if len(union) > 0 else 0
        precision = len(intersection) / len(pred_words) if len(pred_words) > 0 else 0
        recall = len(intersection) / len(gt_words) if len(gt_words) > 0 else 0
        
        # Append metrics to lists
        iou_list.append(iou)
        precision_list.append(precision)
        recall_list.append(recall)
    
    return iou_list, precision_list, recall_list

def safe_chr(c):
    try:
        return chr(c)
    except (ValueError, OverflowError):
        return ''  # or any default character, like '?'

def strings_to_tensor(strings, max_string_length=1024):
    """Convert a list of strings to a tensor of ASCII values, padded to a fixed length."""
    tensor = torch.full((len(strings), max_string_length), fill_value=0, dtype=torch.long)
    for i, string in enumerate(strings):
        encoded_string = [ord(c) for c in string][:max_string_length]
        tensor[i, :len(encoded_string)] = torch.tensor(encoded_string, dtype=torch.long)
    return tensor

def tensor_to_strings(tensor):
    """Convert a tensor of ASCII values back to a list of strings."""
    return ["".join(safe_chr(c) for c in row if 0 < c <= 0x10FFFF) for row in tensor]