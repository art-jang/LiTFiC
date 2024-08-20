import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

def compress_and_average(strings, numbers):
    # Step 2: Create a dictionary to collect numbers associated with each string
    data = defaultdict(list)
    for string, number in zip(strings, numbers):
        data[string].append(number)
    
    # Step 3: Calculate the average for each string
    averages = {key: sum(values) / len(values) for key, values in data.items()}
    
    # Convert the dictionary to two lists
    compressed_strings = list(averages.keys())
    compressed_averages = list(averages.values())
    
    return compressed_strings, compressed_averages


def load_checkpoint_model(
    ckpt_path: str,
    model: nn.Module,
    device: torch.device,
):
    """Load checkpoint if specified in the config."""
  
    # load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    # remove module. from checkpoint keys
    model_state_dict = checkpoint["model_state_dict"]
    model_state_dict = {
        k.replace("module.", ""): v for k, v in model_state_dict.items()}
    # to prevent errors when text encoder is not in chkpt
    model.load_state_dict(model_state_dict, strict=False)
    return model

def group_and_average_tokens(video_tokens, target_indices, target_labels):
    reduced_video_tokens_list = []
    cls_tokens = []
    valid_cls_indices = []

    for seq in range(len(target_indices)):
        seq_indices = target_indices[seq]
        seq_labels = target_labels[seq]
        # seq_word_embds = target_word_embds[seq]
        seq_video_tokens = video_tokens[seq]

        grouped_tokens = []
        
        i = 0
        while i < len(seq_indices):
            current_index = seq_indices[i]
            current_label = seq_labels[i]
            # current_embd = seq_word_embds[i]

            group_indices = [current_index]
            group_tokens = [seq_video_tokens[current_index]]

            while i + 1 < len(seq_indices) and seq_indices[i + 1] == current_index + 1 and seq_labels[i + 1] == current_label:
                i += 1
                current_index = seq_indices[i]
                group_indices.append(current_index)
                group_tokens.append(seq_video_tokens[current_index])

            avg_token = torch.mean(torch.stack(group_tokens), dim=0)
            grouped_tokens.append(avg_token)

            i += 1
    
        if grouped_tokens:

            reduced_video_tokens_list.append(torch.stack(grouped_tokens))

            # Max pool to get the cls_token for the sequence
            cls_token = torch.max(torch.stack(grouped_tokens), dim=0)[0]
            cls_tokens.append(cls_token)
            valid_cls_indices.append(seq)
        
    
    # ipdb.set_trace()
    # Find the max length of the reduced sequences
    
    max_length = max([t.shape[0] for t in reduced_video_tokens_list])

    # Pad the reduced video tokens to the max length
    padded_video_tokens = torch.stack([
        F.pad(t, (0, 0, 0, max_length - t.shape[0])) for t in reduced_video_tokens_list
    ])

    # Create the attention mask
    attention_mask = torch.stack([
        torch.cat([torch.ones(t.shape[0]), torch.zeros(max_length - t.shape[0])]) for t in reduced_video_tokens_list
    ])

    # Convert valid_cls_indices to tensor
    valid_cls_indices = torch.tensor(valid_cls_indices)

    return padded_video_tokens, attention_mask, valid_cls_indices