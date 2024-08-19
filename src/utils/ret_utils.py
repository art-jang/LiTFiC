import numpy as np

def calculate_retrieval_metrics(retrieval_matrix, k_list= [1, 5, 10, 25, 50]):
    """
    Calculate retrieval metrics including recall at different cutoffs and mean/median rank. ddd
    
    Args:
    retrieval_matrix (list of lists): Retrieval matrix where each sublist contains n values.
    k_list (list of int): List of cutoff values to calculate recall.
    
    Returns:
    dict: Dictionary with recall at k values, mean rank, and median rank.
    """
    retrieval_matrix = np.array(retrieval_matrix)
    n = retrieval_matrix.shape[0]
    recall_scores = {k: 0 for k in k_list}
    ranks = []

    for i in range(n):
        row = retrieval_matrix[i]
        correct_index = i  # Assuming the correct match is at the diagonal for simplicity
        sorted_indices = np.argsort(-row)  # Sort indices by descending order of relevance scores
        rank = np.where(sorted_indices == correct_index)[0][0] + 1  # Get rank of the correct item (1-based)
        ranks.append(rank)

        for k in k_list:
            if rank <= k:
                recall_scores[k] += 1

    # Calculate recall percentages
    for k in recall_scores:
        recall_scores[k] = (recall_scores[k] / n) * 100

    mean_rank = np.mean(ranks)
    median_rank = np.median(ranks)

    results = {
        'recall_scores': recall_scores,
        'mean_rank': mean_rank,
        'median_rank': median_rank
    }

    return results

def calculate_average_logit_scores(logits, labels, ignore_index=-100):
    """
    Calculate the average logit scores for each sequence in a batch, ignoring specified indices.
    
    Args:
    logits (torch.Tensor): Tensor of logit scores with shape (batch_size, seq_len, vocab_size).
    labels (torch.Tensor): Tensor of labels with shape (batch_size, seq_len).
    ignore_index (int): Index to ignore in the labels (default: -100).
    
    Returns:
    list: List of average logit scores for each sequence in the batch.
    """
    batch_size, seq_len, vocab_size = logits.shape
    average_scores = []
    
    for i in range(batch_size):
        total_score = 0.0
        valid_count = 0
        
        for j in range(seq_len):
            label = labels[i, j].item()
            if label != ignore_index:
                total_score += logits[i, j, label].item()
                valid_count += 1
        
        if valid_count > 0:
            average_score = total_score / valid_count
        else:
            average_score = 0.0  # or handle the case with no valid labels as needed
        
        average_scores.append(average_score)
    
    return average_scores


def copy_tensor(tensor, n):
        expanded_tensor = tensor.unsqueeze(0)
        expanded_tensor = expanded_tensor.expand(n, *tensor.shape)
        return expanded_tensor


def calculate_overlap_metrics(gt, pred):
    iou_list = []
    precision_list = []
    recall_list = []
    
    for gt_sent, pred_sent in zip(gt, pred):
        # Split sentences into words
        gt_words = set(gt_sent.split())
        pred_words = set(pred_sent.split())
        
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