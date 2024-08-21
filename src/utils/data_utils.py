import random
import math
import ipdb
import string


def sample_pls(data, num_samples):
    """
    Samples words from a list based on their associated probabilities.
    
    Args:
    data (dict): A dictionary with two keys 'words' and 'probs'. 
                 'words' is a list of strings, and 'probs' is a list of probabilities.
    num_samples (int): The number of words to sample.
    
    Returns:
    list: A list of sampled words.
    """
    words = list(data.keys())
    probabilities = list(data.values())
    
    # Sample words based on the given probabilities
    sampled_words = random.choices(words, weights=probabilities, k=num_samples)
    
    return sampled_words


def sample_sub(sentence, shuffle, pct=0.3, replace=False, pl_dist=None):
    words = sentence.split()

    words = [word.strip(string.punctuation).lower() for word in words]
    # Remove stop words
    stop_words = {
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
            }
    filtered_words = [word for word in words if word not in stop_words]

    if not shuffle and pct == 1.0:
        if len(filtered_words) == 0:
            return words
        return filtered_words
            
    if len(filtered_words) == 0:
        # If there are only stop words, use the original sentence
        num_words_to_keep = math.ceil(len(words) * 1)
        selected_words = random.sample(words, num_words_to_keep)
    else:
        # Calculate pct% of the remaining words, rounding up
        num_words_to_keep = math.ceil(len(filtered_words) * pct)
        selected_words = random.sample(filtered_words, num_words_to_keep)
    
    if replace and pl_dist is not None and len(filtered_words) > 0:
        if len(filtered_words) - num_words_to_keep > 0:
            sampled_pls = sample_pls(pl_dist, len(filtered_words) - num_words_to_keep)
            selected_words.extend(sampled_pls)
    
    return selected_words


def unique_ordered_list(lst):
    seen = set()
    unique_list = []
    for item in lst:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list


class CircularBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.start = 0
        self.end = 0
        self.size = 0

    def append(self, item):
        if self.size == self.capacity:
            self.buffer[self.end] = item
            self.start = (self.start + 1) % self.capacity
            self.end = (self.end + 1) % self.capacity
        else:
            self.buffer[self.end] = item
            self.end = (self.end + 1) % self.capacity
            self.size += 1

    def get_all_elements(self):
        elements = []
        idx = self.start
        for _ in range(self.size):
            elements.append(self.buffer[idx])
            idx = (idx + 1) % self.capacity
        return elements

    def clear(self):
        self.buffer = [None] * self.capacity
        self.start = 0
        self.end = 0
        self.size = 0