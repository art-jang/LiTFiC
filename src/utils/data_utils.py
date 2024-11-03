import random
import math
import ipdb
import string
import torch
from operator import itemgetter
import string
import contractions
import re

from bisect import bisect_left, bisect_right

def fix_contractions(text):
    fixed_text = contractions.fix(text)   
    return fixed_text

def remove_punctuation(text):
    punctuation_pattern = r'[^\w\s]'  # Matches any character that is not a word character or whitespace
    text_without_punctuation = re.sub(punctuation_pattern, '', text)

    return text_without_punctuation

def get_annotations_in_time_range(annotations_dict, episode_name, start_time, end_time):
    """
    Retrieves annot_words for a given episode within the specified time range.

    Parameters:
    - annotations_dict: The dictionary containing organized annotations.
    - episode_name: The name of the episode to query.
    - start_time: The start time of the range.
    - end_time: The end time of the range.

    Returns:
    - A list of annot_words within the time range.
    """
    # Retrieve the annotations for the specified episode
    episode_annotations = annotations_dict.get(episode_name, [])

    # Extract the list of annot_times for binary search
    times = [annot[0] for annot in episode_annotations]

    # Find the start and end indices using binary search
    start_index = bisect_left(times, start_time)
    end_index = bisect_right(times, end_time)

    # Slice the list to get annotations within the time range
    relevant_annotations = episode_annotations[start_index:end_index]

    # Extract and return the annot_words
    annot_words = [annot[1] for annot in relevant_annotations]

    return annot_words


def drop_words(word_list, pct):
    pct = random.random() * pct

    num_to_drop = int(len(word_list) * pct / 100)
    indices_to_drop = set(random.sample(range(len(word_list)), num_to_drop))
    return [word for i, word in enumerate(word_list) if i not in indices_to_drop]

def split_word_punctuation(word):
    prefix = ''
    suffix = ''
    while word and word[0] in string.punctuation:
        prefix += word[0]
        word = word[1:]
    while word and word[-1] in string.punctuation:
        suffix = word[-1] + suffix
        word = word[:-1]
    return prefix, word, suffix

def replace_random_word_with_synonym(sentence, syn_dict):
    words = sentence.split()
    indices_in_dict = []
    for i, word in enumerate(words):
        prefix, word_clean, suffix = split_word_punctuation(word)
        if word_clean.lower() in syn_dict:
            indices_in_dict.append(i)
    if indices_in_dict:
        idx = random.choice(indices_in_dict)
        word = words[idx]
        prefix, word_clean, suffix = split_word_punctuation(word)
        synonym = random.choice(syn_dict[word_clean.lower()])
        words[idx] = prefix + synonym + suffix
    return ' '.join(words)


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


def sample_sub(sentence, shuffle, pct=0.3, replace=False, pl_dist=None, drop_stopwords=False, sw_level=0):
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

    if sw_level == 0:
        stop_words = ['i', "you've", 'him', "didn't", 'shouldn', 'yours', 'aren', 'is', 'we', 'haven', 'its', 'themselves', "mightn't", 'his', 'ain', 'not', 'wouldn', 'of', "you'll", "it's", 'mightn', "weren't", 'll', 'such', "you're", 'very', 'needn', 'd', 'the', "wasn't", 'didn', 'having', 'both', 'wasn', 'them', "she's", 'himself', "needn't", "don't", 'won', 'ours', 'herself', 'doing', 'y', 'doesn', "that'll", "should've", 'o', 'couldn', 've', "mustn't", 'shan', 'does', 'itself', 'yourselves', 'than', 'too', 'were', 'as', 'she', 'those', 'but', 'hadn', 'had', "shan't", 'isn', 'into', "couldn't", 'weren', 'these', "shouldn't", 'm', 'a', "you'd", 'hasn', "hadn't", 'was', 'whom', 'did', 're', 't', 'ma', 'it', 'has', "wouldn't", 'hers', 'at', "won't", "doesn't", 'be', "haven't", 'am', 'are', 'don', "hasn't", 's', "aren't", 'theirs', 'ourselves', 'mustn', "isn't", 'an']
    elif sw_level == 1:
        stop_words = ['yourself', 'only', 'i', "you've", 'him', 'during', "didn't", 'shouldn', 'what', 'me', 'yours', 'further', 'aren', 'is', 'above', 'we', 'haven', 'can', 'its', 'themselves', "mightn't", 'his', 'ain', 'and', 'not', 'wouldn', 'off', 'our', 'of', "you'll", 'from', 'their', 'being', "it's", 'myself', 'after', 'mightn', 'with', "weren't", 'will', 'now', 'll', 'such', "you're", 'where', 'over', 'very', 'needn', 'no', 'd', 'her', 'here', 'the', "wasn't", 'some', 'didn', 'then', 'having', 'both', 'each', 'wasn', 'there', 'when', 'before', 'them', 'why', "she's", 'himself', "needn't", 'your', 'other', "don't", 'any', 'won', 'ours', 'against', 'herself', 'doing', 'once', 'y', 'few', 'doesn', 'do', "that'll", 'in', "should've", 'o', 'couldn', 've', 'so', "mustn't", 'shan', 'does', 'by', 'itself', 'own', 'most', 'or', 'yourselves', 'up', 'than', 'again', 'too', 'below', 'between', 'were', 'down', 'nor', 'he', 'as', 'she', 'out', 'how', 'those', 'but', 'hadn', 'should', 'been', 'had', "shan't", 'isn', 'into', 'about', "couldn't", 'weren', 'who', 'under', 'these', "shouldn't", 'my', 'm', 'you', 'a', 'to', 'through', 'more', "you'd", 'hasn', 'just', "hadn't", 'was', 'whom', 'all', 'for', 'that', 'have', 'until', 'did', 're', 't', 'same', 'ma', 'it', 'has', "wouldn't", 'hers', 'at', "won't", 'this', 'if', "doesn't", 'while', 'they', 'be', "haven't", 'am', 'are', 'don', "hasn't", 's', "aren't", 'which', 'theirs', 'on', 'because', 'ourselves', 'mustn', "isn't", 'an']

    if drop_stopwords:
        filtered_words = [word for word in words if word not in stop_words]
    else:
        filtered_words = words
            
    if len(filtered_words) == 0:
        num_words_to_keep = len(words)
        selected_words = random.sample(words, num_words_to_keep)
    else:
        if shuffle:
            num_words_to_keep = math.ceil(len(filtered_words) * pct)
            selected_words = random.sample(filtered_words, num_words_to_keep)
        else:
            randIndex = random.sample(range(len(filtered_words)), math.ceil(len(filtered_words) * pct))
            randIndex.sort()
            selected_words = [filtered_words[i] for i in randIndex]
    
    
    if replace and pl_dist is not None and len(filtered_words) > 0:
        if len(filtered_words) - num_words_to_keep > 0:
            sampled_pls = sample_pls(pl_dist, len(filtered_words) - num_words_to_keep)
            selected_words.extend(sampled_pls)
    
    return selected_words


def sample_sub_prev(sentence, pct=0.5, shuffle=False):
    words = sentence.split()

    filtered_words = [word.strip(string.punctuation).lower() for word in words]  

    pct = 1 - (random.random() * pct)
    
    if shuffle:
        num_words_to_keep = math.ceil(len(filtered_words) * pct)
        selected_words = random.sample(filtered_words, num_words_to_keep)
    else:
        randIndex = random.sample(range(len(filtered_words)), math.ceil(len(filtered_words) * pct))
        randIndex.sort()
        selected_words = [filtered_words[i] for i in randIndex]
    
    sent = ' '.join(selected_words) + '.'

    return sent


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


def cleanup_sub(sent):
    # if the first or last character is " ' ", remove it
    if sent[0] == "'":
        sent = sent[1:]
    if sent[-1] == "'":
        sent = sent[:-1]
    
    sent = sent.lower()
    sent = fix_contractions(sent)
    sent = remove_punctuation(sent)

    return sent


def process_cslr2_pls(labels, probs, inv_vocab):
    
    nl = labels[:,0].tolist()
    np = probs[:,0].tolist()
    
    indices = list(range(len(nl)))

    target_labels = torch.tensor(nl)
    target_indices = torch.tensor(indices)

    pl = list(itemgetter(*nl)(inv_vocab))

    return pl, np, target_labels, target_indices

def get_unique_bg_words(bg_words, drop_sw = False):

    sent = "".join(bg_words.split(".")).strip()
    words = unique_ordered_list(sent.split())

    words = [word.lower() for word in words]

    if drop_sw:
        stop_words = ['yourself', 'only', 'i', "you've", 'him', 'during', "didn't", 'shouldn', 'what', 'me', 'yours', 'further', 'aren', 'is', 'above', 'we', 'haven', 'can', 'its', 'themselves', "mightn't", 'his', 'ain', 'and', 'not', 'wouldn', 'off', 'our', 'of', "you'll", 'from', 'their', 'being', "it's", 'myself', 'after', 'mightn', 'with', "weren't", 'will', 'now', 'll', 'such', "you're", 'where', 'over', 'very', 'needn', 'no', 'd', 'her', 'here', 'the', "wasn't", 'some', 'didn', 'then', 'having', 'both', 'each', 'wasn', 'there', 'when', 'before', 'them', 'why', "she's", 'himself', "needn't", 'your', 'other', "don't", 'any', 'won', 'ours', 'against', 'herself', 'doing', 'once', 'y', 'few', 'doesn', 'do', "that'll", 'in', "should've", 'o', 'couldn', 've', 'so', "mustn't", 'shan', 'does', 'by', 'itself', 'own', 'most', 'or', 'yourselves', 'up', 'than', 'again', 'too', 'below', 'between', 'were', 'down', 'nor', 'he', 'as', 'she', 'out', 'how', 'those', 'but', 'hadn', 'should', 'been', 'had', "shan't", 'isn', 'into', 'about', "couldn't", 'weren', 'who', 'under', 'these', "shouldn't", 'my', 'm', 'you', 'a', 'to', 'through', 'more', "you'd", 'hasn', 'just', "hadn't", 'was', 'whom', 'all', 'for', 'that', 'have', 'until', 'did', 're', 't', 'same', 'ma', 'it', 'has', "wouldn't", 'hers', 'at', "won't", 'this', 'if', "doesn't", 'while', 'they', 'be', "haven't", 'am', 'are', 'don', "hasn't", 's', "aren't", 'which', 'theirs', 'on', 'because', 'ourselves', 'mustn', "isn't", 'an']
        words = [word for word in words if word not in stop_words]
    
    return words