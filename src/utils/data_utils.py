import random
import math
import ipdb
import string
import torch
from operator import itemgetter
import string
# import contractions
import re
from collections import defaultdict

from bisect import bisect_left, bisect_right

# def fix_contractions(text):
#     fixed_text = contractions.fix(text)   
#     return fixed_text

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
    
    if sent[0] == " ":
        sent = sent[1:]
    if sent[-1] == " ":
        sent = sent[:-1]
    sent = re.sub(r'\s+',' ', sent)

    return sent


def get_unique_bg_words(bg_words, drop_sw = False):

    sent = "".join(bg_words.split(".")).strip()
    words = unique_ordered_list(sent.split())

    words = [word.lower() for word in words]

    if drop_sw:
        stop_words = ['yourself', 'only', 'i', "you've", 'him', 'during', "didn't", 'shouldn', 'what', 'me', 'yours', 'further', 'aren', 'is', 'above', 'we', 'haven', 'can', 'its', 'themselves', "mightn't", 'his', 'ain', 'and', 'not', 'wouldn', 'off', 'our', 'of', "you'll", 'from', 'their', 'being', "it's", 'myself', 'after', 'mightn', 'with', "weren't", 'will', 'now', 'll', 'such', "you're", 'where', 'over', 'very', 'needn', 'no', 'd', 'her', 'here', 'the', "wasn't", 'some', 'didn', 'then', 'having', 'both', 'each', 'wasn', 'there', 'when', 'before', 'them', 'why', "she's", 'himself', "needn't", 'your', 'other', "don't", 'any', 'won', 'ours', 'against', 'herself', 'doing', 'once', 'y', 'few', 'doesn', 'do', "that'll", 'in', "should've", 'o', 'couldn', 've', 'so', "mustn't", 'shan', 'does', 'by', 'itself', 'own', 'most', 'or', 'yourselves', 'up', 'than', 'again', 'too', 'below', 'between', 'were', 'down', 'nor', 'he', 'as', 'she', 'out', 'how', 'those', 'but', 'hadn', 'should', 'been', 'had', "shan't", 'isn', 'into', 'about', "couldn't", 'weren', 'who', 'under', 'these', "shouldn't", 'my', 'm', 'you', 'a', 'to', 'through', 'more', "you'd", 'hasn', 'just', "hadn't", 'was', 'whom', 'all', 'for', 'that', 'have', 'until', 'did', 're', 't', 'same', 'ma', 'it', 'has', "wouldn't", 'hers', 'at', "won't", 'this', 'if', "doesn't", 'while', 'they', 'be', "haven't", 'am', 'are', 'don', "hasn't", 's', "aren't", 'which', 'theirs', 'on', 'because', 'ourselves', 'mustn', "isn't", 'an']
        words = [word for word in words if word not in stop_words]
    
    return words

def remove_words(text, max_p=0.5):
    text = text.split()
    max_remove = random.randint(0, int(max_p * len(text)))
    keep_idx = sorted(random.sample(range(len(text)), len(text) - max_remove))
    return ' '.join([text[i] for i in keep_idx])


def compress_and_average(strings, numbers):
    # Step 2: Create a dictionary to collect numbers associated with each string
    if len(strings) == 0:
        return [], []

    data = defaultdict(list)
    for string, number in zip(strings, numbers):
        data[string].append(number)
    
    # Step 3: Calculate the average for each string
    averages = {key: sum(values) / len(values) for key, values in data.items()}
    
    # Convert the dictionary to two lists
    compressed_strings = list(averages.keys())
    compressed_averages = list(averages.values())
    
    return compressed_strings, compressed_averages