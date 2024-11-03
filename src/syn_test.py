import pickle
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import random

sub = "Come immediately to the park!"

syn_dict = pickle.load(open("../cslr2_t/bobsl/syns/synonym_pickle_english_and_signdict_and_signbank.pkl", "rb"))

new_sub = ""
for word in sub.split():
    if word.lower() in syn_dict:
        new_sub += random.choice(syn_dict[word.lower()]) + " "
    else:
        new_sub += word + " "

print(new_sub.strip())