import os
import argparse
import json
import ast
import numpy as np
from tqdm import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["OPENAI_API_KEY"] = "put your openai key here"

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process data file path.')
parser.add_argument('--data_file', type=str, help='Path to the data JSON file')
args = parser.parse_args()

# Load data from the provided file path
with open(args.data_file, "r") as f:
    data = json.load(f)

def eval_each(text_gt, text_pred):
    sys_prompt = (
        "Evaluate how well the candidate sentence aligns with the content and meaning of the reference sentence on a scale of 0 to 5. "
        "Prioritize key nouns and verbs, while giving less importance to subject, pronouns, adjectives, and adverbs.\n\n"
        "Scoring Rules:\n"
        "Score at least 1: If the candidate sentence shares at least one key noun or verb (or their synonyms) with the reference sentence.\n"
        "Score at least 3: If the candidate sentence matches most of the key nouns and verbs (or their synonyms) from the reference sentence.\n"
        "Score at least 5: If the candidate sentence conveys the same overall meaning as the reference sentence, with only minor differences.\n\n"
        "Note: Do not penalize differences in less important words or variations in sentence structure. Focus solely on the essential meaning conveyed by the key nouns and verbs."
        "The candidate sentences are sign language translations of a signer signing the reference sentence. Try to be liberal in the nouns and verbs you consider."
    )

    # Hardcoded examples
    examples = [
        {
            "gt": "It's blind to the genius loci.",
            "pred": "And that's what it means to be dislocated.",
            "score": 0,
            "reason": "No shared key nouns or verbs; the reference mentions 'blind' and 'genius loci', while the candidate mentions 'dislocated'; meanings are different."
        },
        {
            "gt": "She put it by the entrance to the earth so we figure that they like heavy metal or something.",
            "pred": "You've been in a wheelchair for a long time.",
            "score": 0,
            "reason": "No shared key nouns or verbs; the reference talks about 'entrance', 'earth', 'heavy metal', while the candidate mentions 'wheelchair'; meanings are unrelated."
        },
        {
            "gt": "You're coming along to the finale tomorrow?",
            "pred": "I'll have to wait until tomorrow.",
            "score": 1,
            "reason": "Shares the key noun 'tomorrow' but lacks other key content; meanings are somewhat related but differ."
        },
        {
            "gt": "A man's can was open for attack at any point in their life.",
            "pred": "It's not a joke, it's a way of life.",
            "score": 1,
            "reason": "Shares the key noun 'life', but overall meanings are different; reference discusses vulnerability, candidate discusses lifestyle."
        },
        {
            "gt": "Richard called English Nature, who told him that they were natterjack toads, Britain's rarest amphibian.",
            "pred": "Richard, a Nottinghamshire Englishman, is a naturalist.",
            "score": 2,
            "reason": "Shares key nouns 'Richard' and 'English'; candidate omits details about 'natterjack toads' and 'Britain's rarest amphibian.'"
        },
        {
            "gt": "Cromwell treated Ireland like the primitive colony he thought it was, moving the Irish off their farms and using the land to pay his soldiers.",
            "pred": "Cromwell was just one of many areas where the IRA set up roadblocks to stop loyalist paramilitaries and farmers from getting through.",
            "score": 2,
            "reason": "Shares key nouns 'Cromwell', 'Ireland', 'farmers'; reference discusses historical actions, candidate discusses modern events; meanings differ."
        },
        {
            "gt": "He sort of guessed it would be 21 maybe 28 days, ended-up being 35.",
            "pred": "He thought it was 21 days.",
            "score": 3,
            "reason": "Matches key nouns 'he', 'thought', '21 days'; candidate conveys a similar time estimation with fewer details."
        },
        {
            "gt": "They get as high as they can off the ground to push the other male down to the floor, and once that male becomes subservient, he slinks off and the dominant male then goes to his female to breed.",
            "pred": "But when one of them is selected, it's the females that can be changed to breed with the new male.",
            "score": 3,
            "reason": "Shares key nouns 'male', 'female', 'breed'; both discuss breeding behaviors, though specifics differ."
        },
        {
            "gt": "It's a lack of understanding on both sides.",
            "pred": "I don't understand that.",
            "score": 3,
            "reason": "Shares key verb 'understand'; both express lack of understanding; candidate is less specific."
        },
        {
            "gt": "OK, we'll just ring him next time.",
            "pred": "I'll call you back.",
            "score": 4,
            "reason": "Shares key verb 'call' (synonym of 'ring'); both involve making a call; minor differences in context."
        },
        {
            "gt": "Really excited.",
            "pred": "I'm so excited.",
            "score": 5,
            "reason": "Conveys the same overall meaning; both express excitement with minor wording differences."
        },
        {
            "gt": "Every day is totally different.",
            "pred": "You know, every day is different.",
            "score": 5,
            "reason": "Conveys the same overall meaning; both state that each day is different with minor phrasing differences."
        },
    ]

    messages = [
        {"role": "system", "content": sys_prompt},
    ]

    # For each example, add a 'user' message and an 'assistant' message
    for example in examples:
        example_prompt = (
            "Assign a score from 0 to 5 based on the rules provided. "
            "Provide your answer in JSON format with keys \"score\" (0-5) and \"reason\" with a brief explanation.\n\n"
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the JSON string."

            f"Reference Sentence: {example['gt']}\n"
            f"Candidate Sentence: {example['pred']}\n\n"
        )
        example_output = f'{{"score": {example["score"]}, "reason": "{example["reason"]}"}}'

        messages.append({"role": "user", "content": example_prompt})
        messages.append({"role": "assistant", "content": example_output})

    # Now, add the actual input
    prompt = (
        "Assign a score from 0 to 5 based on the rules provided. "
        "Provide your answer in JSON format with keys \"score\" (0-5) and \"reason\" with a brief explanation.\n\n"
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the JSON string."

        f"Reference Sentence: {text_gt}\n"
        f"Candidate Sentence: {text_pred}\n\n"
    )
    messages.append({"role": "user", "content": prompt})

    # Generate the model output
    from openai import OpenAI
    
    # Ensure your API key is set
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    return completion.choices[0].message.content

scores = []
score_list = []

for i in tqdm(range(len(data["gt"])), total=len(data["gt"])):
# for i in tqdm(range(3), total=len(data["gt"])):
    text_gt = data["gt"][i]
    text_pred = data["pred"][i]
    import pdb; pdb.set_trace()
    score_data= eval_each(text_gt, text_pred)
    # score_data = json.loads(score_json)
    parsed = score_data.split("{")[1]
    parsed = "{" + parsed.split("}")[0] + "}"
    # print(parsed)
    # parsed.replace('<|eot_id|>', "")
    # print(parsed)
    try:
        d = ast.literal_eval(parsed)
    except:
        print(parsed)
        print("Error")
        continue
    scores.append(d["score"])
    score_list.append(d)

final_score_mean = np.mean(scores)
final_score_std = np.std(scores)

print(f"Mean score: {final_score_mean}")
print(f"Standard deviation of scores: {final_score_std}")

# Save the results to a JSON file
with open(args.data_file.split(".")[0]+ "_scores.json", "w") as f:
    json.dump({"mean": final_score_mean, "std": final_score_std}, f)
with open(args.data_file.split(".")[0]+ "_score_list.json", "w") as f:
    json.dump(score_list, f)