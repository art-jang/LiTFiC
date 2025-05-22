import os
import ast
import json
import argparse
import numpy as np
from tqdm import tqdm
from openai import OpenAI  # Ensure you have the correct OpenAI import
import concurrent.futures

# Set environment variables
os.environ['HF_HOME'] = '/scratch/shared/beegfs/haran/cache/'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["OPENAI_API_KEY"] = ""

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process data file path.')
    parser.add_argument('data_file', type=str, help='Path to the data JSON file')
    parser.add_argument('--max_threads', type=int, default=10, help='Maximum number of threads to use')
    return parser.parse_args()

def load_data(file_path):
    """
    Load JSON data from the provided file path.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def initialize_openai_client():
    """
    Initialize and return an OpenAI client instance.
    """
    return OpenAI()

def eval_each(text_gt, text_pred, client):
    """
    Evaluate the alignment between ground truth and predicted text using OpenAI's API.

    Parameters:
    - text_gt (str): Ground truth sentence.
    - text_pred (str): Predicted sentence.

    Returns:
    - dict: A dictionary containing 'score' and 'reason'.
    """
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

    # Add hardcoded examples to the messages
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

    # Add the actual input to the messages
    prompt = (
        "Assign a score from 0 to 5 based on the rules provided. "
        "Provide your answer in JSON format with keys \"score\" (0-5) and \"reason\" with a brief explanation.\n\n"
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the JSON string."

        f"Reference Sentence: {text_gt}\n"
        f"Candidate Sentence: {text_pred}\n\n"
    )
    messages.append({"role": "user", "content": prompt})

    try:
        # Generate the model output
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        response = completion.choices[0].message.content

        # Parse the JSON response
        parsed = response.split("{")[1]
        parsed = "{" + parsed.split("}")[0] + "}"
        d = ast.literal_eval(parsed)
        return d
    except Exception as e:
        print(f"Error processing: GT='{text_gt}' | Pred='{text_pred}'")
        print(f"Exception: {e}")
        return None

def process_item(i, data, client):
    """
    Process a single item by evaluating its score.

    Parameters:
    - i (int): Index of the item.
    - data (dict): The data dictionary containing 'gt' and 'pred'.
    - client (OpenAI): An instance of the OpenAI client.

    Returns:
    - tuple: (score, score_data) or (None, None) in case of error.
    """
    text_gt = data["gt"][i]
    text_pred = data["pred"][i]
    score_data = eval_each(text_gt, text_pred, client)
    if score_data is None:
        return None, None
    score_data["gt"] = text_gt
    score_data["pred"] = text_pred
    score_data["index"] = i
    return score_data["score"], score_data

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load data
    data = load_data(args.data_file)

    # Initialize OpenAI client
    client = initialize_openai_client()

    scores = []
    score_list = []

    # Define the number of threads
    max_threads = args.max_threads
    print(f"Using up to {max_threads} threads for processing.")

    # Use ThreadPoolExecutor for multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Create a partial function with fixed data and client
        # Submit all tasks to the executor
        futures = {
            executor.submit(process_item, i, data, client): i for i in range(len(data["gt"]))
        }

        # Initialize tqdm progress bar
        with tqdm(total=len(data["gt"]), desc="Processing", unit="item") as pbar:
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                result = future.result()
                if result != (None, None):
                    score, score_data = result
                    scores.append(score)
                    score_list.append(score_data)
                # Update the progress bar
                pbar.update(1)

    # Compute final statistics
    final_score_mean = np.mean(scores) if scores else 0
    final_score_std = np.std(scores) if scores else 0

    # Save the results to JSON files
    base_filename = os.path.splitext(args.data_file)[0]
    with open(f"{base_filename}_scores_2.json", "w") as f:
        json.dump({"mean": final_score_mean, "std": final_score_std}, f, indent=4)
    with open(f"{base_filename}_score_list_2.json", "w") as f:
        json.dump(score_list, f, indent=4)

    print(f"Processing complete. Mean Score: {final_score_mean}, Std Dev: {final_score_std}")
    print(f"Results saved to '{base_filename}_scores.json' and '{base_filename}_score_list.json'.")

if __name__ == "__main__":
    main()
