from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


all_preds = ['hi', 'hello', 'how are you']
all_gts = ['hi', 'hello', 'how are you']

hypotheses = {str(i): [{'image_id': str(i), 'id':str(i), 'caption': all_preds[i]}] for i in range(len(all_preds))}
references = {str(i): [{'image_id': str(i), 'id':str(i), 'caption': all_gts[i]}] for i in range(len(all_gts))}

tokenizer = PTBTokenizer()


hypotheses = tokenizer.tokenize(hypotheses)
references = tokenizer.tokenize(references)