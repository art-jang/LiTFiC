"""
Creation of sentence_transformer model like for T5 architecture.
https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py
"""
import logging
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm.autonotebook import trange
from transformers import T5EncoderModel, T5Tokenizer

logger = logging.getLogger(__name__)


def make_sentence_model(
    model_name: str = "t5-base",
    root_path: bool = False
) -> nn.Module:
    """
    Setup SentenceTransformer model.

    Args:
        model_name (str): name of the model to use.
        root_path (bool): whether to use the root path of the model.
    """
    if root_path is not None:
        model = T5SentenceTransformer(root_path + model_name)
    else:
        model = T5SentenceTransformer(model_name)
    return model


class T5SentenceTransformer(nn.Module):
    """Loads or create a T5 model, that can be used to map sentences / text to embeddings."""
    def __init__(self, model_path_or_name: str):
        """Initializes a T5 model."""
        super().__init__()
        logger.info("Load pretrained T5 model %s", model_path_or_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path_or_name)
        self.model = T5EncoderModel.from_pretrained(model_path_or_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Use pytorch device: %s", device)
        self._target_device = torch.device(device)

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        Computes sentence embeddings

        Args:
            sentences (Union[str, List[str]]): the sentences to embed
            batch_size (int): the batch size used for the computation
            show_progress_bar (bool): Output a progress bar when encode sentences
            output_value (str): Default, sentence_embeddings, to get sentence embeddings.
                            Can be set to token_embeddings to get wordpiece token embeddings.
            convert_to_numpy (bool): If true, the output is a list of numpy vectors.
                                Else, it is a list of pytorch tensors.
            convert_to_tensor (bool): If true, the output is a list of pytorch tensors.
                                Else, it is a list of numpy vectors.
            device (str): Which torch.device to use for the computation
            normalize_embeddings (bool): If set to true, returned vectors will have length 1.

        Returns:
            By default, a list of tensors is returned.
            If convert_to_tensor, a stacked tensor is returned.
            If convert_to_numpy, a numpy matrix is returned.
        """
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or \
                logger.getEffectiveLevel() == logging.DEBUG
            )
        if convert_to_tensor:
            convert_to_numpy = False
        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False
        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            # cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True
        if device is None:
            device = self._target_device

        self.model.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        pbar = trange(
            0, len(sentences_sorted), batch_size,
            desc="Batches",
            disable=not show_progress_bar,
        )
        for start_index in pbar:
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            features = self.tokenizer(
                sentences_batch,
                padding=True,
                return_tensors="pt",
            )
            features = features.to(device)
            # forward (only the encoder)
            out_features = self.model(**features)
            if output_value == "token_embeddings":
                embeddings = []
                zipped_features = zip(out_features["last_hidden_state"], features["attention_mask"])
                for token_emb, attention in zipped_features:
                    last_mask_id = len(attention) - 1
                    while last_mask_id > 0 and attention[last_mask_id] == 0:
                        last_mask_id -= 1
                    embeddings.append(token_emb[0:last_mask_id + 1])
            elif output_value is None:
                # return all outputs
                embeddings = []
                for sent_idx in range(len(out_features["attentions"])):
                    row = {name: out_features[name][sent_idx] for name in out_features}
                    embeddings.append(row)
            else:
                # sentence embeddings
                # T5 has no pooling/cls token --> take the mean over all tokens
                token_embeddings = out_features["last_hidden_state"]
                input_mask_expanded = features["attention_mask"].unsqueeze(-1).expand(
                    token_embeddings.size()
                ).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / \
                    torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                if convert_to_numpy:
                    embeddings = embeddings.detach()
                    embeddings = embeddings.cpu().numpy()
            all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        if input_was_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings

    def get_max_seq_length(self):
        """
        Returns the maximal sequence length for input the model accepts.
        Longer inputs will be truncated.
        """
        if hasattr(self.tokenizer, "max_seq_length"):
            return self.tokenizer.max_seq_length
        return None

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text.

        Args:
            text (Union[List[int], List[List[int]]]):
                list of ints (which means a signle text as input),
                or a tuple of list of ints (representing several text inputs to the model)
        """
        if isinstance(text, dict):
            # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):
            # object has no len method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):
            # empty string or list of ints
            return len(text)
        else:
            # sum of length of individual strings
            return sum([len(t) for t in text])
