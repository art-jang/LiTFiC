"""Model combining video and text modalities together."""
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

from src.utils.cslr_tools import group_and_average_tokens



class CSLR2(nn.Module):
    """Model combining video and text modalities together."""
    def __init__(
        self,
        video_encoder: nn.Module,
        text_encoder: Union[nn.Module, object],
        video_sequence_ll: Optional[nn.Module] = None,
        video_token_ll: Optional[nn.Module] = None,
        text_sentence_ll: Optional[nn.Module] = None,
        text_word_ll: Optional[nn.Module] = None,
        pooling: str = "max",
        sign_ret: bool = False,
        no_video_encoder: bool = False,
        same_text_ll: bool = False,
        same_video_ll: bool = False,
        avg_baseline: bool = False
    ) -> None:
        """
        Args:
            video_encoder (nn.Module): video encoder model.
            text_encoder (Union[nn.Module, object]): text encoder model.
            video_sequence_ll (Optional[nn.Module]): linear layer for video sequence embeddings.
            video_token_ll (Optional[nn.Module]): linear layer for video token embeddings.
            text_sentence_ll (Optional[nn.Module]): linear layer for text sentence embeddings.
            text_word_ll (Optional[nn.Module]): linear layer for text word embeddings.
            pooling (str): pooling method for video embeddings.
            sign_ret (bool): whether sign retrieval loss is used.
            no_video_encoder (bool): whether to use video encoder.
            same_text_ll (bool): whether to use the same linear layer for text embeddings.
            same_video_ll (bool): whether to use the same linear layer for video embeddings.
        """
        super(CSLR2, self).__init__()
        self.video_encoder = video_encoder
        self.video_sequence_ll = video_sequence_ll
        self.video_token_ll = video_token_ll if sign_ret else None
        self.text_encoder = text_encoder
        self.text_sentence_ll = text_sentence_ll
        self.text_word_ll = text_word_ll if sign_ret else None
        self.pooling = pooling
        self.sign_ret = sign_ret
        self.no_video_encoder = no_video_encoder
        if self.no_video_encoder:
            self.video_encoder = None
        self.same_text_ll = same_text_ll
        self.same_video_ll = same_video_ll
        self.avg_baseline = avg_baseline

    def extract_sentence_embeddings(
        self,
        sentences: Union[str, List[str]],
        device: torch.device,
    ) -> torch.tensor:
        """
        Extract sentence embeddings.

        Args:
            sentences (Union[str, List[str]]): List of sentences or a single sentence.
            device (torch.device): Device to use for the model.

        Returns:
            Sentence embeddings (torch.tensor).
        """
        batch_size = len(sentences) if isinstance(sentences, List) else 1
        embeddings = self.text_encoder.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=device,
        )
        return embeddings

    def project_sentence_embeddings(
        self,
        embeddings: torch.tensor,
    ) -> torch.tensor:
        """Project sentence embeddings (text)."""
        return F.normalize(self.text_sentence_ll(embeddings), dim=-1)

    def project_sequence_embeddings(
        self,
        embeddings: torch.tensor,
    ) -> torch.tensor:
        """Project sequence embeddings (video)."""
        return F.normalize(self.video_sequence_ll(embeddings), dim=-1)

    def project_word_embeddings(
        self,
        embeddings: torch.tensor,
    ) -> torch.tensor:
        """Project word embeddings (text)."""
        assert self.text_word_ll is not None
        if self.same_text_ll:
            return F.normalize(self.text_sentence_ll(embeddings), dim=-1)
        return F.normalize(self.text_word_ll(embeddings), dim=-1)

    def project_token_embeddings(
        self,
        embeddings: torch.tensor,
    ) -> torch.tensor:
        """Project token embeddings (video)."""
        assert self.video_token_ll is not None
        if self.same_video_ll:
            return F.normalize(self.video_sequence_ll(embeddings), dim=-1)
        return F.normalize(self.video_token_ll(embeddings), dim=-1)

    def video_pooling(
        self,
        embeddings: torch.tensor,
        input_features: torch.tensor,
    ) -> torch.tensor:
        """
        Pooling of video embeddings to replace learnable CLS token.

        Args:
            embeddings (torch.tensor): embedded video features.
            input_features (torch.tensor): video features.

        Returns:
            Pooled video embeddings.
        """
        video_mask = (input_features != 0).sum(-1) != 0
        video_mask = video_mask.to(input_features.device, non_blocking=True)
        pool_start_idx = 0 if self.no_video_encoder else 1
        if self.pooling == "mean":
            cls_tokens = (embeddings[:, pool_start_idx:, :]
                          * video_mask[:, :, None])
            cls_tokens = cls_tokens.sum(1) / video_mask.sum(1)[:, None]
        elif self.pooling == "max":
            cls_tokens = embeddings[:, pool_start_idx:, :].max(dim=1)[0]
        elif self.pooling == "median":
            cls_tokens = embeddings[:, pool_start_idx:, :].median(dim=1)[0]
        else:
            # learnable CLS token
            cls_tokens = embeddings[:, 0, :]
        return cls_tokens

    def forward(
        self,
        video_features: torch.tensor,
        video_masks: torch.tensor,
        target_indices, 
        target_labels,
        # predict=False
    ):

        # video side
        # ipdb.set_trace()

        cls_tokens, _ = self.video_encoder(video_features, video_masks)

        video_tokens = cls_tokens[:,1:] 

        attention_mask = video_masks
        valid_ind = None

        if self.avg_baseline:
            video_tokens, attention_mask, valid_ind = group_and_average_tokens(video_tokens, target_indices, target_labels)
       
        # return video_tokens, attention_mask, valid_ind
        return video_tokens, attention_mask # valid_ind

    def forward_sentret(
        self,
        video_features: torch.Tensor,
        subtitles: List[str],
    ):
        """
        Forward function of the model (in the case only sentence-level retrieval is needed).

        Args:
            video_features (torch.tensor): video features.
            subtitles (List[str]): list of subtitles.

        Returns:
            cls_tokens (torch.tensor): video embeddings (sequence level).
            sentence_embds (torch.tensor): sentence embeddings (text).
        """
        device = video_features.device
        # video side
        if not self.no_video_encoder:
            cls_tokens, _ = self.video_encoder(video_features)
        else:
            cls_tokens = video_features

        cls_tokens = self.video_pooling(cls_tokens, video_features)
        if self.video_sequence_ll is not None:
            cls_tokens = self.project_sequence_embeddings(cls_tokens)
        else:
            # normalise
            cls_tokens = F.normalize(cls_tokens, dim=-1)

        # text side
        sentence_embds = self.extract_sentence_embeddings(
            subtitles, device)
        if self.text_sentence_ll is not None:
            sentence_embds = self.project_sentence_embeddings(
                sentence_embds)
        else:
            sentence_embds = F.normalize(sentence_embds, dim=-1)
        return cls_tokens, sentence_embds
