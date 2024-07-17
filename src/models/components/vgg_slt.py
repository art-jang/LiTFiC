import torch
from torch import nn

from src.models.components.vgg_slt_modules.visual_encoder import VisualEncoder
from src.models.components.vgg_slt_modules.mm_projector import MMProjector
from src.models.components.vgg_slt_modules.language_decoder import LanguageDecoder


class VggSLTNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        visual_encoder_config: dict,
        mm_projector_config: dict,
        llm_config: dict,
        load_features: bool = False,
        precision: bool = "float32",
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()
        self.load_features = load_features
        # self.visual_encoder = VisualEncoder(**visual_encoder_config, load_features=load_features, precision=precision)
        self.mm_projector = MMProjector(**mm_projector_config)
        self.language_decoder = LanguageDecoder(**llm_config, precision=precision)

    def forward(self, batch: dict) -> torch.Tensor:
        if self.load_features:
            x = batch["features"]
        masks = batch["attn_masks"]
        subtitles = batch["subtitles"]
        questions = batch["questions"] if batch["questions"][0] is not None else None
        previous_contexts = batch["previous_contexts"] if batch["previous_contexts"][0] is not None else None

        # x = self.visual_encoder(x, masks)        
        x = self.mm_projector(x)
        outputs, labels = self.language_decoder(x, 
                                                video_masks=masks,
                                                subtitles=subtitles,
                                                questions=questions, 
                                                previous_contexts=previous_contexts)

        return outputs, labels


if __name__ == "__main__":
    _ = VggSLTNet()
