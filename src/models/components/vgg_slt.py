import torch
from torch import nn

from src.models.components.vgg_slt_modules.mm_projector import MMProjector
from src.models.components.vgg_slt_modules.language_decoder import LanguageDecoder



class VggSLTNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
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
        self.mm_projector = MMProjector(**mm_projector_config)
        self.language_decoder = LanguageDecoder(**llm_config, precision=precision)


    def forward(self, batch: dict, predict = False, ret = False) -> torch.Tensor:
        if self.load_features:
            x = batch["features"]

        masks = batch["attn_masks"]
        subtitles = batch["subtitles"]
        questions = batch["questions"] if batch["questions"][0] is not None else None
        previous_contexts = batch["previous_contexts"] if batch["previous_contexts"][0] is not None else None

        pls = batch["pls"]

        try:
            rec_prev = batch["rec_prev"]
        except:
            rec_prev = []

        background_description = batch["bg_description"]

        spottings = batch["spottings"]

        if self.load_features:        
            x, masks = self.mm_projector(x, masks=masks)
            if masks is None:
                masks = torch.ones(x.shape[0], x.shape[1]).to(self.language_decoder.decoder.device, dtype=self.language_decoder.torch_dtype)
        else:
            x = torch.zeros(len(pls), 1, 4096).to(self.language_decoder.decoder.device, dtype=self.language_decoder.torch_dtype)
            masks = torch.zeros(len(pls), 1).to(self.language_decoder.decoder.device, dtype=self.language_decoder.torch_dtype)
        
        x = x.to(self.language_decoder.torch_dtype)
        masks = masks.to(self.language_decoder.torch_dtype)


        outputs, labels, gen_sentences= self.language_decoder(x, 
                                                video_masks=masks,
                                                subtitles=subtitles,
                                                questions=questions, 
                                                previous_contexts=previous_contexts,
                                                pls=pls,
                                                background_description=background_description,
                                                rec_prev=rec_prev,
                                                spottings=spottings,
                                                )
        return outputs, labels, gen_sentences


if __name__ == "__main__":
    _ = VggSLTNet()
