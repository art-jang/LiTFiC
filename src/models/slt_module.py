from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

from src.models.components.loss_modules.ce_loss import CELoss
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

import os
import json

from src.utils.data_utils import CircularBuffer
from src.utils.model_utils import strings_to_tensor, tensor_to_strings, calculate_overlap_metrics
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer

from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords, wordnet
import nltk
import pickle



class SLTLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        scheduler_options: dict,
        compile: bool,
        output_dir: str,
        bleurt_path: str,
        context_len: int,
        synonyms_pkl: str,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = CELoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.predict = False

        # for evaluation
        self.bleu = Bleu(4)
        self.rouge = Rouge()
        self.cider = Cider()
        self.tokenizer = PTBTokenizer()

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        self.synonym_dict = pickle.load(open(synonyms_pkl, "rb"))

        self.all_preds=[]
        self.avg_confs=[] 
        self.all_gts = []
        self.pls = []
        self.starts = []
        self.ends = []
        self.names = []
        self.sub_gts = []
        self.prev_context = []
        self.blip_cap = []
        self.probs = []
        self.rec_prev = []
        self.rec_prev_conf = []
        self.prev_pls = []
        self.prev_pls_probs = []
        self.ids = []

        self.context_len = context_len
        self.context_buffer = CircularBuffer(context_len)
        self.context_buffer.ep = None

        self.vis_dir = f"{output_dir}/vis"
        os.makedirs(self.vis_dir, exist_ok=True)

        # bleurt metric
        self.bleurt_model = BleurtForSequenceClassification.from_pretrained(bleurt_path)
        self.bleurt_tokenizer = BleurtTokenizer.from_pretrained(bleurt_path)
    
    def get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def incorporate_syns(self, gts, hyps):
        # Lemmatize and remove stopwords

        gt_final, hyp_final = [], []
        

        for gt, hyp in zip(gts, hyps):
            gt_arr = gt.split()
            hyp_arr = hyp.split()

            gt_arr = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in gt_arr if word not in self.stop_words]
            hyp_arr = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in hyp_arr if word not in self.stop_words]

            hyp_no_repeats = []
            hyp_no_synonynms_lexemes = []
            hyp_match_gt =[]
            synonymes_lexemes = []


            if len(gt_arr) > 0:
                gt_word_2_synonyms = {}
                for gt_word in gt_arr:
                    if gt_word in self.synonym_dict:
                        gt_word_2_synonyms[gt_word] = self.synonym_dict[gt_word]
                
                for word in hyp_arr:
                    if word not in hyp_no_repeats:
                        hyp_no_repeats.append(word)
                
                for i, word in enumerate(hyp_no_repeats):
                    if i == 0:
                        hyp_no_synonynms_lexemes.append(word)
                        if word in self.synonym_dict.keys():
                            synonymes_lexemes += self.synonym_dict[word]
                    if i !=0 and word not in synonymes_lexemes:
                        hyp_no_synonynms_lexemes.append(word)
                        if word in self.synonym_dict.keys():
                            synonymes_lexemes += self.synonym_dict[word]
                ### make the words in hyp match synonym lexeme of ground truth
                for word in hyp_no_synonynms_lexemes:
                    matched_word = word
                    for key in gt_word_2_synonyms.keys():
                        if word in gt_word_2_synonyms[key]:
                            matched_word = key
                    hyp_match_gt.append(matched_word)
                

                gt_final.append(" ".join(gt_arr))
                hyp_final.append(" ".join(hyp_match_gt))
            
        return gt_final, hyp_final
    
    def predict_step(self, batch, batch_idx):
        pass

   
    def get_bleurt_scores(self, references, candidates, batch_size = 16):
        self.bleurt_model.eval()
        with torch.no_grad():
            results = []
            
            for idx in range(0, len(references), batch_size):
                refs = references[idx:idx+batch_size]
                cands = candidates[idx:idx+batch_size]
                inputs = self.bleurt_tokenizer(refs, cands, padding='longest', return_tensors='pt')
                inputs.input_ids = inputs.input_ids.to(self.bleurt_model.device)
                inputs.attention_mask = inputs.attention_mask.to(self.bleurt_model.device)
                inputs.token_type_ids = inputs.token_type_ids.to(self.bleurt_model.device)
                res = self.bleurt_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, token_type_ids=inputs.token_type_ids).logits.flatten().tolist()
                results.extend(res)

        return results

    def forward(self, x: torch.Tensor, ret = False) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """ 
        return self.net(x, ret=ret)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        if not self.training:
            if self.context_buffer.ep is None:
                self.context_buffer.ep = batch["video_names"][0]
            elif self.context_buffer.ep != batch["video_names"][0]:
                self.context_buffer.clear()
            batch["rec_prev"] = [self.context_buffer.get_all_elements()]

        outputs, labels, preds = self.forward(batch)

        
        loss = self.criterion(outputs, labels)
        

        if preds is not None:
            for _, pred in enumerate(preds):
                decoded_pred = self.net.language_decoder.tokenizer.decode(pred, skip_special_tokens=True)
                self.all_preds.append(decoded_pred)
                self.context_buffer.append(decoded_pred)
                
            self.all_gts.extend(batch["subtitles"])
            self.pls.extend(batch['pls'])
            self.starts.extend(batch['start'])
            self.ends.extend(batch['end'])
            self.names.extend(batch['video_names'])
            self.blip_cap.extend(batch['bg_description'])
            self.rec_prev.extend(batch['rec_prev'])
            try:
                self.ids.extend([int(x) for x in batch['ids']])
            except:
                self.ids.extend(batch['ids'])

        return loss, preds, labels

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, _, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0], on_step=True, on_epoch=False)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def gather_and_concatenate(self, tensor: torch.Tensor, dim: int = 0, pad_value: float = 0.0) -> torch.Tensor:
        """
        Gathers tensors of variable size across all processes and concatenates them into a single tensor.

        Args:
            tensor (torch.Tensor): The tensor to gather from each process.
            dim (int): The dimension along which to gather and concatenate. Default is 0.
            pad_value (float): The value used for padding smaller tensors. Default is 0.0.

        Returns:
            torch.Tensor: A concatenated tensor containing all tensors from all processes.
        """
        # Step 1: Gather the size of the tensor along the specified dimension from all processes
        tensor_size = torch.tensor([tensor.size(dim)], device=tensor.device)
        gathered_sizes = self.all_gather(tensor_size)  # Shape: [world_size, 1]
        sizes = gathered_sizes.squeeze(1).tolist()     # Convert to list of integers

        # Step 2: Determine the maximum size across all tensors
        max_size = max(sizes)

        # Step 3: Pad the tensor if it's smaller than the maximum size
        if tensor.size(dim) < max_size:
            pad_amount = max_size - tensor.size(dim)
            # Create a padding tensor filled with pad_value
            pad_shape = list(tensor.size())
            pad_shape[dim] = pad_amount
            padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
            # Concatenate the padding to the original tensor
            padded_tensor = torch.cat([tensor, padding], dim=dim)
        else:
            padded_tensor = tensor

        # Step 4: Perform all_gather on the padded tensors
        gathered_padded = self.all_gather(padded_tensor)  # Shape: [world_size, *padded_tensor.shape]

        # Step 5: Remove padding and collect tensors
        # Initialize a list to hold the unpadded tensors
        unpadded_tensors = []
        for i, size in enumerate(sizes):
            if dim == 0:
                # Slice along the first dimension
                unpadded = gathered_padded[i, :size]
            else:
                # Create slicing objects for other dimensions
                slices = [slice(None)] * gathered_padded.dim()
                slices[dim] = slice(0, size)
                unpadded = gathered_padded[i][tuple(slices)]
            unpadded_tensors.append(unpadded)

        # Step 6: Concatenate all unpadded tensors along the specified dimension
        concatenated = torch.cat(unpadded_tensors, dim=dim)

        return concatenated

    def get_cap_metrics(self):
        
        tensor_preds = strings_to_tensor(self.all_preds)
        tensor_gt = strings_to_tensor(self.all_gts)


        m_preds_tensor = self.gather_and_concatenate(tensor_preds)
        m_gt_tensor = self.gather_and_concatenate(tensor_gt)

        all_preds = tensor_to_strings(m_preds_tensor)
        all_gts = tensor_to_strings(m_gt_tensor)

        hypotheses = {str(i): [{'image_id': str(i), 'id':str(i), 'caption': all_preds[i]}] for i in range(len(all_preds))}
        references = {str(i): [{'image_id': str(i), 'id':str(i), 'caption': all_gts[i]}] for i in range(len(all_gts))}

        hypotheses = self.tokenizer.tokenize(hypotheses)
        references = self.tokenizer.tokenize(references)
        

        bleu_score = self.bleu.compute_score(hypotheses, references)[0][3]
        rouge_score = self.rouge.compute_score(references, hypotheses)[0]
        cider_score = self.cider.compute_score(references, hypotheses)[0]
        bleurt_score = sum(self.get_bleurt_scores(all_gts, all_preds))/len(all_gts)

        all_gts_f = [v[0] for k, v in references.items()]
        all_preds_f = [v[0] for k, v in hypotheses.items()]

        all_gts_f, all_preds_f = self.incorporate_syns(all_gts_f, all_preds_f)
        iou_list, precision_list, recall_list = calculate_overlap_metrics(all_gts_f, all_preds_f)
        iou = sum(iou_list)/len(iou_list)
        precision = sum(precision_list)/len(precision_list)
        recall = sum(recall_list)/len(recall_list)

        try:
            id_tensor = torch.tensor(self.ids)
            m_id_tensor = self.gather_and_concatenate(id_tensor)
            all_ids = m_id_tensor.view(-1).tolist() 
        except:
            all_ids = []

        if self.global_rank == 0:
            cap_dict = {
                "gt": all_gts,
                "pred": all_preds,
                "ids": all_ids,
            }
            os.makedirs(f"{self.vis_dir}/{self.current_epoch}", exist_ok=True)
            with open(f"{self.vis_dir}/{self.current_epoch}/cap.json", 'w') as f:
                json.dump(cap_dict, f)


        return bleu_score, rouge_score, cider_score, bleurt_score, iou, precision, recall
    
    def _eval(self) -> None:
        if self.global_rank == 0 and not self.trainer.testing:
            hypotheses = {'image'+str(i): [self.all_preds[i]] for i in range(len(self.all_preds))}
            references = {'image'+str(i): [self.all_gts[i]] for i in range(len(self.all_gts))}

            iou, precision, recall = calculate_overlap_metrics(self.all_gts, self.all_preds)

            _, rouge_scores  = self.rouge.compute_score(references, hypotheses)

            bleurt_scores = self.get_bleurt_scores(self.all_gts, self.all_preds)

            vis_list = []

            os.makedirs(f"{self.vis_dir}/{self.current_epoch}", exist_ok=True)

            for idx in range(len(rouge_scores)):
                tmp_dict = {}
                tmp_dict['vid'] = self.names[idx]
                tmp_dict['rouge'] = rouge_scores[idx]
                tmp_dict['start'] = self.starts[idx]
                tmp_dict['end'] = self.ends[idx]
                tmp_dict['gt'] = self.all_gts[idx]
                tmp_dict['pred'] = self.all_preds[idx]
                tmp_dict['pls'] = self.pls[idx]
                tmp_dict['bleurt'] = bleurt_scores[idx]
                tmp_dict['iou'] = iou[idx]
                tmp_dict['precision'] = precision[idx]
                tmp_dict['recall'] = recall[idx]
                tmp_dict['blip_cap'] = self.blip_cap[idx]
                tmp_dict['rec_prev'] = self.rec_prev[idx]
            

                if len(self.prev_context) > 0:
                    tmp_dict['prev_contexts'] = self.prev_context[idx] 
                
                vis_list.append(tmp_dict)
                
            with open(f'{self.vis_dir}/{self.current_epoch}/info.json', 'w') as f:
                json.dump(vis_list, f)


        bleu_score, rouge_score, cider_score, bleurt_score, iou, precision, recall = self.get_cap_metrics()
        self.log("bleu", bleu_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("rouge", rouge_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("cider", cider_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("bleurt", bleurt_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("iou", iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("precision", precision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("recall", recall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        

        self.all_preds = []
        self.all_gts = []
        self.pls = []
        self.starts = []
        self.ends = []
        self.names = []
        self.prev_context = []
        self.blip_cap = []
        self.rec_prev = []
        self.context_buffer.clear()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, _, _ = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        self._eval()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, _, _ = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        "Lightning hook that is called when a test epoch ends."
        self._eval()

    def on_predict_start(self):
        self.eval()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            if self.hparams.scheduler_options.interval == "step":
                if isinstance(scheduler, LinearWarmupCosineAnnealingLR):
                    scheduler.max_epochs *= self.trainer.fit_loop._data_source.instance.max_steps
                    scheduler.warmup_epochs *= self.trainer.fit_loop._data_source.instance.max_steps # divide by num gpus

                    n_gpus = self.trainer.num_nodes * self.trainer.num_devices
                    scheduler.max_epochs //= n_gpus
                    scheduler.warmup_epochs //= n_gpus

                    scheduler.max_epochs //= self.hparams.scheduler_options.divide_step
                    scheduler.warmup_epochs //= self.hparams.scheduler_options.divide_step
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.scheduler_options.monitor,
                    "interval": self.hparams.scheduler_options.interval,
                    "frequency": self.hparams.scheduler_options.frequency,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
