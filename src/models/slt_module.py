from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from src.models.components.loss_modules.ce_loss import CELoss
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

import ipdb
import lmdb
import os
import json
from tqdm import tqdm

from src.utils.data_utils import CircularBuffer
from src.utils.gather_utils import strings_to_tensor, tensor_to_strings
from src.utils.vis_utils import save_video
from src.utils.ret_utils import copy_tensor, calculate_average_logit_scores, calculate_retrieval_metrics, calculate_overlap_metrics
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer



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
        frames_path: str,
        output_dir: str,
        bleurt_path: str,
        context_len: int,
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

        self.context_len = context_len
        self.context_buffer = CircularBuffer(context_len)
        self.context_buffer.ep = None

        self.context_conf_buffer = CircularBuffer(context_len)
        self.context_conf_buffer.ep = None 

        self.ret_dataloader = None
        self.rgb_lmdb_env = None

        self.vis_dir = f"{output_dir}/vis"
        os.makedirs(self.vis_dir, exist_ok=True)

        # bleurt metric
        self.bleurt_model = BleurtForSequenceClassification.from_pretrained(bleurt_path)
        self.bleurt_tokenizer = BleurtTokenizer.from_pretrained(bleurt_path)
    
    def predict_step(self, batch, batch_idx):
        pass

    def collect_dataset(self, dataloader):
        result_dict = {}
    
        # Iterate through each dictionary in the list
        for dictionary in dataloader:
            for key, value in dictionary.items():
                if key in result_dict:
                    if isinstance(value, list):
                        result_dict[key].extend(value)
                    elif isinstance(value, torch.Tensor):
                        result_dict[key].extend([value[i] for i in range(value.shape[0])])
                else:
                    if isinstance(value, list):
                        result_dict[key] = value.copy()  # use copy to avoid modifying the original list
                    elif isinstance(value, torch.Tensor):
                        result_dict[key] = []
                        result_dict[key].extend([value[i] for i in range(value.shape[0])])  # use clone to avoid modifying the original tensor
        
        return result_dict

    def perform_retrieval(self, dataloader):
        dataset = self.collect_dataset(dataloader)

        bs = 2

        score_matrix = []
        
        for idx in range(len(dataset["pls"])):

            score_row = []

            try:
                feats = dataset["features"][idx]
                attn_masks = dataset["attn_masks"][idx]

                feats = copy_tensor(feats, bs)
                attn_masks = copy_tensor(attn_masks, bs)

                feats = feats.to(self.device)
                attn_masks = attn_masks.to(self.device)
            
            except:
                feats = torch.zeros(bs, 1, 4096).to(self.device, dtype=self.net.language_decoder.decoder.dtype)
                attn_masks = torch.zeros(bs, 1).to(self.device, dtype=self.net.language_decoder.decoder.dtype)
                

            target_indices = [dataset["target_indices"][idx] for _ in range(bs)]
            target_labels = [dataset["target_labels"][idx] for _ in range(bs)]
            pls = [dataset["pls"][idx] for _ in range(bs)]
            sub_gt = [dataset["sub_gt"][idx] for _ in range(bs)]
            probs = [dataset["probs"][idx] for _ in range(bs)]
            previous_contexts = [dataset["previous_contexts"][idx] for _ in range(bs)]
            questions = [dataset["questions"][idx] for _ in range(bs)]
            bg_description = [dataset["bg_description"][idx] for _ in range(bs)]
            
            for idx2 in tqdm(range(0, len(dataset["pls"]), bs)):

                tmp_batch = {
                        "features": feats,
                        "attn_masks": attn_masks,
                        "subtitles": dataset["subtitles"][idx2:idx2+bs],
                        "questions": questions,
                        "previous_contexts": previous_contexts,
                        "pls": pls,
                        "target_indices": target_indices,
                        "target_labels": target_labels,
                        "start": dataset["start"][idx2:idx2+bs],
                        "end": dataset["end"][idx2:idx2+bs],
                        "video_names": dataset["video_names"][idx2:idx2+bs],
                        "sub_gt": sub_gt,
                        "probs": probs,
                        "bg_description": bg_description
                }
                with torch.no_grad():
                    outputs_list, labels_list, _, _ = self.forward(tmp_batch, ret=True)

                    if len(score_row) == 0:
                        for _ in range(len(outputs_list)):
                            score_row.append([])
                    if len(score_matrix) == 0:
                        for _ in range(len(outputs_list)):
                            score_matrix.append([])
                
                for idx3, (outputs, labels) in enumerate(zip(outputs_list, labels_list)):
                    scores = calculate_average_logit_scores(outputs["logits"], labels)
                    score_row[idx3].extend(scores)

            for idm in range(len(score_row)):
                score_matrix[idm].append(score_row[idm])

        ret_metrics_list = []
        for i in range(len(score_matrix)):
            ret_metrics = calculate_retrieval_metrics(score_matrix[i])
            ret_metrics_list.append(ret_metrics)
        
        return ret_metrics_list  
   
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
            batch["rec_prev_conf"] = [self.context_conf_buffer.get_all_elements()]

        outputs_list, labels_list, preds_list, avg_conf_list = self.forward(batch)

        if len(self.all_preds) == 0 and preds_list is not None:
            self.all_preds = [[] for _ in range(len(preds_list))]
            self.avg_confs = [[] for _ in range(len(preds_list))]
        
        loss = self.criterion(outputs_list[0], labels_list[0])
        

        if preds_list is not None:
            for idx_p, preds in enumerate(preds_list):
                for ip, pred in enumerate(preds):
                    decoded_pred = self.net.language_decoder.tokenizer.decode(pred, skip_special_tokens=True)
                    self.all_preds[idx_p].append(decoded_pred)
                    self.avg_confs[idx_p].append(avg_conf_list[idx_p][ip])
                    if len(self.all_preds) == 1 or idx_p == 1:
                        self.context_buffer.append(decoded_pred)
                        self.context_conf_buffer.append(avg_conf_list[idx_p][ip])
                
            self.all_gts.extend(batch["subtitles"])
            self.pls.extend(batch['pls'])
            self.starts.extend(batch['start'])
            self.ends.extend(batch['end'])
            self.names.extend(batch['video_names'])
            self.sub_gts.extend(batch['sub_gt'])
            self.probs.extend(batch['probs'])
            self.blip_cap.extend(batch['bg_description'])
            self.rec_prev.extend(batch['rec_prev'])
            self.rec_prev_conf.extend(batch['rec_prev_conf'])
            if batch['previous_contexts'] is not None:
                self.prev_context.extend(batch['previous_contexts'])

        return loss, preds_list, labels_list

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

    def get_cap_metrics(self, idx = 0):
        
        tensor_preds = strings_to_tensor(self.all_preds[idx])
        tensor_gt = strings_to_tensor(self.all_gts)

        m_preds_tensor = self.all_gather(tensor_preds)
        m_gt_tensor = self.all_gather(tensor_gt)

        all_preds = tensor_to_strings(m_preds_tensor.view(-1, 1024))
        all_gts = tensor_to_strings(m_gt_tensor.view(-1, 1024))

        hypotheses = {'image'+str(i): [all_preds[i]] for i in range(len(all_preds))}
        references = {'image'+str(i): [all_gts[i]] for i in range(len(all_gts))}

        bleu_score = self.bleu.compute_score(hypotheses, references)[0][3]
        rouge_score = self.rouge.compute_score(references, hypotheses)[0]
        cider_score = self.cider.compute_score(references, hypotheses)[0]
        bleurt_score = sum(self.get_bleurt_scores(all_gts, all_preds))/len(all_gts)
        iou_list, precision_list, recall_list = calculate_overlap_metrics(all_gts, all_preds)
        iou = sum(iou_list)/len(iou_list)
        precision = sum(precision_list)/len(precision_list)
        recall = sum(recall_list)/len(recall_list)

        if self.global_rank == 0:
            cap_dict = {
                "gt": all_gts,
                "pred": all_preds,
            }
            with open(f"{self.vis_dir}/{self.current_epoch}/cap_{idx}.json", 'w') as f:
                json.dump(cap_dict, f)


        return bleu_score, rouge_score, cider_score, bleurt_score, iou, precision, recall
    
    def _eval(self) -> None:
        if self.global_rank == 0:
            hypotheses = {'image'+str(i): [self.all_preds[0][i]] for i in range(len(self.all_preds[0]))}
            references = {'image'+str(i): [self.all_gts[i]] for i in range(len(self.all_gts))}

            iou, precision, recall = calculate_overlap_metrics(self.all_gts, self.all_preds[0])

            _, rouge_scores  = self.rouge.compute_score(references, hypotheses)

            bleurt_scores = self.get_bleurt_scores(self.all_gts, self.all_preds[0])

            if len(self.all_preds) > 1:

                hypotheses = {'image'+str(i): [self.all_preds[1][i]] for i in range(len(self.all_preds[1]))}
                references = {'image'+str(i): [self.all_gts[i]] for i in range(len(self.all_gts))}

                iou_pl, precision_pl, recall_pl = calculate_overlap_metrics(self.all_gts, self.all_preds[1])

                _, rouge_scores_pl  = self.rouge.compute_score(references, hypotheses)

                bleurt_scores_pl = self.get_bleurt_scores(self.all_gts, self.all_preds[1])

            vis_list = []

            os.makedirs(f"{self.vis_dir}/{self.current_epoch}", exist_ok=True)

            for idx in range(len(rouge_scores)):
                tmp_dict = {}
                tmp_dict['vid'] = self.names[idx]
                tmp_dict['rouge'] = rouge_scores[idx]
                tmp_dict['start'] = self.starts[idx]
                tmp_dict['end'] = self.ends[idx]
                tmp_dict['gt'] = self.all_gts[idx]
                tmp_dict['pred'] = self.all_preds[0][idx]
                tmp_dict['conf'] = self.avg_confs[0][idx]
                tmp_dict['pls'] = self.pls[idx]
                tmp_dict['sub_gt'] = self.sub_gts[idx]
                tmp_dict['bleurt'] = bleurt_scores[idx]
                tmp_dict['prob'] = self.probs[idx]
                tmp_dict['iou'] = iou[idx]
                tmp_dict['precision'] = precision[idx]
                tmp_dict['recall'] = recall[idx]
                tmp_dict['blip_cap'] = self.blip_cap[idx]
                tmp_dict['rec_prev'] = self.rec_prev[idx]
                tmp_dict['rec_prev_conf'] = self.rec_prev_conf[idx]
                if len(self.all_preds) > 1:
                    tmp_dict["pred_pls"] = self.all_preds[1][idx]
                    tmp_dict["conf_pls"] = self.avg_confs[1][idx]
                    tmp_dict["bleurt_pls"] = bleurt_scores_pl[idx]
                    tmp_dict["rouge_pls"] = rouge_scores_pl[idx]
                    tmp_dict["iou_pls"] = iou_pl[idx]
                    tmp_dict["precision_pls"] = precision_pl[idx]
                    tmp_dict["recall_pls"] = recall_pl[idx]

                if len(self.prev_context) > 0:
                    tmp_dict['prev_contexts'] = self.prev_context[idx] 
                
                vis_list.append(tmp_dict)
                if self.rgb_lmdb_env is not None:
                    save_video(self.names[idx], self.starts[idx], self.ends[idx], f"{self.vis_dir}/{self.current_epoch}/{idx}.mp4", self.rgb_lmdb_env)
            
            with open(f'{self.vis_dir}/{self.current_epoch}/info.json', 'w') as f:
                json.dump(vis_list, f)


        for idx_p in range(len(self.all_preds)):
            bleu_score, rouge_score, cider_score, bleurt_score, iou, precision, recall = self.get_cap_metrics(idx_p)
            if len(self.all_preds) > 1:
                self.log(f"bleu_{idx_p}", bleu_score, on_step=False, on_epoch=True, prog_bar=True)
                self.log(f"rouge_{idx_p}", rouge_score, on_step=False, on_epoch=True, prog_bar=True)
                self.log(f"cider_{idx_p}", cider_score, on_step=False, on_epoch=True, prog_bar=True)
                self.log(f"bleurt_{idx_p}", bleurt_score, on_step=False, on_epoch=True, prog_bar=True)
                self.log(f"iou_{idx_p}", iou, on_step=False, on_epoch=True, prog_bar=True)
                self.log(f"precision_{idx_p}", precision, on_step=False, on_epoch=True, prog_bar=True)
                self.log(f"recall_{idx_p}", recall, on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.log("bleu", bleu_score, on_step=False, on_epoch=True, prog_bar=True)
                self.log("rouge", rouge_score, on_step=False, on_epoch=True, prog_bar=True)
                self.log("cider", cider_score, on_step=False, on_epoch=True, prog_bar=True)
                self.log("bleurt", bleurt_score, on_step=False, on_epoch=True, prog_bar=True)
                self.log("iou", iou, on_step=False, on_epoch=True, prog_bar=True)
                self.log("precision", precision, on_step=False, on_epoch=True, prog_bar=True)
                self.log("recall", recall, on_step=False, on_epoch=True, prog_bar=True)

        ret_metrics_list = None
        # if not self.trainer.sanity_checking:
        ret_metrics_list = self.perform_retrieval(self.ret_dataloader)

        if ret_metrics_list is not None:
            for idx, ret_metrics in enumerate(ret_metrics_list):
                if len(self.all_preds) > 1:
                    self.log(f"R@1_{idx}", ret_metrics["recall_scores"][1], on_step=False, on_epoch=True, prog_bar=True)
                    self.log(f"R@5_{idx}", ret_metrics["recall_scores"][5], on_step=False, on_epoch=True, prog_bar=True)
                    self.log(f"R@10_{idx}", ret_metrics["recall_scores"][10], on_step=False, on_epoch=True, prog_bar=True)
                    self.log(f"R@25_{idx}", ret_metrics["recall_scores"][25], on_step=False, on_epoch=True, prog_bar=True)
                    self.log(f"R@50_{idx}", ret_metrics["recall_scores"][50], on_step=False, on_epoch=True, prog_bar=True)
                    self.log(f"mean_rank_{idx}", ret_metrics["mean_rank"], on_step=False, on_epoch=True, prog_bar=True)
                    self.log(f"median_rank_{idx}", ret_metrics["median_rank"], on_step=False, on_epoch=True, prog_bar=True)
                else:
                    self.log("R@1", ret_metrics["recall_scores"][1], on_step=False, on_epoch=True, prog_bar=True)
                    self.log("R@5", ret_metrics["recall_scores"][5], on_step=False, on_epoch=True, prog_bar=True)
                    self.log("R@10", ret_metrics["recall_scores"][10], on_step=False, on_epoch=True, prog_bar=True)
                    self.log("R@25", ret_metrics["recall_scores"][25], on_step=False, on_epoch=True, prog_bar=True)
                    self.log("R@50", ret_metrics["recall_scores"][50], on_step=False, on_epoch=True, prog_bar=True)
                    self.log("mean_rank", ret_metrics["mean_rank"], on_step=False, on_epoch=True, prog_bar=True)
                    self.log("median_rank", ret_metrics["median_rank"], on_step=False, on_epoch=True, prog_bar=True)

        self.all_preds = []
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
        self.context_buffer.clear() # log the rec_prev_context
        self.context_conf_buffer.clear()

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
        if self.ret_dataloader is None:
            self.ret_dataloader = self.trainer.datamodule.ret_dataloader()
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
