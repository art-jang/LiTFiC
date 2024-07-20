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

from src.utils.gather_utils import strings_to_tensor, tensor_to_strings
from src.utils.vis_utils import save_video



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

        self.all_preds =[]
        self.all_gts = []
        self.pls = []
        self.starts = []
        self.ends = []
        self.names = []

        self.rgb_lmdb_env = None

        if self.global_rank == 0:
            self.rgb_lmdb_env = lmdb.open(
                frames_path, readonly=True, lock=False, max_readers=512
            )

        self.vis_dir = f"{output_dir}/vis"
        os.makedirs(self.vis_dir, exist_ok=True)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """ 
        return self.net(x, predict=self.predict)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        if not self.predict:
            outputs, labels = self.forward(batch)
            loss = self.criterion(outputs, labels)
        else:
            preds, labels = self.forward(batch)
            loss = None
        if self.predict:

            for idx, (pred, gt) in enumerate(zip(preds, labels)):
                 
                decoded_pred = self.net.language_decoder.tokenizer.decode(pred, skip_special_tokens=True)

                self.all_preds.append(decoded_pred)
                self.all_gts.append(gt)
        else:
            preds = None

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
        loss, preds, labels = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0], on_step=True, on_epoch=False)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        if not self.trainer.sanity_checking:
            self.eval()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass
    

    def eval_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        # for all the keys in the batch if the value is a tenor, transfer it to the self.device
        for key in list(batch.keys()):
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device, dtype =next(self.net.parameters()).dtype)

        _, _, _ = self.model_step(batch)

        self.pls.extend(batch['pls'])
        self.starts.extend(batch['start'])
        self.ends.extend(batch['end'])
        self.names.extend(batch['video_names'])


    def eval(self):
        self.predict = True
        for batch_idx, batch in enumerate(self.trainer.datamodule.eval_dataloader()):
            self.eval_step(batch, batch_idx)
        self.predict = False

        self.eval_epoch_end()
    
    def eval_epoch_end(self):
        if self.global_step > 0:

            if self.global_rank == 0:
                hypotheses = {'image'+str(i): [self.all_preds[i]] for i in range(len(self.all_preds))}
                references = {'image'+str(i): [self.all_gt[i]] for i in range(len(self.all_gt))}

                _, rouge_scores  = self.rouge.compute_score(references, hypotheses)

                vis_list = []

                os.makedirs(f"{self.vis_dir}/{self.current_epoch}", exist_ok=True)

                for idx in range(len(rouge_scores)):
                    tmp_dict = {}
                    tmp_dict['vid'] = self.names[idx]
                    tmp_dict['rouge'] = rouge_scores[idx]
                    tmp_dict['start'] = self.starts[idx]
                    tmp_dict['end'] = self.ends[idx]
                    tmp_dict['gt'] = self.all_gt[idx]
                    tmp_dict['pred'] = self.all_preds[idx]
                    tmp_dict['pls'] = self.pls[idx]
                    vis_list.append(tmp_dict)
                    save_video(self.names[idx], self.starts[idx], self.ends[idx], f"{self.vis_dir}/{self.current_epoch}/{idx}.mp4", self.rgb_lmdb_env)
                
                with open(f'{self.vis_dir}/{self.current_epoch}/info.json', 'w') as f:
                    json.dump(vis_list, f)
        
            tensor_preds = strings_to_tensor(self.all_preds)
            tensor_gt = strings_to_tensor(self.all_gt)

            m_preds_tensor = self.all_gather(tensor_preds)
            m_gt_tensor = self.all_gather(tensor_gt)

            self.all_preds = tensor_to_strings(m_preds_tensor.view(-1, 1024))
            self.all_gt = tensor_to_strings(m_gt_tensor.view(-1, 1024))

            hypotheses = {'image'+str(i): [self.all_preds[i]] for i in range(len(self.all_preds))}
            references = {'image'+str(i): [self.all_gts[i]] for i in range(len(self.all_gts))}

            bleu_score = self.bleu.compute_score(hypotheses, references)[0][3]
            rouge_score = self.rouge.compute_score(references, hypotheses)[0]
            cider_score = self.cider.compute_score(references, hypotheses)[0]

            self.log("bleu", bleu_score, on_step=False, on_epoch=True, prog_bar=True)
            self.log("rouge", rouge_score, on_step=False, on_epoch=True, prog_bar=True)
            self.log("cider", cider_score, on_step=False, on_epoch=True, prog_bar=True)

            self.all_preds = []
            self.all_gts = []
            self.pls = []
            self.starts = []
            self.ends = []
            self.names = []


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
