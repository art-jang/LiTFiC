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

        # for evaluation
        self.bleu = Bleu(4)
        self.rouge = Rouge()
        self.cider = Cider()

        self.all_preds =[]
        self.all_gts = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

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
        outputs, labels = self.forward(batch)
        loss = self.criterion(outputs, labels)
        if not self.training:
            preds = outputs.logits.argmax(dim=-1)
            for pred, gt in zip(preds, labels):
                valid_len = gt[gt != -100].size(0)
                decoded_pred = self.net.language_decoder.tokenizer.decode(pred[-valid_len:-1], skip_special_tokens=True)
                decoded_gt = self.net.language_decoder.tokenizer.decode(gt[-valid_len:-1], skip_special_tokens=True)
                self.all_preds.append(decoded_pred)
                self.all_gts.append(decoded_gt)
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
        "Lightning hook that is called when a training epoch ends."
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
        if self.global_step > 0:
            hypotheses = {'image'+str(i): [self.all_preds[i]] for i in range(len(self.all_preds))}
            references = {'image'+str(i): [self.all_gts[i]] for i in range(len(self.all_gts))}

            bleu_score = self.bleu.compute_score(hypotheses, references)[0][3]
            rouge_score = self.rouge.compute_score(references, hypotheses)[0]
            cider_score = self.cider.compute_score(references, hypotheses)[0]

            self.log("val/bleu", bleu_score, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/rouge", rouge_score, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/cider", cider_score, on_step=False, on_epoch=True, prog_bar=True)

            self.all_preds = []
            self.all_gts = []

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
