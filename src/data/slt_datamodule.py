from typing import Any, Dict, Optional, Callable
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
import torch


class SLTDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: str = "mnist",
        dataset_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        collate_fn: Optional[str] = None,
        eval_data_size: int = 1000,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_eval: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            if self.hparams.dataset == 'bobsl':
                from ..data.components.sentence import Sentences
                self.data_train = Sentences(**self.hparams.dataset_config, setname="train")
                self.data_val = Sentences(**self.hparams.dataset_config, setname="val")
                self.data_eval = Subset(
                                    self.data_val,
                                    torch.randperm(
                                        len(self.data_val)
                                    )[:self.hparams.eval_data_size],
                                )

        if self.hparams.collate_fn is not None:
            from importlib import import_module
            module_, func = self.hparams.collate_fn.rsplit(".", maxsplit=1)
            m = import_module(module_)
            self.collate_fn = getattr(m, func)
        else:
            self.collate_fn = None

        a, b = divmod(len(self.data_train), self.hparams.batch_size)
        self.max_steps = a + int(b > 0)
            

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def eval_dataloader(self) -> DataLoader[Any]:
        """Create and return the evaluation dataloader.

        :return: The evaluation dataloader.
        """
        return DataLoader(
            dataset=self.data_eval,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = MNISTDataModule()
