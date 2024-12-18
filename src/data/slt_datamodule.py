from typing import Any, Dict, Optional, Callable
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from ..data.components.sentence import Sentences

import torch
import json


class SLTDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: str,
        val_episode_ind_path,
        test_episode_ind_path,
        dataset_config: Optional[Dict[str, Any]] = None,
        extra_dataset_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        collate_fn: Optional[str] = None,
        eval_data_size: int = 1000,
        ret_data_size: int = 500,
        train_data_fraction: int = 1.0,
        test_setname: str = "val",
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.eval_data_size = eval_data_size
        self.train_data_fraction = train_data_fraction
        self.test_setname = test_setname

        self.dataset = dataset

        with open(val_episode_ind_path, 'r') as f:
            self.val_episode_ind = json.load(f)

        with open(test_episode_ind_path, 'r') as f:
            self.test_episode_ind = json.load(f)

        if self.dataset == "how2sign":
            with open(test_episode_ind_path, 'r') as f:
                self.val_episode_ind = json.load(f)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            self.batch_size_per_device = self.hparams.batch_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            
            ########### extra training data ############

            self.data_train = Sentences(**self.hparams.dataset_config, setname="train")
            if self.dataset == "how2sign":
                self.data_val = Sentences(**self.hparams.dataset_config, setname="test")
            else:
                self.data_val = Sentences(**self.hparams.dataset_config, setname="val")

            self.data_test = Sentences(**self.hparams.dataset_config, setname=self.test_setname)
            # if self.hparams.extra_dataset_config is not None:
            #     extra_data_train = Sentences(**self.hparams.extra_dataset_config, setname="train")
            #     self.data_train = torch.utils.data.ConcatDataset([self.data_train, extra_data_train])

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

        if self.trainer.num_devices>1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.data_train, shuffle=True)
        else:
            sampler = None
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            sampler=sampler,
            shuffle=(sampler is None)
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        rank = self.trainer.global_rank
        world_size = self.trainer.world_size

        total_episodes = len(self.val_episode_ind["idx"])
        episodes_per_gpu = total_episodes // world_size
        start_index = episodes_per_gpu * rank
        end_index = start_index + episodes_per_gpu if rank != world_size - 1 else total_episodes

        if len(self.data_val) < self.val_episode_ind["idx"][-1]:
            return DataLoader(
                dataset=self.data_val,
                batch_size=1,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                collate_fn=self.collate_fn,
                sampler=None
            )

        # Get the actual dataset indices
        dataset_start_idx = self.val_episode_ind["idx"][start_index]

        if self.eval_data_size > 0 and not self.trainer.testing:
            dataset_end_idx = dataset_start_idx + (self.eval_data_size // world_size)
        else:
            dataset_end_idx = self.val_episode_ind["idx"][end_index] if end_index < total_episodes else len(self.data_val)


        dataset = Subset(self.data_val, range(dataset_start_idx, dataset_end_idx))

        return DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
            sampler=None
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        rank = self.trainer.global_rank
        world_size = self.trainer.world_size

        total_episodes = len(self.test_episode_ind["idx"])
        episodes_per_gpu = total_episodes // world_size
        start_index = episodes_per_gpu * rank
        end_index = start_index + episodes_per_gpu if rank != world_size - 1 else total_episodes

        # Get the actual dataset indices

        dataset_start_idx = self.test_episode_ind["idx"][start_index]


        dataset_end_idx = self.test_episode_ind["idx"][end_index] if end_index < total_episodes else len(self.data_test)


        dataset = Subset(self.data_test, range(dataset_start_idx, dataset_end_idx))

        return DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
            sampler=None
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
    _ = SLTDataModule()
