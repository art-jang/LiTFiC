from typing import Any, Dict, Optional, Callable
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import json


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
        ret_data_size: int = 500,
        train_data_fraction: int = 1.0,
        val_episode_ind_path: Optional[str] = None,
        test_setname: str = "val",
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_ret: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.eval_data_size = eval_data_size
        self.train_data_fraction = train_data_fraction
        self.ret_data_size = ret_data_size
        self.test_setname = test_setname

        self.val_episode_ind = None
        self.val_episode_ind_path = val_episode_ind_path
        with open(val_episode_ind_path, 'r') as f:
            self.val_episode_ind = json.load(f)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            self.batch_size_per_device = self.hparams.batch_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            if self.hparams.dataset == 'bobsl':
                from ..data.components.sentence import Sentences
                self.data_train = Sentences(**self.hparams.dataset_config, setname="train")
                self.data_val = Sentences(**self.hparams.dataset_config, setname="man_val")
                self.data_ret = Subset(self.data_val,
                                    torch.randperm(
                                        len(self.data_val)
                                    )[:self.ret_data_size],
                                )
                self.data_test = Sentences(**self.hparams.dataset_config, setname=self.test_setname)
                # self.data_val = Subset(Sentences(**self.hparams.dataset_config, setname="val"), torch.randperm(
                # len(self.data_val))[:self.eval_data_size])

                self.data_train = Subset(
                                    self.data_train,
                                    torch.randperm(
                                        len(self.data_train)
                                    )[:int(len(self.data_train) * self.train_data_fraction)],
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

        if self.trainer.num_devices>1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.data_train, shuffle=True)
        
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            sampler=sampler if self.trainer.num_devices > 1 else None
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        
        rank = self.trainer.global_rank
        world_size = self.trainer.world_size

        if self.val_episode_ind_path.split("/")[-1] != "val_how2sign_indices.json":
            episode_ind = json.load(open("/lustre/fswork/projects/rech/vvh/upk96qz/datasets/bobsl/val_start_indices_manual.json", "rb"))
        else:
            episode_ind = self.val_episode_ind

        total_episodes = len(episode_ind["idx"])
        episodes_per_gpu = total_episodes // world_size
        start_index = episodes_per_gpu * rank
        end_index = start_index + episodes_per_gpu if rank != world_size - 1 else total_episodes

        if len(self.data_val) < episode_ind["idx"][-1]:
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

        dataset_start_idx = episode_ind["idx"][start_index]

        if self.eval_data_size > 0 and not self.trainer.testing:
            dataset_end_idx = dataset_start_idx + (self.eval_data_size // world_size)
        else:
            dataset_end_idx = episode_ind["idx"][end_index] if end_index < total_episodes else len(self.data_val)


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

        episode_ind = self.val_episode_ind
        if self.test_setname == "man_val":
            episode_ind = json.load(open("/lustre/fswork/projects/rech/vvh/upk96qz/datasets/bobsl/val_start_indices_manual.json", "rb"))
        
        elif self.test_setname == "public_test":
            episode_ind = json.load(open("/lustre/fswork/projects/rech/vvh/upk96qz/datasets/bobsl/test_start_indices.json", "rb"))
        
        elif self.test_setname == "train":
            episode_ind = json.load(open("/lustre/fswork/projects/rech/vvh/upk96qz/datasets/bobsl/train_start_indices.json", "rb"))

        total_episodes = len(episode_ind["idx"])
        episodes_per_gpu = total_episodes // world_size
        start_index = episodes_per_gpu * rank
        end_index = start_index + episodes_per_gpu if rank != world_size - 1 else total_episodes

        # Get the actual dataset indices

        dataset_start_idx = episode_ind["idx"][start_index]


        dataset_end_idx = episode_ind["idx"][end_index] if end_index < total_episodes else len(self.data_test)


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

    def ret_dataloader(self) -> DataLoader[Any]:
        """Create and return the retrieval dataloader.

        :return: The retrieval dataloader.
        """
        return DataLoader(
            dataset=self.data_ret,
            batch_size=self.batch_size_per_device,
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
