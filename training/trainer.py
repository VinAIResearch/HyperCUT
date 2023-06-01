import logging
import os
import os.path as osp

import torch
import utils.training_utils as training_utils
from data import get_datasets, get_datasets_b
from models import get_model
from resources.consts import (
    CHECKPOINT_PATH,
    MODEL_SAVE_PATH_FORMAT,
    TRAIN_SAMPLE_SAVE_PATH,
    TRAINING_STATE_SAVE_PATH_FORMAT,
    VAL_SAMPLE_SAVE_PATH,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.common_utils import is_scalar


class Trainer:
    def __init__(self, args):
        self.args = args

        if args.exp_name is None:
            self.exp_name = args.dataset_name + "_" + args.model_name
        else:
            timestamp = training_utils.get_timestamp()
            self.exp_name = args.exp_name + "_" + timestamp

        self.device = torch.device("cuda")
        self.initialize_training_folders(args.load_epoch == -1)
        self.use_parallel = args.use_parallel

        if args.dataset_name in ["b-aist++"]:
            train_set, val_set, tral_set, _ = get_datasets_b(**vars(args))
        else:
            train_set, val_set, tral_set, _ = get_datasets(**vars(args))

        val_debug_save_path = VAL_SAMPLE_SAVE_PATH.format(self.exp_name)
        train_debug_save_path = TRAIN_SAMPLE_SAVE_PATH.format(self.exp_name)

        val_set.debug(val_debug_save_path)
        tral_set.debug(train_debug_save_path)

        self.epoch_length = len(train_set) // args.batch_size

        self.train_dataloader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        self.val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
        self.tral_dataloader = DataLoader(tral_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

        self.model = get_model(args.model_name, args.model_kwargs).to(self.device)

        self.optimizer = training_utils.get_optimizer(self.model, args)
        self.scheduler = training_utils.get_scheduler(self.optimizer, args)

        self.num_epochs = args.num_epochs

        self.img_log_freq = args.img_log_freq

        self.load(args.load_epoch)

        if args.use_parallel:
            self.model = torch.nn.DataParallel(self.model)
        self.logger.info(self.exp_name + "\n")

    def initialize_training_folders(self, from_scratch):
        exp_path = osp.join(CHECKPOINT_PATH, self.exp_name)
        if from_scratch and osp.isdir(exp_path):
            os.rename(exp_path, osp.join(osp.dirname(exp_path), self.exp_name))

        exp_path = osp.join(CHECKPOINT_PATH, self.exp_name)

        if from_scratch:
            os.makedirs(exp_path, exist_ok=True)
            os.makedirs(osp.join(exp_path, "models"), exist_ok=True)
            os.makedirs(osp.join(exp_path, "training_states"), exist_ok=True)

        self.writer = SummaryWriter(log_dir=exp_path)

        training_utils.setup_logger("base", exp_path, screen=True, tofile=True)
        self.logger = logging.getLogger("base")

    def training_loop(self):
        self.validation()
        while self.current_epoch < self.num_epochs:
            self.train_one_epoch()
            self.scheduler.step()
            self.current_epoch += 1

            if self.current_epoch % self.args.val_epoch == 0:
                self.save()
                self.validation()

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def train_one_epoch(self):
        model = self.model.module if self.use_parallel else self.model

        self.model.train()

        self.logger.info(f"Running epoch {self.current_epoch}/{self.num_epochs}")
        for batch_idx, data_point in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            out = model(data_point)

            loss = model.calc_loss(out, data_point, "train")["total"]

            loss.backward()
            self.optimizer.step()

            if batch_idx % self.args.train_step == 0:
                self.logger.info(
                    f"step: {batch_idx}/{self.epoch_length} lr: {self.get_lr():.2e}, loss: {loss.item():.4f}"
                )
            self.current_iter += 1

            self.writer.add_scalar("loss", loss.item(), self.current_iter)

    def save(self):
        """
        Save training state during training,
        which will be used for resuming
        """
        self.logger.info(f"Saving epoch #{self.current_epoch}...")

        state = {
            "epoch": self.current_epoch,
            "scheduler": self.scheduler.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "current_iter": self.current_iter,
        }

        save_name = self.current_epoch

        training_state_save_path = TRAINING_STATE_SAVE_PATH_FORMAT.format(self.exp_name, save_name)
        save_path = MODEL_SAVE_PATH_FORMAT.format(self.exp_name, save_name)

        torch.save(state, training_state_save_path)

        if self.args.use_parallel:
            torch.save(self.model.module.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)

    def load(self, load_epoch):
        if load_epoch == -1:
            if self.args.pretrained_path is not None:
                self.logger.info(f"Loading pretrained models from {self.args.pretrained_path}")
                self.model.load_state_dict(torch.load(self.args.pretrained_path))
            else:
                self.logger.info("Training model from scratch...")
            self.current_epoch = 0
            self.current_iter = 0
            return

        self.logger.info(f"Resuming training from epoch #{load_epoch}")

        state_path = TRAINING_STATE_SAVE_PATH_FORMAT.format(self.exp_name, load_epoch)
        model_path = MODEL_SAVE_PATH_FORMAT.format(self.exp_name, load_epoch)

        if not osp.isfile(state_path):
            raise ValueError(f"Training state for epoch #{load_epoch} not found!")
        if not osp.isfile(model_path):
            raise ValueError(f"Weights for epoch #{load_epoch} not found!")

        state = torch.load(state_path)

        resume_optimizer = state["optimizer"]
        resume_scheduler = state["scheduler"]
        self.current_epoch = state["epoch"]
        self.current_iter = state["current_iter"]

        self.model.load_state_dict(torch.load(model_path))

        self.current_epoch = load_epoch + 1

        self.optimizer.load_state_dict(resume_optimizer)
        self.scheduler.load_state_dict(resume_scheduler)

    def validation(self):
        if self.val_dataloader is None:
            self.logger.warning("No validation dataloader was given. Skipping validation...")
            return

        self.logger.info("Evaluating metric")

        model = self.model.module if self.use_parallel else self.model

        train_save_root = TRAIN_SAMPLE_SAVE_PATH.format(self.exp_name, self.current_epoch)
        val_save_root = VAL_SAMPLE_SAVE_PATH.format(self.exp_name, self.current_epoch)

        train_scores = model.validation(
            self.tral_dataloader, self.args.display_step, limit=500, save_root=train_save_root
        )
        val_scores = model.validation(self.val_dataloader, self.args.display_step, limit=None, save_root=val_save_root)

        for k, v in train_scores.items():
            self.logger.info(f"{k} on train subset: {v}")
            if is_scalar(v):
                self.writer.add_scalar("train_{k}", v, self.current_epoch)
            else:
                self.writer.add_scalar("train_{k}", v.mean(), self.current_epoch)

        for k, v in val_scores.items():
            self.logger.info(f"{k} on val subset: {v}")
            if is_scalar(v):
                self.writer.add_scalar("val_{k}", v, self.current_epoch)
            else:
                self.writer.add_scalar("val_{k}", v.mean(), self.current_epoch)
