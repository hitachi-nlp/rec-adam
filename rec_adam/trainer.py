from typing import Optional

import torch
from transformers import Trainer
from transformers.utils import is_sagemaker_mp_enabled
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

from .optimizer import build_rec_adam_optimizer


class RecAdamTrainer(Trainer):

    def __init__(
        self,
        *args,
        rec_adam_target_task_weight=1.0,
        rec_adam_fisher_coef=3000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rec_adam_target_task_weight = rec_adam_target_task_weight
        self.rec_adam_fisher_coef = rec_adam_fisher_coef

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = build_optimizer_from_trainer(self,
                                                          self.rec_adam_target_task_weight,
                                                          self.rec_adam_fisher_coef)
        return self.optimizer


def build_optimizer_from_trainer(trainer: Trainer, rec_adam_target_task_weight=1.0, rec_adam_fisher_coef=3000):
    optimizer = build_rec_adam_optimizer(
        trainer.model_wrapped if is_sagemaker_mp_enabled() else trainer.model,

        learning_rate=trainer.args.learning_rate,
        adam_epsilon=trainer.args.adam_epsilon,
        adam_beta1=trainer.args.adam_beta1,
        adam_beta2=trainer.args.adam_beta2,
        weight_decay=trainer.args.weight_decay,

        target_task_weight=rec_adam_target_task_weight,
        fisher_coef=rec_adam_fisher_coef,
    )
    if is_sagemaker_mp_enabled():
        optimizer = smp.DistributedOptimizer(optimizer)
    return optimizer
