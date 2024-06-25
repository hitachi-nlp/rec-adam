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
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            self.optimizer = build_rec_adam_optimizer(
                opt_model,

                learning_rate=self.args.learning_rate,
                adam_epsilon=self.args.adam_epsilon,
                adam_beta1=self.args.adam_beta1,
                adam_beta2=self.args.adam_beta2,
                weight_decay=self.args.weight_decay,

                target_task_weight=self.rec_adam_target_task_weight,
                fisher_coef=self.rec_adam_fisher_coef,
            )
            
        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)
        return self.optimizer
