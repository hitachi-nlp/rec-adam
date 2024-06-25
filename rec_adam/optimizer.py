import logging
import math
from typing import Optional

import numpy as np
import torch
from torch.optim import Optimizer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS



logger = logging.getLogger(__name__)


class RecAdam(Optimizer):
    """ Re-Implementation of RecAdam optimizer, a variant of Adam optimizer. 

    Notes:
        * We recommend to initialize this optimizer throught "build()" rather than directly calling the constructor.
          as the initialization is a bit complicated.
        * "fisher_coef" should be tuned model-wisely, especially when you vary model size.
          The default value of 3000 is the best for llama3-8B.
    """

    def __init__(self,

                 params,

                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-6,
                 weight_decay=0.0,

                 target_task_weight=1.0,
                 fisher_coef=3000,

                 correct_bias=True,
                 regularization='l2',
                 anneal_type='sigmoid',
                 anneal_t0=0,
                 anneal_tau=0):

        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'correct_bias': correct_bias,
            'regularization': regularization,
            'anneal_type': anneal_type,
            'anneal_t0': anneal_t0,
            'anneal_tau': anneal_tau,
            'target_task_weight': target_task_weight,
            'fisher_coef': fisher_coef,
        }
        if target_task_weight < 0.0 or target_task_weight > 1.0:
            raise ValueError('Invalid target_task_weight value: %f' % target_task_weight)
        super().__init__(params, defaults)
        self._pretrain_params = {}

    def step(self, closure=None, only_pretrain_task=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:

            if self._pretrain_params is None:
                logger.info('initializing initial_params...')
                for p in group["params"]:
                    if p.shape is self._pretrain_params:  # p.shape is duplicated for some parameters.
                        raise Exception('Unexpected.')
                    self._pretrain_params[p.shape] = p.detach().clone()

            # for i, (p, pp) in enumerate(zip(group["params"], self.pretrain_params)):
            for i, p in enumerate(group["params"]):

                param_shape = tuple(p.shape)
                if param_shape not in self._pretrain_params:
                    # logger.warning('Initializing pre-trained parameters which was not found at the first step() call.'
                    #                ' This is a heuristic and may not work properly.')
                    pp = p.detach().clone()
                    self._pretrain_params[param_shape] = pp

                """
                Guess the corresponding pre-trained parameters from the current parameters' shape.
                We do this because deepspeed send us chunks of parameters asynchronusly.
                But, this trick may not work if the two or more parameters have the same shape....
                """
                pp = self._pretrain_params[param_shape]

                if p.data.shape != pp.data.shape:
                    raise Exception('Unexpected, may be bug')

                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                if group['target_task_weight'] >= 0.0:
                    target_task_factor = self._get_target_task_factor(
                        state["step"],
                        anneal_type=group['anneal_type'],
                        anneal_t0=group['anneal_t0'],
                        anneal_tau=group['anneal_tau'],
                        target_task_weight=group['target_task_weight']
                    )

                    p.data.addcdiv_(- step_size * target_task_factor, exp_avg, denom)  # target task loss

                    regularization = group['regularization']
                    lr = group["lr"]
                    # pretrain_task_factor = 1.0 - target_task_factor
                    pretrain_task_factor = 1.0
                    if regularization == 'l1':
                        with torch.no_grad():
                            p.data = torch.sign(p.data) * torch.clamp(p.data.abs() - lr * pretrain_task_factor * group["fisher_coef"], min=0)
                    elif regularization == 'l2':
                        # Add the quadratic penalty to simulate the pretraining tasks
                        p.data.add_(- lr * pretrain_task_factor * group["fisher_coef"], p.data - pp.data)
                    else:
                        raise ValueError('Invalid regularization type: %s' % regularization)

                    if state["step"] % 10 == 0:
                        logger.info(
                            ('  regularization: %s'
                             '  anneal_type: %s'
                             '  anneal_tau: %f'
                             '  anneal_t0: %d'
                             '  target_task_weight: %f'
                             '  fisher_coef: %f'
                             '  step: %d'
                             '  target_task_factor: %f'
                             '  pretrain_task_factor: %f'),
                            regularization,
                            group['anneal_type'],
                            group['anneal_tau'],
                            group['anneal_t0'],
                            group['target_task_weight'],
                            group["fisher_coef"],
                            state["step"],
                            target_task_factor,
                            pretrain_task_factor,
                        )


                else:
                    p.data.addcdiv_(-step_size, exp_avg, denom)

                if group["weight_decay"] > 0.0:
                    p.data.add_(-group["lr"] * group["weight_decay"], p.data)

        return loss

    def _get_target_task_factor(self,
                               step: int,
                               anneal_type: Optional[str] = None,
                               anneal_t0: Optional[float] = None,
                               anneal_tau: Optional[float] = None,
                               target_task_weight: Optional[float] = None) -> float:
        target_task_weight = target_task_weight or self.defaults['target_task_weight']
        anneal_lambda = self._get_anneal_lambda(step,
                                               anneal_type=anneal_type,
                                               anneal_t0=anneal_t0,
                                               anneal_tau=anneal_tau)
        return target_task_weight * anneal_lambda

    def _get_anneal_lambda(self,
                           step: int,
                           anneal_type: Optional[str] = None,
                           anneal_t0: Optional[float] = None,
                           anneal_tau: Optional[float] = None) -> float:
        anneal_type = anneal_type or self.defaults['anneal_type']
        anneal_t0 = anneal_t0 or self.defaults['anneal_t0']
        anneal_tau = anneal_tau or self.defaults['anneal_tau']

        if anneal_type == 'sigmoid':
            return float(1 / (1 + np.exp(- (step - anneal_t0) / (anneal_tau + 0.0001) )))
        elif anneal_type == 'linear':
            return min(1, step / (anneal_t0 + 0.0001))
        elif anneal_type == 'constant':
            return 1.0
        else:
            ValueError('Invalid anneal_type: %s' % anneal_type)


def build_rec_adam_optimizer(
    model,

    learning_rate=1e-3,
    adam_epsilon=1e-6,
    adam_beta1=0.9,
    adam_beta2=0.999,
    weight_decay=0.0,

    target_task_weight=1.0,
    fisher_coef=3000,
):
    anneal_schedule = 'immediately_from_beginning'
    anneal_type = 'sigmoid'
    anneal_t0: Optional[int] = None
    anneal_tau: Optional[int] = None
    max_steps: Optional[int] = None


    # taken from trainer.py
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    if anneal_t0 is not None:
        if anneal_tau is None:
            raise ValueError('anneal_tau must be specified if anneal_t0 is specified')
    elif anneal_tau is not None:
        if anneal_t0 is None:
            raise ValueError('anneal_t0 must be specified if anneal_tau is specified')
    else:
        if anneal_schedule == 'immediately_from_beginning':
            _anneal_t0 = 0
            _anneal_tau = 0

        elif anneal_schedule == 'gradually_from_middle':
            if max_steps is None:
                raise ValueError('max_steps must be specified for anneal_schedule')
            _anneal_t0 = int(max_steps / 2)
            _anneal_tau = int(_anneal_t0 / 3)  # factor will be 0.95 at steps = 2 x t0

        else:
            raise ValueError('Invalid anneal_schedule: %s' % anneal_schedule)

        anneal_t0 = _anneal_t0
        anneal_tau = _anneal_tau

        logger.info('annealing schedule parameters: anneal_t0=%d, anneal_tau=%d', anneal_t0, anneal_tau)

    def should_decay_param(n):
        return n in decay_parameters

    def is_original_arch_param(n):
        # return model_args.model_type in n
        return True   # TODO: implement logic to judge whether the parameter is from the original architecture or added one.

    # initial_parameters = [(n, p.detach()) for n, p in opt_model.named_parameters() if p.requires_grad]
    update_parameter = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    # we do not set initial_params here, as deepspeed may alter the shape of "params"
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in update_parameter if should_decay_param(n) and is_original_arch_param(n)],
            "weight_decay": weight_decay,
            "target_task_weight": target_task_weight,
        },
        {
            "params": [p for n, p in update_parameter if should_decay_param(n) and not is_original_arch_param(n)],
            "weight_decay": weight_decay,
            "target_task_weight": -1.0,
        },
        {
            "params": [p for n, p in update_parameter if not should_decay_param(n) and is_original_arch_param(n)],
            "weight_decay": 0.0,
            "target_task_weight": target_task_weight,
        },
        {
            "params": [p for n, p in update_parameter if not should_decay_param(n) and not is_original_arch_param(n)],
            "weight_decay": 0.0,
            "target_task_weight": -1.0,
        }
    ]

    return RecAdam(
        optimizer_grouped_parameters,

        lr=learning_rate,
        eps=adam_epsilon,
        betas=(adam_beta1, adam_beta2),
        weight_decay=weight_decay,
        target_task_weight=target_task_weight,
        fisher_coef=fisher_coef,

        regularization='l2',
        anneal_type=anneal_type,
        anneal_tau=anneal_tau,
        anneal_t0=anneal_t0,
    )
