#!/bin/bash
import logging
import math
import numpy as np
from typing import Optional

import torch
from torch.optim import Optimizer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS



logger = logging.getLogger(__name__)


class RecAdam(Optimizer):
    """ Implementation of RecAdam optimizer, a variant of Adam optimizer.

    Just adding the square of the weights to the loss function is *not*
    the correct way of using L2 regularization/weight decay with Adam,
    since that will interact with the m and v parameters in strange ways.
    
    Instead we want to decay the weights in a manner that doesn't interact
    with the m/v parameters. This is equivalent to adding the square
    of the weights to the loss with plain (non-momentum) SGD.
    Add weight decay at the end (fixed version)

    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
        anneal_type (str): a hyperparam for the anneal function, decide the function of the curve. Default 'sigmoid'.
        anneal_tau (float): a hyperparam for the anneal function, decide the slop of the curve. Choice: [0.05, 0.1, 0.2, 0.5, 1]
        anneal_t0 (float): a hyperparam for the anneal function, decide the middle point of the curve. Choice: [100, 250, 500, 1000]
        target_task_weight (float): a hyperparam for the anneal function, decide the scale of the curve. Default 1.0.
        fisher_coef (float): the coefficient of the quadratic penalty. Default 5000.0.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-6,
                 weight_decay=0.0,
                 correct_bias=True,
                 target_task_weight=1.0,
                 regularization='l2',
                 anneal_type='sigmoid',
                 anneal_tau=0,
                 anneal_t0=0,
                 fisher_coef=5000.0):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'correct_bias': correct_bias,
            'regularization': regularization,
            'anneal_type': anneal_type,
            'anneal_tau': anneal_tau,
            'anneal_t0': anneal_t0,
            'target_task_weight': target_task_weight,
            'fisher_coef': fisher_coef,
        }
        if target_task_weight < 0.0 or target_task_weight > 1.0:
            raise ValueError('Invalid target_task_weight value: %f' % target_task_weight)
        super().__init__(params, defaults)

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

            if "initial_params" not in group:   # the first 
                logger.info('initializing initial_params...')
                group["initial_params"] = [p.detach().clone() for p in group["params"]]

            for p, pp in zip(group["params"], group["initial_params"]):
                if p.data.shape != pp.data.shape:
                    import pudb; pudb.set_trace()

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
                    target_task_factor = self.get_target_task_factor(
                        state["step"],
                        anneal_type=group['anneal_type'],
                        anneal_tau=group['anneal_tau'],
                        anneal_t0=group['anneal_t0'],
                        target_task_weight=group['target_task_weight']
                    )

                    p.data.addcdiv_(- step_size * target_task_factor, exp_avg, denom)  # target task loss

                    regularization = group['regularization']
                    lr = group["lr"]
                    pretrain_task_factor = 1.0 - target_task_factor
                    if regularization == 'l1':
                        with torch.no_grad():
                            p.data = torch.sign(p.data) * torch.clamp(p.data.abs() - lr * pretrain_task_factor * group["fisher_coef"], min=0)
                    elif regularization == 'l2':
                        # Add the quadratic penalty to simulate the pretraining tasks
                        p.data.add_(- lr * pretrain_task_factor * group["fisher_coef"], p.data - pp.data)
                    else:
                        raise ValueError('Invalid regularization type: %s' % regularization)

                    logger.info(
                        '  regularization: %s'
                        '  anneal_type: %s'
                        '  anneal_t0: %d'
                        '  anneal_tau: %f'
                        '  target_task_weight: %f'
                        '  step: %d'
                        '  target_task_factor: %f'
                        '  pretrain_task_factor: %f',
                        regularization,
                        group['anneal_type'],
                        group['anneal_t0'],
                        group['anneal_tau'],
                        group['target_task_weight'],
                        state["step"],
                        target_task_factor,
                        pretrain_task_factor
                    )


                else:
                    p.data.addcdiv_(-step_size, exp_avg, denom)

                if group["weight_decay"] > 0.0:
                    p.data.add_(-group["lr"] * group["weight_decay"], p.data)

        return loss

    def get_target_task_factor(self,
                               step: int,
                               anneal_type: Optional[str] = None,
                               anneal_t0: Optional[float] = None,
                               anneal_tau: Optional[float] = None,
                               target_task_weight: Optional[float] = None) -> float:
        target_task_weight = target_task_weight or self.defaults['target_task_weight']
        anneal_lambda = self.get_anneal_lambda(step,
                                               anneal_type=anneal_type,
                                               anneal_t0=anneal_t0,
                                               anneal_tau=anneal_tau)
        return target_task_weight * anneal_lambda

    def get_anneal_lambda(self,
                          step: int,
                          anneal_type: Optional[str] = None,
                          anneal_t0: Optional[float] = None,
                          anneal_tau: Optional[float] = None) -> float:
        anneal_type = anneal_type or self.defaults['anneal_type']
        anneal_tau = anneal_tau or self.defaults['anneal_tau']
        anneal_t0 = anneal_t0 or self.defaults['anneal_t0']

        if anneal_type == 'sigmoid':
            return float(1 / (1 + np.exp(- (step - anneal_t0) / (anneal_tau + 0.0001) )))
        elif anneal_type == 'linear':
            return min(1, step / (anneal_t0 + 0.0001))
        elif anneal_type == 'constant':
            return 1.0
        else:
            ValueError


def build(args,
          opt_model,
          target_task_weight=1.0,
          regularization='l2',
          anneal_type='sigmoid',
          anneal_tau=0,
          anneal_t0=0,
          fisher_coef=5000.0):

    # taken from trainer.py
    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    def should_decay_param(n):
        return n in decay_parameters

    def is_original_arch_param(n):
        # return model_args.model_type in n
        return True   # TODO: implement logic to judge whether the parameter is from the original architecture or added one.

    # initial_parameters = [(n, p.detach()) for n, p in opt_model.named_parameters() if p.requires_grad]
    update_parameter = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad]

    # we do not set initial_params here, as deepspeed may alter the shape of "params"
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in update_parameter if should_decay_param(n) and is_original_arch_param(n)],
            "weight_decay": args.weight_decay,
            "target_task_weight": target_task_weight,
            # "initial_params": [p_p for p_n, p_p in initial_parameters if should_decay_param(p_n) and is_original_arch_param(p_n)]
        },
        {
            "params": [p for n, p in update_parameter if should_decay_param(n) and not is_original_arch_param(n)],
            "weight_decay": args.weight_decay,
            "target_task_weight": -1.0,
            # "initial_params": [p_p for p_n, p_p in initial_parameters if should_decay_param(p_n) and not is_original_arch_param(p_n)]
        },
        {
            "params": [p for n, p in update_parameter if not should_decay_param(n) and is_original_arch_param(n)],
            "weight_decay": 0.0,
            "target_task_weight": target_task_weight,
            # "initial_params": [p_p for p_n, p_p in initial_parameters if not should_decay_param(p_n) and is_original_arch_param(p_n)]
        },
        {
            "params": [p for n, p in update_parameter if not should_decay_param(n) and not is_original_arch_param(n)],
            "weight_decay": 0.0,
            "target_task_weight": -1.0,
            # "initial_params": [p_p for p_n, p_p in initial_parameters if not should_decay_param(p_n) and not is_original_arch_param(p_n)]
        }
    ]

    if anneal_t0 is None:
        if args.max_steps is None:
            raise ValueError('rec_adam_anneal_t0 must be specified if max_steps is not specified')

        logger.info(f'rec_adam_anneal_t0 is not specified, set to the half of the max_steps: {args.max_steps / 2}')
        anneal_t0 = args.max_steps / 2

    if anneal_tau is None:
        logger.info(f'rec_adam_anneal_tau is not specified, set to the 1/3 of the rec_adam_anneal_t0: {anneal_t0 / 3}')
        anneal_tau = anneal_t0 / 3  # factor will be 0.95 at steps = 2 x t0

    return RecAdam(
        optimizer_grouped_parameters,

        lr=args.learning_rate,
        eps=args.adam_epsilon,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,

        regularization=regularization,
        anneal_type=anneal_type,
        anneal_tau=anneal_tau,
        anneal_t0=anneal_t0,

        fisher_coef=fisher_coef,
    )
