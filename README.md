# RecAdam
Re-implementation of [Sanyuan-Chen/RecAdam](https://github.com/Sanyuan-Chen/RecAdam).




## Features
* Simpler interfaces with less tuning parameters.
* Compatible with deepspeed.




## Installation
```console
pip install git+https://github.com/MorishT/rec-adam.git@honoka-dev
```




## How to use


### Initializing the optimizer using the factory method
```python
from rec_adam import build_rec_adam_optimizer

your_model = (...)
optimizer = build_rec_adam_optimizer(
    your_model,
    learning_rate=1e-3,
    weight_decay=0.0,
    target_task_weight=1.0,
    fisher_coef=3000,
)
```
The loss will become something like `loss = loss_original + target_task_weight * (fisher_coef * l2_term)`.
Note that `target_task_weight` works differently from the original implementation,
where the loss will become somthing like `loss = (1 - target_task_weight) * loss_original + (...)`.  

`fisher_coef` should be tuned for each model, especially when you vary model size.
The default value of 3000 is the best for llama3-8B.


### Using the optimizer via huggingface's Trainer interface
```python
from rec_adam import RecAdamTrainer
trainer = RecAdamTrainer(Trainer):
    training_args,
    rec_adam_target_task_weight=1.0,
    rec_adam_fisher_coef=3000,
):
```


### (Not recommended) Initializing the optimizer directly via its constructor
We do not recommend to initialize the optimizer directly using its constructor,
as setting the suitable arguments is complex.

Although, you can do it like this:
```python
from rec_adam import RecAdam
optimizer = RecAdam(...)
```
