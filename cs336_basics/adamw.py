from collections.abc import Callable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError(f"Invalid betas: {betas}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            betas = group["betas"] # Get beta1 and beta2
            eps = group["eps"] # Get epsilon
            weight_decay = group["weight_decay"] # Get lambda

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                # Get first moment vector, or zeros
                if "m" not in state:
                    m = torch.zeros_like(p.data)
                else:
                    m = state.get("m")
                # Get second moment vector, or zeros
                if "v" not in state:
                    v = torch.zeros_like(p.data)
                else:
                    v = state.get("v")
                grad = p.grad.data # Get the gradient of loss with respect to p.
                lr_t = lr * math.sqrt(1-betas[1]**t) / (1 - betas[0]**t) # Compute adjusted alpha for iteration t
                p.data -= lr * weight_decay * p.data # Apply weight decay
                m = betas[0] * m + (1 - betas[0]) * grad # Update first moment estimate
                v = betas[1]*v + (1 - betas[1]) * grad**2 # Update second moment estimate
                p.data -= lr_t * m / (torch.sqrt(v) + eps) # Apply moment-adjusted weight updates
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss
                
                
