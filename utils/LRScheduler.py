from torch.optim.lr_scheduler import LambdaLR
import math

def get_cosine_schedule_with_warmup(optimizer,warmup_steps:int,max_steps:int,min_lr_ratio:float = 0.1):

    def lr_lambda(step: int) -> float:

        if step < warmup_steps:

            return step / max(1, warmup_steps)

        if step > max_steps:

            return min_lr_ratio

        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)