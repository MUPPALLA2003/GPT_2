from typing import Optional
import wandb

class WandbLogger:

    def __init__(self,project:str,run_name:Optional[str] = None,config:Optional[dict] = None,enabled:bool = True,is_main_process: bool = True):

        self.enabled = enabled and is_main_process
        self.run = None

        if self.enabled:

            self.run = wandb.init(project=project,name=run_name,config=config)

    def log(self,metrics:dict,step:Optional[int] = None) -> None:

        if self.enabled:

            wandb.log(metrics,step=step)

    def watch(self,model,log:str = "gradients",log_freq:int = 100) -> None:

        if self.enabled:
            
            wandb.watch(model,log=log,log_freq=log_freq)

    def finish(self) -> None:

        if self.enabled and self.run is not None:

            wandb.finish()

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self.finish()

        return False