import os
import numpy as np
import torch

DATA_DIR = r"E:\GPT2_data" 

class DataLoaderLite:
    
    def __init__(self,B:int,T:int,process_rank:int,num_processes:int,split:str = "train",data_dir:str = DATA_DIR):

        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {"train","val"}

        filepath = os.path.join(data_dir,f"{split}.bin")

        assert os.path.exists(filepath),f"no {split}.bin found at {filepath}"

        self.tokens = np.memmap(filepath, dtype=np.uint16, mode="r")

        if process_rank == 0:

            print(f"loaded {split} split: {len(self.tokens):,} tokens from {filepath}")

        self.reset()

    def reset(self) -> None:

        self.curr_pos = self.B * self.T * self.process_rank

    def next_batch(self):

        B, T = self.B, self.T
        needed = B * T + 1

        if self.curr_pos + needed > len(self.tokens):

            self.curr_pos = self.B * self.T * self.process_rank

        chunk = self.tokens[self.curr_pos: self.curr_pos + needed]
        chunk = torch.from_numpy(chunk.astype(np.int64))
        x = chunk[:-1].view(B, T)
        y = chunk[1:].view(B, T)
        self.curr_pos += B * T * self.num_processes

        return x, y

    def state_dict(self) -> dict:
      
        return {"curr_pos": self.curr_pos}

    def load_state_dict(self, state: dict) -> None:

        self.curr_pos = state["curr_pos"]