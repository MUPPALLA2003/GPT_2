import multiprocessing as mp
import os

os.environ.setdefault("HF_HOME", r"E:\hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", r"E:\hf_cache\datasets")

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = r"E:\GPT2_data"
REMOTE_NAME = "sample-10BT"
VAL_TOKENS = int(1e8)   
CHUNK_SIZE = 16        

enc = tiktoken.get_encoding("gpt2")
EOS = enc._special_tokens["<|endoftext|>"]

def tokenize(doc: dict) -> np.ndarray:
  
    tokens = [EOS]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)

    assert (tokens_np >= 0).all() and (tokens_np < 2**16).all(), "token id doesn't fit in uint16"

    return tokens_np.astype(np.uint16)

def main():

    os.makedirs(DATA_DIR, exist_ok=True)
    val_path = os.path.join(DATA_DIR, "val.bin")
    train_path = os.path.join(DATA_DIR, "train.bin")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu",name=REMOTE_NAME,split="train")
    n_procs = max(1, os.cpu_count() // 2)
    tokens_written = 0
    val_done = False

    with mp.Pool(n_procs) as pool, \
         open(val_path, "wb") as val_f, \
         open(train_path, "wb") as train_f, \
         tqdm(total=VAL_TOKENS, unit="tokens", desc="val") as val_bar:

        train_bar = None

        for tokens in pool.imap(tokenize, dataset, chunksize=CHUNK_SIZE):

            if not val_done:

                remaining = VAL_TOKENS - tokens_written
                head, tail = tokens[:remaining], tokens[remaining:]
                val_f.write(head.tobytes())
                val_bar.update(len(head))
                tokens_written += len(head)

                if tokens_written >= VAL_TOKENS:

                    val_done = True
                    train_bar = tqdm(unit="tokens", desc="train")

                    if len(tail) > 0:

                        train_f.write(tail.tobytes())
                        train_bar.update(len(tail))

            else:

                train_f.write(tokens.tobytes())
                train_bar.update(len(tokens))

if __name__ == "__main__":
    main()