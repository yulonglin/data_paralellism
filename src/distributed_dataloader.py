import os
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torchtext
from typing import Optional
from transformers import GPT2Tokenizer

import common
from common import DEVICES


class DistributedDataLoader:
    """
    DataLoader that loads minibatches of data to be distributed from leader node to worker nodes.
    """
    def __init__(self, rank: int, world_size: int,
                 mini_batch_size: int,
                 random_seed: Optional[int] = 0) -> None:
        super().__init__()

        self.rank = rank
        self.world_size = world_size
        self.mini_batch_size = mini_batch_size
        self.random_seed = random_seed
        self.leader_rank = common.LEADER_RANK
        self.max_seq_len = common.MAX_SEQ_LEN
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        train_data, valid_data, test_data = torchtext.datasets.WikiText2('data', split=('train', 'valid', 'test'))

        # Only load data from leader node
        if self.rank == self.leader_rank:
            batch_size = mini_batch_size * world_size
            self.train_dataloader = DataLoader(train_data,
                                               batch_size=batch_size)
            self.test_dataloader = DataLoader(test_data,
                                              batch_size=batch_size)

    def __iter__(self):
        minibatch = torch.zeros([self.mini_batch_size, self.max_seq_len], dtype=torch.long).to(DEVICES[self.rank])
        # If leader
        if self.rank == self.leader_rank:
            # Load 1 batch for each worker
            minibatches = []
            batch = next(iter(self.train_dataloader))
            for i in range(self.world_size):
                non_padded_seqs = self.tokenizer(batch[i * self.mini_batch_size:(i + 1) * self.mini_batch_size])[
                    'input_ids']  # List of List of Tokens(int)

                non_padded_seqs = [seq[:self.max_seq_len] for seq in non_padded_seqs]

                padded_seqs = torch.LongTensor(
                    [sent + [self.tokenizer(" ")["input_ids"][0]] * (self.max_seq_len - len(sent)) for sent in
                     non_padded_seqs])

                minibatches.append(padded_seqs.to(DEVICES[self.rank]))
        else:
            # Not load anything
            minibatches = None

        # Scatter minibatches
        dist.scatter(minibatch, minibatches, src=self.leader_rank)

        yield minibatch

    def tokenize(self, batch):
        non_padded_seqs = self.tokenizer(batch)['input_ids']  # List of List of Tokens(int)

        non_padded_seqs = [seq[:self.max_seq_len] for seq in non_padded_seqs]

        padded_seqs = torch.LongTensor(
            [sent + [self.tokenizer(" ")["input_ids"][0]] * (self.max_seq_len - len(sent)) for sent in non_padded_seqs])

        return padded_seqs.to(DEVICES[self.rank])
