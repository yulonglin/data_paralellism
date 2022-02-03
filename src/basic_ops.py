"""An example of how basic operations like broadcast, scatter and all-reduce can be implemented."""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import common
from common import DEVICES, InvalidDistributedOperationError


OPERATION = common.BROADCAST


def run(rank, size):
    """ Distributed function implementing broadcast, scatter and all-reduce."""
    shape = [2, 2]
    device = DEVICES[rank]

    if OPERATION == common.BROADCAST:
        # Broadcast from the worker of rank 1 to other workers
        x = torch.zeros(shape).to(device)
        if rank == 1:
            x = torch.randn(shape).to(device)
        dist.broadcast(x, src=1)
        print(x)
    elif OPERATION == common.SCATTER:
        # Scatter list of tensors y from the worker of rank 1,
        #   to the other workers at tensor x
        x = torch.zeros(shape).to(device)
        if rank == 1:
            y = [torch.zeros(shape).to(device) + 1 + 2 * i for i in range(size)]
        else:
            y = None
        dist.scatter(x, y, src=1)
        print(x)
    elif OPERATION == common.ALL_REDUCE:
        # Starting with unique tensors x on each device,
        #   sum them together on all devices
        x = torch.zeros(shape).to(device) + rank
        dist.all_reduce(x, dist.ReduceOp.SUM)
        print(x)
    else:
        raise InvalidDistributedOperationError(OPERATION)


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = len(DEVICES)
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run, "gloo"))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
