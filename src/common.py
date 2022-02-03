DEVICES = [
    "cuda:0", "cuda:1", "cuda:2", "cuda:3",
]

BROADCAST = 'broadcast'
SCATTER = 'scatter'
ALL_REDUCE = 'all-reduce'

LEADER_RANK = 0
MAX_SEQ_LEN = 10


class InvalidDistributedOperationError(Exception): ...
