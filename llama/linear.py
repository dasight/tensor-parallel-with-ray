import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class Split(torch.autograd.Function):
    # forward: split the input along the last dimension. (..., idim) -> (..., idim/n)
    # backward: cat the upstream gradient along the last dim. (..., idim/n) -> (..., idim)
    @staticmethod
    def forward(ctx, x: torch.Tensor, tp_rank=None, tp_size=1) -> torch.Tensor:
        print(f'x.requires_grad: {x.requires_grad}')
        dim = x.size(-1)
        ctx.tp_rank = tp_rank
        ctx.tp_size = tp_size
        assert dim % tp_size == 0
        dim = dim // tp_size
        start, end = tp_rank * dim, tp_rank * dim + dim
        x = x[..., start:end].contiguous()
        return x
        
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        print(f'Grad Split@{ctx.tp_rank}:', grad_output.shape)
        grad_output_list = [grad_output.new_zeros(grad_output.size()) for _ in range(ctx.tp_size)]
        dist.all_gather(grad_output_list, grad_output)
        output = torch.cat(grad_output_list, dim=-1)
        return output, None, None
    

class Reduce(torch.autograd.Function):
    # forward: reduce the input along the last dimension. (..., idim) -> (..., idim)
    # backward: directly return the upstream gradient
    @staticmethod
    def forward(ctx, x: torch.Tensor, tp_rank=None, tp_size=1) -> torch.Tensor:
        ctx.tp_rank = tp_rank
        ctx.tp_size = tp_size
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # print(f'Grad Reduce@{ctx.tp_rank}:', grad_output.shape)
        return grad_output, None, None
    

class Copy(torch.autograd.Function):
    # forward: directly return the input
    # backward: reduce the grad_output
    # Notes: Copy is the opposite operation of Reduce
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        return grad_output


class Gather(torch.autograd.Function):
    # forward: gather the input along the last dimension: (..., idim/n) -> (..., idim)
    # backward: split the upstream grad_output along the last dim: (..., idim) -> (..., idim/n)
    # Notes: Gather is the opposite operation of Split
    @staticmethod
    def forward(ctx, x: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
        ctx.tp_rank = tp_rank
        ctx.tp_size = tp_size
        input_list = [x.new_zeros(x.size()) for _ in range(tp_size)]
        input_list[tp_rank] = x
        dist.all_gather(input_list, x)
        return torch.cat(input_list, dim=-1).contiguous()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        tp_rank = ctx.tp_rank
        tp_size = ctx.tp_size
        assert grad_output.size(-1) % tp_size == 0
        grad_arr = torch.split(grad_output, grad_output.size(-1) // tp_size, dim=-1)
        return grad_arr[tp_rank].contiguous(), None, None
    

class AvgGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(grad_output, op=dist.ReduceOp.AVG)
        return grad_output


class ColumnParallelLinear(nn.Module):
    def __init__(self, idim: int, odim: int, add_bias=True, tp_rank=-1, tp_size=1, gather_output: bool = False):
        # forward: (b, idim) -[linear]> (b, odim/n) -[gather]> (b, odim)
        # weight shape: (idim/n, odim)
        super().__init__()
        self.idim, self.odim = idim, odim
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.gather_output = gather_output

        assert odim % tp_size == 0
        tp_odim = odim // tp_size
        self.weight = nn.Parameter(torch.Tensor(tp_odim, idim))
        self.add_bias = add_bias
        if add_bias:
            self.bias = nn.Parameter(torch.Tensor(tp_odim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: copy (this is for backward correctness)
        x = Copy.apply(x)
        # Step 2: linear, (..., idim) -> (..., odim/n)
        x = F.linear(x, self.weight)
        # Step 3 (optional): add bias
        if self.add_bias:
            x = x + self.bias
        # Step 4: gather linear outputs, i.e. [(.., odim/n), ...] -> (..., odim)
        if self.gather_output and self.tp_size > 1:
            x = Gather.apply(x, self.tp_rank, self.tp_size)
        return x

class RowParallelLinear(nn.Module):
    def __init__(self, idim: int, odim: int, add_bias=False, tp_rank=-1, tp_size=1, split_input: bool = True):
        # forward: (b, idim) -[split]> (b, idim/n) -[linear]> (b, odim) -[gather]> (b, odim)
        # weight shape: (idim/n, odim)
        super().__init__()
        self.idim, self.odim = idim, odim
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.split_input = split_input

        assert idim % tp_size == 0
        self.weight = nn.Parameter(torch.Tensor(odim, idim // tp_size))
        if add_bias:
            self.bias = nn.Parameter(torch.Tensor(self.odim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.split_input and self.tp_size > 1:
            x = Split.apply(x, self.tp_rank, self.tp_size)
        # print(f'RowParallelLinear@{self.tp_rank} input:', x.shape)
        # print(f'RowParallelLinear@{self.tp_rank} weight:', self.weight.shape)
        x = F.linear(x, self.weight)
        x = Reduce.apply(x, self.tp_rank, self.tp_size)
        if self.bias:
            x = x + self.bias
        return x


def tp_paracol(x, weight, tp_rank, tp_size):
    import os
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "23333"
    dist.init_process_group(backend='gloo', init_method='env://', world_size=tp_size, rank=tp_rank)
    print(f"TP Rank {tp_rank} initialized.")

    odim, idim = weight.shape
    tp_odim = odim // tp_size
    layer = ColumnParallelLinear(idim, odim, add_bias=False, tp_rank=tp_rank, tp_size=tp_size)
    start, end = tp_rank * tp_odim, (tp_rank + 1) * tp_odim
    layer.weight = nn.Parameter(weight[start:end, :])
    y = layer(x)
    print(f'Result@{tp_rank}:', y)

    y = y.sum()
    y.backward()
    print(f'Grad@{tp_rank}:', layer.weight.grad)


def tp_pararow(x, weight, tp_rank, tp_size):
    import os
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "23333"
    dist.init_process_group(backend='gloo', init_method='env://', world_size=tp_size, rank=tp_rank)
    print(f"TP Rank {tp_rank} initialized.")

    odim, idim = weight.shape
    dim = idim // tp_size
    layer = RowParallelLinear(idim, odim, add_bias=False, tp_rank=tp_rank, tp_size=tp_size)
    start, end = tp_rank * dim, (tp_rank + 1) * dim
    layer.weight = nn.Parameter(weight[:, start:end])
    y = layer(x)
    print(f'Result@{tp_rank}:', y)

    y = y.sum()
    y.backward()
    print(f'Grad@{tp_rank}:', layer.weight.grad)


if __name__ == "__main__":
    import multiprocessing as mp

    tp_size = 2
    tp_group = 0
    weight = torch.randn(6, 8, requires_grad=True)
    x = torch.randn(2, 8)
    y = F.linear(x, weight)
    print('Normal forward:', y)
    y = y.sum()
    y.backward()
    print('Normal backward:', weight.grad)

    workers = [mp.Process(target=tp_pararow, args=(x, weight, tp_rank, tp_size)) for tp_rank in range(tp_size)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
