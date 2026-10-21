"""Grouped timestep sampling and differentiable per-timestep energy estimates."""

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce


def sample_grouped_timesteps(batch_size, num_timesteps, groups, device,
                             sync_distributed=False):
    """Uniformly choose distinct times, then repeat each across a balanced group.

    A short final minibatch uses fewer groups if necessary. In DDP, all ranks
    receive the same times, so their energy estimates can be pooled correctly.
    """
    if batch_size < 1 or num_timesteps < 1 or groups < 1:
        raise ValueError("batch_size, num_timesteps and groups must be positive")
    group_count = min(groups, batch_size, num_timesteps)
    distributed = sync_distributed and dist.is_available() and dist.is_initialized()
    if not distributed or dist.get_rank() == 0:
        times = torch.randperm(num_timesteps, device=device)[:group_count]
    else:
        times = torch.empty(group_count, device=device, dtype=torch.long)
    if distributed:
        dist.broadcast(times, src=0)
    counts = torch.full((group_count,), batch_size // group_count,
                        device=device, dtype=torch.long)
    counts[:batch_size % group_count] += 1
    return torch.repeat_interleave(times, counts)


def per_timestep_energy(output, t, sync_distributed=False):
    """Return sorted times, mean squared prediction per coordinate, and counts.

    With DDP, use an autograd-aware SUM for energies. Its backward SUM and DDP's
    gradient average together give the gradient of the global pooled objective.
    This requires matching timestep sets on every rank (provided by the sampler).
    """
    if output.shape[0] != t.numel() or output.shape[0] == 0:
        raise ValueError("output and t must contain the same nonzero sample count")
    # Keep reductions in float32 for float16/bfloat16 predictions.
    values = output.float() if output.dtype in (torch.float16, torch.bfloat16) else output
    sample_energy = values.flatten(1).square().mean(dim=1)
    times, inverse, counts = torch.unique(t, sorted=True, return_inverse=True,
                                         return_counts=True)
    sums = sample_energy.new_zeros(times.numel()).scatter_add(0, inverse, sample_energy)
    counts = counts.to(sums.dtype)
    if sync_distributed and dist.is_available() and dist.is_initialized():
        sums = all_reduce(sums, op=dist.ReduceOp.SUM)
        counts = counts.clone()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return times, sums / counts, counts
