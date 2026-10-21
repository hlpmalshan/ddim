import torch
from functions.trace_energy import per_timestep_energy

def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False, reg = 0.0):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    batch_mean_isotropy = output.square().mean(dim=(0, 1, 2, 3))
    batch_standard_deviation = output.square().std(dim=(0, 1, 2, 3))
    if keepdim:
        return (e - output).square().mean(dim=(1, 2, 3))
    else:
        return (e - output).square().mean(dim=(1, 2, 3)).mean(dim=0), batch_mean_isotropy, batch_standard_deviation
'''
def noise_estimation_loss_iso(model,
                              x0: torch.Tensor,
                              t: torch.LongTensor,
                              e: torch.Tensor,
                              b: torch.Tensor,
                              keepdim=False, 
                              reg = 0.0, weighting_type='constant', snr_gamma=5.0):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    batch_norm_mean = output.square().mean(dim=(1, 2, 3)).mean(dim=0)
    batch_norm_standard_deviation = output.square().mean(dim=(1, 2, 3)).std(dim=0)
    # Compute SNR per sample
    snr = a / (1.0 - a + 1e-8)
    # compute weighting factor for SNR weighting
    # clipping the SNR by ``snr_gamma`` corresponds to the min‑SNR
    # weighting proposed in the Improved DDPMs paper.  When the
    # ``weighting`` argument to :meth:`train_step` is ``'snr'`` this
    # factor will be applied; otherwise the unweighted loss is used.
    snr_weights = torch.minimum(snr, torch.full_like(snr, snr_gamma)) / snr
    weights = snr_weights if weighting_type == 'snr' else torch.ones_like(snr_weights)
    if keepdim:
        base_loss = (e - output).square().mean(dim=(1, 2, 3))
        iso_loss =  (1 - output.square().mean(dim=(1, 2, 3))).square()
        return base_loss  + reg * iso_loss
    else:
        sample_mse_loss = (e - output).square().mean(dim=(1, 2, 3)) # MSE loss of each sample
        base_loss = (weights * sample_mse_loss).mean()              # Weighted mean MSE loss of the batch
        # iso_loss = (1 - output.square().mean(dim=0).mean(dim=(0, 1, 2))).square()
        # iso_loss = (1 - output.flatten().square().mean()).square()
        # Flatten c, h, w -> (b, c*h*w)
        output_flat = output.view(output.size(0), -1)
        # Square and take mean over flattened dims
        output_mean = output_flat.square().mean(dim=1)   # shape: (b,)
        sample_iso_loss = (1 - output_mean).square()
        iso_loss = (weights * sample_iso_loss).mean()
        return base_loss + reg * iso_loss, batch_norm_mean, batch_norm_standard_deviation

loss_registry = {
    'simple': noise_estimation_loss,
    'iso': noise_estimation_loss_iso,
}
'''
# Copy of working iso loss
def noise_estimation_loss_iso(model,
                              x0: torch.Tensor,
                              t: torch.LongTensor,
                              e: torch.Tensor,
                              b: torch.Tensor,
                              keepdim=False, reg = 0.0):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    batch_norm_mean = output.square().mean(dim=(1, 2, 3)).mean(dim=0)
    batch_norm_standard_deviation = output.square().mean(dim=(1, 2, 3)).std(dim=0)
    
    # Base loss
    if keepdim:
        base_loss = (e - output).square().mean(dim=(1, 2, 3))
        iso_loss =  (1 - output.square().mean(dim=(1, 2, 3))).square()
        return base_loss  + reg * iso_loss
    else:
        base_loss = (e - output).square().mean(dim=(1, 2, 3)).mean(dim=0)
        # iso_loss = (1 - output.square().mean(dim=0).mean(dim=(0, 1, 2))).square()
        iso_loss = (1 - output.flatten().square().mean()).square()
        return base_loss + reg * iso_loss, batch_norm_mean, batch_norm_standard_deviation


def noise_estimation_loss_iso_per_timestep(model,
                                          x0: torch.Tensor,
                                          t: torch.LongTensor,
                                          e: torch.Tensor,
                                          b: torch.Tensor,
                                          keepdim=False, reg=0.0,
                                          sync_distributed=False,
                                          return_details=False):
    """MSE plus mean_t[(mean_{samples,coordinates at t}(eps_hat**2)-1)**2].

    A per-sample keepdim loss has no equivalent semantics for this grouped
    objective; callers must request the scalar loss instead.
    """

    if keepdim:
        raise ValueError("per-timestep regularization requires keepdim=False")
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    values = output.float() if output.dtype in (torch.float16, torch.bfloat16) else output
    base_loss = (e - values).square().mean()
    times, energies, counts = per_timestep_energy(output, t, sync_distributed)
    energy_loss = (energies - 1.0).square().mean()
    sample_energy = values.flatten(1).square().mean(dim=1)
    result = (base_loss + reg * energy_loss,
              sample_energy.mean(), sample_energy.std(unbiased=False))
    if return_details:
        details = {"mse": base_loss.detach(), "energy_loss": energy_loss.detach(),
                   "timesteps": times.detach(), "energies": energies.detach(),
                   "counts": counts.detach()}
        return result + (details,)
    return result


loss_registry = {
        'simple': noise_estimation_loss,
        'iso': noise_estimation_loss_iso,
        'iso_per_timestep': noise_estimation_loss_iso_per_timestep
        }        
