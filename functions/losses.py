import torch


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

loss_registry = {
    'simple': noise_estimation_loss,
    'iso': noise_estimation_loss_iso,
}