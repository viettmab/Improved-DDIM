import torch

def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    
def calculate_alpha(beta):
    alphas_cumprod = (1 - beta).cumprod(dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1).to(alphas_cumprod.device), alphas_cumprod[:-1]], dim=0)
    alphas_cumprod_next = torch.cat([alphas_cumprod[1:], torch.zeros(1).to(alphas_cumprod.device)], dim=0)
    recip_noise_coef = torch.sqrt(1-alphas_cumprod) * torch.sqrt(1-beta) / beta
    return {
        "alphas_cumprod": alphas_cumprod, 
        "alphas_cumprod_prev": alphas_cumprod_prev, 
        "alphas_cumprod_next": alphas_cumprod_next, 
        "recip_noise_coef": recip_noise_coef,
    }

def extract_into_tensor(x, t):
    return x.index_select(0, t.long()).view(-1, 1, 1, 1)

def q_posterior_mean_variance(x_start, xt, t, b, eta = 0):
    assert x_start.shape == xt.shape
    coef = calculate_alpha(b)
    at_bar = extract_into_tensor(coef["alphas_cumprod"],t)
    at_bar_prev = extract_into_tensor(coef["alphas_cumprod_prev"],t)

    sigma = (eta * torch.sqrt((1 - at_bar / at_bar_prev) * (1 - at_bar_prev) / (1 - at_bar)))
    posterior_mean = (
        torch.sqrt(at_bar_prev) * x_start 
        + torch.sqrt(1-at_bar_prev-sigma**2) 
        * ((xt-torch.sqrt(at_bar)*x_start)/torch.sqrt(1-at_bar))
    )
    return posterior_mean

def p_mean_variance(model, residual_connection_net, xt, t, b, residual_x_start=None):
    eps_predict = model(xt, t.float())
    coef = calculate_alpha(b)
    at_bar = extract_into_tensor(coef["alphas_cumprod"],t)
    pred_xstart = (xt - eps_predict * torch.sqrt(1 - at_bar)) / torch.sqrt(at_bar)

    #residual_val_raw = torch.tensor([-1]*xt.shape[0])
    # if residual_x_start is not None:
    #     if residual_connection_net is not None:
    #         residual_val_raw = residual_connection_net(
    #             residual_x_start, t
    #         ).squeeze()
    #         while len(residual_val_raw.shape) < len(pred_xstart.shape):
    #             residual_val_raw = residual_val_raw[..., None]
    #         residual_val = residual_val_raw.expand(pred_xstart.shape)
    #         pred_xstart = (1.0 - residual_val) * pred_xstart + residual_val * residual_x_start
    if residual_x_start is not None:
        residual_val = torch.empty(pred_xstart.shape, dtype=torch.float32, device=pred_xstart.device)
        residual_val.fill_(0.5)
        pred_xstart = (1.0 - residual_val) * pred_xstart + residual_val * residual_x_start

    model_mean = q_posterior_mean_variance(pred_xstart,xt,t,b)

    assert(
        model_mean.shape == pred_xstart.shape == xt.shape
    )
    
    return model_mean, pred_xstart, eps_predict


def train2step_loss(model, residual_connection_net,
                    x0: torch.Tensor,
                    t: torch.LongTensor,
                    e: torch.Tensor,
                    b: torch.Tensor, keepdim=False):
    coef = calculate_alpha(b)
    at_bar = extract_into_tensor(coef["alphas_cumprod"],t)
    xt = x0 * torch.sqrt(at_bar) + e * torch.sqrt(1.0 - at_bar)
    mean_prediction, x0_pred, eps_prediction = p_mean_variance(model,residual_connection_net,xt,t,b,None)
    true_mean = q_posterior_mean_variance(x0, xt, t, b)
    mse = (extract_into_tensor(coef["recip_noise_coef"],t) * (true_mean - mean_prediction)).square().sum(dim=(1, 2, 3)).mean(dim=0)
 
    # #First residual
    at_bar_prev = extract_into_tensor(coef["alphas_cumprod_prev"],t)
    e_1 = torch.randn_like(e)
    xt_1_first = x0 * torch.sqrt(at_bar_prev) + e_1 * torch.sqrt(1.0 - at_bar_prev)
    t_prev = torch.clamp(t - 1.0, min=0)
    mean_prediction_1, _, _ = p_mean_variance(model,residual_connection_net,xt_1_first,t_prev,b,x0_pred)
    true_mean_1 = q_posterior_mean_variance(x0, xt_1_first,t_prev,b)

    mse_first = (extract_into_tensor(coef["recip_noise_coef"],t_prev)*(true_mean_1 - mean_prediction_1)).square().sum(dim=(1, 2, 3)).mean(dim=0)

    # #Second residual
    eta = 0
    sigma = eta * torch.sqrt((1 - at_bar / at_bar_prev) * (1 - at_bar_prev) / (1 - at_bar))
    c2 = torch.sqrt(1 - at_bar_prev - sigma ** 2)
    xt_1_second = torch.sqrt(at_bar_prev) * x0_pred + sigma * torch.randn_like(xt) + c2 * eps_prediction
    mean_prediction_2, _, _ = p_mean_variance(model,residual_connection_net,xt_1_second,t_prev,b,x0_pred)
    true_mean_2 = q_posterior_mean_variance(x0, xt_1_second,t_prev,b)

    mse_second = (extract_into_tensor(coef["recip_noise_coef"],t_prev)*(true_mean_2 - mean_prediction_2)).square().sum(dim=(1, 2, 3)).mean(dim=0)
    mse += (mse_first+mse_second)/2
    return mse

def mismatch_loss(model, residual_connection_net,
                    x0: torch.Tensor,
                    t: torch.LongTensor,
                    e: torch.Tensor,
                    b: torch.Tensor, gamma=1):
    coef = calculate_alpha(b)
    at_bar = extract_into_tensor(coef["alphas_cumprod"],t)
    xt = x0 * torch.sqrt(at_bar) + e * torch.sqrt(1.0 - at_bar)
    eps_prediction = model(xt, t.float())
    x0_pred = (xt - eps_prediction * torch.sqrt(1 - at_bar)) / torch.sqrt(at_bar)
    mse = (e - eps_prediction).square().sum(dim=(1, 2, 3)).mean(dim=0)
 
    # Interpolation
    # residual_val_raw = torch.tensor([-1]*xt.shape[0])
    # if residual_connection_net is not None:
    #     residual_val_raw = residual_connection_net(
    #         x0_pred, t
    #     ).squeeze()
    #     while len(residual_val_raw.shape) < len(x0_pred.shape):
    #         residual_val_raw = residual_val_raw[..., None]
    #     residual_val = residual_val_raw.expand(x0_pred.shape)
    #     x0_tilde = (1.0 - residual_val) * x0_pred + residual_val * x0
    # residual_val = torch.empty(x0_pred.shape, dtype=torch.float32, device=x0_pred.device)
    # residual_val.fill_(0.5)
    # x0_tilde = (1.0 - residual_val) * x0_pred + residual_val * x0

    # Perfect xt_1
    # e_1 = torch.randn_like(e)
    at_bar_prev = extract_into_tensor(coef["alphas_cumprod_prev"],t)
    # xt_1= x0 * torch.sqrt(at_bar_prev) + e_1 * torch.sqrt(1.0 - at_bar_prev)
    t_prev = torch.clamp(t - 1.0, min=0)
    # eps_prediction_1 = model(xt_1, t_prev.float())
    # mse_1 = (e_1 - eps_prediction_1).square().sum(dim=(1, 2, 3)).mean(dim=0)

    # Consistency loss
    eta = 1
    sigma = eta * torch.sqrt((1 - at_bar / at_bar_prev) * (1 - at_bar_prev) / (1 - at_bar))
    c2 = torch.sqrt(1 - at_bar_prev - sigma ** 2)
    xt_1_tilde = torch.sqrt(at_bar_prev) * x0_pred + c2 * eps_prediction + sigma * torch.randn_like(xt)
    eps_prediction_1_tilde = model(xt_1_tilde, t_prev.float())

    # Define a small epsilon value to handle division by zero
    z = 1e-5
    numerator = xt_1_tilde-torch.sqrt(at_bar_prev)*x0
    denominator = torch.sqrt(1 - at_bar_prev)
    # Replace zeros in denominator with the epsilon value to avoid division by zero
    denominator_safe = torch.where(denominator != 0, denominator, torch.tensor(z, dtype=torch.float32))
    # Perform element-wise division
    eps_target_1_tilde = torch.where(denominator != 0, numerator / denominator_safe, torch.tensor(0, dtype=torch.float32))
    eps_prediction_1_tilde = torch.where(denominator != 0, eps_prediction_1_tilde, torch.tensor(0, dtype=torch.float32))
    mse_2 = (eps_prediction_1_tilde - eps_target_1_tilde).square().sum(dim=(1, 2, 3)).mean(dim=0)
    mse += mse_2
    return mse / 2

def get_residual_value(model, residual_connection_net,
                    x0: torch.Tensor,
                    t: torch.LongTensor,
                    e: torch.Tensor,
                    b: torch.Tensor):
    coef = calculate_alpha(b)
    at_bar = extract_into_tensor(coef["alphas_cumprod"],t)
    xt = x0 * torch.sqrt(at_bar) + e * torch.sqrt(1.0 - at_bar)
    eps_predict = model(xt, t.float())
    pred_xstart = (xt - eps_predict * torch.sqrt(1 - at_bar)) / torch.sqrt(at_bar)
    residual_val_raw = torch.tensor([-1]*xt.shape[0])
    if residual_connection_net is not None:
        residual_val_raw = residual_connection_net(
            pred_xstart, t
        ).squeeze()
    return residual_val_raw


loss_registry = {
    'simple': noise_estimation_loss,
    'train2steps': train2step_loss,
    'get_residual_value': get_residual_value,
    'train_mismatch': mismatch_loss
}