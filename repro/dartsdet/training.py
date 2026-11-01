"""K-fold denoising score matching of the conditional score.

Fold j holds out every K-th usable pixel of one fixed random order as validation centres; they stay
visible in the other pixels' contexts, and K folds hold out every usable pixel once. A training
step takes a random crop (a window plus the context margin, under one of the 8 dihedral
transforms), computes its contexts, and draws a noise level and a noise per training centre in it.
AdamW with a linear warmup and a cosine decay, gradient clipping, and an EMA of the weights; the EMA
state with the lowest held-out loss, at draws fixed for the run, is kept, and training stops after
`patience` validations without a new minimum.

The fold index seeds the model's initialization and every training draw; the fold partition
itself is one fixed order, drawn with seed 0. Fold j of a scene is therefore a pure function of the
scene, the recipe and j, up to the device's arithmetic.
"""

import math
import time
from functools import partial

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from dartsdet.model import ContextFreeScore, ScoreModel, Standardize, vp_coefficients


def held_out(usable: torch.Tensor, fold: int, folds: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(held-out, kept) pixel indices of fold `fold` of `folds` over the usable pixels (n,) bool."""
    rows = torch.nonzero(usable).squeeze(1)
    order = rows[torch.randperm(len(rows), generator=torch.Generator().manual_seed(0))]
    chosen = torch.zeros(len(rows), dtype=torch.bool)
    chosen[fold % folds::folds] = True
    return order[chosen], order[~chosen]


def log_uniform(n: int, low: float, high: float, generator: torch.Generator) -> torch.Tensor:
    """n noise levels, log-uniform on [low, high]."""
    return low * (high / low) ** torch.rand(n, generator=generator)


def lr_factor(step: int, steps: int, warmup: int) -> float:
    """Learning-rate multiplier after `step` steps: linear warmup to 1 over `warmup` steps, times a
    cosine decay from 1 at step 0 to 0 at `steps`."""
    return min(1.0, (step + 1) / max(warmup, 1)) * 0.5 * (1 + math.cos(math.pi * step / steps))


def optimization(model: torch.nn.Module, train: dict) -> tuple[
        torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR, AveragedModel]:
    """(AdamW, its warmup-cosine schedule, an EMA of the model) for the recipe's `train` block."""
    scheme = train["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["lr"],
                                  weight_decay=scheme["weight_decay"])
    factor = partial(lr_factor, steps=train["max_steps"], warmup=scheme["warmup"])
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(scheme["ema"]), use_buffers=True)
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor), ema


def dsm_loss(model: ScoreModel | ContextFreeScore, zc: torch.Tensor, context: torch.Tensor,
             sigma: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """(eps_hat - eps)^2 / a^2, the v-parametrization's loss, averaged over centres and bands:
    standardized centres zc (n, B) with contexts (n, C), noise levels sigma (n,), noises (n, B)."""
    a, s = vp_coefficients(sigma[:, None])
    return ((model.epsilon(a * zc + s * noise, context, sigma) - noise) ** 2 / a**2).mean()


def validation_loss(model: ScoreModel, z: torch.Tensor, mask: torch.Tensor, rows: torch.Tensor,
                    sigma: torch.Tensor, noise: torch.Tensor) -> float:
    """`dsm_loss` of the held-out centres `rows` at fixed draws, contexts from the whole image."""
    with torch.no_grad():
        context = model.encoder(z, mask)[0].flatten(1).T[rows.to(z.device)]
        zc = z[0].flatten(1).T[rows.to(z.device)]
        return float(dsm_loss(model, zc, context, sigma, noise))


def crop_batch(model: ScoreModel, padded: list[torch.Tensor], grid: tuple[int, int], train: dict,
               generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """(centres (n, B), contexts (n, C)) of the training centres in one random crop, at most
    `centres` of them. padded: the image, the mask and the training-centre mask, each padded by
    the encoder's margin."""
    W, H = grid
    margin = model.encoder.margin()
    cw, ch = min(train["crop"], W), min(train["crop"], H)
    r0 = int(torch.randint(W - cw + 1, (1,), generator=generator))
    c0 = int(torch.randint(H - ch + 1, (1,), generator=generator))
    k = int(torch.randint(8, (1,), generator=generator))
    zw, mw, kw = (v[:, :, r0:r0 + cw + 2 * margin, c0:c0 + ch + 2 * margin] for v in padded)
    context = dihedral(model.encoder(dihedral(zw, k), dihedral(mw, k)), k, inverse=True)
    inner = (slice(None), slice(None), slice(margin, margin + cw), slice(margin, margin + ch))
    context, zc = context[inner], zw[inner]
    chosen = kw[inner][0, 0].flatten().nonzero().squeeze(1)
    if len(chosen) > train["centres"]:
        draw = torch.randperm(len(chosen), generator=generator)[: train["centres"]]
        chosen = chosen[draw.to(chosen.device)]
    return zc[0].flatten(1).T[chosen], context[0].flatten(1).T[chosen]


def setup(fx: torch.Tensor, usable: torch.Tensor, grid: tuple[int, int], fold: int, spec: dict,
          device: str) -> tuple[ScoreModel, torch.Tensor, torch.Tensor, list, torch.Tensor]:
    """(model, image, mask, padded crops' inputs, held-out rows) of fold `fold`, untrained."""
    val, kept = held_out(usable, fold, spec["train"]["folds"])
    coordinates = Standardize.fit(fx[kept].double()).float()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fold)
        model = ScoreModel(coordinates, spec)
    model.to(device)
    z, mask = model.image(fx.float().to(device), usable.to(device), grid)
    centres = torch.zeros(len(fx), dtype=torch.bool)
    centres[kept] = True
    centres = centres.reshape(grid)[None, None].float().to(device)
    m = model.encoder.margin()
    padded = [F.pad(v, (m, m, m, m)) for v in (z, mask, centres)]
    return model, z, mask, padded, val


def fit(fx: torch.Tensor, usable: torch.Tensor, grid: tuple[int, int], fold: int, spec: dict,
        device: str) -> ScoreModel:
    """Fold `fold` of the recipe `spec` on whitened spectra fx (P, B) in pixel order,
    trained on the usable pixels (P,) bool: the EMA state with the lowest held-out loss."""
    train = spec["train"]
    generator = torch.Generator().manual_seed(fold)
    model, z, mask, padded, val = setup(fx, usable, grid, fold, spec, device)
    val_sigma = log_uniform(len(val), model.sigma_min, model.sigma_max, generator).to(device)
    val_noise = torch.randn(len(val), z.shape[1], generator=generator).to(device)
    optimizer, scheduler, ema = optimization(model, train)
    best, best_state, window, start = math.inf, None, [], time.time()
    for step in range(1, train["max_steps"] + 1):
        zc, context = crop_batch(model, padded, grid, train, generator)
        sigma = log_uniform(len(zc), model.sigma_min, model.sigma_max, generator).to(device)
        noise = torch.randn(len(zc), zc.shape[1], generator=generator).to(device)
        loss = dsm_loss(model, zc, context, sigma, noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train["training"]["clip"])
        optimizer.step()
        scheduler.step()
        ema.update_parameters(model)
        window.append(float(loss.detach()))
        if step % train["eval_every"]:
            continue
        current = validation_loss(ema.module, z, mask, val, val_sigma, val_noise)
        model.history.append(sum(window) / len(window))
        model.validation.append((step, current))
        window = []
        if current < best:
            best, model.best_step = current, step
            best_state = {k: v.detach().clone() for k, v in ema.module.state_dict().items()}
        print(f"[fit] fold {fold} step {step} train {model.history[-1]:.4f} val {current:.4f} "
              f"best {model.best_step} ({time.time() - start:.0f} s)", flush=True)
        if (step - model.best_step) // train["eval_every"] >= train["patience"]:
            break
    if best_state is None:
        raise RuntimeError(f"fold {fold}: no finite validation loss in {step} steps")
    model.load_state_dict(best_state)
    return model


def dihedral(x: torch.Tensor, k: int, inverse: bool = False) -> torch.Tensor:
    """The k-th of the 8 dihedral transforms of the last two axes (k >= 4: transpose first), or
    its inverse."""
    if not inverse:
        return torch.rot90(x.transpose(-1, -2) if k >= 4 else x, k % 4, (-2, -1))
    y = torch.rot90(x, -(k % 4), (-2, -1))
    return y.transpose(-1, -2) if k >= 4 else y
