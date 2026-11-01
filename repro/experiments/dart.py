"""DART, Experiment 3's context-free model: denoising score matching of the score head on
whitened spectra alone, with early stopping by a held-out run and a refit on every pixel.

Phase 1 trains on all but a seeded `validation_fraction` of the pixels, validating every epoch,
until the validation loss has gone `patience` epochs without a new minimum; E* is the epoch of the
minimum. Phase 2 trains from the same seed on every pixel and stops at epoch E*. Both follow the
warmup and cosine of a `max_epochs` schedule, and one epoch is round(pixels / batch_size) steps.
The model is the EMA of the weights at the last step. Training noise levels are log-normal,
truncated to [sigma_min, sigma_max]; validation draws are log-uniform on the same range. The
network predicts v = a eps - s z (its training loss), and the validation loss is the eps loss.
"""

import math
from collections.abc import Callable

import torch

from dartsdet.model import ContextFreeScore, Standardize, vp_coefficients
from dartsdet.training import optimization

VALIDATION_CHUNK = 65536  # held-out pixels per validation block; the sum runs per block


def truncated_log_normal(n: int, generator: torch.Generator, spec: dict) -> torch.Tensor:
    """n noise levels (float64), log sigma ~ N(log median, log_std^2) truncated to
    [sigma_min, sigma_max], by the inverse distribution function of seeded uniforms."""
    center, spread = math.log(spec["sampling"]["median"]), spec["sampling"]["log_std"]
    limits = torch.tensor([math.log(spec["sigma_min"]), math.log(spec["sigma_max"])],
                          dtype=torch.float64)
    lower, upper = torch.special.ndtr((limits - center) / spread)
    p = lower + (upper - lower) * torch.rand(n, generator=generator, dtype=torch.float64)
    return (center + spread * torch.special.ndtri(p)).exp().clamp(spec["sigma_min"],
                                                                  spec["sigma_max"])


def v_loss(model: ContextFreeScore, z: torch.Tensor, sigma: torch.Tensor,
           noise: torch.Tensor) -> torch.Tensor:
    """Mean squared error of the v prediction, v = a eps - s z, at given draws."""
    a, s = vp_coefficients(sigma[:, None])
    noised = a * z + s * noise
    out = model.network(noised, sigma.log())
    return torch.nn.functional.mse_loss(out, a * noise - s * z)


def validation_loss(model: ContextFreeScore, z: torch.Tensor, sigma: torch.Tensor,
                    noise: torch.Tensor) -> float:
    """The eps loss, mean squared error of eps_hat, over every held-out draw."""
    total = 0.0
    with torch.no_grad():
        for zb, sb, nb in zip(*(v.split(VALIDATION_CHUNK) for v in (z, sigma, noise)),
                              strict=True):
            a, s = vp_coefficients(sb[:, None])
            eps = model.epsilon(a * zb + s * nb, zb.new_zeros(len(zb), 0), sb)
            loss = torch.nn.functional.mse_loss(eps, nb)
            total += float(loss) * zb.numel()
    return total / z.numel()


def run(x: torch.Tensor, seed: int, spec: dict, steps: int, eval_every: int,
        held: float, stop: Callable[[int, list[float]], bool]) -> tuple[ContextFreeScore,
                                                                     list[float]]:
    """(model, losses) of one training run on spectra x (n, B): `steps` of AdamW with a share
    `held` held out, evaluated every `eval_every` steps and at the last; `stop(step, losses)` may
    end it early. The losses are the validation losses, or without a hold-out the mean training
    losses since the previous evaluation."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(x), generator=generator) if held else torch.arange(len(x))
    cut = math.ceil(held * len(x))
    val, kept = order[:cut].to(x.device), order[cut:].to(x.device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ContextFreeScore(Standardize.fit(x[kept].cpu()), spec["width"], spec["depth"],
                                 spec["network"]).to(x.device)
    z_val, z = model.coordinates(x[val]), model.coordinates(x[kept])
    u = torch.rand(len(z_val), generator=generator, dtype=torch.float64)
    val_sigma = (spec["sigma_min"] * (spec["sigma_max"] / spec["sigma_min"]) ** u).to(z_val)
    val_noise = torch.randn(z_val.shape, generator=generator, dtype=torch.float64).to(z_val)
    train = {"lr": spec["schedule"]["lr"], "max_steps": steps, "training": spec["training"]}
    optimizer, scheduler, ema = optimization(model, train)
    batch, losses, window = spec["schedule"]["batch_size"], [], []
    for step in range(1, steps + 1):
        rows = torch.randint(len(z), (batch,), generator=generator).to(z.device)
        sigma = truncated_log_normal(batch, generator, spec).to(z)
        noise = torch.randn(batch, z.shape[1], generator=generator, dtype=torch.float64).to(z)
        loss = v_loss(model, z[rows], sigma, noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec["training"]["clip"])
        optimizer.step()
        window.append(loss.detach().clone())
        scheduler.step()
        ema.update_parameters(model)
        if step % eval_every and step != steps:
            continue
        losses.append(validation_loss(ema.module, z_val, val_sigma, val_noise) if len(z_val)
                      else float(torch.stack(window).mean()))
        window = []
        if stop(step, losses):
            break
    model.load_state_dict(ema.module.state_dict())
    return model, losses


def fit(x: torch.Tensor, seed: int, spec: dict) -> tuple[ContextFreeScore, int]:
    """(DART on spectra x (n, B), E*): phase 1 to find E*, then phase 2 (module docstring)."""
    rule, batch = spec["early_stopping"], spec["schedule"]["batch_size"]
    first = max(1, round((len(x) - math.ceil(rule["validation_fraction"] * len(x))) / batch))

    def patience_ran_out(_step: int, losses: list[float]) -> bool:
        return len(losses) - 1 - first_minimum(losses) >= rule["patience"]

    _, losses = run(x, seed, spec, rule["max_epochs"] * first, first,
                    rule["validation_fraction"], patience_ran_out)
    if not any(math.isfinite(v) for v in losses):
        raise RuntimeError(f"DART seed {seed}: no finite validation loss in phase 1")
    epochs, every = first_minimum(losses) + 1, max(1, round(len(x) / batch))
    model, _ = run(x, seed, spec, rule["max_epochs"] * every, every, 0.0,
                   lambda step, _: step >= epochs * every)
    return model, epochs


def first_minimum(values: list[float]) -> int:
    """Index of the first smallest value: a later equal value is no improvement."""
    return min(range(len(values)), key=values.__getitem__)
