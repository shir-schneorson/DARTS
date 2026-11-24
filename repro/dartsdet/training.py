"""Denoising score matching of the conditional score model (Sections V-C and V-E, Algorithm 1).

A fifth of the usable (unscreened) pixels are held out: fold j of 5 of one fixed random order,
drawn with seed 0. Held-out pixels stay visible to the contexts but are never training centres.
Each step takes `crops` random crops of the image, each under a random dihedral transform and
with its own noise level, log-uniform on [sigma_min, sigma_max]. The centres of a crop are noised,
u = a z + s eps, while its contexts read the clean a z. The loss is eq. (dsm-loss) with
w(sigma) = a^-2, over the crop's training centres. AdamW with a linear warmup, gradient clipping,
and an exponential moving average (EMA) of the weights.

Every `validate_every` steps, the EMA model's loss on the held-out pixels is computed at fixed
noise levels and fixed draws. Once the warmup and the EMA's horizon 1 / (1 - decay) have passed,
training stops when that loss has not improved for `patience` steps. The model kept is the EMA
state at the last multiple of `checkpoint_every` steps at or before the lowest held-out loss. With
`patience` null, as for Experiment 2 and Pavia University in Experiment 3, training runs every
step and keeps the final EMA state.

The fold seeds the initialization and every draw, so a run is a function of the scene, the
recipe, the learning rate and the fold, up to the device's arithmetic.
"""

import time

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from dartsdet.model import DARTS, pixels, vp
from dartsdet.whitening import Inputs


def held_out(usable: torch.Tensor, fold: int, folds: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(held-out, kept) pixel indices: fold `fold` of `folds` of the usable pixels (P,) bool, in
    one fixed random order drawn with seed 0."""
    rows = torch.nonzero(usable).squeeze(1)
    order = rows[torch.randperm(len(rows), generator=torch.Generator().manual_seed(0))]
    chosen = torch.zeros(len(rows), dtype=torch.bool)
    chosen[fold % folds::folds] = True
    return order[chosen], order[~chosen]


def crops(images: list[torch.Tensor], size: int, count: int,
          generator: torch.Generator) -> list[torch.Tensor]:
    """`count` random crops of side `size` (at most the image's) of each of `images`
    (1, ., W, H), all of them cut at the same window and dihedral transform (k >= 4: transposed,
    then k % 4 quarter turns); concatenated, (count, ., w, h). A non-square crop keeps its shape:
    identity or half turn."""
    W, H = images[0].shape[2:]
    w, h = min(size, W), min(size, H)
    out = []
    for _ in range(count):
        r0 = int(torch.randint(W - w + 1, (1,), generator=generator))
        c0 = int(torch.randint(H - h + 1, (1,), generator=generator))
        k = int(torch.randint(8 if w == h else 4, (1,), generator=generator))
        k = k if w == h else 2 * (k % 2)
        windows = [image[:, :, r0:r0 + w, c0:c0 + h] for image in images]
        out.append([torch.rot90(v.transpose(-1, -2) if k >= 4 else v, k % 4, (-2, -1))
                    for v in windows])
    return [torch.cat(parts) for parts in zip(*out, strict=True)]


def image_epsilon(model: DARTS, u: torch.Tensor, image: torch.Tensor, mask: torch.Tensor,
                  sigma: torch.Tensor, create_graph: bool) -> torch.Tensor:
    """eps_hat (N, B, W, H) at every pixel of the inputs u (N, B, W, H), each context read from
    the clean a z `image` (N, B, W, H) at the pixels of `mask` (N, 1, W, H); sigma (N,), one level
    per image."""
    n, b, w, h = u.shape
    with torch.set_grad_enabled(create_graph):
        c = pixels(model.contexts(image, mask, sigma.log()))
    eps_hat = model.epsilon(pixels(u), c, sigma.repeat_interleave(w * h), create_graph)
    return eps_hat.reshape(n, w, h, b).permute(0, 3, 1, 2)


def dsm_error(eps_hat: torch.Tensor, noise: torch.Tensor, a: float | torch.Tensor,
              weight: torch.Tensor) -> torch.Tensor:
    """Eq. (dsm-loss): |eps_hat - eps|^2 / (B a^2) averaged over the pixels of `weight`
    (N, 1, W, H); eps_hat and noise (N, B, W, H)."""
    error = ((eps_hat - noise) ** 2).mean(1, keepdim=True) / a**2
    return (error * weight).sum() / weight.sum()


def validation_loss(model: DARTS, z: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor,
                    sigmas: list[float], noises: list[torch.Tensor]) -> float:
    """The mean over `sigmas` of `dsm_error` on the pixels of `weight` (1, 1, W, H), the whole
    image z (1, B, W, H) noised with the fixed draws `noises`."""
    losses = []
    for sigma, noise in zip(sigmas, noises, strict=True):
        a, s = vp(sigma)
        noised = (a * z + s * noise) * mask
        level = torch.full((1,), sigma, device=z.device)
        eps_hat = image_epsilon(model, noised, a * z * mask, mask, level, False)
        losses.append(float(dsm_error(eps_hat, noise, a, weight)))
    return sum(losses) / len(losses)


def setup(inputs: Inputs, recipe: dict, fold: int, device: str) -> tuple:
    """(untrained model, standardized image z (1, B, W, H), mask (1, 1, W, H), training and
    held-out pixel weights (1, 1, W, H) each, the run's generator, the fixed validation noises) of
    fold `fold`. The standardization is that of the training pixels."""
    spec = recipe["training"]
    val, kept = held_out(inputs.usable, fold, spec["folds"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fold)
        model = DARTS(inputs.fx.shape[1], recipe["model"])
    x = inputs.fx[kept].double()
    model.mean.copy_(x.mean(dim=0))
    model.std.copy_(x.std(dim=0, correction=0))
    model.to(device)
    z, mask = model.image(inputs.fx, inputs.usable, inputs.grid)
    weights = []
    for rows in (kept, val):
        weight = torch.zeros(len(inputs.usable), device=device)
        weight[rows.to(device)] = 1
        weights.append(weight.reshape(inputs.grid)[None, None])
    generator = torch.Generator().manual_seed(fold)
    noises = [torch.randn(z.shape, generator=generator).to(device)
              for _ in spec["validation_sigmas"]]
    return model, z, mask, weights, generator, noises


def train(inputs: Inputs, recipe: dict, lr: float, fold: int, device: str) -> tuple[DARTS, dict]:
    """(the model kept, its record: `step`, `best_step`, `held_out_loss`, `stopped`) of fold
    `fold`, trained on the usable pixels of `inputs` with the recipe (configs/darts.json) at
    learning rate `lr`. If training ends before any held-out loss is recorded, the final EMA."""
    spec = recipe["training"]
    model, z, mask, (centres, held), generator, noises = setup(inputs, recipe, fold, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=spec["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / spec["warmup"]))
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(spec["ema"]), use_buffers=True)
    counted_from = spec["warmup"] + round(1 / (1 - spec["ema"]))
    best, best_step, chosen, snapshot, window, start = float("inf"), 0, None, None, [], time.time()
    for step in range(1, spec["steps"] + 1):
        clean, m, weight = crops([z, mask, centres], spec["crop"], spec["crops"], generator)
        low, high = spec["sigma_min"], spec["sigma_max"]
        sigma = (low * (high / low) ** torch.rand(spec["crops"], generator=generator)).to(z.device)
        noise = torch.randn(clean.shape, generator=generator).to(z.device)
        a, s = vp(sigma[:, None, None, None])
        eps_hat = image_epsilon(model, (a * clean + s * noise) * m, a * clean * m, m, sigma, True)
        loss = dsm_error(eps_hat, noise, a, weight)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec["clip"])
        optimizer.step()
        scheduler.step()
        ema.update_parameters(model)
        window.append(float(loss.detach()))
        if step % spec["validate_every"]:
            continue
        current = validation_loss(ema.module, z, mask, held, spec["validation_sigmas"], noises)
        if step % spec["checkpoint_every"] == 0:
            snapshot = {k: v.detach().clone() for k, v in ema.module.state_dict().items()}
        print(f"[train] fold {fold} step {step} train {sum(window) / len(window):.6f} held-out "
              f"{current:.6f} best {best_step} ({time.time() - start:.0f} s)", flush=True)
        window = []
        if step >= counted_from and current < best:
            best, best_step, chosen = current, step, snapshot
        elif (spec["patience"] is not None and step >= counted_from
              and step - best_step >= spec["patience"]):
            break
    if spec["patience"] is None:
        chosen = None
    model.load_state_dict(ema.module.state_dict() if chosen is None else chosen)
    every = spec["checkpoint_every"]
    record = {"step": step if chosen is None else best_step // every * every,
              "best_step": best_step, "held_out_loss": best, "stopped": step}
    return model.eval(), record
