# DARTS

Code for *From Diffusion Models to Score Tests: Detecting Weak Signals in Non-Gaussian Noise*.

DARTS detects a target of known signature in a hyperspectral scene with the Rao test. For weak
additive targets this test is locally most powerful, and it needs the background only through its
score function. DARTS learns that score from the scene itself, by denoising score matching,
conditioned on each pixel's spatial neighbourhood:

```
A(x) = -t' s_sigma(x | c)
```

Here `t` is the unit signature, `c` the pixel's context, and `s_sigma` the score of the background
smoothed at noise level `sigma`. The context comes from a blind-spot convolutional encoder, which
never sees the pixel itself or its 3 x 3 neighbourhood. One noise level per scene is chosen without
labels (Section VII of the paper, `dartsdet/noise_level.py`). The detector is an ensemble of five
models, each trained with one fold of the pixels held out.

## Installation

The paper's numbers were produced with Python 3.11.16 and the versions in `requirements.txt`.

```bash
pip install -r requirements.txt
pip install -e ".[figures,test]"
```

The library needs numpy, scipy, scikit-learn and PyTorch; the figures also need matplotlib, and
the tests pytest.

## Data

The scenes are not redistributed here. Place them under `data/`, or pass `--data DIR`, and check
them with `shasum -a 256 -c SHA256SUMS`.

| Scene | Files under `data/` | Source |
|---|---|---|
| San Diego | `Sandiego.mat` (`data` 100 x 100 x 189, `map`) | `datasets/Sandiego.mat` in the HTD-Mamba repository, github.com/shendb2022/HTD-Mamba |
| ABU-beach-4 | `abu-beach-4.mat` (`data` 150 x 150 x 102, `map`) | `datasets/abu-beach-4.mat` in the same repository |
| ABU-airport-2 (Experiment 1) | `abu-airport-2.mat` (`data` 100 x 100 x 205, `map`) | `datasets/abu-airport-2.mat` in the same repository |
| Pavia University | `pavia_university/PaviaU.mat`, `pavia_university/PaviaU_gt.mat` | the Pavia University scene and its ground truth (M. Graña, M. A. Veganzones and B. Ayerdi, "Hyperspectral Remote Sensing Scenes – Pavia University (ROSIS, data courtesy of Prof. Paolo Gamba, Univ. of Pavia)," Grupo de Inteligencia Computacional (GIC), UPV/EHU, 2011; https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes); the checksums identify the files used |
| MUUFL Gulfport | `MUUFLGulfport/MUUFLGulfportDataCollection/muufl_gulfport_campus_w_lidar_1.mat`, `.../tgt_lab_spectra.mat` | the MUUFL Gulfport data collection (P. Gader et al., Tech. Rep. REP-2013-570, University of Florida, 2013; doi:10.5281/zenodo.1186326) |

The MUUFL target halos (`dartsdet/muufl.py`) port Bullwinkle's scoring windows: T. Glenn, A. Zare,
P. Gader, D. Dranishnikov. (2016). Bullwinkle: Scoring Code for Sub-pixel Targets (Version 1.0)
[Software]. GatorSense/MUUFLGulfport.

San Diego and ABU-beach-4 have real targets. Pavia University and MUUFL receive implanted ones:
the mean spectrum of class 9 at amplitude 0.75 in Pavia University, whose 59 pixels clipped at
8000 DN are left out, and lab signature 0 (Soil Brown) at amplitude 0.1 in the first 220 columns
of MUUFL campus 1. The implants follow one protocol (`dartsdet/scenes.py`): one implant per 100
scored background pixels, at least 3 pixels apart, with the sites drawn from seed 0. Pixels of the
implanted material's real instances are not scored.

## Reproducing the paper

Run from this directory. `--help` lists each command's options and defaults.

| Result | Command | Output |
|---|---|---|
| Fig. 1 (Experiment 1) | `python -m experiments.experiment1` | `results/experiment1/runs.csv` |
| | `python -m figures.figure1` | `results/figure1.pdf` |
| Fig. 2 (Experiment 2) | `python -m experiments.experiment2` | `results/experiment2/` |
| | `python -m figures.figure2` | `results/figure2.pdf` |
| Fig. 3 (Experiment 3) | `python -m experiments.experiment3` | `results/experiment3/seed<s>.csv` |
| | `python -m figures.figure3` | `results/figure3.pdf` |
| Fig. 4 (Experiment 4) | `python -m experiments.experiment4 pavia`, then `muufl` | `results/experiment4/<background>.csv` |
| | `python -m figures.figure4` | `results/figure4.pdf` |
| Fig. 5 (Experiment 5) | `python -m experiments.experiment5` | `results/experiment5/curves.npz` |
| | `python -m figures.figure5` | `results/figure5.pdf` |
| Tables I and II, DARTS rows (Experiment 6) | `python -m experiments.experiment6` | `results/experiment6/<scene>.json` |

Experiment 1 uses the real-target scenes and MUUFL campus 1 with its own cloth targets, without
implants. In Experiment 2, panel (a) of Fig. 2 is computed exactly; the networks behind panels (b)
and (c) are seeded here, while the paper's were initialized from an unseeded generator, so a rerun
repeats exactly but does not regenerate the paper's values of those two panels.

Each Experiment 6 file holds `sigma`, the noise level chosen, and the AUC, PD at a false-alarm rate
of 1e-3 and standardized partial AUC up to that rate under `labels`. For MUUFL it also holds them
under `masked`, which leaves out the pixels around every emplaced cloth panel; Table II reports
those. From the paper's fold models, Experiment 6 gives:

| Scene | sigma | AUC | PD at 1e-3 | partial AUC |
|---|---|---|---|---|
| San Diego | 0.1651 | 0.9995 | 0.914 | 0.847 |
| ABU-beach-4 | 0.1033 | 0.9435 | 0.588 | 0.757 |
| Pavia University | 0.0866 | 0.9979 | 0.875 | 0.858 |
| MUUFL campus 1 (`masked`) | 0.1052 | 0.9989 | 0.956 | 0.923 |

### Models and run time

A fold model is trained the first time it is needed, and loaded from `--checkpoints` afterwards:

```
checkpoints/experiment3/seed<s>/{darts,dart}.pt
checkpoints/experiment4/<background>/fold0.pt
checkpoints/experiment5/n<n>_s<seed>.pt
checkpoints/experiment6/<scene>/fold<j>.pt
```

A saved model is refused unless it was trained with the current recipe and seed, and a DARTS
fold model also unless it was trained as the same fold on the same pixels.

The paper's trained models (44 files, 67 MiB) are not part of this repository. With them in place,
the experiments reproduce the paper's numbers as described under Checks. Without them, the
experiments train their own. Experiment 5's models then come out identical to the paper's, which
were trained on the CPU. The other models were trained on a GPU with PyTorch's MPS backend, so
retrained ones agree with them only up to that arithmetic; how far the reported numbers then move
has not been measured.

Training is best done on a GPU. A Pavia University fold took about an hour on an Apple M4 Pro GPU.
After training, Experiment 6 takes on a CPU about 45 s for San Diego, 90 s for ABU-beach-4, 4 min
for MUUFL and 13 min for Pavia University. Experiment 4 computes 65 GLRTs, about 95 s each for
Pavia University and 35 s for MUUFL on the same GPU, and several times longer on a CPU.
Experiments 1, 2 and 5 run on the CPU: Experiment 5, training included, takes about 25 minutes,
and Experiments 1 and 2 each took under an hour while sharing the CPU with other jobs.

Every draw is seeded: implant sites, folds, model initialization, training batches, and the
rule's pixel samples and noise. On the CPU a rerun with the same number of threads repeats bit for
bit; Experiments 2 and 5 fix theirs (8 and 4), since the thread count moves float reductions. GPU
kernels are not guaranteed to be deterministic, so on a GPU a retrained model can differ in the
last bits, and early stopping can turn that into a different stopping step.

## Using DARTS on your own scene

```python
import json
from pathlib import Path

from dartsdet import evaluation, noise_level, pipeline, preprocessing
from dartsdet.scenes import Scene

# cube (H, W, B); labels, exists (H W,) in Fortran pixel order; target (B,); see dartsdet/scenes.py
scene = Scene(cube, labels, exists, target, additive=False)
inputs = preprocessing.prepare(scene)
spec = json.loads(Path("configs/darts.json").read_text())
sigmas = noise_level.grid(spec["sigma_max"])
models = [pipeline.fold_model(inputs, spec, j, "cuda", Path(f"checkpoints/mine/fold{j}.pt"))
          for j in range(spec["train"]["folds"])]
reads = [pipeline.rule_reads(m, inputs, scene.exists, j, sigmas, "cuda")
         for j, m in enumerate(models)]
sigma = pipeline.sigma_hat(reads, sigmas)
score = evaluation.ensemble([pipeline.statistic(m, inputs, sigma, "cuda") for m in models],
                            scene.exists)
detection_map = score.reshape(cube.shape[:2], order="F")
```

Labels are only needed for `evaluation.detection_metrics`; without them, pass zeros.

## Checks

```bash
pytest
```

Every read is tested against a Gaussian background, where it has a closed form. The Rao statistic
is the adaptive matched filter of the smoothed Gaussian, the curvature is the projected precision,
the log ratio is quadratic, and the GLRT is the squared one-sided AMF. The other tests cover:

- the encoder's blind spot;
- the folds and a pinned short training run;
- the implant protocol and the MUUFL halos;
- each part of the noise-level rule;
- the metrics, including the AUC convention;
- the per-scene pipeline end to end;
- the experiments' own models: the Gaussian and Student-t mixtures against their closed forms,
  the synthetic scene's exact detectors in the Gaussian limit, and each of Experiment 2's routes
  to a log ratio with exact networks.

The package was also checked on a CPU against the research code that produced the paper's
numbers.

- **From the paper's checkpoints:** it reproduces every DARTS entry of Tables I and II, and the
  Experiment 3 and 4 entries checked, to the printed digit, with the same noise levels to the
  last bit. One AUC differs in the ninth decimal, from the CPU's arithmetic against the GPU's.
- **Experiment 5:** retrained from scratch on the CPU, it reproduces every stored curve, noise
  level and PD exactly.
- **Experiment 1:** rerun in full, it reproduces every stored line exactly.
- **Figures:** drawn from the stored results, or from the reruns of Experiments 1 and 5, Figs. 1
  to 5 match the paper's.
- **Training:** on the CPU, `dartsdet.training.fit` and Experiment 5's fits match the originals
  bit for bit.

## Layout

| Module | Contents |
|---|---|
| `dartsdet/scenes.py`, `dartsdet/muufl.py` | scenes, labels (Fortran pixel order) and the implant protocol |
| `dartsdet/preprocessing.py` | Ledoit-Wolf ZCA whitening, the signature, the one-sided AMF and the screen |
| `dartsdet/encoder.py`, `network.py`, `model.py` | the blind-spot context encoder, the noise-conditional head, the score models |
| `dartsdet/training.py` | five-fold denoising score matching with EMA and early stopping |
| `dartsdet/reads.py` | the Rao statistic, its curvature, the log-likelihood ratio and the GLRT |
| `dartsdet/noise_level.py` | the noise-level rule |
| `dartsdet/pipeline.py` | the per-scene steps the experiments share |
| `dartsdet/evaluation.py` | the fold ensemble and the metrics |
| `experiments/` | one driver per experiment, and what only one experiment uses: Experiment 1's mixtures and calibration test, Experiment 2's networks and routes, Experiment 3's scene and DART, Experiment 5's background |
| `figures/` | the figures, drawn from the experiments' outputs |
| `configs/` | the DARTS recipe, the scenes, and Experiments 1 and 3's settings |

## Baselines

The comparison detectors in Tables I and II were run with their authors' released code, with the
settings described in the paper. That code is not included here.
