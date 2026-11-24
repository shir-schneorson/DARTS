# DARTS

Code for *From Diffusion Models to Score Tests: Target Detection under General Background
Distributions*.

The paper detects a known signature `t` of unknown amplitude in a non-Gaussian background with the
Rao (locally most powerful) test. The test needs the background only through its score function,
and DARTS learns that score from the scene itself by denoising score matching. A blind-spot
encoder conditions each pixel's score on its spatial neighbourhood, so the test reads

```
U_sigma(x | c) = -t' s_sigma(x | c)
```

with `s_sigma` the score of the background smoothed at noise level `sigma`, and `c` the pixel's
context. The noise level is not a nuisance: smoothing tunes the Rao test to a target amplitude. A
two-stage rule chooses it without labels. It first estimates the amplitude from how the scene's
screened pixels respond across levels, then picks the level at which held-out pixels with
implants of that amplitude are detected best (`dartsdet/noise_level.py`).

## Install

The paper's numbers were produced with Python 3.11.16 and the versions in `requirements.txt` on an
Apple M4 Pro: the statistics on its CPU, training and Table II's DARTS timings on its GPU, through
PyTorch's MPS backend.

```bash
pip install -r requirements.txt
pip install -e .
```

## Data

The scenes are not redistributed. From this directory, run

```bash
python -m dartsdet.fetch
```

This downloads San Diego and ABU-beach-4 from the HTD-Mamba repository, and MUUFL Gulfport from
the GatorSense repository, into `../data/`. Each file is checked against `SHA256SUMS`. The source of
Pavia University refuses scripted downloads, so put `PaviaU.mat` and `PaviaU_gt.mat` in
`../data/pavia_university/` by hand. The page is listed below, and a second run of the command
checks them.

| Scene | Files under `data/` | Source |
|---|---|---|
| San Diego | `Sandiego.mat` | `datasets/` of github.com/shendb2022/HTD-Mamba |
| ABU-beach-4 | `abu-beach-4.mat` | the same repository; the scene is from X. Kang, X. Zhang, S. Li, K. Li, J. Li and J. A. Benediktsson, "Hyperspectral Anomaly Detection With Attribute and Edge-Preserving Filters," IEEE Trans. Geosci. Remote Sens., 55(10):5600–5611, 2017 |
| Pavia University | `pavia_university/PaviaU.mat`, `PaviaU_gt.mat` | M. Graña, M. A. Veganzones and B. Ayerdi, "Hyperspectral Remote Sensing Scenes – Pavia University (ROSIS, data courtesy of Prof. Paolo Gamba, Univ. of Pavia)," GIC, UPV/EHU, 2011, https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes |
| MUUFL Gulfport | `MUUFLGulfport/MUUFLGulfportDataCollection/muufl_gulfport_campus_w_lidar_1.mat`, `tgt_lab_spectra.mat` | P. Gader et al., Tech. Rep. REP-2013-570, University of Florida, 2013; github.com/GatorSense/MUUFLGulfport |

The MUUFL scoring windows follow Bullwinkle: T. Glenn, A. Zare, P. Gader and D. Dranishnikov,
GatorSense/MUUFLGulfport, 2016.

## Reproducing the paper

Run from this directory. Each experiment scores the committed checkpoints (`checkpoints/`; check
them with `shasum -a 256 -c checkpoints/SHA256SUMS`, which pytest also does) on the CPU and writes
to `outputs/`. The scripts in `paper/` draw the tables and figures from `results/`, the committed
reference outputs; pass `--results outputs/experimentN` to draw them from your own run instead.
`paper.figure1` also recomputes DARTS's San Diego map, so it needs the data, and it uses the model
of a `--train` run when `--results` holds one.

| Paper | Commands | Time (CPU) |
|---|---|---|
| Table II, Fig. 1 | `python -m experiments.experiment1`, then `python -m paper.table2` and `python -m paper.figure1` | 8 min |
| Fig. 2 | `python -m experiments.experiment2`, then `python -m paper.figure2` | 3 min |
| Fig. 3 | `python -m experiments.experiment3`, then `python -m paper.figure3` | 13 min |
| Table III, Fig. 4 | `python -m experiments.experiment4`, then `python -m paper.table3` and `python -m paper.figure4` | 6 min |

`python -m paper.web` redraws the project page's figures and tables (`site/`) from `results/`.

The experiments hold one scene in memory at a time; the largest run needs about 4 GB of RAM.
Experiment 4 scores the shipped Experiment 1 and 3 models and reads their noise levels from
`results/experiment3/`; it refuses the output of a `--train` run. Each script's `--help` prints its
docstring: what it computes, from which files, and where it writes. Table II's DARTS rows are its
Rao test at the rule's level sigma-hat and, as a reference, its plug-in GLRT at sigma-hat_G, the
level the rule's second stage picks from the GLRT's held-out detection probability:

| Scene | sigma-hat | AUC | PD at PF = 1e-3 | partial AUC | sigma-hat_G | GLRT PD |
|---|---|---|---|---|---|---|
| San Diego | 0.6431 | 0.9987 | 0.897 | 0.843 | 0.1710 | 0.879 |
| ABU-beach-4 | 0.2659 | 0.9332 | 0.647 | 0.778 | 0.0707 | 0.603 |
| Pavia University (class 9, amplitude 0.75) | 0.1100 | 0.9978 | 0.904 | 0.906 | 0.0707 | 0.909 |
| MUUFL campus 1 (signature 0, amplitude 0.1) | 0.4135 | 0.9996 | 0.972 | 0.963 | 0.1710 | 0.974 |

**Stored rows.** Some numbers are stored, not computed:
- the detectors run with their authors' code: LRao, OS-VAE, TSTTD, CS-TTD, HTD-Mamba, E-CEM and
  hCEM. Their Table II rows are in `results/experiment1/baselines.csv`, and the E-CEM and CS-TTD
  maps of Fig. 1 in `sandiego-maps.npz`. Each was run with its authors' own recipe.
- Fig. 3's curves of the DARTS network trained with LRao's objective
  (`results/experiment3/<background>-lrao-objective.csv`). Three of the runs' checkpoints were
  lost, so that column cannot be recomputed.

**Retraining.** `--train` on Experiments 1, 2 and 3 trains the DARTS models anew from
`configs/`, on `--device` (MPS or CUDA by default when available), for at most 20,000 steps. As in
the paper, training stops early on the held-out loss in Experiment 1 and on MUUFL in Experiment 3;
Experiment 2 (4,000 steps) and Pavia University in Experiment 3 run their full length and keep the
final weights. On the M4 Pro's GPU, a model of Experiment 1 or 3 took about two hours, and
Experiment 2's ten small networks about 75 minutes in all. Training is seeded, and so is every draw the experiments make. On the CPU, a rerun with the
same number of threads repeats bit for bit (a test checks it). GPU kernels are not guaranteed to
be deterministic, and early stopping reads a held-out loss computed on the device, so a retrain
on another backend or build can stop at another step and give other weights.

**Notes on reproducing exactly.**
- Times in Table II depend on the machine. As the caption says, DARTS, TSTTD, CS-TTD and
  HTD-Mamba were timed on the M4 Pro's GPU (the package times DARTS on `--device`), the other rows
  on the CPU.
- Table II's DARTS models are the last checkpoint, at a multiple of 500 steps, at or before the
  lowest held-out loss. Pavia University's is the step-12000 checkpoint of a run stopped by hand
  at step 12400, so `--train`, which stops by patience, trains it longer.
- The paper's Fig. 1 was drawn from the San Diego model's step-4900 checkpoint, the package's from
  Table II's step 4500; the false alarms and misses are the same, only the gray shading differs.
- Fig. 4 is set in Arial, like the paper's; without it, matplotlib falls back to DejaVu Sans.
- Table III's factor omega uses the paper's draws: 30,000 background pixels per scene from one
  seeded generator per background (`configs/experiment4.json`).

## Checks

```bash
pytest
```

The statistics are tested against closed forms. With a zero potential, the model's score is
affine. Its Rao statistic is then the Ledoit-Wolf AMF, its GLRT the squared one-sided AMF, its
log ratio quadratic, and its Rao gradient constant. The synthetic scene's oracles are tested in
the Gaussian limit, and the matched noise level in a closed-form case. Further tests cover:
- the encoder's blind spot;
- the rule's stages, including its selection and its GLRT variant (sigma-hat_G) in closed form;
- training, its stopping rule, and that it repeats bit for bit;
- the shipped checkpoints against `checkpoints/SHA256SUMS`;
- Experiment 4's omega, threshold and level choices;
- the implant protocol;
- the metrics;
- the fetch script's checksums.

The package was checked on a CPU against the research code behind the paper's numbers:
- Table II's metric cells;
- Table III;
- every curve of Figs. 2–4;
- Fig. 3, pixel for pixel.

## Layout

| Path | Contents |
|---|---|
| `dartsdet/` | data and implants, whitening and screening, the model, training, the statistics, the noise-level rule, the metrics, the fetch script |
| `baselines/` | the classical detectors and the Gaussian mixture |
| `experiments/` | one driver per experiment; `synthetic.py` is Experiment 2's scene and its exact detectors |
| `paper/` | the tables and figures |
| `configs/` | the DARTS recipe, the scenes, and each experiment's settings |
| `checkpoints/`, `results/` | the trained models and the reference outputs |

## License

MIT; see `LICENSE`.
