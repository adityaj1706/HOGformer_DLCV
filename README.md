# Soft-Voting HOG Conditioning for HOGformer (Team Kernel Crew)

Course project for *Deep Learning for Computer Vision* (DA 6021). We reproduce **HOGformer**
(Wu et al., *Gradient as Conditions: Rethinking HOG for All-in-one Image Restoration*, AAAI 2026,
[arXiv 2504.09377](https://arxiv.org/abs/2504.09377), upstream code
[Fire-friend/HOGformer](https://github.com/Fire-friend/HOGformer)) under a reduced compute budget and test
one modification of its HOG descriptor. The original README is kept as [`README_upstream.md`](README_upstream.md).

> **Headline result (honest summary).** Soft voting did **not** give a consistent improvement: it helped on Snow100K-L
> (+0.32 dB PSNR) but was slightly worse on Test1 (-0.28 dB) and RainDrop (-0.09 dB) and indistinguishable on Snow100K-S;
> the mean change over the four test sets is -0.03 dB. Details and caveats are in section 6.

---

## 1. Project overview

**HOGformer** conditions its self-attention on a Histogram-of-Oriented-Gradients (HOG) descriptor computed per
8x8 patch of the feature map (9 orientation bins). In the original code every pixel is put into **exactly one**
orientation bin (a hard integer decision).

**Our modification: soft orientation voting.** Each pixel splits its gradient magnitude between its two nearest
orientation bins with linear weights. Only `Attention_DHOGSA.apply_hog_to_patch` changes; there are no new
parameters (16.65 M for both models), and the pixel-sorting step keeps the original hard rule.

Why we expect it to help (a hypothesis we test, not an assumption):

* Hard binning has zero gradient with respect to the orientation and makes the histogram jump by a pixel's full
  magnitude when its orientation crosses a bin edge. Soft voting makes the histogram continuous
  (`|dH/dtheta| <= (B/2pi) m/|P|`) and lets gradient flow through the orientation (bounded by `1 + B/2pi`).
* Verified numerically in [`settingI/tests/test_soft_vote.py`](settingI/tests/test_soft_vote.py): sweeping the
  orientation of a synthetic edge, the largest histogram change between neighbouring angles is **2.000 (hard) vs
  0.005 (soft)**, and `|dLoss/dtheta|` is **0 (hard) vs 1.432 = B/2pi (soft)**.
* With the flag off the model is **bit-identical** to the original code (checked: max output difference 0.0).

The one-page report is in [`report/main.tex`](report/main.tex).

**Important scope notes (deviations from the paper):**

* Training is **shortened to 12,000 iterations** on **64x64 crops, batch 8** (the released config uses 300k
  iterations with progressive patch sizes up to 400x400) because we only had free Colab/Kaggle T4 GPUs. Absolute
  PSNR is therefore far below the paper's. We compare *our* baseline against *our* modified model, trained
  under identical conditions; we do not compare with the paper's table.
* The baseline's training was resumed once from its iteration-10,000 checkpoint (Colab -> Kaggle).
* Only Setting I (all-weather: Snow100K / Outdoor-Rain / RainDrop) was run. `settingIII&IV/` is unchanged upstream code.

## 2. Environment and dependencies

* **Hardware used:** NVIDIA T4 (16 GB) on Google Colab (baseline, iterations 0-10k) and Kaggle (T4 x2, only one GPU
  used; baseline 10k-12k, proposed model, evaluation). Training needs about 9.4 GB of GPU memory at batch 8 / 64x64.
  About 0.9-1 s per iteration; the proposed run took 3 h 03 min of training (3:03:02 in its log) plus 4 RainDrop validations of about 2.6 min each.
* **Software:** PyTorch 2.11.0 (CUDA 13.0 build) and TorchVision 0.26.0, as recorded in both training logs (Colab and Kaggle used the same
  versions), BasicSR 1.2.0, plus the other packages in [`settingI/requirements_colab.txt`](settingI/requirements_colab.txt)
  (not pinned; Python 3.10+). `torch`/`torchvision` are intentionally not in that file: install the CUDA build for your machine.
  The upstream `requirements.txt` is UTF-16 and lists Windows-only `pywin32` and `dlib`; do not use it on Linux.

```bash
git clone https://github.com/adityaj1706/HOGformer_DLCV.git
cd HOGformer_DLCV
pip install torch torchvision                 # skip on Colab/Kaggle (preinstalled)
pip install -r settingI/requirements_colab.txt
```

No CUDA extension needs to be built (we run from the source tree: `cd settingI` first).

## 3. Dataset preparation

Setting I uses the **All-Weather** composite training set (from TransWeather/Histoformer) and four test sets:

| Split | Content | Pairs |
|---|---|---|
| train | All-Weather composite (Snow100K 9,001 / Outdoor-Rain 8,250 / RainDrop 818, by filename) | 18,069 |
| test | Outdoor-Rain **Test1** | 750 |
| test | **RainDrop** test_a | 58 |
| test | **Snow100K-S** / **Snow100K-L** | 16,611 / 16,801 (we score a fixed random subset, see section 5) |

One command downloads and arranges everything (about 35 GB of temporary disk, about 15 GB at the end):

```bash
bash scripts/download_data.sh ./data          # -> ./data/weather/{train,test}/...
```

Options (environment variables): `SKIP_SNOW=1` (skip the Snow100K test sets, not needed for training),
`SKIP_TRAIN=1` (evaluation only, skip the 11 GB training set), `ALLWEATHER_ZIP=/path/allweather.zip` (use a local copy),
`ALLWEATHER_ID=<Google Drive file id>` (use your own shared copy). **Google Drive may refuse the shared links with
"Too many users have viewed or downloaded this file"**; in that case copy the file to your own Drive and use one of the last two options.

What the script does besides downloading: it extracts the archives, renames RainDrop `data/` to `input/` and
`N_clean.png` to `N_rain.png` (the data loader pairs files by identical name), and checks the file counts and name matching.
Expected layout:

```
data/weather/
  train/{gt,input}/                              18069 pairs
  test/Test1/{gt,input}/                         750
  test/RainDrop/test_a/{gt,input}/               58
  test/Snow100K-Test/Snow100K-{S,L}/{gt,synthetic}/
```

The configs read the data from `/content/data/weather/`. To use another location either symlink it
(`sudo mkdir -p /content && sudo ln -s "$(pwd)/data" /content/data`) or edit the `dataroot_gt` / `dataroot_lq` lines of the YAML.

## 4. Training instructions

Run from `settingI/`. Both models use the same data, seed (100), schedule and losses (L1 + Pearson + HOG loss);
**the only difference between the two YAML files is the run name and `soft_vote: true`.**

```bash
cd settingI
# Baseline (original hard voting)
python basicsr/train.py -opt Allweather/Options/Allweather_HOGformer_baseline.yml --launcher none
# Proposed (soft voting)
python basicsr/train.py -opt Allweather/Options/Allweather_HOGformer_softvote.yml --launcher none
```

Key hyperparameters (identical for both): 12,000 iterations; 64x64 random crops with flips/rotations; batch 8;
AdamW (lr 3e-4, weight decay 1e-4, betas 0.9/0.999) with gradient clipping; learning rate held at 3e-4 for 4,000
iterations, then cosine decay to 1e-6 over 8,000; network dim 36, blocks [4,4,6,8], heads [1,2,4,8]; validation on RainDrop
every 3,000 iterations; checkpoint every 2,000.
`use_checkpoint: true` (activation checkpointing, same math, less memory) is available if the GPU has less than about 10 GB.

* **Resuming:** rerunning the same command resumes automatically from the newest `experiments/<name>/training_states/<N>.state`
  (it needs the matching `models/net_g_<N>.pth` too).
* **Pretrained weights:** none are used; both models train from scratch. (The authors' weights are only on BaiduYun.)
* **Notebooks:** [`colab/hogformer_baseline_colab.ipynb`](colab/hogformer_baseline_colab.ipynb),
  [`kaggle/hogformer_kaggle.ipynb`](kaggle/hogformer_kaggle.ipynb) (baseline) and
  [`kaggle/hogformer_kaggle_softvote.ipynb`](kaggle/hogformer_kaggle_softvote.ipynb) (proposed). For unattended Kaggle runs use
  *Save & Run All (Commit)*; an idle interactive session is shut down by Kaggle.
* Training logs go to `experiments/<name>/train_*.log` and a heartbeat (speed, memory, ETA, last PSNR) to `experiments/<name>/progress.txt`.

## 5. Inference and evaluation

[`settingI/evaluate.py`](settingI/evaluate.py) scores **every** image of Test1 and RainDrop and a **fixed random subset**
(seed 0, default 300) of Snow100K-S/-L, with PSNR/SSIM on the Y channel (crop 0), reflect-padding to a multiple of 16, plus
parameters, FLOPs (fvcore, 128x128; not every op is counted) and latency (one 480x720 image). It saves per-image scores.

```bash
cd settingI
python evaluate.py --config Allweather/Options/Allweather_HOGformer_baseline.yml \
    --weights ../checkpoints/baseline_net_g_latest.pth --out results_baseline.json --name baseline --data_root ../data/weather
python evaluate.py --config Allweather/Options/Allweather_HOGformer_softvote.yml \
    --weights ../checkpoints/softvote_net_g_latest.pth --out results_softvote.json --name softvote --data_root ../data/weather
python compare_results.py results_baseline.json results_softvote.json --md comparison.md
```

`compare_results.py` prints the per-dataset PSNR difference (proposed - baseline) with a 95 % paired-bootstrap confidence interval and the
share of images that improved. Reproduce everything on Kaggle with [`kaggle/hogformer_eval_kaggle.ipynb`](kaggle/hogformer_eval_kaggle.ipynb)
(about 1-2 h on a T4; inference is about 2.6 s per 480x720 image).
Use `--snow_n N` to change the Snow100K subset size and `--datasets test1 raindrop` to restrict the datasets.
We evaluate the **last** checkpoint (`net_g_latest.pth`, iteration 12,000), not the best-on-test one.

**Note:** the training-time validation (`basicsr/models/image_restoration_model.py`) only scores the first 60 images of a set; this is
harmless for RainDrop (58 images) but is why `evaluate.py` exists. Unit checks of the modification:
`cd settingI && PYTHONPATH=. python tests/test_soft_vote.py`.

## 6. Checkpoints and results

**Checkpoints, configs and logs** (public, no login needed):

| Item | Link |
|---|---|
| Baseline `net_g_latest.pth` (67.2 MB) | [download](https://github.com/adityaj1706/HOGformer_DLCV/releases/download/v1.0/baseline_net_g_latest.pth), SHA-256 `6a4114c5a561d40b0506082d709db54fb22ce4b5c6b963e6e508df5bd5b2d2e6` |
| Proposed (soft voting) `net_g_latest.pth` (67.2 MB) | [download](https://github.com/adityaj1706/HOGformer_DLCV/releases/download/v1.0/softvote_net_g_latest.pth), SHA-256 `2705299124773bba3e5e9048c4fde32af62a43d39001b6d3a573d8699ce26934` |
| Training log, proposed (iterations 0-12,000, Kaggle T4) | [`results/logs/train_softvote_kaggle_iter0-12000_20261104_231803.log`](results/logs/train_softvote_kaggle_iter0-12000_20261104_231803.log) |
| Training log, baseline, iterations 0-11,290 (Colab T4) | [`results/logs/train_baseline_colab_iter0-11290_20261104_130937.log`](results/logs/train_baseline_colab_iter0-11290_20261104_130937.log) |
| Training log, baseline, iterations 10,000-12,000 (Kaggle resume) | not included; the final baseline result is in `results/results_baseline.json` and reproducible with the released checkpoint |
| Evaluation log and per-image results | [`results/logs/eval_stdout.log`](results/logs/eval_stdout.log), [`results/results_baseline.json`](results/results_baseline.json), [`results/results_softvote.json`](results/results_softvote.json) |
| Configs | [`baseline`](settingI/Allweather/Options/Allweather_HOGformer_baseline.yml), [`softvote`](settingI/Allweather/Options/Allweather_HOGformer_softvote.yml) |

Download and check the released weights, then evaluate them (about 1-2 h on a T4, see section 5):

```bash
mkdir -p checkpoints && cd checkpoints
wget https://github.com/adityaj1706/HOGformer_DLCV/releases/download/v1.0/baseline_net_g_latest.pth
wget https://github.com/adityaj1706/HOGformer_DLCV/releases/download/v1.0/softvote_net_g_latest.pth
sha256sum *.pth        # must match the table above
cd ../settingI         # then run the commands of section 5 with --weights ../checkpoints/<name>_net_g_latest.pth
```

Sanity check we ran: scoring one full-size RainDrop image with each *downloaded* file reproduced the per-image PSNR stored in
`results/*.json` to within 0.02 dB (CPU vs GPU arithmetic).

**Training-time RainDrop validation PSNR (Y channel, 58 images, from the training logs):**

| Iteration | Baseline | Soft voting |
|---|---|---|
| 3,000 | 25.81 | 25.83 |
| 6,000 | 25.51 | 25.66 |
| 9,000 | 26.43 | 26.31 |
| 12,000 (final) | 26.72 | 26.63 |

**Final evaluation** (`evaluate.py`, last checkpoint, PSNR / SSIM on the Y channel; raw table in
[`results/comparison.md`](results/comparison.md)):

| Method | Test1 (750) | RainDrop (58) | Snow-S (300) | Snow-L (300) | Params | Latency (T4, 480x720) |
|---|---|---|---|---|---|---|
| Baseline (hard voting) | **21.80** / 0.8067 | **26.72** / **0.8866** | **32.52** / 0.9259 | 26.70 / 0.8510 | 16.65 M | 2626 ms |
| Soft voting (ours) | 21.52 / **0.8109** | 26.63 / 0.8863 | 32.47 / **0.9261** | **27.02** / **0.8538** | 16.65 M | **2543 ms** |

Paired per-image PSNR difference, proposed minus baseline (95 % bootstrap CI over images):

| Dataset | n | dPSNR (95 % CI) | Images improved |
|---|---|---|---|
| Test1 | 750 | -0.281 [-0.334, -0.227] | 29 % |
| RainDrop | 58 | -0.094 [-0.161, -0.026] | 36 % |
| Snow-S (subset) | 300 | -0.058 [-0.134, +0.023] | 41 % |
| Snow-L (subset) | 300 | +0.322 [+0.269, +0.374] | 78 % |

**Interpretation.** There is no consistent improvement: the mean of the four PSNR deltas is -0.03 dB. Soft voting helps on dense snow
(Snow-L) and hurts on rain/haze (Test1) and raindrops; SSIM changes by at most +0.004. Parameters are identical, and the measured latency is 3 % lower,
a difference we did not test against timing noise. The numerical checks in section 1 show the descriptor behaves as designed, so the end metric is
sensitive to the change but not uniformly positive. A plausible but **untested** explanation: rain streaks give a sharply peaked orientation histogram that hard
bins already encode stably, and soft voting spreads that peak over two bins; dispersed snow edges may benefit from the smoothing.
**Caveats:** one training run per model (the intervals capture image sampling, not training-seed variance, so differences of a few tenths of a dB may partly be run-to-run noise),
a 12k-iteration schedule, and 300-image Snow subsets. FLOPs counted by fvcore (128x128) are 28.86 GMACs for both models (not every op is counted).

## 7. Code organization and reproducibility

Files we added or changed (everything else is upstream HOGformer / BasicSR code):

| File | Purpose |
|---|---|
| `settingI/basicsr/models/archs/hogformer_arch.py` | **The modification:** `soft_vote` flag in `Attention_DHOGSA.apply_hog_to_patch` (off by default). Also an optional `use_checkpoint` flag (activation checkpointing, identical numerics). |
| `settingI/Allweather/Options/Allweather_HOGformer_{baseline,softvote}.yml` | Single-GPU, shortened-schedule configs (differ only in `soft_vote`). |
| `settingI/evaluate.py`, `settingI/compare_results.py` | Full-image evaluation and paired comparison. |
| `settingI/tests/test_soft_vote.py` | Checks of the continuity and gradient-flow claims. |
| `settingI/basicsr/train.py`, `settingI/basicsr/utils/logger.py` | Logging only: removed duplicated log lines, heartbeat, validation timing. No effect on training. |
| `scripts/download_data.sh`, `settingI/requirements_colab.txt` | Dataset download/arrangement and a Linux dependency list. |
| `colab/`, `kaggle/`, `report/` | Notebooks used for the runs and the one-page report source. |

Reproducibility notes: both runs use `manual_seed: 100` and the same data order. GPU kernels (and `cudnn.benchmark`) are not
bit-deterministic, so a rerun will differ slightly; one training run per model is a limitation (the confidence intervals above
capture image sampling, not training-seed variation). The shortened schedule means the conclusions apply to this short-training regime only.

## License and acknowledgments

MIT (see `LICENSE`, upstream copyright retained). Built on HOGformer, [BasicSR](https://github.com/xinntao/BasicSR),
[Histoformer](https://github.com/sunshangquan/Histoformer), [DiffUIR](https://github.com/iSEE-Laboratory/DiffUIR) and
[AdaIR](https://github.com/c-yn/AdaIR).
