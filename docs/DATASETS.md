# Datasets & How to Run

This is the operator's guide for the BitNet jet tagger: where the data lives,
what HDF5 keys each file exposes, and the commands to train / evaluate / export.

All paths below are the **defaults baked into `bnjettag/config.py`** (the
May-2026 merged dataset on russelld's area). Every path is overridable with a
CLI flag — you only need flags if you move the data.

---

## 1. Data locations

Common root:

```
/home/users/russelld/TOOLLIP_TESTS/cmssw-tests/clean_SCRAM/CMSSW_15_1_0_pre4/src/L1LLPJetTag/data
```

| Role | File | CLI flag | HDF5 key | Shape |
|------|------|----------|----------|-------|
| Signal — particles | `train_merged/merged_trainPart.h5` | `--sig-part` | `jet_constituents` | N × 141 |
| Signal — jet-level | `train_merged/merged_trainJet.h5` | `--sig-jet`  | `train_jet_data`   | N × 4 |
| Background (QCD) — particles | `QCD_Pt15To3000_Flat_PU200/Bkg_train.h5` | `--bkg-part` | `jet_constituents` | N × 141 |
| Background (QCD) — jet-level | `QCD_Pt15To3000_Flat_PU200/Bkg_trainJets.h5` | `--bkg-jet` | `train_jet_data` | N × 4 |
| Background test — particles | `QCD_Pt15To3000_Flat_PU200/Bkg_test.h5` | `--bkg-test-part` | `jet_constituents` | M × 141 |
| Signal test — directory | `test_merged/` | `--test-dir` | see §3 | — |

> **Particle shape note:** the raw `jet_constituents` key is `N × 141`
> (10 particles × 14 features + 1 trailing column). The loaders reshape the
> leading 140 columns into `(N, 10, 14)` = `(N, N_PART_PER_JET, N_FEAT)`.

### Per-particle feature layout (the 14 features)

```
[0..7]  one-hot particle type
[8]     dZ
[9]     dX
[10]    dY
[11]    pT   (jet-relative)        ← IDX_PT
[12]    Δη   (jet-relative)        ← IDX_ETA
[13]    Δφ   (signed delta)        ← IDX_PHI
```

The pair-feature attention bias (ParT `--arch particle`) uses indices 11/12/13.

---

## 2. Signal test set — categories

The signal test files are split by **LLP mass** and **final state**. Two LLPs
each decay to a pair of quarks, giving the 4-quark final states below.

- **Mass:** `phi15`, `phi30`, `phi60`  (the number is the LLP mass in GeV)
- **Decay:** `bbbb`, `cccc`, `uuuu`  (4b, 4c, 4u)

That is the full 9-category grid defined in `config.TEST_CATEGORIES`.

### File naming under `test_merged/`

```
{mass}_{decay}_merged_testPart.h5   →  key: jet_constituents  (N × 141)   particles
{mass}_{decay}_merged_testJet.h5    →  key: test_jet_data     (N × 4)     jet-level
```

e.g. `phi30_cccc_merged_testPart.h5`. Per-category evaluation pairs each signal
category against the single QCD background test file (`--bkg-test-part`).

---

## 3. How to run

The entrypoint is the top-level **`train.py`** (the package `bnjettag/` holds
the model/layer/loss/data code; you do not run it directly).

> **Environment note:** this env requires
> `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (a protobuf/tensorboard
> version quirk). Export it once in your shell, or prefix each command.

### Sanity check (no data needed)

```bash
python train.py --sanity                  # bitnet transformer (default)
python train.py --sanity --arch particle  # ParT-style pair-bias + [CLS]
python train.py --sanity --arch deepsets  # attention-free, hls4ml-friendly
```

Prints the model summary, verifies ternary/FP-edge assignment, checks the int8
activation path, and reports a rough FPGA LUT estimate.

### Train (defaults to the May-2026 dataset)

```bash
python train.py                           # bitnet, all-default paths
python train.py --arch particle           # pick an architecture
python train.py --arch deepsets
```

Override any data path only if you've moved the files:

```bash
python train.py \
  --sig-part  /path/merged_trainPart.h5 \
  --sig-jet   /path/merged_trainJet.h5 \
  --bkg-part  /path/Bkg_train.h5 \
  --bkg-jet   /path/Bkg_trainJets.h5 \
  --bkg-test-part /path/Bkg_test.h5 \
  --test-dir  /path/test_merged
```

### Weights & Biases

```bash
python train.py --wandb                            # log a single run
python train.py --wandb --wandb-offline            # no network
python train.py --wandb-sweep --num-agents 4 --epochs-sw 5
python train.py --wandb-sweep --sweep-id abc123 --num-agents 2
```

### ROC plots

```bash
python ROC.py    # reloads a saved .h5 via bnjettag.layers custom objects
```

---

## 4. Architectures

| `--arch` | Builder | Notes |
|----------|---------|-------|
| `bitnet`   | `build_bitnet_jet_tagger`   | Default. Ternary MHSA + FFN transformer. |
| `particle` | `build_particle_bitnet_tagger` | ParT-style pair-feature attention bias + `[CLS]` readout. |
| `deepsets` | `build_deepsets_jet_tagger` | Attention-free; most hls4ml-friendly. |

All three are **W1A8**: 1-bit (ternary {-1,0,+1}) weights, int8 activations,
with the input projection and output head kept in FP32 (BitNet b1.58 edge rule).
