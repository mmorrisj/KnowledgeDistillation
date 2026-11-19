# Labs

Each module has one notebook. Run them on your GPU machine, not on the machine
that serves the course site.

## What you need

- One NVIDIA A10G (24 GB) or larger, with bf16 support
- About 40 GB of free disk for models and cached data
- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/)

## Set up (once)

On the GPU machine, in the repo folder:

```bash
make install-labs
```

## Start Jupyter

On the GPU machine:

```bash
uv run --group labs jupyter lab --no-browser --port 8888
```

On your own computer, forward the port over SSH:

```bash
ssh -N -L 8888:localhost:8888 <user>@<gpu-host>
```

Then open the address Jupyter printed (it starts with `http://localhost:8888`).
To make the course site's lab buttons open notebooks for you, save
`http://localhost:8888` on the site's Settings page.

## Models

The labs download models from Hugging Face on first use. If the GPU machine
cannot reach Hugging Face, download the models elsewhere, copy them over, and
point the labs at the folder:

```bash
export KD_COURSE_MODEL_DIR=/data/models
```

The labs look for `<folder>/<model name>`, for example
`/data/models/Qwen2.5-7B-Instruct` and `/data/models/Qwen2.5-0.5B-Instruct`.

## Data

The labs download the Banking77 dataset from Hugging Face on first use. To use
a local copy, put `train.parquet` and `test.parquet` in a folder named
`banking77` and point the labs at its parent:

```bash
export KD_COURSE_DATA_DIR=/data/datasets
```

## Results

Labs save scores and predictions to a `results` folder beside the notebook.
Later labs read them, so keep that folder. It is not committed to git.

## Notebooks

| Module | Notebook | Time |
|---|---|---|
| 1. Foundations | `01_foundations/01_setup_check.ipynb` | 5 to 10 min after download |
| 2. Baselines and evaluation | `02_baselines/01_baselines.ipynb` | 20 to 30 min |

Clear all outputs before committing a notebook.
