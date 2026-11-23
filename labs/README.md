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

## Using an API teacher (optional)

Lab 4 runs with the local teacher by default. To use a teacher behind an
OpenAI-style chat completions API instead, set these on the GPU machine and
change the one line in the notebook that creates the teacher:

```bash
export KD_COURSE_API_BASE=https://your-provider.example/v1
```

```bash
export KD_COURSE_API_KEY=your-key
```

```bash
export KD_COURSE_API_MODEL=model-name
```

Check the provider's terms before training on its outputs (lesson 1.5), and
never commit the key.

## Serving a student (Module 7)

Lab 7 saves a student to `results/students/classifier` and asks you to start a
small web service for it in a terminal:

```bash
uv run --group labs kd-student-serve
```

Settings, all optional: `KD_STUDENT_DIR` (folder holding the saved student),
`KD_STUDENT_DEVICE` (`cuda` or `cpu`), `KD_STUDENT_QUANTIZE=1` (serve the 8-bit
version on the CPU), `HOST` (default `127.0.0.1`) and `PORT` (default `8100`).
The service has no authentication; keep it on `127.0.0.1` unless you put it
behind something that does.

Lab 8 uses the Oxford-IIIT Pet images (`timm/oxford-iiit-pet`). For a local
copy, put its `train.parquet` and `test.parquet` in a folder named
`oxford-iiit-pet` under the same `KD_COURSE_DATA_DIR`.

## Postgres with pgvector (optional, Module 9)

Lab 9 stores embeddings in Postgres if you give it a database, and in memory if
you do not. To use Postgres, the server needs the
[pgvector](https://github.com/pgvector/pgvector) extension installed. On
Debian or Ubuntu with Postgres 16 that is the `postgresql-16-pgvector` package.
Then create a database and a user for the course, and set the address before
starting Jupyter:

```bash
export KD_COURSE_DATABASE_URL=postgresql+psycopg://user:password@localhost/dbname
```

The lab creates the extension in that database if it is not there yet, which
needs a user allowed to do so. It creates and drops its own tables, whose names
start with `kd_`. Do not point it at a database where that could clash.

## Results

Labs save scores and predictions to a `results` folder at the top of the repo
(set `KD_COURSE_RESULTS_DIR` to put it elsewhere). Later labs read them, so keep
that folder. It is not committed to git.

## What is checked without a GPU

Every notebook's own code cells are run on CPU by the test suite, with tiny
models and toy data swapped in for the real ones
(`tests/labs/test_notebook_smoke.py`). The lab code is also run against toy-size
copies of the real model designs, Qwen2 and ModernBERT
(`tests/kd_course/test_real_architectures.py`).

These checks catch mistakes in the notebooks and the code. They cannot tell you
how fast a lab runs, how much memory it needs, or how good the real models are.
For that, each lab still has to be run once on a GPU.

## Notebooks

| Module | Notebook | Time |
|---|---|---|
| 1. Foundations | `01_foundations/01_setup_check.ipynb` | 5 to 10 min after download |
| 2. Baselines and evaluation | `02_baselines/01_baselines.ipynb` | 20 to 30 min |
| 3. White-box distillation | `03_white_box/01_white_box.ipynb` | about 45 min; needs the Module 2 results |
| 4. Black-box distillation | `04_black_box/01_black_box.ipynb` | about 45 min; needs the Module 2 results |
| 5. LoRA and QLoRA | `05_lora/01_lora.ipynb` | about 40 min; needs the Module 2 and 3 results |
| 6. Distilling generative models | `06_generative/01_generative.ipynb` | about 50 min; stands alone |
| 7. Compress and serve | `07_serve/01_serve.ipynb` | about 45 min; needs the Module 2 and 3 results |
| 8. Distilling vision models | `08_vision/01_vision.ipynb` | about 40 min; stands alone; downloads about 800 MB of images |
| 9. Embedding distillation and search | `09_embeddings/01_embeddings.ipynb` | about 30 min; stands alone; Postgres optional |

Clear all outputs before committing a notebook.
