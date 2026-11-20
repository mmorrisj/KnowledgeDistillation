# Distilling Frontier Models

A self-paced course that teaches data scientists to take what a large
"teacher" model can do on one task and move it into a small "student" model.

- **Students:** read the lessons on the course site, then run each lab
  notebook from `labs/` on your GPU machine.
- **Running the site yourself:** see the [operator guide](docs/operator-guide.md).
  The short version:

```bash
make install && make build && make serve
```

## What's here

| Path | Contents |
|---|---|
| `site/` | Lesson site (Astro, MDX, React components) |
| `server/` | FastAPI app that serves the built site and the API |
| `src/kd_course/` | Python package shared by the labs |
| `labs/` | Lab notebooks, run on a GPU machine ([setup](labs/README.md)) |
| `tests/` | Python tests |
| `docs/` | Operator guide, design and plans |

## Status

The site platform and all nine modules are in place: Foundations, Baselines
and evaluation, White-box distillation, Black-box distillation, LoRA and QLoRA,
Distilling generative models, Compress and serve, Distilling vision models, and
Embedding distillation and search. The lab notebooks have not yet been run on a
real A10G. Instructor tracking is planned; see
`docs/superpowers/specs/2026-11-19-kd-course-platform-design.md`.
