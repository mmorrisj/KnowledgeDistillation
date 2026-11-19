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
| `tests/` | Python tests |
| `docs/` | Operator guide, design and plans |

## Status

The site platform and a component sampler lesson are in place. Course modules,
labs and instructor tracking are planned next; see
`docs/superpowers/specs/2026-11-19-kd-course-platform-design.md`.
