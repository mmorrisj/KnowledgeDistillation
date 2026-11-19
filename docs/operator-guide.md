# Operator guide

How to run the course site, on your own machine or on a server. No containers
are needed.

## What you need

- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/)
- Node 22 or newer (only to build the site; not needed to serve it)

## Install, build, serve

```bash
make install
```

```bash
make build
```

```bash
make serve
```

Open http://127.0.0.1:8000. `make build` writes the site to `site/dist`;
`make serve` starts one Python process that serves those files and the API.

If you start the server before building, it stops with a message telling you
to build first.

## Settings

Set these as environment variables, or in a `.env` file in the repo root.

| Variable | Default | Meaning |
|---|---|---|
| `HOST` | `127.0.0.1` | Address to listen on. Use `0.0.0.0` to accept connections from other machines. |
| `PORT` | `8000` | Port to listen on. |
| `SITE_DIST` | `site/dist` | Folder holding the built site. |
| `TRACKING_ENABLED` | `false` | Instructor tracking. Not implemented yet; leave it off. |

With tracking off, the site has no accounts and no database. Each student's
progress and quiz answers are saved in their own browser.

## Running as a service (Linux)

Build the site once, then create `/etc/systemd/system/kd-course.service`:

```ini
[Unit]
Description=Knowledge distillation course site
After=network.target

[Service]
User=kdcourse
WorkingDirectory=/opt/KnowledgeDistillation
Environment=HOST=0.0.0.0
Environment=PORT=8000
ExecStart=/usr/local/bin/uv run --no-dev kd-course-serve
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now kd-course
```

To publish new lessons: pull, run `make build`, then restart the service.

## Building on one machine, serving on another

The serving machine does not need Node. Build on any machine with Node, copy
`site/dist` to the server, and point `SITE_DIST` at it.

## Labs

Labs are Jupyter notebooks in `labs/`. Students run them on a GPU machine they
reach over SSH. The site only links to them: on the Settings page a student can
save the address they use to open Jupyter (for example `http://localhost:8888`
with an SSH port forward), and lab buttons then open the notebook directly.

## Writing lessons

- Lessons are MDX files in `site/src/content/lessons/<module>/<lesson>.mdx`.
- Modules are listed in `site/src/content/modules.ts`.
- `site/src/content/lessons/00-sample/01-components.mdx` shows every building
  block. Use `client:load` on interactive components.
- `make dev` gives a live-reloading preview while you write.

## Checks

```bash
make test
```

```bash
make e2e
```

`make test` runs the linter, Python tests, site unit tests and type check.
`make e2e` builds the site and runs a browser test against the real server.
