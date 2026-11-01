# RelFlow documentation

The standalone documentation for [RelFlow](https://github.com/relflow/relflow).
Examples start with payments, then add account history and larger workflows.
Published at [relflow.ai](https://relflow.ai/).

The site, its assets, canonical experiments, runners, and recorded results all
live here. Building does not require the RelFlow package or
another checkout.

## Work locally

Install Python 3.12 or newer and uv for this documentation build, then run
(the RelFlow runtime itself supports Python 3.11 or newer):

```bash
uv sync --locked
make dev
```

The preview opens at `http://127.0.0.1:4200`. Edit pages under [docs](docs/index.qmd);
the website project starts at [docs/_quarto.yml](docs/_quarto.yml).

```bash
make check
make check-docs
make render
```

`make check` checks static examples, links, and evidence
reporting. `make check-docs` also renders all pages and checks their links,
anchors, and Typst diagrams. HTML output goes to `docs/site/`.

Quarto is pinned to **1.10.18** through an isolated uv tool invocation and uses
its bundled Typst. To use that exact Quarto version already installed locally,
pass `QUARTO=quarto` to make. Python tooling installs only PyYAML, pytest, and
Ruff. Building never imports RelFlow, executes model examples, trains models,
or downloads model weights.

## Write documentation

Follow [AGENTS.md](AGENTS.md) for the authoring contract. Keep introductory
examples small and use the same payment fields throughout a topic. Add details
where readers need them, and link to their canonical explanation.

Use the shared [Typst tree renderer](docs/assets/typst/README.md) for structural
visuals. Branding, fonts, styles, mobile scrolling behavior, and the vendored
Typst extension live here. Regenerating branding is an optional maintenance
task described in [its README](docs/assets/branding/README.md).

## Evidence

Canonical experiments, the protocol, lab runner, and results live under
[`evidence/`](evidence/README.md). Each script owns its metadata and current result in frontmatter; `evidence/render.py` generates ignored pages under
`docs/evidence/` directly from local canonical sources. There is no vendored
snapshot or import/export step, and rendering never executes experiments.

The reusable `evidence-lab.yml` workflow runs all 79 experiments against the
library's exact base and candidate wheels on lab over Tailscale. The library
pins the trusted workflow separately from the suite and requires both the full
library test matrix and evidence through `CI / required` before merging into main.
The [lab guide](evidence/lab.md) describes its OIDC trust and isolation.

`publish-evidence.yml` retains a selected complete run in an immutable Release
in this repository, with JSON results, logs, an HTML report, tested wheels,
runtime requirements, and checksums. It also validates a candidate update to
the experiment frontmatter and uploads a review patch. Apply the patch on a branch
at the recorded docs base and open a normal PR to publish those measurements.
Ordinary builds read the committed results and need no runtime or cloud credentials.

Runtime tests are explicit and excluded from default checks. See
[evidence/README.md](evidence/README.md) for execution commands. Each experiment publishes one selected full run with its complete provenance. Old Proofs routes are no longer published.

## Publication

Pull requests render and validate once, uploading a review artifact. Use manual
workflow dispatch to preview a branch without a PR. Pushes to `main` publish the validated site through GitHub
Pages. This repository enables that deployment with the Actions variable
`PUBLISH_PAGES=true`; forks leave it unset to build without publishing.

The build does not inspect RelFlow branches or check out another repository.
Local preview and rendering work without enabling publication.

RelFlow-authored material retains the [Apache 2.0 license](LICENSE) and
[NOTICE](NOTICE). The vendored Typst extension retains its
[MIT license](docs/_extensions/mcanouil/typst-render/LICENSE), and the bundled
font retains its [SIL Open Font License](docs/assets/fonts/OFL.txt).
