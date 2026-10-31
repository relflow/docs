# RelFlow documentation

The standalone documentation for [RelFlow](https://github.com/relflow/relflow).
Examples start with payments, then add account history and larger workflows.
Published at [relflow.github.io/docs](https://relflow.github.io/docs/).

The saved [API contract](contracts/relflow.json) records the source used to
review the public API.
The site and its assets live here. Canonical experiments and report archives
live in [relflow/evidence](https://github.com/relflow/evidence); this checkout
contains a verified publication snapshot. Building does not require the RelFlow package or
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

`make check` checks static examples and links, the saved API contract, and evidence
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

[contracts/relflow.json](contracts/relflow.json) records the public options,
value states, and attention modes used by reference checks. Refresh it when
targeting a new package version; tests do not import RelFlow to discover options.
[migration.json](migration.json) records the documentation extraction.

## Evidence

Canonical experiments, their protocol, runtime tests, and report archives live
in [relflow/evidence](https://github.com/relflow/evidence). This repository owns
the website presentation and a checked-in snapshot under `vendor/evidence/`.
[contracts/evidence.json](contracts/evidence.json) pins the source revision,
selected results, and file digests. `scripts/render_evidence.py` verifies the
snapshot and generates ignored pages under `docs/evidence/` without importing
or executing experiments. Preserve archived provenance when importing reports.

To import a selected GitHub Release export, supply its reviewed SHA-256:

```sh
uv run python scripts/import_evidence.py --release <release-tag> --sha256 <bundle-sha256>
```

The import downloads `evidence-docs.tar.gz` from `relflow/evidence`, verifies
it, and replaces the local publication snapshot. Review the snapshot and
contract changes together. A local bundle can be supplied with `--bundle`.
Ordinary checks, preview, and rendering read local files and require no GitHub
or Modal credentials. Historical Proofs URLs redirect to Evidence pages.

## Publication

Pull requests and branches render and validate independently, uploading a
review artifact. Pushes to `main` publish the validated site through GitHub
Pages. This repository enables that deployment with the Actions variable
`PUBLISH_PAGES=true`; forks leave it unset to build without publishing.

The build does not inspect RelFlow branches or check out another repository.
Local preview and rendering work without enabling publication.

RelFlow-authored material retains the [Apache 2.0 license](LICENSE) and
[NOTICE](NOTICE). The vendored Typst extension retains its
[MIT license](docs/_extensions/mcanouil/typst-render/LICENSE), and the bundled
font retains its [SIL Open Font License](docs/assets/fonts/OFL.txt).
