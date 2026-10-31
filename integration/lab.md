# Lab evidence execution over Tailscale

The library workflow builds candidate and baseline wheels on GitHub-hosted
workers. A pinned reusable workflow joins the tailnet using GitHub OIDC,
dispatches the suite to lab over Tailscale SSH, retrieves the report, and leaves
the ephemeral node. Lab runs experiments in separate bounded Docker containers.
There is no general GitHub runner installed on lab and no cloud compute account
or stored Tailscale client secret is required.

## Lab prerequisites

- Docker Engine available to the chosen SSH user.
- Python 3.10 or newer with PyYAML for the trusted coordinator.
- Tailscale SSH enabled and sufficient disk space for the cached runtime image.
- Eight workers use at most 16 CPU cores and 32 GiB of container memory. Each
  experiment keeps its original seed, budget, and CPU behavior profile. The
  host GPU is not used for the CPU behavioral gate.

The dispatcher writes only under `~/relflow-evidence/runs/<run-id>/`, with unique
run directories. It never installs RelFlow or its runtime dependencies on the
host. Containers mount the pinned source and the selected wheel read-only,
plus a per-task output directory. They have no network, dropped capabilities,
no new privileges, a read-only root, and explicit CPU, memory, and PID limits.
Interruptions clean up containers belonging to that run and retain incomplete
reports; incomplete execution cannot pass the required gate.

## Tailscale trust configuration

Create a federated identity in the Tailscale admin console's Trust credentials
page. Configure:

| Setting | Value |
| --- | --- |
| Issuer | GitHub Actions, `https://token.actions.githubusercontent.com` |
| Subject | `repo:relflow/relflow:*` |
| Custom claim `repository` | `relflow/relflow` |
| Custom claim `job_workflow_ref` | `relflow/evidence/.github/workflows/lab.yml@<exact-helper-commit>` |
| Scope | `auth_keys` write, restricted to `tag:relflow-evidence-ci` |
| Device tag | `tag:relflow-evidence-ci` |

The helper commit is the complete SHA in the library workflow's `uses:` line.
Keep the custom workflow claim exact; do not replace it with a wildcard. The
immutable helper validates the calling repository and PR source, then checks
out a separately pinned, tested evidence implementation. A fork cannot gain
tailnet access by changing its own workflow.

Grant the CI tag access to lab's SSH service and allow noninteractive Tailscale
SSH for the selected account using an `accept` rule. Retain existing human
access rules. Review the existing policy when limiting destinations: an older
allow-all rule can grant broader network access than a new narrow rule.
Interactive SSH `check` rules require a person and cannot serve unattended CI.

Copy the generated Client ID and audience. These identifiers are public
configuration, not credentials. The workflow uses the generated audience
`api.tailscale.com/<client-id>`.

References: [Tailscale workload identity federation](https://tailscale.com/docs/features/workload-identity-federation),
[the official GitHub Action](https://github.com/tailscale/github-action).

## Library repository configuration

Set these Actions variables in `relflow/relflow`:

- `TAILSCALE_OIDC_CLIENT_ID`: the federated identity's Client ID.
- `EVIDENCE_LAB_TARGET`: the permitted SSH account and machine, such as
  `evidence@lab`.
- `EVIDENCE_LAB_HOST`: optional connectivity probe target; defaults to `lab`.

Install `integration/evidence.yml` as `.github/workflows/evidence.yml`, replacing
`__EVIDENCE_REVISION__` with the exact helper commit. The workflow builds wheels
without tailnet credentials, invokes the trusted helper, and always evaluates
the final `Evidence / required` job. Configure branch protection only after a
real successful run exists, requiring that check and an up-to-date base or
merge queue. Preserve the repository's existing requirements and bypass policy.

Missing trust configuration fails the job with a clear setup message. It never
turns a skipped or missing evidence run into a successful required check.

## Manual validation

Build both wheels from the desired Git revisions and export the candidate's
locked runtime requirements. From a clean evidence checkout:

```sh
uv run python ssh_run.py --target evidence@lab \
  --candidate-wheel /path/to/candidate/relflow.whl \
  --baseline-wheel /path/to/baseline/relflow.whl \
  --candidate-sha <candidate-sha> --baseline-sha <base-sha> \
  --requirements /path/to/requirements.txt --output runs/<unique-run-id>
```

Supply actual wheel filenames from the build. Add `--ids P001 --steps 2` for a
smoke test. It produces execution records but deliberately fails the learning
gate because it is partial and uses a reduced budget. Complete runs produce
the same JSON/HTML report contract as the optional Modal backend.
