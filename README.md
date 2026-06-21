# nanogptpro

Release repository for NanoGPT Pro packages.

The source of truth is the private/source development repository. This
repository carries release-facing assets:

- GitHub Release artifacts for the compiled `nanogptpro` runtime wheel.
- The vendored `lm-evaluation-harness/` checkout with NanoGPT Pro evaluation
  bridge patches.

## Runtime Wheel

Install the published wheel from this repository's matching GitHub Release:

```bash
uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install nanogptpro-0.0.1-cp314-cp314-linux_x86_64.whl
```

The wheel is binary-only. It contains the compiled `nanogptpro` runtime and
console entrypoints such as `nanogptpro-generate` and
`nanogptpro-train-openwebtext`.

## Evaluation Harness

Install the bundled evaluation harness from this checkout when running
checkpoint benchmarks:

```bash
source .venv/bin/activate
uv pip install -e "./lm-evaluation-harness[hf]"
lm_eval --help
```

The harness is kept in this repository so evaluation users do not need the
source development checkout. It expects the matching `nanogptpro` runtime wheel
to be installed in the active environment.
