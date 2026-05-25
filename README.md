# NanoGPT Pro — `website` branch

This orphan branch hosts **only** the project homepage for the yfz.ai blog:

- `index.html` — the NanoGPT Pro landing page
- `.nojekyll` — serve assets verbatim

## Why this branch exists

`https://yfz.ai` is deployed by **Cloudflare Pages**, which serves the entire
recursive submodule checkout from the repo root and **fails any deployment with
more than 20,000 files**. The `main` branch of this repo ships the full release
tree (`config/`, `data/`, and the ~16k-file `lm-evaluation-harness/`). If the
blog submodule tracked `main`, it would blow past that cap and break the deploy
for the whole blog.

So the `blog/nanogptpro` submodule in
[`yifanzhang-pro/yfz.ai`](https://github.com/yifanzhang-pro/yfz.ai) tracks **this
`website` branch** (`.gitmodules` → `branch = website`), keeping the checkout at
two files. Same pattern as that blog's `TPA` entry.

## Maintenance

1. Edit `index.html` here on `website`. Keep it lightweight — homepage assets
   only. **Never merge `main` into this branch.**
2. In `yfz.ai`, re-pin the submodule and push a normal (non-force) commit:
   ```bash
   cd blog/nanogptpro && git fetch origin website && git checkout <new-sha>
   cd ../.. && git add blog/nanogptpro && git commit -m "blog: update NanoGPT Pro homepage" && git push
   ```
   Cloudflare Pages often skips force-pushed/amended history; a fresh
   fast-forward commit reliably triggers a rebuild and cache invalidation.
3. Verify with `git clone --recurse-submodules` of the yfz.ai repo: total files
   must stay `< 20000`.
