# NanoGPT Pro — `website` branch

This orphan branch hosts **only** the project homepage for GitHub Pages:

- `index.html` — the NanoGPT Pro landing page
- `.nojekyll` — serve assets verbatim

Primary deployment:

- `https://math-ai-org.github.io/nanogptpro/`

The same lightweight page may also be consumed by `yfz.ai/blog`, but that is a
secondary mirror/use case rather than the branch's purpose.

## Why this branch exists

The `main` branch of this repo ships the full release tree (`config/`, `data/`,
`nanogptpro/`, and the ~16k-file `lm-evaluation-harness/`). The homepage does
not need that payload. Keeping the GitHub Pages branch as a tiny orphan branch
makes the public website fast to fetch and deploy.

As a secondary benefit, the `blog/nanogptpro` submodule in `yfz.ai` can track
this `website` branch without pulling the full release checkout into the blog
deployment.

## Maintenance

1. Edit `index.html` here on `website`. Keep it lightweight — homepage assets
   only. **Never merge `main` into this branch.**
2. Deploy or refresh `https://math-ai-org.github.io/nanogptpro/` from this
   branch.
3. If the page is mirrored into `yfz.ai`, re-pin that submodule and push a
   normal (non-force) commit:
   ```bash
   cd blog/nanogptpro && git fetch origin website && git checkout <new-sha>
   cd ../.. && git add blog/nanogptpro && git commit -m "blog: update NanoGPT Pro homepage" && git push
   ```
   Cloudflare Pages often skips force-pushed/amended history; a fresh
   fast-forward commit reliably triggers a rebuild and cache invalidation.
4. Verify with `git clone --recurse-submodules` of the yfz.ai repo: total files
   must stay `< 20000`.
