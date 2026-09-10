# Pages site

<!-- markdownlint-disable MD013 -->

**Live:** https://ocha-dap.github.io/pa-aa-bfa-drought/

Landing page plus per-analysis pages, using the team's Pages template (HDX v2 tokens,
particle hero, cards — lineage `ds-storm-impact-harmonisation` → `ds-geospatial-impact-estimates`
→ `ds-seas5-skill`). There is no build step: `.github/workflows/deploy-pages.yml` copies
`pages/` to the site root on push to `main` or `2026-revision` touching `pages/**`, or on
manual dispatch.

| Site path | Source |
|---|---|
| `/` | `pages/index.html` (landing, hand-edited cards) |
| `/external-sources-sep-2026/` | `pages/external-sources-sep-2026/index.html` |

## Adding a page

Create a directory under `pages/` with an `index.html`, paste the back-to-home snippet as the
first element in `<body>` (copy it from an existing page), and add a card to `pages/index.html`.
Keep source documents out of the repo — the repo is public; summarise, don't republish.

## The September 2026 external-sources comparison

`pages/external-sources-sep-2026/index.html` is hand-authored HTML. The two inline SVG
heatmaps are generated from the JRC ASAP `warnings_l2_ts` archive filtered to the four AOI
provinces (asap2 ids 3820, 3824, 3825, 3827). To refresh after a new dekad:

```bash
curl -sL -o /tmp/warnings_l2_ts.zip https://agricultural-production-hotspots.ec.europa.eu/files/warnings_l2_ts.zip
unzip -p /tmp/warnings_l2_ts.zip | grep -E '^(asap0_id|219;[0-9]+;(3820|3824|3825|3827);)' > /tmp/aoi.csv
```

then re-run the figure script (see the page's footer for the data vintage) and paste the SVGs
back into the HTML in place of the existing `<svg>` blocks.

## Deploy notes

- The `github-pages` environment must allow deployments from both `main` and `2026-revision`
  (Settings → Environments → github-pages → deployment branches), otherwise the deploy job fails
  immediately.
- `workflow_dispatch` only registers once the workflow file is on the default branch.

## Checking locally

```bash
python3 -m http.server -d pages 8000
```
