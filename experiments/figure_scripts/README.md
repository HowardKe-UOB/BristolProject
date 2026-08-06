# Figure generation

Scripts that render the dissertation figures. Output goes to `figures/` (not committed). Named `figure_scripts` rather than `figures` because `.gitignore` excludes any directory called `figures/`.

Run everything from the repository root, e.g. `python experiments\figure_scripts\make_figures.py`.

| Script | What it answers | Result archived as |
|---|---|---|
| `make_figures.py` | Generates the five bilingual (Chinese+English) paper figures as PNGs from hardcoded artifacts2 numbers. | - |
| `make_figures_en.py` | English-only variant of make_figures.py producing figures/*_en_v1.png with the same data and palette. | - |
| `make_methods_figs_en.py` | Draws the English-only method schematic figures: fig6 overall pipeline diagram and fig7 three mechanism demonstrations. | - |
