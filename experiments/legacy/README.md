# Superseded code

Pre-ladder experiments kept for provenance. Nothing in the reported results depends on them.

Run everything from the repository root, e.g. `python experiments\legacy\multiseed_leaveout.py`.

| Script | What it answers | Result archived as |
|---|---|---|
| `multiseed_leaveout.py` | Phase-2-era multi-seed error bars: reruns the finetune+IICS+CA-Jaccard+crossview-mining config across seeds on leave-out 66.130, superseded by the 5-seed CAP stage. | - |
| `push_sota.py` | Pre-ladder training experiment: bigger DINOv2 ViT-B backbone + CA-Jaccard re-ranking + longer schedule, run both supervised and unsupervised on leave-out 66.130. | - |
| `train_advanced.py` | Early-phase apples-to-apples comparison of frozen DINOv2 vs SSL head vs IICS vs supervised upper bound, all trained as small heads over the same cached frozen features. | - |
