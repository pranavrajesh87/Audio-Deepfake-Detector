# Research & Training Scripts

This folder contains the experimental and training scripts used to build and evaluate the model.
These are **not part of the production API** — they are kept here for reproducibility and reference.

| Script | Purpose |
|---|---|
| `phase1_data_pipeline.py` | Data loading, balancing, and PyTorch Dataset class for ASVspoof 2019 |
| `phase3_stress_test.py` | Cross-domain stress testing (Edge TTS fakes, self-recorded mic clips) |
| `create_subset.py` | Creates a smaller balanced subset of the dataset for faster iteration |
| `generate_fakes.py` | Generates synthetic fake audio clips using Edge TTS |
| `convert_m4a_to_wav.py` | Batch-converts `.m4a` microphone recordings to `.wav` format |
| `test_asvspoof_bonafide.py` | Tests the model against ASVspoof 2019 real (bonafide) voice clips |
| `test_clean_only.py` | Tests the model against clean studio-quality recordings |

> **Note**: These scripts contain hardcoded paths that point to local data directories.
> They will not run as-is on a different machine without updating the paths at the top of each file.
> The production inference service lives in `../app/`.
