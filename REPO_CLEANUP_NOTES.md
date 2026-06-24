# Repository Cleanup Notes

## Current Main Pipeline

Text-FSFuse v10 uses the current frequency-spatial dual-domain route:

```text
VIS / IR
-> SharedEncoder
-> DualDomainTextIntentGenerator generates I_deg / I_fus
-> TGSFF uses I_deg for frequency amplitude/phase token routing and interaction
-> TextConditionedSpatialFusion uses I_fus for spatial adaptive fusion
-> DDA aggregates F_freq and F_spa
-> FusionDecoder outputs the fused image
```

## Deleted Files

* `net/fusion/base_fusion.py`: removed after cache cleanup because it was not referenced by the current core pipeline.
* `net/fusion/spatial_compensation.py`: removed after cache cleanup because it was not referenced by the current core pipeline.
* `net/intent_frequency_fusion.py`: removed after cache cleanup because it was only an obsolete compatibility re-export and was not referenced by the current core pipeline.

## Archived Files

* `archive/THREE_LEVEL_DUAL_DOMAIN_FREQUENCY_NOTES.md`: old three-level frequency route notes.
* `archive/THREE_LEVEL_SPG_TEXTIF_NOTES.md`: old spatial/Text-IF route notes.
* `archive/BEGINNER_STRUCTURE_GUIDE.md`: replaced by the current v10 structure guide at the repository root.
* `archive/START_MODIFY_NOTES.md`: historical notes mentioning old BaseFusion / SpatialResidualCompensation route.

Archived documents describe old routes and should not be used as current method documentation.

## Moved Utility Scripts

* `tools/dataprocessing.py`
  * Example:
    ```powershell
    D:\anaconda\envs\Image_Fusion\python.exe tools\dataprocessing.py --ir_dir <path_to_ir_train> --vi_dir <path_to_vi_train> --out_dir ./data --data_name MSRS_train --img_size 128 --stride 200
    ```
  * H5 groups remain compatible with `utils/dataset.py`: `ir_patchs` and `vis_patchs`.
* `tools/visualization/enlarge_image.py`
  * Standalone visualization helper; not used by `train.py` or `test.py`.
* `tools/visualization/tocolor.py`
  * Standalone visualization/color helper; not used by `train.py` or `test.py`.

## Files Kept for Safety

No candidate old structure file was kept for safety. The removed files had no current core pipeline references after cache cleanup.

## Safety Decisions

* No checkpoint, data, result, or CLIP weight file was deleted.
* Current `train.py` / `test.py` forward pipeline was not changed.
* Loss weights and training hyperparameters were not changed.
* Deleted files were not referenced by current core pipeline after cache cleanup.
* Archived documents describe old routes and should not be used as current method documentation.
* `.gitignore` updates only affect future untracked files; no tracked checkpoint/data/result file was untracked.

## Verification

Reference checks before cache cleanup showed matches from `__pycache__` files and historical notes.

After cache cleanup:

* `git grep -n "SimpleBaseFusion"`: no matches.
* `git grep -n "SpatialResidualCompensation"`: only `archive/START_MODIFY_NOTES.md`.
* `git grep -n "intent_frequency_fusion"`: no matches.
* `git grep -n "base_fusion"`: no matches.
* `git grep -n "spatial_compensation"`: no matches.
* `git grep -n "dataprocessing"`: no matches in train/test or current core modules.
* `git grep -n "enlarge_image"`: no matches in train/test or current core modules.
* `git grep -n "tocolor"`: no matches in train/test or current core modules.

Final verification:

* Core `py_compile`: passed for `train.py`, `test.py`, `net/Network.py`, `net/intent.py`, `net/dda.py`, encoder, decoder, spatial fusion, and frequency fusion.
* Tool `py_compile`: passed for `tools/dataprocessing.py`, `tools/visualization/enlarge_image.py`, and `tools/visualization/tocolor.py`.
* Core imports: `Core imports OK`.
* Smoke test: `Smoke test OK: torch.Size([1, 1, 64, 64])`.
* `git diff -- train.py test.py utils/loss.py`: no output.

## Known Dependency Notes

The smoke test uses `use_learnable_prompt_embedding=True` so it does not depend on a local CLIP checkpoint. Training and testing defaults were not changed.
