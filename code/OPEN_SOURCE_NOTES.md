# Open-Source Cleanup Notes

Generated from an internal experiment directory as a cleaned release tree.

Removed from this copy:

- experiment outputs and VBench result videos
- private training artifacts and logs
- base pretrained checkpoints
- backup files, pid files, and Python bytecode caches
- heavy third-party example assets
- private absolute path references found in generated outputs

Included after cleanup:

- current best released project-specific latent expert transformer
- matching latent residual router
- best latent route configuration notes

Post-clean checks to run before publishing:

- no private absolute path references
- no `.bak`, `.pyc`, `.log`, or `.pid`
- base pretrained models remain external
- shell scripts pass `bash -n`

Before public release:

- add a top-level project license
- review third-party Pi3/SAM3 licenses
- use Git LFS or a model-release host for large weights
- run a clean-environment install/inference smoke test
