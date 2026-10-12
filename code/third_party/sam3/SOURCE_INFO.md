# SAM3 Source Info

This directory vendors the SAM3 code used by Embody4D's automatic front-mask generation.

- Upstream: https://github.com/facebookresearch/sam3.git
- Server mirror observed during integration: https://ghfast.top/github.com/facebookresearch/sam3.git
- Commit: `11dec2936de97f2857c1f76b66d982d5a001155d`
- `.git`, Python caches, build outputs, and local test outputs are intentionally excluded.

The SAM3 checkpoint is stored outside this code directory:

```bash
weights/sam3/sam3.pt
```
