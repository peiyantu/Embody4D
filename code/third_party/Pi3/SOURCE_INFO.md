# Pi3 Source Info

This directory vendors the Pi3 working tree used by Embody4D's Pi3 warp backend.

- Upstream: https://github.com/yyfz/Pi3.git
- Server mirror observed during integration: https://ghfast.top/github.com/yyfz/Pi3.git
- Commit: `b56ef4b9cc9be05b8590604861aa3906db0ef45e`
- `.git`, Python caches, and bytecode files are intentionally excluded.

Embody4D's backend imports `run_pi3_warp_two_videos_firstframe_pose.py` from this directory for Pi3X model loading and source-video reconstruction helpers. Although that bridge script still contains its legacy standalone CLI example, Embody4D uses Pi3 only for depth/reconstruction and then applies the same `target_pose` / trajectory Warper path as the depth backend.

To recreate this directory from upstream, run:

```bash
bash scripts/setup_pi3.sh
```
