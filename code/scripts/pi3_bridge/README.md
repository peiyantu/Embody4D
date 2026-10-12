# Pi3 Bridge

`run_pi3_warp_two_videos_firstframe_pose.py` is the Embody4D bridge used by `scripts/setup_pi3.sh` when Pi3 is cloned from upstream instead of shipped as `third_party/Pi3`.

The current inference path imports its Pi3X helpers for model loading and source-video reconstruction, then uses Embody4D's existing Warper for `target_pose` / trajectory rendering.
