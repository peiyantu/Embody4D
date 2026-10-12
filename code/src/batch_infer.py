#!/usr/bin/env python3
import argparse
import csv
import os
import queue
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock, Thread


PRINT_LOCK = Lock()


def safe_print(*args, **kwargs):
    with PRINT_LOCK:
        print(*args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TrajCrafter inference over a video folder with one worker per GPU."
    )
    parser.add_argument(
        "--input-dir",
        default="../assets/example",
        help="Folder containing input .mp4 files.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional CSV file to sample inference inputs from. When set, folder traversal is skipped.",
    )
    parser.add_argument(
        "--selected-rows-csv",
        default=None,
        help=(
            "Optional selected_rows.csv produced by a previous run. When set, "
            "tasks are loaded exactly from this file instead of sampling --csv-path."
        ),
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=50,
        help="Number of CSV rows to sample.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=43,
        help="Random seed used for CSV row sampling.",
    )
    parser.add_argument(
        "--csv-video-column",
        default="input",
        help="CSV column used as inference --video_path.",
    )
    parser.add_argument(
        "--csv-prompt-column",
        default="prompt",
        help="CSV column used as inference --prompt.",
    )
    parser.add_argument(
        "--csv-front-column",
        default="front_mask",
        help="CSV column used as inference --frontvideo when front guidance is enabled.",
    )
    parser.add_argument(
        "--use-existing-warp-mask",
        action="store_true",
        help=(
            "In CSV mode, use each selected row's existing warp/mask videos "
            "instead of letting inference.py rebuild warp and mask."
        ),
    )
    parser.add_argument(
        "--csv-warp-column",
        default="warp",
        help="CSV column used as inference --warp_path when --use-existing-warp-mask is set.",
    )
    parser.add_argument(
        "--csv-mask-column",
        default="mask",
        help="CSV column used as inference --mask_path when --use-existing-warp-mask is set.",
    )
    parser.add_argument(
        "--enable-front-guidance",
        action="store_true",
        help="Pass --enable_front_guidance and per-row --frontvideo from the CSV front column.",
    )
    parser.add_argument(
        "--auto-sam3-front-mask",
        action="store_true",
        help="Let inference.py generate --frontvideo from each input video with SAM3 when no front mask is provided.",
    )
    parser.add_argument(
        "--frontvideo",
        default=None,
        help="Global front guidance video used when tasks do not provide per-row front videos.",
    )
    parser.add_argument(
        "--out-dir",
        default="cam_type_9_outputs",
        help="Output directory passed to inference.py.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Log directory. Defaults to <out-dir>/_logs.",
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Comma or space separated physical GPU ids, for example: 0,1,2,3.",
    )
    parser.add_argument(
        "--inference-script",
        default="inference.py",
        help="Path to inference.py, relative paths are resolved from the current working directory.",
    )
    parser.add_argument(
        "--exp-name-prefix",
        default="cam_type_9",
        help="Prefix for stable per-video exp_name values. Use an empty string to use only the video stem.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--camera", default="target")
    parser.add_argument("--target-pose", default="0 -20 0 0 0")
    parser.add_argument(
        "--target-poses",
        nargs="+",
        default=None,
        help=(
            "Optional list of target poses. In CSV mode each sampled input is "
            "expanded once per pose. Example: --target-poses '-20 0 0 0 0' "
            "'20 0 0 0 0'."
        ),
    )
    parser.add_argument(
        "--pose-labels",
        nargs="+",
        default=None,
        help="Optional labels matching --target-poses, used in exp_name values.",
    )
    parser.add_argument("--mode", choices=["gradual", "direct", "bullet", "zoom"], default="gradual")
    parser.add_argument("--traj-txt", default="test/trajs/loop2.txt")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--blip-path", required=True)
    parser.add_argument("--unet-path", required=True)
    parser.add_argument("--pre-train-path", required=True)
    parser.add_argument("--transformer-path", required=True)
    parser.add_argument("--video-length", type=int, default=49)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run even when <out-dir>/<exp_name>/gen.mp4 already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching inference.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Keep dispatching tasks after failures. This is the default behavior for running workers.",
    )
    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    return args, extra_args


def split_gpus(gpus):
    result = [item for item in re.split(r"[,\s]+", gpus.strip()) if item]
    if not result:
        raise ValueError("No GPU ids were provided.")
    return result


def make_folder_task(video):
    return {
        "video_path": Path(video),
        "prompt": None,
        "frontvideo": None,
        "warp_path": None,
        "mask_path": None,
        "exp_name": None,
        "row_index": None,
        "sample_index": None,
        "source_row": {},
    }


def exp_name_for(task, prefix):
    explicit = task.get("exp_name")
    if explicit:
        return explicit
    video = task["video_path"]
    if prefix:
        return f"{prefix}_{video.stem}"
    return video.stem


def pose_label_for(target_pose, index):
    tokens = str(target_pose).replace(",", " ").split()
    try:
        values = [float(token) for token in tokens]
    except ValueError:
        values = []
    if values:
        parts = []
        names = ("theta", "phi", "r", "x", "y")
        for name, value in zip(names, values):
            if abs(value) > 1e-9:
                sign = "p" if value > 0 else "m"
                mag = f"{abs(value):g}".replace(".", "p")
                parts.append(f"{name}{sign}{mag}")
        if parts:
            return "_".join(parts)
    return f"pose{index:02d}"


def build_command(args, extra_args, task, exp_name):
    cmd = [
        sys.executable,
        args.inference_script,
        "--video_path",
        str(task["video_path"]),
        "--stride",
        str(args.stride),
        "--out_dir",
        str(args.out_dir),
        "--exp_name",
        exp_name,
        "--radius_scale",
        str(args.radius_scale),
        "--camera",
        args.camera,
        "--target_pose",
        str(task.get("target_pose") or args.target_pose),
        "--mode",
        args.mode,
        "--model_name",
        args.model_name,
        "--blip_path",
        args.blip_path,
        "--unet_path",
        args.unet_path,
        "--pre_train_path",
        args.pre_train_path,
        "--transformer_path",
        args.transformer_path,
        "--traj_txt",
        args.traj_txt,
        "--video_length",
        str(args.video_length),
        "--fps",
        str(args.fps),
        "--seed",
        str(args.seed),
        "--device",
        "cuda:0",
    ]
    prompt = task.get("prompt")
    if prompt:
        cmd.extend(["--prompt", prompt])
    frontvideo = task.get("frontvideo") or args.frontvideo
    if args.enable_front_guidance:
        cmd.append("--enable_front_guidance")
        if frontvideo:
            cmd.extend(["--frontvideo", str(frontvideo)])
        elif args.auto_sam3_front_mask:
            cmd.append("--sam3_front_mask")
        else:
            raise ValueError(
                "Front guidance requires a frontvideo unless --auto-sam3-front-mask is set."
            )
    warp_path = task.get("warp_path")
    mask_path = task.get("mask_path")
    if warp_path and mask_path:
        cmd.extend(["--warp_path", str(warp_path)])
        cmd.extend(["--mask_path", str(mask_path)])
    return cmd + extra_args


def shell_join(cmd):
    return " ".join(subprocess.list2cmdline([part]) for part in cmd)


def run_one(args, extra_args, task, gpu, log_dir):
    exp_name = exp_name_for(task, args.exp_name_prefix)
    save_dir = Path(args.out_dir) / exp_name
    gen_path = save_dir / "gen.mp4"
    if gen_path.is_file() and gen_path.stat().st_size > 0 and not args.overwrite:
        safe_print(f"[skip] {exp_name}: {gen_path} exists", flush=True)
        return 0, task, gpu

    log_path = log_dir / f"{exp_name}.gpu{gpu}.log"
    cmd = build_command(args, extra_args, task, exp_name)
    if args.dry_run:
        safe_print(f"[dry-run][gpu {gpu}] {shell_join(cmd)}", flush=True)
        return 0, task, gpu

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    start = time.time()
    safe_print(f"[start][gpu {gpu}] {exp_name} -> {save_dir}", flush=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"video: {task['video_path']}\n")
        if task.get("row_index") is not None:
            log_file.write(f"csv_row: {task['row_index']}\n")
        if task.get("sample_index") is not None:
            log_file.write(f"sample_index: {task['sample_index']}\n")
        if task.get("target_pose"):
            log_file.write(f"target_pose: {task['target_pose']}\n")
        if task.get("pose_label"):
            log_file.write(f"pose_label: {task['pose_label']}\n")
        if task.get("prompt"):
            log_file.write(f"prompt: {task['prompt']}\n")
        if task.get("frontvideo"):
            log_file.write(f"frontvideo: {task['frontvideo']}\n")
        if task.get("warp_path"):
            log_file.write(f"warp_path: {task['warp_path']}\n")
        if task.get("mask_path"):
            log_file.write(f"mask_path: {task['mask_path']}\n")
        log_file.write(f"gpu: {gpu}\n")
        log_file.write(f"save_dir: {save_dir}\n")
        log_file.write(f"command: {shell_join(cmd)}\n\n")
        log_file.flush()
        proc = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )

    elapsed = time.time() - start
    if proc.returncode == 0:
        safe_print(f"[done][gpu {gpu}] {exp_name} ({elapsed / 60:.1f} min)", flush=True)
    else:
        safe_print(
            f"[fail][gpu {gpu}] {exp_name} rc={proc.returncode}; log={log_path}",
            flush=True,
        )
    return proc.returncode, task, gpu


def worker(gpu, tasks, args, extra_args, log_dir, results):
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return
        try:
            results.append(run_one(args, extra_args, task, gpu, log_dir))
        finally:
            tasks.task_done()


def require_columns(fieldnames, columns, csv_path):
    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")


def _optional_int(value):
    if value is None or value == "":
        return None
    return int(value)


def load_selected_rows_tasks(args):
    csv_path = Path(args.selected_rows_csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Selected rows CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        require_columns(
            fieldnames,
            [
                "exp_name",
                "target_pose",
                args.csv_video_column,
                args.csv_prompt_column,
            ],
            csv_path,
        )
        if args.enable_front_guidance and not args.auto_sam3_front_mask and not args.frontvideo:
            require_columns(fieldnames, [args.csv_front_column], csv_path)
        if args.use_existing_warp_mask:
            require_columns(fieldnames, [args.csv_warp_column, args.csv_mask_column], csv_path)
        rows = list(reader)

    tasks = []
    for row in rows:
        video_path = Path((row.get(args.csv_video_column) or "").strip())
        if not video_path.is_file():
            safe_print(f"[skip-row] missing input video for exp={row.get('exp_name')}: {video_path}", flush=True)
            continue
        frontvideo = None
        if (
            args.enable_front_guidance
            and not args.auto_sam3_front_mask
            and not args.frontvideo
        ):
            front_candidate = Path((row.get(args.csv_front_column) or "").strip())
            if front_candidate.is_file():
                frontvideo = front_candidate
        warp_path = None
        mask_path = None
        if args.use_existing_warp_mask:
            warp_candidate = Path((row.get(args.csv_warp_column) or "").strip())
            mask_candidate = Path((row.get(args.csv_mask_column) or "").strip())
            if warp_candidate.is_file() and mask_candidate.is_file():
                warp_path = warp_candidate
                mask_path = mask_candidate

        tasks.append(
            {
                "video_path": video_path,
                "prompt": (row.get(args.csv_prompt_column) or "").strip(),
                "frontvideo": frontvideo,
                "warp_path": warp_path,
                "mask_path": mask_path,
                "target_pose": (row.get("target_pose") or args.target_pose).strip(),
                "pose_label": (row.get("pose_label") or "").strip(),
                "exp_name": (row.get("exp_name") or "").strip(),
                "row_index": _optional_int(row.get("csv_row")),
                "sample_index": _optional_int(row.get("sample_index")),
                "pose_index": None,
                "source_row": row,
            }
        )
    return tasks


def load_csv_tasks(args, out_dir):
    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive.")

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        require_columns(
            reader.fieldnames or [],
            [args.csv_video_column, args.csv_prompt_column],
            csv_path,
        )
        if args.enable_front_guidance and not args.auto_sam3_front_mask and not args.frontvideo:
            require_columns(reader.fieldnames or [], [args.csv_front_column], csv_path)
        if args.use_existing_warp_mask:
            require_columns(
                reader.fieldnames or [],
                [args.csv_warp_column, args.csv_mask_column],
                csv_path,
            )
        rows = [(index, row) for index, row in enumerate(reader, start=2)]

    candidates = []
    for row_index, row in rows:
        video_path = Path((row.get(args.csv_video_column) or "").strip())
        if not video_path.is_file():
            continue
        candidates.append((row_index, row, video_path))

    if len(candidates) < args.sample_count:
        raise ValueError(
            f"Only {len(candidates)} CSV rows have valid {args.csv_video_column} videos"
            f"; need {args.sample_count}."
        )

    rng = random.Random(args.sample_seed)
    selected = rng.sample(candidates, args.sample_count)

    missing_front = []
    if args.enable_front_guidance and not args.auto_sam3_front_mask and not args.frontvideo:
        for row_index, row, _video_path in selected:
            front_path = Path((row.get(args.csv_front_column) or "").strip())
            if not front_path.is_file():
                missing_front.append((row_index, front_path))
        if missing_front:
            preview = ", ".join(
                f"row {row_index}: {front_path}" for row_index, front_path in missing_front[:5]
            )
            raise FileNotFoundError(
                f"{len(missing_front)} sampled rows are missing valid "
                f"{args.csv_front_column} files. First missing: {preview}"
            )

    missing_warp_mask = []
    if args.use_existing_warp_mask:
        for row_index, row, _video_path in selected:
            warp_path = Path((row.get(args.csv_warp_column) or "").strip())
            mask_path = Path((row.get(args.csv_mask_column) or "").strip())
            if not warp_path.is_file() or not mask_path.is_file():
                missing_warp_mask.append((row_index, warp_path, mask_path))
        if missing_warp_mask:
            preview = ", ".join(
                f"row {row_index}: warp={warp_path}, mask={mask_path}"
                for row_index, warp_path, mask_path in missing_warp_mask[:5]
            )
            raise FileNotFoundError(
                f"{len(missing_warp_mask)} sampled rows are missing valid "
                f"{args.csv_warp_column}/{args.csv_mask_column} files. "
                f"First missing: {preview}"
            )

    target_poses = args.target_poses or [args.target_pose]
    if args.pose_labels is not None and len(args.pose_labels) != len(target_poses):
        raise ValueError("--pose-labels must have the same length as --target-poses.")
    pose_labels = args.pose_labels or [
        pose_label_for(target_pose, index)
        for index, target_pose in enumerate(target_poses)
    ]

    selected_rows_path = out_dir / "selected_rows.csv"
    fieldnames = [
        "sample_index",
        "csv_row",
        "exp_name",
        "pose_label",
        "target_pose",
        *list((rows[0][1].keys() if rows else [])),
    ]
    tasks = []
    with selected_rows_path.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index, (row_index, row, video_path) in enumerate(selected):
            prompt = (row.get(args.csv_prompt_column) or "").strip()
            frontvideo = None
            if (
                args.enable_front_guidance
                and not args.auto_sam3_front_mask
                and not args.frontvideo
            ):
                frontvideo = Path((row.get(args.csv_front_column) or "").strip())
                if not frontvideo.is_file():
                    frontvideo = None
            warp_path = None
            mask_path = None
            if args.use_existing_warp_mask:
                warp_path = Path((row.get(args.csv_warp_column) or "").strip())
                mask_path = Path((row.get(args.csv_mask_column) or "").strip())
            for pose_index, (target_pose, pose_label) in enumerate(zip(target_poses, pose_labels)):
                exp_name = (
                    f"{args.exp_name_prefix}_{pose_label}_"
                    f"{sample_index:03d}_row{row_index}"
                )
                task = {
                    "video_path": video_path,
                    "prompt": prompt,
                    "frontvideo": frontvideo,
                    "warp_path": warp_path,
                    "mask_path": mask_path,
                    "target_pose": target_pose,
                    "pose_label": pose_label,
                    "exp_name": exp_name,
                    "row_index": row_index,
                    "sample_index": sample_index,
                    "pose_index": pose_index,
                    "source_row": row,
                }
                tasks.append(task)
                writer.writerow(
                    {
                        "sample_index": sample_index,
                        "csv_row": row_index,
                        "exp_name": exp_name,
                        "pose_label": pose_label,
                        "target_pose": target_pose,
                        **row,
                    }
                )
    return tasks, selected_rows_path, len(rows), len(candidates)


def main():
    args, extra_args = parse_args()
    input_dir = Path(args.input_dir) if args.input_dir else None
    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir) if args.log_dir else out_dir / "_logs"
    if input_dir is not None:
        args.input_dir = str(input_dir)
    args.out_dir = str(out_dir)

    gpus = split_gpus(args.gpus)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.frontvideo and not Path(args.frontvideo).is_file():
        raise FileNotFoundError(f"frontvideo not found: {args.frontvideo}")
    if (
        args.enable_front_guidance
        and not (args.csv_path or args.selected_rows_csv)
        and not args.frontvideo
        and not args.auto_sam3_front_mask
    ):
        raise ValueError(
            "Folder mode with --enable-front-guidance requires a global --frontvideo "
            "or --auto-sam3-front-mask."
        )
    if args.use_existing_warp_mask and not (args.csv_path or args.selected_rows_csv):
        raise ValueError("--use-existing-warp-mask requires --csv-path or --selected-rows-csv.")

    if args.selected_rows_csv:
        tasks_list = load_selected_rows_tasks(args)
        safe_print(
            f"Loaded {len(tasks_list)} tasks from selected rows: {args.selected_rows_csv}",
            flush=True,
        )
    elif args.csv_path:
        tasks_list, selected_rows_path, total_rows, valid_rows = load_csv_tasks(args, out_dir)
        safe_print(
            f"Selected {len(tasks_list)} rows from {args.csv_path} "
            f"(valid {valid_rows}/{total_rows}, seed {args.sample_seed})",
            flush=True,
        )
        safe_print(f"Selected rows: {selected_rows_path}", flush=True)
    else:
        if input_dir is None or not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        videos = sorted(input_dir.glob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No .mp4 files found in: {input_dir}")
        tasks_list = [make_folder_task(video) for video in videos]
        safe_print(f"Found {len(tasks_list)} videos in {input_dir}", flush=True)

    safe_print(f"Using GPUs: {', '.join(gpus)}", flush=True)
    safe_print(f"Output dir: {out_dir}", flush=True)
    safe_print(f"Log dir: {log_dir}", flush=True)

    tasks = queue.Queue()
    for task in tasks_list:
        tasks.put(task)

    results = []
    threads = [
        Thread(
            target=worker,
            args=(gpu, tasks, args, extra_args, log_dir, results),
            daemon=False,
        )
        for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failed = [(rc, task, gpu) for rc, task, gpu in results if rc != 0]
    succeeded = [(rc, task, gpu) for rc, task, gpu in results if rc == 0]
    safe_print(f"Finished: {len(succeeded)} ok, {len(failed)} failed", flush=True)
    if failed:
        for rc, task, gpu in failed:
            safe_print(
                f"  rc={rc} gpu={gpu} exp_name={exp_name_for(task, args.exp_name_prefix)} "
                f"video={task['video_path']}",
                flush=True,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
