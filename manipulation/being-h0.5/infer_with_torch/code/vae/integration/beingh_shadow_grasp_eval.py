# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared imports and constants for Being-H closed-loop evaluation."""

from dataclasses import dataclass

from vae.integration.beingh_shadow_grasp_common import *  # noqa: F401,F403
from vae.integration.beingh_shadow_grasp_primitives import *  # noqa: F401,F403
from vae.integration.beingh_shadow_grasp_policies import *  # noqa: F401,F403


def _add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--deployment-profile",
        choices=("legacy", "safe_smooth"),
        default="safe_smooth",
        help=(
                 'safe_smooth selects synchronous 16-step chunk execution, fixed query noise, chun'
                 'k velocity/acceleration smoothing, SO(3)/native-q motion limits, normalized wris'
                 't clipping, and a safety guard. legacy keeps the historical defaults unless indi'
                 'vidual options are supplied.'
             ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT
        / "outputs/shadow_grasp_bottle_1071"
        / (
              'train-shadow_grasp_0725_core_bottle_1071_Being-H05-2B_freeze-mllm-True_chunk-16_'
              'tok-8192_norm-wrist_euler_minmax_zraw_wristw-1.0_tdelta-0.0_mpg-True_20260730_10'
              '4141'
          )
        / "0020000",
        help=(
                 'Fine-tuned checkpoint directory containing config.json, tokenizer files, model s'
                 'afetensors, and shadow_grasp_posttrain_metadata.json.'
             ),
    )


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    data_config_help = (
        "Override the data transform stored with the checkpoint. "
        "Use "
        f"{Q99_DATA_CONFIG_NAME!r} for q01/q99 checkpoints, "
        f"{MINMAX_DATA_CONFIG_NAME!r} for full min-max checkpoints, "
        f"{WRIST_MINMAX_ZRAW_DATA_CONFIG_NAME!r} for wrist-only axis-angle checkpoints, "
        f"{WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME!r} for continuous Euler wrist checkpoints, "
        f"{WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME!r} for internal Rot6D wrist checkpoints, "
        f"{TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME!r} for the 22D physical-joint baseline, "
        f"and {RAW_DATA_CONFIG_NAME!r} for legacy raw checkpoints."
    )
    parser.add_argument(
        "--data-config-name",
        choices=SUPPORTED_DATA_CONFIG_NAMES,
        default=None,
        help=data_config_help,
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help=(
                 'Natural-language instruction override. When omitted, the episode task is read au'
                 'tomatically from an evaluation JSONL or from the LeRobot meta/tasks.jsonl annota'
                 'tion.'
             ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hand",
        choices=SUPPORTED_HANDS,
        default="shadow_hand_right",
        help="Target hand used by NativeVAE decoding and the MuJoCo environment.",
    )
    parser.add_argument("--vae-checkpoint", type=Path, default=VAE_ROOT / "checkpoints/native_n2_epoch800_inference.pt")
    parser.add_argument(
        "--joint-retargeting",
        choices=("auto", "none", "geometry"),
        default="auto",
        help=(
                 'For a physical Shadow-joint checkpoint, auto uses direct joints on Shadow and ge'
                 'ometry optimization on Gaia/Sharpa. z_gesture checkpoints continue to use Native'
                 'VAE.'
             ),
    )


def _add_model_hand_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--geometry-retargeting-profile",
        choices=("raw", "stable"),
        default="raw",
        help=(
                 'raw minimizes only the NativeVAE-compatible 60D geometry; stable also regularize'
                 's normalized target-joint velocity/acceleration.'
             ),
    )
    parser.add_argument(
        "--geometry-action-chunk-mode",
        choices=("auto", "batch", "sequential"),
        default="auto",
        help=(
                 'auto uses one batched geometry solve per action chunk for synchronous chunk sele'
                 'ction, and otherwise falls back to per-executed-step retargeting.'
             ),
    )
    parser.add_argument("--geometry-max-iterations", type=int, default=12)
    parser.add_argument("--geometry-learning-rate", type=float, default=0.8)
    parser.add_argument("--geometry-tolerance", type=float, default=1e-07)
    parser.add_argument("--geometry-temporal-weight", type=float, default=0.002)
    parser.add_argument("--geometry-acceleration-weight", type=float, default=0.0005)
    parser.add_argument(
        "--scene-xml",
        type=Path,
        default=None,
        help="Optional generated hand/object scene. The selected hand template is used if omitted.",
    )
    parser.add_argument(
        "--scene-contains-object",
        action="store_true",
        help=(
                 'Keep the object already embedded in --scene-xml instead of replacing it from the'
                 ' evaluation manifest. The manifest still supplies task, provenance, and success '
                 'metadata.'
             ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
                 'Portable JSONL manifest, LeRobot dataset directory, or source NPZ. When omitted,'
                 ' it is selected from the checkpoint data config.'
             ),
    )


def _add_episode_args(parser: argparse.ArgumentParser) -> None:
    episode_group = parser.add_mutually_exclusive_group()
    episode_group.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Evaluate one episode. Episode 4 is used when no episode option is given.",
    )
    episode_group.add_argument(
        "--episodes",
        "--episode-indices",
        dest="episode_indices",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate multiple episodes while loading Being-H and the VAE only once.",
    )
    episode_group.add_argument(
        "--episode-range",
        dest="episode_range",
        type=int,
        nargs=2,
        metavar=("A", "B"),
        default=None,
        help="Evaluate every episode in the inclusive index range [A, B] while loading Being-H and the VAE only once.",
    )
    episode_group.add_argument(
        "--all-episodes", action="store_true", help="Evaluate every episode available in the dataset or manifest."
    )


def _add_dataset_rollout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dataset", type=Path, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
                 'Closed-loop policy steps. For a JSONL manifest this defaults to the episode fram'
                 'e count; otherwise it defaults to 300.'
             ),
    )
    parser.add_argument(
        "--success-profile",
        choices=tuple(SUCCESS_PROFILES),
        default="strict",
        help=(
                 'Success criterion: strict requires a 0.20 m lift for 10 consecutive control fram'
                 'es; loose succeeds once lift reaches 0.10 m; dex-demo uses the source TissueBox '
                 "benchmark's 0.08 m threshold."
             ),
    )
    parser.add_argument(
        "--replan-every",
        type=int,
        default=16,
        help=(
                 'For chunk selection, actions executed before the next query. For temporal ensemb'
                 'le, the model query interval in environment steps.'
             ),
    )
    parser.add_argument(
        "--action-selection",
        choices=("chunk", "temporal_ensemble"),
        default="chunk",
        help=(
                 'chunk executes actions directly from one predicted chunk. temporal_ensemble quer'
                 'ies every replan_every steps and averages overlapping chunk predictions aligned '
                 'to the current execution time.'
             ),
    )


def _add_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--inference-mode",
        choices=("sync", "async"),
        default="sync",
        help=(
                 'async runs Being-H in one background worker while buffered actions continue; it '
                 'blocks only when a query slot or valid action is unavailable.'
             ),
    )
    parser.add_argument(
        "--temporal-ensemble-decay",
        type=float,
        default=0.1,
        help="Exponential age decay k in weight=exp(-k*age); larger values favor newer predictions.",
    )
    parser.add_argument(
        "--temporal-ensemble-max-history",
        type=int,
        default=None,
        help=(
                 'Action horizon in environment steps for temporal aggregation; old chunks expire '
                 'after this many steps. Defaults to the model chunk length.'
             ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise-mode",
        choices=("fixed_per_query", "rollout"),
        default="fixed_per_query",
        help=(
                 'fixed_per_query reuses one flow-noise template for a deterministic conditional p'
                 'olicy; rollout consumes a new noise sample per query.'
             ),
    )
    parser.add_argument(
        "--execution-mode",
        choices=("raw", "rate_limited"),
        default="raw",
        help=(
                 'raw evaluates model commands without hard motion limits. rate_limited constrains'
                 ' wrist and decoded-joint velocity/acceleration to the training distribution and '
                 'logs interventions.'
             ),
    )


def _add_smoothing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chunk-velocity-smoothing-weight",
        type=float,
        default=None,
        help="Quadratic first-difference penalty applied inside each predicted chunk.",
    )
    parser.add_argument(
        "--chunk-acceleration-smoothing-weight",
        type=float,
        default=None,
        help="Quadratic second-difference penalty applied inside each predicted chunk.",
    )
    parser.add_argument(
        "--clip-normalized-wrist-action",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
                 'Clamp normalized action.eef_position/action.eef_rotation to [-1,1] before invers'
                 'e normalization; z_gesture is never clipped here.'
             ),
    )
    parser.add_argument(
        "--bounded-wrist-euler",
        action="store_true",
        help=(
                 'Restore the legacy [-pi,pi] MuJoCo wrist hinge limits. By default wrist Euler jo'
                 'ints are continuous so nearest-branch commands can cross +/-pi without clipping '
                 'or taking a full-turn path.'
             ),
    )


def _add_control_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--control-reference-dataset",
        type=Path,
        default=None,
        help=(
                 'LeRobot dataset used for initialization and deployment limits. Defaults to the m'
                 'atching 2-camera dataset for shadow_grasp_2cam_* checkpoints and the legacy core'
                 '-bottle dataset otherwise.'
             ),
    )
    parser.add_argument(
        "--cross-hand-initialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
                 'For Gaia/Sharpa, initialize wrist and native joints from the matching Shadow LeR'
                 "obot episode's first wrist + z_gesture. Shadow evaluation is unchanged."
             ),
    )
    parser.add_argument(
        "--wrist-action-source",
        choices=("model", "dataset"),
        default="model",
        help=(
                 'model executes Being-H wrist predictions. dataset replaces only the executed wri'
                 'st6 with action[0:6] from the matching Shadow episode while retaining model-pred'
                 'icted z_gesture/finger actions.'
             ),
    )


def _add_rate_safety_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--latent-observation-mode",
        choices=("encoded", "commanded"),
        default="encoded",
        help=(
                 "For z_gesture checkpoints, encoded re-encodes the target hand's actual joints an"
                 'd commanded feeds back the previous selected latent action. For cross-hand physi'
                 'cal-joint geometry baselines, encoded retargets actual target joints back to Sha'
                 'dow and commanded feeds back the previous selected canonical Shadow-joint action'
                 '.'
             ),
    )
    parser.add_argument("--rate-limit-quantile", type=float, default=None)
    parser.add_argument(
        "--native-joint-rate-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
                 'Apply target-hand native-joint velocity/acceleration limits after VAE decoding o'
                 'r geometry retargeting. Use --no-native-joint-rate-limit to disable only this li'
                 'miter while keeping wrist rate limits, chunk smoothing, wrist clipping, and safe'
                 'ty unchanged.'
             ),
    )
    parser.add_argument(
        "--limit-z-gesture",
        action="store_true",
        help=(
                 'Also rate-limit z_gesture before VAE decoding. Disabled by default because decod'
                 'ed native joints are already rate-limited; enable only to reproduce the legacy d'
                 'ouble-limited controller.'
             ),
    )
    parser.add_argument(
        "--safety-mode",
        choices=("off", "hold", "terminate"),
        default=None,
        help=(
                 'On an abnormal observation/action, off raises normally, hold keeps the current p'
                 'hysical command for the remaining rollout, and terminate executes one hold comma'
                 'nd then ends the rollout.'
             ),
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--safety-max-wrist-position-error-m", type=float, default=0.25)
    parser.add_argument("--safety-max-wrist-rotation-error-deg", type=float, default=90.0)
    parser.add_argument("--safety-max-object-drop-m", type=float, default=0.3)
    parser.add_argument("--safety-max-workspace-radius-m", type=float, default=0.6)
    parser.add_argument("--num-inference-timesteps", type=int, default=None)
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=2,
        help=(
                 'Discard this many full VAE+Being-H queries before the first episode. The policy '
                 'and RNG are reset afterward, so rollout actions are unchanged.'
             ),
    )
    parser.add_argument("--disable-mpg", action="store_true")
    parser.add_argument("--mpg-refinement-iters", type=int, default=None)
    parser.add_argument(
        "--output", type=Path, default=None, help="Single-episode NPZ path or multi-episode rollout directory."
    )
    parser.add_argument(
        "--output-video", type=Path, default=None, help="Single-episode MP4 path or multi-episode video directory."
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=None,
        help="Single-episode metadata JSON path or multi-episode metadata directory.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Multi-episode summary JSON path; an automatic path is used by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/sim_evaluation",
        help="Directory used for automatic output paths.",
    )
    parser.add_argument(
        "--no-record-images", action="store_true", help="Do not keep rendered images in memory during evaluation."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Being-H05 Shadow z_gesture or physical-joint policy in MuJoCo."
    )
    _add_profile_args(parser)
    _add_data_args(parser)
    _add_model_hand_args(parser)
    _add_episode_args(parser)
    _add_dataset_rollout_args(parser)
    _add_execution_args(parser)
    _add_smoothing_args(parser)
    _add_control_args(parser)
    _add_rate_safety_args(parser)
    _add_output_args(parser)
    return parser.parse_args()


def apply_deployment_profile(args: argparse.Namespace) -> None:
    if args.deployment_profile == "safe_smooth":
        args.inference_mode = "sync"
        args.action_selection = "chunk"
        args.replan_every = 16
        args.noise_mode = "fixed_per_query"
        args.execution_mode = "rate_limited"
        if args.chunk_velocity_smoothing_weight is None:
            args.chunk_velocity_smoothing_weight = 1.0
        if args.chunk_acceleration_smoothing_weight is None:
            args.chunk_acceleration_smoothing_weight = 4.0
        if args.clip_normalized_wrist_action is None:
            args.clip_normalized_wrist_action = True
        if args.rate_limit_quantile is None:
            args.rate_limit_quantile = 0.995
        if args.safety_mode is None:
            args.safety_mode = "terminate"
    else:
        if args.chunk_velocity_smoothing_weight is None:
            args.chunk_velocity_smoothing_weight = 0.0
        if args.chunk_acceleration_smoothing_weight is None:
            args.chunk_acceleration_smoothing_weight = 0.0
        if args.clip_normalized_wrist_action is None:
            args.clip_normalized_wrist_action = False
        if args.rate_limit_quantile is None:
            args.rate_limit_quantile = 0.999
        if args.safety_mode is None:
            args.safety_mode = "off"
    LOGGER.info(
        f"[deployment] profile={args.deployment_profile} infer={args.inference_mode} "
        f"selection={args.action_selection} h={args.replan_every} noise={args.noise_mode} "
        f"execution={args.execution_mode} "
        f"chunk_v={args.chunk_velocity_smoothing_weight:g} "
        f"chunk_a={args.chunk_acceleration_smoothing_weight:g} "
        f"clip_wrist={args.clip_normalized_wrist_action} "
        f"native_q_limit={args.native_joint_rate_limit} safety={args.safety_mode}"
    )


def apply_success_profile(args: argparse.Namespace) -> None:
    criteria = SUCCESS_PROFILES[args.success_profile]
    args.success_lift_m = float(criteria["lift_m"])
    args.success_frames = int(criteria["frames"])
    LOGGER.info(
        f"[success criterion] profile={args.success_profile} "
        f"lift>={args.success_lift_m:g}m for {args.success_frames} consecutive control frames"
    )


def resolve_joint_retargeting_mode(args: argparse.Namespace, direct_shadow_joints: bool) -> None:
    if args.joint_retargeting == "auto":
        args.joint_retargeting = "geometry" if direct_shadow_joints and args.hand != "shadow_hand_right" else "none"
    if args.geometry_action_chunk_mode == "auto":
        args.geometry_action_chunk_mode = (
            "batch"
            if args.joint_retargeting == "geometry"
            and args.inference_mode == "sync"
            and (args.action_selection == "chunk")
            else "sequential"
        )
    LOGGER.info(
        f"[joint retargeting] mode={args.joint_retargeting} "
        f"profile={args.geometry_retargeting_profile} "
        f"action_chunk={args.geometry_action_chunk_mode}"
    )


def _validate_checkpoint_inputs(args: argparse.Namespace, direct_shadow_joints: bool) -> None:
    direct_shadow_joints = args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES
    required_model_files = (args.model_path / "config.json", args.model_path / "shadow_grasp_posttrain_metadata.json")
    for path in required_model_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required checkpoint file: {path}")
    if not list(args.model_path.glob("*.safetensors")):
        raise FileNotFoundError(f"No .safetensors file found in {args.model_path}")
    if not direct_shadow_joints and (not args.vae_checkpoint.is_file()):
        raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_checkpoint}")
    if args.scene_xml is not None and (not args.scene_xml.is_file()):
        raise FileNotFoundError(f"Scene XML not found: {args.scene_xml}")
    if args.scene_contains_object and args.scene_xml is None:
        raise ValueError("--scene-contains-object requires --scene-xml")
    if not args.dataset.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {args.dataset}")
    if args.replan_every <= 0:
        raise ValueError("--replan-every must be positive")
    if args.success_lift_m <= 0:
        raise ValueError("success lift threshold must be positive")
    if args.success_frames <= 0:
        raise ValueError("success frame count must be positive")
    if args.chunk_velocity_smoothing_weight < 0:
        raise ValueError("--chunk-velocity-smoothing-weight must be non-negative")
    if args.chunk_acceleration_smoothing_weight < 0:
        raise ValueError("--chunk-acceleration-smoothing-weight must be non-negative")
    if args.inference_mode == "async" and (
        args.chunk_velocity_smoothing_weight > 0 or args.chunk_acceleration_smoothing_weight > 0
    ):
        raise ValueError("Chunk smoothing currently requires --inference-mode sync")
    if not 0.0 < args.rate_limit_quantile <= 1.0:
        raise ValueError("--rate-limit-quantile must be in (0,1]")


def _validate_runtime_positive_options(args: argparse.Namespace) -> None:
    options = (
        "safety_max_wrist_position_error_m", "safety_max_wrist_rotation_error_deg",
        "safety_max_object_drop_m", "safety_max_workspace_radius_m",
    )
    for option in options:
        if getattr(args, option) <= 0:
            raise ValueError(f"--{option.replace('_', '-')} must be positive")


def _validate_runtime_args(args: argparse.Namespace) -> None:
    _validate_runtime_positive_options(args)
    if (
        args.action_selection == "temporal_ensemble"
        and args.temporal_ensemble_max_history is not None
        and args.replan_every > args.temporal_ensemble_max_history
    ):
        raise ValueError("--replan-every cannot exceed --temporal-ensemble-max-history")
    if args.inference_mode == "async" and args.action_selection != "temporal_ensemble":
        raise ValueError("--inference-mode async requires --action-selection temporal_ensemble")
    if (
        args.temporal_ensemble_decay < 0
        or (
            args.temporal_ensemble_max_history is not None
            and args.temporal_ensemble_max_history <= 0
        )
    ):
        raise ValueError("temporal ensemble parameters must be non-negative and max history positive")
    if args.num_inference_timesteps is not None and args.num_inference_timesteps <= 0:
        raise ValueError("--num-inference-timesteps must be positive")
    if args.warmup_queries < 0 or (args.mpg_refinement_iters is not None and args.mpg_refinement_iters < 0):
        raise ValueError("warmup and MPG refinement values must be non-negative")
    for option in ("geometry_max_iterations", "geometry_learning_rate", "geometry_tolerance"):
        if getattr(args, option) <= 0:
            raise ValueError(f"--{option.replace('_', '-')} must be positive")
    if args.geometry_temporal_weight < 0 or args.geometry_acceleration_weight < 0:
        raise ValueError("geometry temporal weights must be non-negative")


def _validate_geometry_args(args: argparse.Namespace, direct_shadow_joints: bool) -> None:
    needs_control_reference = (
        args.execution_mode == "rate_limited"
        or args.wrist_action_source == "dataset"
        or args.latent_observation_mode == "commanded"
        or (args.data_config_name in TWO_CAMERA_DATA_CONFIG_NAMES)
        or (args.hand != "shadow_hand_right" and args.cross_hand_initialization)
    )
    if needs_control_reference and (not args.control_reference_dataset.exists()):
        raise FileNotFoundError(f"Control reference dataset not found: {args.control_reference_dataset}")
    if args.limit_z_gesture and args.execution_mode != "rate_limited":
        raise ValueError("--limit-z-gesture requires --execution-mode rate_limited")
    if not direct_shadow_joints and args.joint_retargeting != "none":
        raise ValueError("--joint-retargeting is only applicable to a physical-joint checkpoint")


def _validate_physical_hand_args(args: argparse.Namespace, direct_shadow_joints: bool) -> None:
    if not direct_shadow_joints:
        return
    model_hand = DIRECT_JOINT_CONFIG_HANDS[args.data_config_name]
    if model_hand != "shadow_hand_right" and args.hand != model_hand:
        raise ValueError(f"Target-native joint checkpoint {args.data_config_name} must use --hand {model_hand}")
    if model_hand != "shadow_hand_right" and args.joint_retargeting != "none":
        raise ValueError("Target-native joint checkpoints require --joint-retargeting none")
    if model_hand == "shadow_hand_right" and args.hand != "shadow_hand_right" and args.joint_retargeting != "geometry":
        raise ValueError("Cross-hand physical-joint evaluation requires geometry retargeting")
    if model_hand == "shadow_hand_right" and args.hand == "shadow_hand_right" and args.joint_retargeting != "none":
        raise ValueError("Shadow evaluation requires --joint-retargeting none")


def _is_batch_geometry_config(args: argparse.Namespace) -> bool:
    return (
        args.joint_retargeting == "geometry"
        and args.inference_mode == "sync"
        and args.action_selection == "chunk"
    )


def _validate_geometry_chunk_args(args: argparse.Namespace) -> None:
    if args.geometry_action_chunk_mode == "batch" and not _is_batch_geometry_config(args):
        raise ValueError("Batch geometry retargeting requires sync chunk inference")


def _validate_commanded_feedback_args(args: argparse.Namespace, direct_shadow_joints: bool) -> None:
    if not direct_shadow_joints or DIRECT_JOINT_CONFIG_HANDS[args.data_config_name] != "shadow_hand_right":
        return
    cross_hand = args.hand != "shadow_hand_right"
    commanded = args.latent_observation_mode == "commanded"
    if commanded and cross_hand and args.joint_retargeting != "geometry":
        raise ValueError("Cross-hand commanded feedback requires geometry retargeting")
    if commanded and cross_hand and not args.cross_hand_initialization:
        raise ValueError("Geometry commanded feedback requires --cross-hand-initialization")


def _validate_control_args(args: argparse.Namespace, direct_shadow_joints: bool) -> None:
    _validate_physical_hand_args(args, direct_shadow_joints)
    _validate_geometry_chunk_args(args)
    _validate_commanded_feedback_args(args, direct_shadow_joints)
    if direct_shadow_joints and args.limit_z_gesture:
        raise ValueError("--limit-z-gesture is not applicable to a physical-joint checkpoint")


def validate_inputs(args: argparse.Namespace) -> None:
    direct_shadow_joints = args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES
    _validate_checkpoint_inputs(args, direct_shadow_joints)
    _validate_runtime_args(args)
    _validate_geometry_args(args, direct_shadow_joints)
    _validate_control_args(args, direct_shadow_joints)


def _jsonl_episode_indices(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as handle:
        return list(dict.fromkeys(int(json.loads(line)["episode_index"]) for line in handle if line.strip()))


def available_episode_indices(dataset: Path) -> list[int]:
    dataset = dataset.expanduser().resolve()
    if dataset.is_file() and dataset.suffix == ".jsonl":
        return _jsonl_episode_indices(dataset)
    if dataset.is_file() and dataset.suffix == ".npz":
        with np.load(dataset, allow_pickle=True) as payload:
            return list(range(int(np.asarray(payload["object_id"]).shape[0])))
    manifest_path = dataset / "episode_manifest.csv"
    if dataset.is_dir() and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            return list(dict.fromkeys(int(row["lerobot_episode_index"]) for row in csv.DictReader(handle)))
    raise ValueError(f"Cannot enumerate episodes from {dataset}; expected JSONL, NPZ, or episode_manifest.csv")


def resolve_episode_indices(args: argparse.Namespace, available: list[int]) -> list[int]:
    if args.all_episodes:
        selected = list(available)
    elif args.episode_range is not None:
        start, end = args.episode_range
        if start > end:
            raise ValueError(f"--episode-range requires A <= B, but received {start} {end}")
        selected = list(range(start, end + 1))
    elif args.episode_indices is not None:
        selected = list(dict.fromkeys(args.episode_indices))
    elif args.episode is not None:
        selected = [args.episode]
    else:
        selected = [4]
    available_set = set(available)
    invalid = [index for index in selected if index not in available_set]
    if invalid:
        invalid_preview = invalid[:20]
        invalid_suffix = f" ... ({len(invalid)} unavailable in total)" if len(invalid) > 20 else ""
        if available:
            availability = f"available index range is {min(available)}..{max(available)} ({len(available)} episodes)"
        else:
            availability = "the dataset contains no episodes"
        raise ValueError(f"Episodes {invalid_preview}{invalid_suffix} are unavailable; {availability}")
    if not selected:
        raise ValueError("At least one episode must be selected")
    return selected


def _batch_directory(path: Path, option: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.suffix:
        raise ValueError(f"In multi-episode mode {option} must be a directory, not a file")
    return resolved



DATA_CONFIG_LABELS = {
    Q99_DATA_CONFIG_NAME: "q99",
    MINMAX_DATA_CONFIG_NAME: "minmax",
    WRIST_MINMAX_ZRAW_DATA_CONFIG_NAME: "wrist-minmax_zraw",
    WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME: "wrist-euler-minmax-zraw",
    WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME: "wrist-rot6d-minmax-zraw",
    TWO_CAMERA_WRIST_EULER_MINMAX_ZRAW_DATA_CONFIG_NAME: "2cam-wrist-euler-minmax-zraw",
    TWO_CAMERA_WRIST_ROT6D_MINMAX_ZRAW_DATA_CONFIG_NAME: "2cam-wrist-rot6d-minmax-zraw",
    TWO_CAMERA_WRIST_ROT6D_MINMAX_JOINTS_DATA_CONFIG_NAME: "2cam-wrist-rot6d-minmax-joints",
    SHARPA_JOINT_DATA_CONFIG_NAME: "sharpa-2cam-wrist-rot6d-minmax-joints",
    GAIA_JOINT_DATA_CONFIG_NAME: "gaia-2cam-wrist-rot6d-minmax-joints",
    RAW_DATA_CONFIG_NAME: "raw",
}


def _execution_output_label(args: argparse.Namespace) -> str:
    if args.execution_mode != "rate_limited":
        return args.execution_mode
    if args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES:
        z_mode = "direct-joints"
    else:
        z_mode = "zlimited" if args.limit_z_gesture else "zraw"
    native = "nativeq" if args.native_joint_rate_limit else "nativeq-off"
    return f"rate_limited-q{args.rate_limit_quantile:g}-{z_mode}-{native}"


def _latent_output_label(args: argparse.Namespace) -> str:
    if args.latent_observation_mode == "encoded":
        return ""
    prefix = "jobs" if args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES else "zobs"
    return f"_{prefix}-{args.latent_observation_mode}"


def automatic_output_root(args: argparse.Namespace) -> Path:
    normalization = DATA_CONFIG_LABELS[args.data_config_name]
    mpg = "nompg" if args.disable_mpg else "mpg"
    denoise = str(args.num_inference_timesteps) if args.num_inference_timesteps is not None else "model"
    wrist_source = "" if args.wrist_action_source == "model" else "_wrist-dataset"
    latent = _latent_output_label(args)
    retarget = "" if args.joint_retargeting == "none" else (
        f"_retarget-{args.joint_retargeting}-{args.geometry_retargeting_profile}-"
        f"{args.geometry_action_chunk_mode}"
    )
    config = (
        f"{args.deployment_profile}{wrist_source}{latent}{retarget}_{normalization}_{args.inference_mode}_"
        f"{args.action_selection}_h{args.replan_every}_{args.noise_mode}_"
        f"seed{args.seed}_"
        f"{_execution_output_label(args)}_smooth-v"
        f"{args.chunk_velocity_smoothing_weight:g}-a"
        f"{args.chunk_acceleration_smoothing_weight:g}_"
        f"clipw-{int(args.clip_normalized_wrist_action)}_safety-{args.safety_mode}_"
        f"success-{args.success_profile}-lift{args.success_lift_m:g}-f{args.success_frames}_"
        f"{'wrist-bounded' if args.bounded_wrist_euler else 'wrist-unwrapped'}_{mpg}_d{denoise}"
    )
    return (
        args.output_dir.expanduser().resolve() / args.model_path.parent.name
        / args.model_path.name / args.hand / config
    )


def automatic_output_paths(args: argparse.Namespace, episode_index: int, num_episodes: int) -> tuple[Path, Path, Path]:
    episode_directory_name = f"episode_{episode_index:06d}"
    default_episode_directory = automatic_output_root(args) / episode_directory_name
    if num_episodes == 1:
        output = (
            args.output.expanduser().resolve() if args.output is not None else default_episode_directory / "rollout.npz"
        )
        output_video = (
            args.output_video.expanduser().resolve()
            if args.output_video is not None
            else default_episode_directory / "rollout.mp4"
        )
        output_metadata = (
            args.output_metadata.expanduser().resolve()
            if args.output_metadata is not None
            else default_episode_directory / "metadata.json"
        )
    else:
        rollout_directory = (
            _batch_directory(args.output, "--output") / episode_directory_name
            if args.output is not None
            else default_episode_directory
        )
        video_directory = (
            _batch_directory(args.output_video, "--output-video") / episode_directory_name
            if args.output_video is not None
            else default_episode_directory
        )
        metadata_directory = (
            _batch_directory(args.output_metadata, "--output-metadata") / episode_directory_name
            if args.output_metadata is not None
            else default_episode_directory
        )
        output = rollout_directory / "rollout.npz"
        output_video = video_directory / "rollout.mp4"
        output_metadata = metadata_directory / "metadata.json"
    return (output.resolve(), output_video.resolve(), output_metadata.resolve())


def warmup_closed_loop_policy(
    env: GraspEnv, policy: AuditedNativePolicy, num_queries: int, initial_action: np.ndarray | None = None
) -> None:
    if num_queries <= 0:
        return
    observation, _ = env.reset(initial_action=initial_action)
    latencies: list[float] = []
    LOGGER.info(f"[warmup] running {num_queries} discarded full-policy queries...")
    for index in range(num_queries):
        policy.reset()
        started_at = time.perf_counter()
        action = np.asarray(policy.predict(observation), dtype=np.float32)
        if not np.all(np.isfinite(action)):
            raise ValueError("Warm-up policy returned NaN or Inf")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started_at
        latencies.append(elapsed)
        LOGGER.info(f"[warmup] query={index + 1}/{num_queries} latency={elapsed * 1000.0:.1f}ms")
    policy.reset()
    LOGGER.info(f"[warmup] complete: {json.dumps(latency_summary_seconds(latencies))}")



def _stack_actions(values, width):
    return np.stack(values).astype(np.float32) if values else np.empty((0, width), dtype=np.float32)


def _collect_policy_arrays(latent_policy, policy, direct, args):
    predicted = np.stack(latent_policy.predicted_chunks).astype(np.float32)
    smoothed = np.stack(latent_policy.smoothed_predicted_chunks).astype(np.float32)
    observed = np.stack(latent_policy.observed_latent_states).astype(np.float32)
    encoded = _stack_actions(policy.policy.encoded_latent_states, policy.policy.z_dim)
    policy_states = _stack_actions(policy.policy.policy_latent_states, policy.policy.z_dim)
    if len(encoded) != len(policy_states):
        raise RuntimeError(
            "Encoded and policy-observation latent audit lengths differ: "
            f"{len(encoded)} != {len(policy_states)}"
        )
    gap = policy_states - encoded
    gap_norm = np.linalg.norm(gap, axis=1)
    if direct and args.hand == "shadow_hand_right":
        audit = {
            "mode": "direct_native_shadow_joints",
            "num_policy_steps": int(len(observed)),
            "joint_state_source": "MuJoCo observation.state[6:28]",
        }
    else:
        audit = {
            "mode": args.latent_observation_mode,
            "num_policy_steps": int(len(policy_states)),
            "commanded_vs_encoded_l2_mean": float(gap_norm.mean()) if len(gap_norm) else None,
            "commanded_vs_encoded_l2_max": float(gap_norm.max()) if len(gap_norm) else None,
        }
    return {
        "predicted": predicted, "smoothed": smoothed, "observed": observed,
        "encoded": encoded, "policy_states": policy_states, "gap": gap, "audit": audit,
        "latest_query": _stack_actions(latent_policy.latest_query_actions, predicted.shape[-1]),
        "temporal": _stack_actions(latent_policy.temporal_ensemble_actions, predicted.shape[-1]),
        "candidate_counts": np.asarray(latent_policy.temporal_ensemble_candidate_counts, dtype=np.int64),
        "raw_selected": _stack_actions(latent_policy.raw_selected_actions, predicted.shape[-1]),
        "branch_aligned": _stack_actions(latent_policy.branch_aligned_actions, predicted.shape[-1]),
        "executed": _stack_actions(latent_policy.executed_latent_actions, predicted.shape[-1]),
    }


def _collect_native_arrays(policy, geometric_adapter, hand_model, args):
    model_native = _stack_actions(policy.model_native_actions, 6 + len(hand_model.joint_names(args.hand)))
    raw_native = _stack_actions(policy.raw_native_actions, model_native.shape[-1])
    executed_native = _stack_actions(policy.executed_native_actions, model_native.shape[-1])
    wrist_requested = _stack_actions(policy.dataset_wrist_actions_requested, 6)
    native_delta = executed_native - raw_native
    empty = np.empty((0, len(hand_model.joint_names(args.hand))), dtype=np.float32)
    return {
        "model_native": model_native, "raw_native": raw_native,
        "executed_native": executed_native, "wrist_requested": wrist_requested,
        "native_delta": native_delta,
        "native_interventions": int(np.any(np.abs(native_delta) > 1e-6, axis=1).sum()),
        "shadow_observation": _stack_actions(
            geometric_adapter.retargeted_shadow_observation_joints, 22
        ) if geometric_adapter is not None else empty,
        "model_shadow": _stack_actions(
            geometric_adapter.model_shadow_action_joints, 22
        ) if geometric_adapter is not None else empty,
        "target_shadow": _stack_actions(
            geometric_adapter.retargeted_target_action_joints, len(hand_model.joint_names(args.hand))
        ) if geometric_adapter is not None else empty,
    }


def _collect_rollout_data(latent_policy, policy, hand_model, args, direct):
    arrays = _collect_policy_arrays(latent_policy, policy, direct, args)
    geometry = policy.policy if isinstance(policy.policy, GeometricShadowJointPolicyAdapter) else None
    native = _collect_native_arrays(policy, geometry, hand_model, args)
    delta = arrays["executed"] - arrays["branch_aligned"]
    interventions = int(np.any(np.abs(delta) > 1e-6, axis=1).sum())
    return {
        "predicted_chunks": arrays["predicted"],
        "smoothed_predicted_chunks": arrays["smoothed"],
        "observed_latent_states": arrays["observed"],
        "encoded_native_latent_states": arrays["encoded"],
        "policy_observation_latent_states": arrays["policy_states"],
        "latent_observation_gap": arrays["gap"],
        "latent_observation_audit": arrays["audit"],
        "latest_query_actions": arrays["latest_query"],
        "temporal_ensemble_actions": arrays["temporal"],
        "temporal_ensemble_candidate_counts": arrays["candidate_counts"],
        "raw_selected_actions": arrays["raw_selected"],
        "branch_aligned_actions": arrays["branch_aligned"],
        "executed_latent_actions": arrays["executed"],
        "model_native_actions": native["model_native"],
        "raw_native_actions": native["raw_native"],
        "executed_native_actions": native["executed_native"],
        "dataset_wrist_actions_requested": native["wrist_requested"],
        "native_filter_delta": native["native_delta"],
        "native_intervention_steps": native["native_interventions"],
        "retargeted_shadow_observation_joints": native["shadow_observation"],
        "model_shadow_action_joints": native["model_shadow"],
        "retargeted_target_action_joints": native["target_shadow"],
        "filter_delta": delta,
        "intervention_steps": interventions,
    }


def _episode_setup(context):
    args = context["args"]
    episode_index = context["episode_index"]
    num_episodes = context["num_episodes"]
    vae = context["vae"]
    geometry = context["geometry_retargeter"]
    latent_policy = context["latent_policy"]
    manifest = load_jsonl_metadata(args.dataset, episode_index)
    episode = load_dataset_object_episode(args.dataset, episode_index, source_dataset=args.source_dataset)
    instruction, source, task_index = resolve_episode_instruction(
        instruction_override=args.instruction, episode_index=episode_index,
        manifest_metadata=manifest, control_reference_dataset=args.control_reference_dataset,
    )
    max_steps = int(manifest.get("frames", 300)) if args.max_steps is None else args.max_steps
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    output, video, metadata = automatic_output_paths(args, episode_index, num_episodes)
    initial, init_metadata = build_cross_hand_initial_action(
        args=args, episode_index=episode_index, manifest_metadata=manifest, vae=vae,
        geometry_retargeter=geometry, policy_wrist_world_origin=latent_policy.policy_wrist_world_origin,
    )
    wrist_actions, wrist_metadata = load_dataset_wrist_action_trajectory(
        args=args, episode_index=episode_index, manifest_metadata=manifest,
        required_steps=max_steps, policy_wrist_world_origin=latent_policy.policy_wrist_world_origin,
    )
    return {
        "episode_index": episode_index, "manifest": manifest, "episode": episode, "instruction": instruction,
        "instruction_source": source, "task_index": task_index, "max_steps": max_steps,
        "output": output, "output_video": video, "output_metadata": metadata,
        "initial_action": initial, "initialization_metadata": init_metadata,
        "dataset_wrist_actions": wrist_actions, "wrist_action_metadata": wrist_metadata,
    }


def _resolve_commanded_initial(args, setup, vae, latent_policy, direct):
    metadata = setup["initialization_metadata"]
    if args.latent_observation_mode != "commanded":
        return None
    if direct:
        reference = metadata.get("reference_shadow_joint_position")
        if reference is None:
            raise RuntimeError(
                "Geometry commanded feedback requires reference Shadow joints "
                "from cross-hand initialization"
            )
        metadata["joint_observation_reference"] = "matching Shadow episode first observation.state[6:28]"
        return np.asarray(reference, dtype=np.float32)
    reference = metadata.get("reference_z_gesture")
    if reference is None:
        value, ref_metadata = load_reference_initial_z_gesture(
            args=args, episode_index=setup["episode_index"], manifest_metadata=setup["manifest"]
        )
        metadata["latent_observation_reference"] = ref_metadata
        metadata["reference_z_gesture"] = value.tolist()
        return value
    return np.asarray(reference, dtype=np.float32)


def _episode_env_config(args, setup, latent_policy):
    episode = setup["episode"]
    camera = "ego_opposite" if "ego_opposite" in latent_policy.required_cameras else latent_policy.required_cameras[0]
    return GraspEnvConfig(
        hand=args.hand, scene_xml=args.scene_xml, camera=camera,
        observation_cameras=latent_policy.required_cameras, width=320, height=240, fps=30,
        continuous_wrist_rotation=not args.bounded_wrist_euler, max_steps=setup["max_steps"],
        success_lift_m=args.success_lift_m, success_frames=args.success_frames,
        object_id=None if args.scene_contains_object else episode.object_id,
        object_scale=episode.scale, object_position=episode.position, object_quaternion=episode.quaternion,
    )



def _audit_initialization(context):
    args = context["args"]
    initial_action = context["initial_action"]
    actual_state = context["actual_state"]
    metadata = context["metadata"]
    latent_policy = context["latent_policy"]
    vae = context["vae"]
    direct = context["direct"]
    if initial_action is None:
        return np.empty((0,), dtype=np.float32)
    target_wrist = actual_state[:6].copy()
    target_wrist[:3] -= latent_policy.policy_wrist_world_origin
    target_wrist[3:6] += np.asarray(evaluation_policy_wrist_offset(args))
    if not direct:
        if vae is None:
            raise RuntimeError("z_gesture audit requires NativeVAE")
        actual_z = vae.encode(actual_state[6:], args.hand)[0].detach().cpu().numpy().astype(np.float32)
        reference = np.asarray(metadata["reference_z_gesture"], dtype=np.float32)
        denom = float(np.linalg.norm(reference) * np.linalg.norm(actual_z))
        metadata.update({
            "initial_native_state_actual": actual_state.tolist(),
            "initial_policy_wrist_actual": target_wrist.tolist(),
            "initial_z_gesture_reencoded": actual_z.tolist(),
            "z_cycle_l2": float(np.linalg.norm(actual_z - reference)),
            "z_cycle_cosine": float(np.dot(reference, actual_z) / denom) if denom > 0 else 0.0,
        })
        return actual_z
    reference_q = np.asarray(metadata["target_q_requested"], dtype=np.float32)
    metadata.update({
        "initial_native_state_actual": actual_state.tolist(),
        "initial_policy_wrist_actual": target_wrist.tolist(),
        "initial_target_joint_position_actual": actual_state[6:].tolist(),
        "joint_initialization_l2": float(np.linalg.norm(actual_state[6:] - reference_q)),
    })
    return np.empty((0,), dtype=np.float32)


def _latent_rollout_payload(context):
    latent_policy = context["latent_policy"]
    args = context["args"]
    direct = context["direct"]
    predicted = context["predicted_chunks"]
    return {
        "observed_latent_states": context["observed_latent_states"],
        "observed_policy_states": context["observed_latent_states"],
        "encoded_native_latent_states": context["encoded_native_latent_states"],
        "policy_observation_latent_states": context["policy_observation_latent_states"],
        "policy_observation_minus_encoded_latent": context["latent_observation_gap"],
        "encoded_shadow_joint_states": (
            context["encoded_native_latent_states"]
            if direct and args.hand != "shadow_hand_right" else np.empty((0, 22), dtype=np.float32)
        ),
        "policy_observation_shadow_joint_states": (
            context["policy_observation_latent_states"]
            if direct and args.hand != "shadow_hand_right" else np.empty((0, 22), dtype=np.float32)
        ),
        "raw_predicted_chunks": predicted,
        "smoothed_predicted_chunks": context["smoothed_predicted_chunks"],
        "query_step_indices": np.asarray(latent_policy.query_step_indices, dtype=np.int64),
        "latest_query_latent_actions": context["latest_query_actions"],
        "temporal_ensemble_latent_actions": context["temporal_ensemble_actions"],
        "temporal_ensemble_candidate_counts": context["temporal_ensemble_candidate_counts"],
        "raw_selected_latent_actions": context["raw_selected_actions"],
        "branch_aligned_latent_actions": context["branch_aligned_actions"],
        "executed_latent_actions": context["executed_latent_actions"],
        "action_filter_delta": context["filter_delta"],
    }


def _native_rollout_payload(context):
    return {
        "model_native_actions": context["model_native_actions"],
        "raw_native_actions": context["raw_native_actions"],
        "executed_native_actions": context["executed_native_actions"],
        "dataset_wrist_actions_requested": context["dataset_wrist_actions_requested"],
        "native_action_filter_delta": context["native_filter_delta"],
        "retargeted_shadow_observation_joints": context["retargeted_shadow_observation_joints"],
        "model_shadow_action_joints": context["model_shadow_action_joints"],
        "retargeted_target_action_joints": context["retargeted_target_action_joints"],
    }


def _initial_rollout_payload(context):
    metadata = context["initialization_metadata"]
    initial = context["initial_action"]
    return {
        "initial_native_action_requested": np.asarray(
            initial if initial is not None else [], dtype=np.float32
        ),
        "initial_native_state_actual": context["actual_initial_state"],
        "initial_reference_z_gesture": np.asarray(
            metadata.get("reference_z_gesture") or [], dtype=np.float32
        ),
        "initial_observed_z_gesture": context["actual_initial_z"],
        "initial_reference_shadow_joint_position": np.asarray(
            metadata.get("reference_shadow_joint_position") or [], dtype=np.float32
        ),
    }


def _save_rollout_arrays(context):
    output = context["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "states": context["result"].states,
        "actions": context["result"].actions,
        "object_poses": context["result"].object_poses,
        "beingh_query_latency_s": np.asarray(context["latent_policy"].query_latencies_s, dtype=np.float64),
        "closed_loop_policy_step_latency_s": np.asarray(context["policy"].predict_latencies_s, dtype=np.float64),
    }
    payload.update(_latent_rollout_payload(context))
    payload.update(_native_rollout_payload(context))
    payload.update(_initial_rollout_payload(context))
    np.savez_compressed(output, **payload)
    LOGGER.info(f"Saved rollout to {output}")


def _save_rollout_videos(context):
    args = context["args"]
    result = context["result"]
    env_config = context["env_config"]
    output_video = context["output_video"]
    saved_video_paths: dict[str, str] = {}
    if args.no_record_images:
        LOGGER.info("Video was not saved because --no-record-images was set.")
    else:
        images_by_camera = result.images_by_camera or {env_config.camera: result.images}
        for camera_name, camera_images in images_by_camera.items():
            camera_video = (
                output_video
                if camera_name == env_config.camera
                else output_video.with_name(f"{output_video.stem}_{camera_name}{output_video.suffix}")
            )
            write_mp4(camera_video, camera_images, env_config.fps)
            saved_video_paths[camera_name] = str(camera_video.resolve())
            LOGGER.info(f"Saved {camera_name} H.264 video to {camera_video}")
    return saved_video_paths


def _metadata_identity(c):
    args, episode, latent = c["args"], c["episode"], c["latent_policy"]
    env_config = c["env_config"]
    return {
        "model_path": str(args.model_path.resolve()),
        "deployment_profile": args.deployment_profile,
        "data_config_name": args.data_config_name,
        "data_config_source": args.data_config_source,
        "hand_action_representation": latent.hand_action_representation,
        "shadow_joint_contract": getattr(args, "shadow_joint_contract", None),
        "vae_checkpoint": None if c["direct"] else str(args.vae_checkpoint.resolve()),
        "joint_retargeting": args.joint_retargeting,
        "dataset": str(args.dataset.resolve()),
        "episode": c["episode_index"],
        "source_episode_index": episode.source_episode_index,
        "object_id": episode.object_id,
        "object_scale": episode.scale,
        "object_position": episode.position,
        "object_quaternion_wxyz": episode.quaternion,
        "scene_xml": None if args.scene_xml is None else str(args.scene_xml.resolve()),
        "scene_contains_object": args.scene_contains_object,
        "instruction": c["instruction"],
        "instruction_source": c["instruction_source"],
        "task_index": c["task_index"],
        "target_hand": args.hand,
        "target_hand_joint_names": list(c["hand_model"].joint_names(args.hand)),
        "target_native_action_dim": 6 + len(c["hand_model"].joint_names(args.hand)),
        "policy_wrist_euler_offset": list(evaluation_policy_wrist_offset(args)),
        "policy_wrist_world_origin": latent.policy_wrist_world_origin.tolist(),
        "initialization": c["initialization_metadata"],
        "latent_observation": c["latent_observation_audit"],
        "wrist_action_source": args.wrist_action_source,
        "wrist_action_source_metadata": c["wrist_action_metadata"],
        "camera": env_config.camera,
        "cameras": list(env_config.observation_cameras),
        "policy_video_source_columns": latent.video_source_columns,
        "fps": env_config.fps,
        "max_steps": c["max_steps"],
    }


def _metadata_runtime(c):
    args, latent = c["args"], c["latent_policy"]
    return {
        "success_criterion": {
            "profile": args.success_profile,
            "lift_m": args.success_lift_m,
            "consecutive_control_frames": args.success_frames,
        },
        "warmup_queries_configured": args.warmup_queries,
        "warmup_queries_applied": c["warmup_queries"],
        "render_on_query": (args.no_record_images and args.inference_mode == "sync"
                            and args.action_selection == "chunk"),
        "replan_every": args.replan_every,
        "action_selection": args.action_selection,
        "inference_mode": args.inference_mode,
        "temporal_ensemble_query_interval": args.replan_every if args.action_selection == "temporal_ensemble" else None,
        "temporal_ensemble_decay": args.temporal_ensemble_decay,
        "temporal_ensemble_max_history": latent.temporal_ensemble_max_history,
        "temporal_ensemble_candidate_count_max": int(c["temporal_ensemble_candidate_counts"].max()),
        "seed": args.seed,
        "noise_mode": args.noise_mode,
        "execution_mode": args.execution_mode,
        "chunk_smoothing": latent.chunk_smoother.metadata(),
        "clip_normalized_wrist_action": args.clip_normalized_wrist_action,
        "continuous_wrist_rotation": not args.bounded_wrist_euler,
        "num_queries": len(c["predicted_chunks"]),
        "async_inference": {
            "submitted_queries": latent.async_submitted_queries,
            "blocking_waits": latent.async_blocking_waits,
            "blocking_wait_seconds": latent.async_blocking_wait_seconds,
        },
        "timing": {
            "beingh_query": latency_summary_seconds(latent.query_latencies_s),
            "closed_loop_policy_step": latency_summary_seconds(c["policy"].predict_latencies_s),
        },
        "inference": {
            "num_inference_timesteps": int(latent.policy.model.num_inference_timesteps),
            "use_mpg": bool(latent.policy.model.use_mpg),
            "mpg_refinement_iters": int(latent.policy.model.mpg_refinement_iters),
        },
    }


def _metadata_metrics(c):
    args, policy, result = c["args"], c["policy"], c["result"]
    limiter = c["rate_limiter"]
    return {
        "geometry_retargeting": (
            policy.policy.retargeting_metadata()
            if isinstance(policy.policy, GeometricShadowJointPolicyAdapter) else None
        ),
        "rate_limiter": None if limiter is None else {
            **limiter.metadata(),
            "decoded_native_joint_limiter": args.native_joint_rate_limit,
        },
        "intervention_steps": c["intervention_steps"],
        "native_joint_rate_limit": args.native_joint_rate_limit,
        "native_q_step_limits": None if c["q_step_limits"] is None else c["q_step_limits"].tolist(),
        "native_q_acceleration_limits": (
            None if c["q_acceleration_limits"] is None
            else c["q_acceleration_limits"].tolist()
        ),
        "native_intervention_steps": c["native_intervention_steps"],
        "safety": {
            "mode": args.safety_mode,
            "events": policy.safety_events,
            "termination_requested": policy.termination_requested,
            "termination_reason": result.termination_reason,
            "max_wrist_position_error_m": args.safety_max_wrist_position_error_m,
            "max_wrist_rotation_error_deg": args.safety_max_wrist_rotation_error_deg,
            "max_object_drop_m": args.safety_max_object_drop_m,
            "max_workspace_radius_m": args.safety_max_workspace_radius_m,
        },
        "raw_selected_wrist_motion": wrist_motion_metrics(c["raw_selected_actions"]),
        "executed_wrist_motion": wrist_motion_metrics(c["executed_latent_actions"]),
        "model_native_wrist_motion": wrist_motion_metrics(c["model_native_actions"]),
        "dataset_wrist_motion": (
            wrist_motion_metrics(c["dataset_wrist_actions_requested"])
            if len(c["dataset_wrist_actions_requested"]) else None
        ),
        "actual_wrist_motion": wrist_motion_metrics(result.states[:-1, :6]),
        "success": result.success,
        "steps": result.steps,
        "max_lift_m": result.max_lift_m,
        "rollout": str(c["output"].resolve()),
        "video": None if args.no_record_images else c["saved_video_paths"].get(c["env_config"].camera),
        "videos": c["saved_video_paths"],
        "metadata": str(c["output_metadata"].resolve()),
    }


def _execute_episode(context):
    args = context["args"]
    env_config = context["env_config"]
    policy = context["policy"]
    warmup_queries = context["warmup_queries"]
    initial_action = context["initial_action"]
    max_steps = context["max_steps"]
    with GraspEnv(env_config) as env:
        warmup_closed_loop_policy(env, policy, warmup_queries, initial_action=initial_action)
        return PolicyEvaluationClient(env, policy).run(
            initial_action=initial_action,
            max_steps=max_steps,
            record_images=not args.no_record_images,
            observation_images_required=context["latent_policy"].will_query_next,
        )


def _evaluate_episode_body(c):
    args, episode_index = c["args"], c["episode_index"]
    latent_policy, policy = c["latent_policy"], c["policy"]
    hand_model = c["hand_model"]
    rate_limiter = c["rate_limiter"]
    q_step_limits, q_acceleration_limits = c["q_step_limits"], c["q_acceleration_limits"]
    warmup_queries = c["warmup_queries"]
    direct = args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES
    setup = _episode_setup(c)
    episode, instruction = setup["episode"], setup["instruction"]
    instruction_source, task_index = setup["instruction_source"], setup["task_index"]
    wrist_action_metadata = setup["wrist_action_metadata"]
    output, output_video, output_metadata = setup["output"], setup["output_video"], setup["output_metadata"]
    initial_action, initialization_metadata = setup["initial_action"], setup["initialization_metadata"]
    max_steps = setup["max_steps"]
    policy.set_wrist_action_trajectory(setup["dataset_wrist_actions"])
    initial_z = _resolve_commanded_initial(args, setup, c["vae"], latent_policy, direct)
    policy.set_initial_commanded_z(initial_z)
    latent_policy.instruction = instruction
    env_config = _episode_env_config(args, setup, latent_policy)
    result = _execute_episode({**c, "env_config": env_config, "initial_action": initial_action, "max_steps": max_steps})
    actual_initial_state = np.asarray(result.states[0], dtype=np.float32)
    actual_initial_z = _audit_initialization({
        "args": args, "initial_action": initial_action, "actual_state": actual_initial_state,
        "metadata": initialization_metadata, "latent_policy": latent_policy,
        "vae": c["vae"], "direct": direct,
    })
    latent_policy.finish_episode()
    data = _collect_rollout_data(latent_policy, policy, c["hand_model"], args, direct)
    context = {**locals(), **data}
    _save_rollout_arrays(context)
    context["saved_video_paths"] = _save_rollout_videos(context)
    metadata = {**_metadata_identity(context), **_metadata_runtime(context), **_metadata_metrics(context)}
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )
    LOGGER.info(f"Saved metadata to {output_metadata}")
    return metadata


def evaluate_episode(
    *,
    args: argparse.Namespace,
    episode_index: int,
    num_episodes: int,
    vae: NativeVAE | None,
    hand_model: Any,
    geometry_retargeter: GeometryRetargeter | None,
    latent_policy: BeingHGesturePolicy,
    policy: AuditedNativePolicy,
    rate_limiter: TrainingDistributionRateLimiter | None,
    q_step_limits: np.ndarray | None,
    q_acceleration_limits: np.ndarray | None,
    warmup_queries: int,
) -> dict[str, Any]:
    return _evaluate_episode_body(locals())


def _prepare_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    apply_deployment_profile(args)
    apply_success_profile(args)
    args.data_config_name, args.data_config_source = resolve_checkpoint_data_config(
        args.model_path, args.data_config_name, SUPPORTED_DATA_CONFIG_NAMES
    )
    direct = args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES
    resolve_joint_retargeting_mode(args, direct)
    if args.dataset is None:
        args.dataset = (
            TWO_CAMERA_EVALUATION_DATASET
            if args.data_config_name in TWO_CAMERA_DATA_CONFIG_NAMES
            else LEGACY_EVALUATION_DATASET
        )
    if args.control_reference_dataset is None:
        args.control_reference_dataset = {
            SHARPA_JOINT_DATA_CONFIG_NAME: SHARPA_CONTROL_REFERENCE_DATASET,
            GAIA_JOINT_DATA_CONFIG_NAME: GAIA_CONTROL_REFERENCE_DATASET,
        }.get(
            args.data_config_name,
            TWO_CAMERA_CONTROL_REFERENCE_DATASET
            if args.data_config_name in TWO_CAMERA_DATA_CONFIG_NAMES
            else LEGACY_CONTROL_REFERENCE_DATASET,
        )
    validate_inputs(args)
    available = available_episode_indices(args.dataset)
    episode_indices = resolve_episode_indices(args, available)
    seed_everything(args.seed)
    include = args.data_config_name in ROT6D_DATA_CONFIG_NAMES
    rate_limiter = None
    if args.execution_mode == "rate_limited":
        rate_limiter = TrainingDistributionRateLimiter.from_lerobot_dataset(
            args.control_reference_dataset, args.rate_limit_quantile,
            limit_z_gesture=args.limit_z_gesture, include_xyz_equivalent=include,
            direct_native_joints=direct,
        )
    return {
        "args": args, "direct": direct, "include": include,
        "episode_indices": episode_indices, "rate_limiter": rate_limiter,
    }


def _load_hand_runtime(args: argparse.Namespace, direct: bool) -> dict[str, Any]:
    geometry = None
    vae = None
    if direct:
        geometry = GeometryRetargeter(
            device=args.device,
            config=GeometryRetargeterConfig(
                profile=args.geometry_retargeting_profile,
                max_iterations=args.geometry_max_iterations,
                learning_rate=args.geometry_learning_rate,
                tolerance=args.geometry_tolerance,
                temporal_weight=args.geometry_temporal_weight,
                acceleration_weight=args.geometry_acceleration_weight,
            ),
        )
        hand_model = geometry
    else:
        vae = NativeVAE.from_pretrained(checkpoint=args.vae_checkpoint, device=args.device)
        hand_model = vae
    if args.hand not in hand_model.hand_names:
        raise ValueError(f"Hand runtime does not support {args.hand!r}; available={hand_model.hand_names}")
    if direct:
        model_hand = DIRECT_JOINT_CONFIG_HANDS[args.data_config_name]
        if model_hand == "shadow_hand_right":
            args.shadow_joint_contract = validate_shadow_joint_contract(
                args.control_reference_dataset, hand_model.joint_names("shadow_hand_right")
            )
        else:
            dim = DIRECT_JOINT_CONFIG_DIMS[args.data_config_name]
            args.shadow_joint_contract = {
                "dataset_to_mujoco_signs": [1.0] * dim,
                "dataset_joint_names": list(hand_model.joint_names(model_hand)),
                "mujoco_joint_names": list(hand_model.joint_names(model_hand)),
                "naming": "target_native_identity",
            }
    else:
        args.shadow_joint_contract = None
    return {"geometry": geometry, "vae": vae, "hand_model": hand_model}


def _build_latent_policy(args, episode_indices, include, rate_limiter):
    manifest = load_jsonl_metadata(args.dataset, episode_indices[0])
    instruction, source, _ = resolve_episode_instruction(
        instruction_override=args.instruction, episode_index=episode_indices[0],
        manifest_metadata=manifest, control_reference_dataset=args.control_reference_dataset,
    )
    smoother = ChunkTemporalSmoother(
        args.chunk_velocity_smoothing_weight, args.chunk_acceleration_smoothing_weight,
        include_xyz_equivalent=include,
    )
    policy = BeingHGesturePolicy(
        model_path=args.model_path, data_config_name=args.data_config_name,
        instruction=instruction, device=args.device, replan_every=args.replan_every,
        seed=args.seed, noise_mode=args.noise_mode, action_selection=args.action_selection,
        inference_mode=args.inference_mode, temporal_ensemble_decay=args.temporal_ensemble_decay,
        temporal_ensemble_max_history=args.temporal_ensemble_max_history, rate_limiter=rate_limiter,
        chunk_smoother=smoother, clip_normalized_wrist_action=args.clip_normalized_wrist_action,
        use_mpg=False if args.disable_mpg else None, mpg_refinement_iters=args.mpg_refinement_iters,
        num_inference_timesteps=args.num_inference_timesteps,
    )
    return policy


def _build_gesture_policy(c):
    args, direct = c["args"], c["direct"]
    latent, vae = c["latent"], c["vae"]
    geometry, hand_model = c["geometry"], c["hand_model"]
    offset = np.asarray(evaluation_policy_wrist_offset(args))
    if direct and (args.hand == "shadow_hand_right" or DIRECT_JOINT_CONFIG_HANDS[args.data_config_name] == args.hand):
        return ShadowJointPolicyAdapter(
            latent, target_hand=args.hand, joint_names=hand_model.joint_names(args.hand),
            dataset_to_mujoco_signs=np.asarray(args.shadow_joint_contract["dataset_to_mujoco_signs"], dtype=np.float32),
            policy_wrist_euler_offset=offset, policy_wrist_world_origin=latent.policy_wrist_world_origin,
            observation_mode=args.latent_observation_mode,
        )
    if direct:
        if geometry is None:
            raise RuntimeError("Geometry retargeter was not initialized")
        return GeometricShadowJointPolicyAdapter(
            latent, retargeter=geometry, target_hand=args.hand,
            dataset_to_mujoco_signs=np.asarray(args.shadow_joint_contract["dataset_to_mujoco_signs"], dtype=np.float32),
            policy_wrist_euler_offset=offset, policy_wrist_world_origin=latent.policy_wrist_world_origin,
            action_chunk_mode=args.geometry_action_chunk_mode, observation_mode=args.latent_observation_mode,
        )
    if vae is None:
        raise RuntimeError("NativeVAE was not initialized")
    return GesturePolicyAdapter(
        latent, vae=vae, target_hand=args.hand, encode_observation=True,
        latent_observation_mode=args.latent_observation_mode, policy_wrist_euler_offset=offset,
        policy_wrist_world_origin=latent.policy_wrist_world_origin,
    )


def _build_audited_policy(args, gesture, vae, geometry, direct):
    q_step = q_accel = None
    if args.execution_mode == "rate_limited" and args.native_joint_rate_limit:
        q_step, q_accel = AuditedNativePolicy.q_motion_limits_from_lerobot_dataset(
            args.control_reference_dataset, vae, args.hand, args.rate_limit_quantile,
            direct_shadow_joints=direct, geometry_retargeter=geometry,
        )
    policy = AuditedNativePolicy(
        gesture, q_step, q_accel, safety_mode=args.safety_mode,
        safety_max_wrist_position_error_m=args.safety_max_wrist_position_error_m,
        safety_max_wrist_rotation_error_deg=args.safety_max_wrist_rotation_error_deg,
        safety_max_object_drop_m=args.safety_max_object_drop_m,
        safety_max_workspace_radius_m=args.safety_max_workspace_radius_m,
    )
    return policy, q_step, q_accel


def _run_episode_batch(context):
    args = context["args"]
    indices = context["indices"]
    vae = context["vae"]
    hand_model = context["hand_model"]
    geometry = context["geometry"]
    latent = context["latent"]
    policy = context["policy"]
    limiter = context["limiter"]
    q_step, q_accel = context["q_step"], context["q_accel"]
    results = []
    try:
        for index in indices:
            results.append(evaluate_episode(
                args=args, episode_index=index, num_episodes=len(indices), vae=vae,
                hand_model=hand_model, geometry_retargeter=geometry, latent_policy=latent,
                policy=policy, rate_limiter=limiter, q_step_limits=q_step,
                q_acceleration_limits=q_accel,
                warmup_queries=args.warmup_queries if not results else 0,
            ))
    finally:
        latent.close()
    return results


def _summary_configuration(args) -> dict[str, Any]:
    return {
        "hand": args.hand, "replan_every": args.replan_every,
        "action_selection": args.action_selection, "inference_mode": args.inference_mode,
        "noise_mode": args.noise_mode, "execution_mode": args.execution_mode,
        "success_criterion": {
            "profile": args.success_profile, "lift_m": args.success_lift_m,
            "consecutive_control_frames": args.success_frames,
        },
        "continuous_wrist_rotation": not args.bounded_wrist_euler,
        "wrist_action_source": args.wrist_action_source,
        "latent_observation_mode": args.latent_observation_mode,
        "cross_hand_initialization": args.cross_hand_initialization,
        "control_reference_dataset": str(args.control_reference_dataset.resolve()),
        "rate_limit_quantile": args.rate_limit_quantile,
        "limit_z_gesture": args.limit_z_gesture,
        "native_joint_rate_limit": args.native_joint_rate_limit,
        "seed": args.seed,
    }


def _write_batch_summary(args, indices, results, latent):
    if len(results) <= 1:
        return
    successes = [bool(item["success"]) for item in results]
    successful = [int(item["episode"]) for item in results if bool(item["success"])]
    failed = [int(item["episode"]) for item in results if not bool(item["success"])]
    lifts = np.asarray([float(item["max_lift_m"]) for item in results], dtype=np.float64)
    summary = {
        "model_path": str(args.model_path.resolve()), "data_config_name": args.data_config_name,
        "data_config_source": args.data_config_source,
        "hand_action_representation": latent.hand_action_representation,
        "shadow_joint_contract": args.shadow_joint_contract,
        "vae_checkpoint": str(args.vae_checkpoint.resolve()), "dataset": str(args.dataset.resolve()),
        "episode_indices": indices, "num_episodes": len(results),
        "num_successes": len(successful), "num_failures": len(failed),
        "success_rate": float(np.mean(successes)), "successful_episode_indices": successful,
        "failed_episode_indices": failed, "mean_max_lift_m": float(lifts.mean()),
        "max_lift_m": float(lifts.max()),
        "total_steps": int(sum(int(item["steps"]) for item in results)),
        "total_queries": int(sum(int(item["num_queries"]) for item in results)),
        "configuration": _summary_configuration(args), "episodes": results,
    }
    path = args.summary_output.expanduser().resolve() if args.summary_output else (
        automatic_output_root(args) / f"summary_episodes_{indices[0]:06d}_{indices[-1]:06d}_n{len(indices)}.json"
    )
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    LOGGER.info(f"Saved batch summary to {path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepared = _prepare_evaluation(parse_args())
    args, indices = prepared["args"], prepared["episode_indices"]
    direct, include, limiter = prepared["direct"], prepared["include"], prepared["rate_limiter"]
    runtime = _load_hand_runtime(args, direct)
    latent = _build_latent_policy(args, indices, include, limiter)
    gesture = _build_gesture_policy({
        "args": args, "direct": direct, "latent": latent, "vae": runtime["vae"],
        "geometry": runtime["geometry"], "hand_model": runtime["hand_model"],
    })
    policy, q_step, q_accel = _build_audited_policy(
        args, gesture, runtime["vae"], runtime["geometry"], direct
    )
    results = _run_episode_batch({
        "args": args, "indices": indices, "vae": runtime["vae"],
        "hand_model": runtime["hand_model"], "geometry": runtime["geometry"],
        "latent": latent, "policy": policy, "limiter": limiter,
        "q_step": q_step, "q_accel": q_accel,
    })
    _write_batch_summary(args, indices, results, latent)


if __name__ == "__main__":
    main()
