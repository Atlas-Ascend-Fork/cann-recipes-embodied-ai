# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared imports and constants for Being-H closed-loop evaluation."""

from dataclasses import dataclass

from .beingh_shadow_grasp_common import *  # noqa: F401,F403


def evaluation_policy_wrist_offset(args: argparse.Namespace) -> np.ndarray:
    """Return the wrist-frame offset used by the model being evaluated.

    Shadow-canonical VAE/joint models use the Being-H/Shadow policy frame and
    therefore need the target-hand offset. Target-native joint baselines were
    trained directly on the target hand's native wrist coordinates, so their
    model boundary must not apply the Shadow->target offset a second time.
    """
    if (
        args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES
        and DIRECT_JOINT_CONFIG_HANDS[args.data_config_name] == args.hand
        and (args.hand != "shadow_hand_right")
    ):
        return np.zeros(3, dtype=np.float32)
    return np.asarray(POLICY_WRIST_EULER_OFFSETS[args.hand], dtype=np.float32)


SHADOW_DATASET_NEGATED_JOINT_NAMES = frozenset({"THJ2", "THJ1", "THJ0"})
LEGACY_CONTROL_REFERENCE_DATASET = REPO_ROOT / "data/shadow_grasp_0725_core_bottle_1071"
TWO_CAMERA_CONTROL_REFERENCE_DATASET = REPO_ROOT / "data/shadow_grasp_bottle22249179_aug100_2cam"
SHARPA_CONTROL_REFERENCE_DATASET = REPO_ROOT / "data/sharpa_grasp_bottle22249179_geo_visual100_2cam"
GAIA_CONTROL_REFERENCE_DATASET = REPO_ROOT / "data/gaia_grasp_bottle22249179_geo_visual100_2cam"
LEGACY_EVALUATION_DATASET = VAE_ROOT / "evaluation/object_episodes/shadow_grasp_0725.jsonl"
TWO_CAMERA_EVALUATION_DATASET = VAE_ROOT / "evaluation/object_episodes/shadow_grasp_bottle22249179_aug100_2cam.jsonl"
INSTRUCTION_TEMPLATE = (
    "According to the instruction '{task_description}', what's the micro-step actions in the next {k} steps?"
)
DEFAULT_INSTRUCTION = "Grasp the object and lift it up."


@dataclass(frozen=True)
class _DatasetInitialActionRequest:
    args: argparse.Namespace
    episode_index: int
    metadata: Mapping[str, Any]
    vae: NativeVAE | None
    geometry_retargeter: GeometryRetargeter | None
    world_origin: np.ndarray


@dataclass(frozen=True)
class _ReferenceTarget:
    target_q: np.ndarray
    direct: bool
    dataset_q: np.ndarray | None
    shadow_q: np.ndarray | None
    reference_z: np.ndarray | None
    retargeted: Any | None
    wrist: np.ndarray


def json_default(value: Any) -> Any:
    """Convert NumPy scalars/arrays and Paths for JSON metadata output."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _dataset_joint_names(info_path: Path) -> tuple[str, ...]:
    if not info_path.is_file():
        raise FileNotFoundError(f"Physical-joint evaluation requires LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    state_names = features.get("observation.state", {}).get("names")
    action_names = features.get("action", {}).get("names")
    if not isinstance(state_names, list) or not isinstance(action_names, list):
        raise ValueError(f"Physical-joint evaluation requires named state/action features in {info_path}")
    state_joints = tuple(str(name) for name in state_names[6:28])
    action_joints = tuple(str(name) for name in action_names[6:28])
    if state_joints != action_joints:
        raise ValueError(f"Dataset state/action Shadow joint orders differ: {state_joints} vs {action_joints}")
    return state_joints


def _joint_name_mapping(mujoco_joint_names: tuple[str, ...]) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    zero_based = []
    mapping = []
    for index, name in enumerate(mujoco_joint_names):
        prefix = name.rstrip("0123456789")
        suffix = name[len(prefix):]
        if not suffix or int(suffix) <= 0:
            raise ValueError(f"Cannot convert MuJoCo Shadow joint name {name!r}")
        source = f"{prefix}{int(suffix) - 1}"
        zero_based.append(source)
        mapping.append({"dataset": source, "mujoco": name, "index": index,
                        "dataset_to_mujoco_sign": -1.0 if source in SHADOW_DATASET_NEGATED_JOINT_NAMES else 1.0})
    return tuple(zero_based), mapping


def validate_shadow_joint_contract(dataset: Path, mujoco_joint_names: tuple[str, ...]) -> dict[str, Any]:
    dataset_names = _dataset_joint_names(dataset.expanduser().resolve() / "meta/info.json")
    zero_based, mapping = _joint_name_mapping(mujoco_joint_names)
    if dataset_names == tuple(mujoco_joint_names):
        naming = "identical"
    elif dataset_names == zero_based:
        naming = "dataset_zero_based_mujoco_one_based"
    else:
        raise ValueError(
            "Dataset Shadow joints are not positionally compatible with MuJoCo. "
            f"dataset={dataset_names}, mujoco={mujoco_joint_names}, expected_zero_based={zero_based}"
        )
    contract = {"naming": naming, "dataset_joint_names": list(dataset_names),
                "mujoco_joint_names": list(mujoco_joint_names), "mapping": mapping}
    contract["dataset_to_mujoco_signs"] = [item["dataset_to_mujoco_sign"] for item in mapping]
    LOGGER.info("[shadow joint contract] naming=%s dims=%s positional=true", naming, len(dataset_names))
    return contract


def mujoco_camera_name(source_column: str) -> str:
    prefix = "observation.images."
    if not source_column.startswith(prefix):
        raise ValueError(f"Unsupported simulation video source column {source_column!r}; expected {prefix!r} prefix")
    camera_name = source_column[len(prefix):]
    if not camera_name:
        raise ValueError(f"Empty MuJoCo camera name in {source_column!r}")
    return camera_name


def _read_jsonl_record(path: Path, episode_index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record["episode_index"]) == episode_index:
                return dict(record)
    raise IndexError(f"Episode {episode_index} is not in {path}")


def _task_record(line: str) -> tuple[int, str] | None:
    if not line.strip():
        return None
    record = json.loads(line)
    return int(record["task_index"]), str(record["task"])


def _load_task_index(tasks_path: Path) -> dict[int, str]:
    tasks = {}
    with tasks_path.open("r", encoding="utf-8") as file:
        for line in file:
            pair = _task_record(line)
            if pair is not None:
                index, task = pair
                tasks[index] = task
    return tasks


def _resolve_task_metadata(metadata: dict[str, Any], tasks_by_index: Mapping[int, str]) -> dict[str, Any]:
    task_index = metadata.get("task_index")
    if task_index is None and metadata.get("task_indices"):
        task_index = metadata["task_indices"][0]
    task = tasks_by_index.get(int(task_index)) if task_index is not None else None
    if task is None and metadata.get("tasks"):
        task = str(metadata["tasks"][0])
        matches = [index for index, value in tasks_by_index.items() if value == task]
        if len(matches) == 1:
            task_index = matches[0]
    if task is None and metadata.get("task"):
        task = str(metadata["task"])
    if task is None and len(tasks_by_index) == 1:
        task_index, task = next(iter(tasks_by_index.items()))
    if task is not None:
        metadata["task"], metadata["task_source"] = task, "meta/tasks.jsonl"
    if task_index is not None:
        metadata["task_index"] = int(task_index)
    return metadata


def _load_dataset_metadata(dataset: Path, episode_index: int) -> dict[str, Any]:
    episodes_path, tasks_path = dataset / "meta/episodes.jsonl", dataset / "meta/tasks.jsonl"
    if not episodes_path.is_file() or not tasks_path.is_file():
        return {}
    metadata = _read_jsonl_record(episodes_path, episode_index)
    return _resolve_task_metadata(metadata, _load_task_index(tasks_path))


def load_jsonl_metadata(dataset: Path, episode_index: int) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve()
    if dataset.is_file() and dataset.suffix == ".jsonl":
        metadata = _read_jsonl_record(dataset, episode_index)
        if metadata.get("task"):
            metadata["task_source"] = "evaluation_jsonl.task"
        return metadata
    if not dataset.is_dir():
        return {}
    return _load_dataset_metadata(dataset, episode_index)


def resolve_episode_instruction(
    *,
    instruction_override: str | None,
    episode_index: int,
    manifest_metadata: Mapping[str, Any],
    control_reference_dataset: Path,
) -> tuple[str, str, int | None]:
    """Resolve the exact language conditioning used by the training sample.

    The control-reference LeRobot dataset is authoritative. The evaluation
    manifest describes the object/scene and may be generated separately, so
    its task text is only a fallback. ``source_episode_index`` maps a generated
    evaluation episode back to the corresponding LeRobot episode.
    """
    if instruction_override is not None:
        return (instruction_override, "cli:--instruction", None)
    reference_candidates: list[int] = []
    for key in ("lerobot_episode_index", "reference_episode_index", "source_episode_index"):
        value = manifest_metadata.get(key)
        if value is not None:
            candidate = int(value)
            if candidate not in reference_candidates:
                reference_candidates.append(candidate)
    if episode_index not in reference_candidates:
        reference_candidates.append(int(episode_index))
    reference_dataset = control_reference_dataset.expanduser().resolve()
    for reference_episode_index in reference_candidates:
        try:
            reference_metadata = load_jsonl_metadata(reference_dataset, reference_episode_index)
        except IndexError:
            continue
        task = reference_metadata.get("task")
        if task:
            source = reference_metadata.get("task_source", "control reference metadata")
            return (
                str(task),
                f"control_reference:{source} (episode={reference_episode_index})",
                reference_metadata.get("task_index"),
            )
    task = manifest_metadata.get("task")
    if task:
        return (
            str(task),
            str(manifest_metadata.get("task_source", "evaluation manifest")),
            manifest_metadata.get("task_index"),
        )
    return (DEFAULT_INSTRUCTION, "built-in fallback", None)


def load_reference_frame(
    args: argparse.Namespace, episode_index: int, manifest_metadata: Mapping[str, Any], columns: list[str]
) -> tuple[Any, Path, int, Path]:
    """Load the matching LeRobot episode once for all initialization modes."""
    import pandas as pd

    dataset = args.control_reference_dataset.expanduser().resolve()
    candidates = [int(episode_index)]
    candidates.extend(
        (
            int(manifest_metadata[key])
            for key in ("lerobot_episode_index", "reference_episode_index")
            if key in manifest_metadata and int(manifest_metadata[key]) not in candidates
        )
    )
    for candidate in candidates:
        paths = sorted((dataset / "data").glob(f"chunk-*/episode_{candidate:06d}.parquet"))
        if paths:
            frame = pd.read_parquet(paths[0], columns=columns)
            if frame.empty:
                raise ValueError(f"Reference episode is empty: {paths[0]}")
            return (frame, paths[0], candidate, dataset)
    raise FileNotFoundError(
        f"No reference episode for evaluation episode {episode_index}; tried {candidates} below {dataset / 'data'}"
    )


def _manifest_initial_action(args, metadata, vae, world_origin):
    explicit = metadata.get("initial_native_action")
    if explicit is None:
        return None
    if args.hand != "shadow_hand_right":
        raise ValueError("initial_native_action describes the packaged Shadow scene")
    initial = np.asarray(explicit, dtype=np.float32).reshape(-1)
    if initial.shape != (28,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_native_action must contain 28 finite Shadow MuJoCo controls")
    target_q = initial[6:].copy()
    direct = args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES
    reference_z = None
    if not direct:
        if vae is None:
            raise RuntimeError("NativeVAE is required to encode a packaged native pose")
        reference_z = vae.encode(target_q, args.hand)[0].detach().cpu().numpy().astype(np.float32)
    policy_wrist = initial[:6].copy()
    policy_wrist[:3] -= world_origin
    policy_wrist[3:6] += np.asarray(evaluation_policy_wrist_offset(args), dtype=np.float32)
    result = {
        "mode": "evaluation_manifest_initial_native_action",
        "source": str(metadata.get("initial_native_action_source", "evaluation manifest")),
        "policy_wrist_world_origin": world_origin.tolist(),
        "reference_shadow_wrist_policy_coordinates": policy_wrist.tolist(),
        "reference_shadow_joint_position": target_q.tolist(),
        "reference_z_gesture": None if reference_z is None else reference_z.tolist(),
        "target_q_requested": target_q.tolist(),
        "target_q_source": "evaluation_manifest.initial_native_action[6:28]",
        "decoded_target_q_requested": None if direct else target_q.tolist(),
        "target_wrist_world_requested": initial[:6].tolist(),
    }
    LOGGER.info(
        "[manifest init] hand=%s world_wrist=%s source=%s",
        args.hand, initial[:6].tolist(), result["source"],
    )
    return initial, result


def _resolve_reference_target(args, reference_state, reference_path, vae, geometry_retargeter):
    direct = args.data_config_name in JOINT_ACTION_DATA_CONFIG_NAMES
    required = 6 + DIRECT_JOINT_CONFIG_DIMS[args.data_config_name] if direct else 52
    if reference_state.shape[0] < required:
        raise ValueError(
            f"Expected at least {required} state dimensions in {reference_path}, "
            f"got {reference_state.shape[0]}"
        )
    if not np.all(np.isfinite(reference_state[:required])):
        raise ValueError(f"Reference first-frame state contains NaN/Inf: {reference_path}")
    wrist = reference_state[:6].copy()
    if not direct:
        if vae is None:
            raise RuntimeError("z_gesture initialization requires NativeVAE")
        z = reference_state[28:52].copy()
        return _ReferenceTarget(
            vae.decode(z, args.hand)[0].detach().cpu().numpy().astype(np.float32),
            direct, None, None, z, None, wrist,
        )
    joint_dim = DIRECT_JOINT_CONFIG_DIMS[args.data_config_name]
    dataset_q = reference_state[6:6 + joint_dim].copy()
    signs = np.asarray(args.shadow_joint_contract["dataset_to_mujoco_signs"], dtype=np.float32)
    shadow_q = dataset_q * signs
    model_hand = DIRECT_JOINT_CONFIG_HANDS[args.data_config_name]
    if model_hand == args.hand:
        return _ReferenceTarget(shadow_q.copy(), direct, dataset_q, shadow_q, None, None, wrist)
    if model_hand != "shadow_hand_right":
        raise ValueError(
            f"Native-joint checkpoint for {model_hand} must be evaluated "
            f"with --hand {args.hand}"
        )
    if geometry_retargeter is None:
        raise RuntimeError(
            "Cross-hand physical-joint initialization requires the geometry retargeter"
        )
    retargeted = geometry_retargeter.retarget(
        shadow_q, "shadow_hand_right", args.hand, stream="initialization", update_state=False
    )
    return _ReferenceTarget(retargeted.target_q, direct, dataset_q, shadow_q, None, retargeted, wrist)


def _dataset_initial_action(request: _DatasetInitialActionRequest):
    args = request.args
    episode_index = request.episode_index
    metadata = request.metadata
    vae = request.vae
    geometry_retargeter = request.geometry_retargeter
    world_origin = request.world_origin
    frame, path, ref_index, dataset = load_reference_frame(
        args, episode_index, metadata, ["observation.state"]
    )
    reference_state = np.asarray(frame.iloc[0]["observation.state"], dtype=np.float32).reshape(-1)
    target = _resolve_reference_target(args, reference_state, path, vae, geometry_retargeter)
    target_q, direct = target.target_q, target.direct
    dataset_q, shadow_q = target.dataset_q, target.shadow_q
    reference_z, retargeted, wrist = target.reference_z, target.retargeted, target.wrist
    target_wrist = wrist.copy()
    target_wrist[:3] += world_origin
    target_wrist[3:6] -= evaluation_policy_wrist_offset(args)
    initial = np.concatenate([target_wrist, target_q]).astype(np.float32)
    result = {
        "mode": (
            "shadow_first_frame_relative_wrist_and_direct_joints"
            if direct else "shadow_first_frame_relative_wrist_and_z_decode"
        ),
        "reference_dataset": str(dataset), "reference_episode_index": ref_index,
        "reference_episode_path": str(path.resolve()),
        "reference_state_layout": (
            {"wrist": [0, 6], "shadow_joint_position": [6, 28]}
            if direct else {"wrist": [0, 6], "z_gesture": [28, 52]}
        ),
        "policy_wrist_world_origin": world_origin.tolist(),
        "reference_shadow_wrist_policy_coordinates": wrist.tolist(),
        "reference_shadow_joint_position": shadow_q.tolist() if direct else None,
        "reference_shadow_joint_position_dataset_coordinates": None if dataset_q is None else dataset_q.tolist(),
        "reference_z_gesture": None if reference_z is None else reference_z.tolist(),
        "target_q_requested": target_q.tolist(),
        "target_q_source": (
            "signed_transform(observation.state[6:28])"
            if args.hand == "shadow_hand_right"
            else "geometry_retarget(shadow_hand_right->target_hand)"
            if direct else "VAE.decode(observation.state[28:52])"
        ),
        "initialization_geometry_retarget": None if retargeted is None else retargeted.metadata(),
        "decoded_target_q_requested": None if direct else target_q.tolist(),
        "target_wrist_world_requested": target_wrist.tolist(),
    }
    LOGGER.info("[dataset init] hand=%s reference_episode=%s source=%s", args.hand, ref_index, path)
    return initial, result


def build_cross_hand_initial_action(
    *, args: argparse.Namespace, episode_index: int, manifest_metadata: Mapping[str, Any],
    vae: NativeVAE | None, geometry_retargeter: GeometryRetargeter | None,
    policy_wrist_world_origin: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Initialize a cross-hand episode from an explicit action or reference data."""
    world_origin = np.asarray(policy_wrist_world_origin, dtype=np.float32)
    if world_origin.shape != (3,) or not np.all(np.isfinite(world_origin)):
        raise ValueError("policy_wrist_world_origin must be finite xyz [3]")
    manifest = _manifest_initial_action(args, manifest_metadata, vae, world_origin)
    if manifest is not None:
        return manifest
    if args.hand == "shadow_hand_right" and np.allclose(world_origin, 0.0):
        return None, {
            "mode": "environment_default",
            "reason": "legacy zero-origin Shadow evaluation",
            "policy_wrist_world_origin": world_origin.tolist(),
        }
    if args.hand != "shadow_hand_right" and not args.cross_hand_initialization:
        return None, {
            "mode": "environment_default",
            "reason": "cross-hand initialization disabled by CLI",
            "policy_wrist_world_origin": world_origin.tolist(),
        }
    return _dataset_initial_action(
        _DatasetInitialActionRequest(
            args=args,
            episode_index=episode_index,
            metadata=manifest_metadata,
            vae=vae,
            geometry_retargeter=geometry_retargeter,
            world_origin=world_origin,
        )
    )


def load_reference_initial_z_gesture(
    *, args: argparse.Namespace, episode_index: int, manifest_metadata: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the matching Shadow episode's first z_gesture without changing reset."""
    frame, reference_path, reference_episode_index, reference_dataset = load_reference_frame(
        args, episode_index, manifest_metadata, ["observation.state"]
    )
    reference_state = np.asarray(frame.iloc[0]["observation.state"], dtype=np.float32).reshape(-1)
    if reference_state.shape[0] < 52:
        raise ValueError(f"Expected at least 52 state dimensions in {reference_path}, got {reference_state.shape[0]}")
    reference_z = reference_state[28:52].copy()
    if not np.all(np.isfinite(reference_z)):
        raise ValueError(f"Reference first-frame z_gesture contains NaN/Inf: {reference_path}")
    return (
        reference_z,
        {
            "source": "shadow_reference_first_observation",
            "reference_dataset": str(reference_dataset),
            "reference_episode_index": reference_episode_index,
            "reference_episode_path": str(reference_path.resolve()),
            "reference_state_slice": [28, 52],
        },
    )


def load_dataset_wrist_action_trajectory(
    *,
    args: argparse.Namespace,
    episode_index: int,
    manifest_metadata: Mapping[str, Any],
    required_steps: int,
    policy_wrist_world_origin: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load policy-coordinate wrist6 and convert xyz to MuJoCo world."""
    if args.wrist_action_source == "model":
        return (None, {"source": "model"})
    frame, reference_path, reference_episode_index, reference_dataset = load_reference_frame(
        args, episode_index, manifest_metadata, ["action"]
    )
    if len(frame) < required_steps:
        actual_steps = len(frame)
        raise ValueError(
            f"Dataset wrist trajectory {reference_path} has {actual_steps} steps, "
            f"but evaluation requires {required_steps}"
        )
    actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(f"Expected action [T,D>=6] in {reference_path}, got {actions.shape}")
    wrist_actions = actions[:required_steps, :6].copy()
    if not np.all(np.isfinite(wrist_actions)):
        raise ValueError(f"Dataset wrist trajectory contains NaN/Inf: {reference_path}")
    world_origin = np.asarray(policy_wrist_world_origin, dtype=np.float32)
    if world_origin.shape != (3,) or not np.all(np.isfinite(world_origin)):
        raise ValueError("policy_wrist_world_origin must be finite xyz [3]")
    wrist_actions[:, 0:3] += world_origin
    wrist_actions[:, 3:6] -= np.asarray(evaluation_policy_wrist_offset(args))
    metadata = {
        "source": "dataset",
        "reference_dataset": str(reference_dataset),
        "reference_episode_index": reference_episode_index,
        "reference_episode_path": str(reference_path.resolve()),
        "slice": [0, 6],
        "num_steps": int(len(wrist_actions)),
        "source_xyz_coordinates": "relative_to_wrist_world_origin",
        "executed_xyz_coordinates": "mujoco_world",
        "policy_wrist_world_origin": world_origin.tolist(),
        "target_hand_wrist_euler_offset_removed": list(evaluation_policy_wrist_offset(args)),
    }
    LOGGER.info(
        f"[wrist action] source=dataset reference_episode={reference_episode_index} "
        f"steps={len(wrist_actions)} source={reference_path}",
    )
    return (wrist_actions, metadata)


def write_mp4(path: Path, images: tuple[np.ndarray, ...], fps: int) -> None:
    if not images:
        raise ValueError("No RGB frames were rendered.")
    first = np.asarray(images[0])
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB frames, got {first.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = int(first.shape[1])
        stream.height = int(first.shape[0])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for image in images:
            frame_array = np.ascontiguousarray(image, dtype=np.uint8)
            if frame_array.shape != first.shape:
                raise ValueError(f"All video frames must have shape {first.shape}, got {frame_array.shape}")
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_attributes(instance: Any, names: tuple[str, ...]) -> None:
    for name in names:
        getattr(instance, name).clear()


def align_euler_branch(
    target: np.ndarray, reference: np.ndarray, *, include_xyz_equivalent: bool = False
) -> np.ndarray:
    """Choose the intrinsic-XYZ Euler solution nearest the current qpos.

    Legacy checkpoints retain component-wise 2*pi alignment. Rot6D checkpoints
    additionally consider the second equivalent XYZ solution
    ``(x + pi, pi - y, z + pi)`` before choosing the nearest joint command.
    """
    target = np.asarray(target)
    reference = np.asarray(reference)
    primary = reference + (target - reference + np.pi) % (2.0 * np.pi) - np.pi
    if not include_xyz_equivalent:
        return primary
    alternate_raw = np.asarray([target[0] + np.pi, np.pi - target[1], target[2] + np.pi], dtype=target.dtype)
    alternate = reference + (alternate_raw - reference + np.pi) % (2.0 * np.pi) - np.pi
    if np.linalg.norm(alternate - reference) < np.linalg.norm(primary - reference):
        return alternate
    return primary


def latency_summary_seconds(values: list[float]) -> dict[str, float | int]:
    samples = np.asarray(values, dtype=np.float64)
    if len(samples) == 0:
        return {"count": 0, "total_s": 0.0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "count": int(len(samples)),
        "total_s": float(samples.sum()),
        "mean_ms": float(samples.mean() * 1000.0),
        "p50_ms": float(np.quantile(samples, 0.5) * 1000.0),
        "p95_ms": float(np.quantile(samples, 0.95) * 1000.0),
        "max_ms": float(samples.max() * 1000.0),
    }


def wrist_motion_metrics(actions: np.ndarray) -> dict[str, float]:
    if len(actions) < 2:
        return {
            "position_step_mean_m": 0.0,
            "position_step_max_m": 0.0,
            "rotation_step_mean_deg": 0.0,
            "rotation_step_max_deg": 0.0,
        }
    position_steps = np.linalg.norm(np.diff(actions[:, 0:3], axis=0), axis=1)
    rotations = Rotation.from_euler("XYZ", actions[:, 3:6])
    rotation_steps = (rotations[:-1].inv() * rotations[1:]).magnitude()
    return {
        "position_step_mean_m": float(position_steps.mean()),
        "position_step_max_m": float(position_steps.max()),
        "rotation_step_mean_deg": float(np.degrees(rotation_steps).mean()),
        "rotation_step_max_deg": float(np.degrees(rotation_steps).max()),
    }


def _quadratic_temporal_smooth(
    values: np.ndarray, anchor: np.ndarray, velocity_weight: float, acceleration_weight: float
) -> np.ndarray:
    """Smooth a complete chunk while anchoring its first derivatives to state."""
    values = np.asarray(values, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if values.ndim != 2 or anchor.shape != (values.shape[1],):
        raise ValueError(f"Expected values [T,D] and anchor [D], got {values.shape}, {anchor.shape}")
    horizon = len(values)
    if horizon <= 1 or (velocity_weight <= 0 and acceleration_weight <= 0):
        return values.astype(np.float32, copy=True)
    identity = np.eye(horizon, dtype=np.float64)
    system = identity.copy()
    rhs = values.copy()
    if velocity_weight > 0:
        first = np.eye(horizon, dtype=np.float64)
        if horizon > 1:
            first[1:, :-1] -= np.eye(horizon - 1, dtype=np.float64)
        first_target = np.zeros((horizon, values.shape[1]), dtype=np.float64)
        first_target[0] = anchor
        system += velocity_weight * (first.T @ first)
        rhs += velocity_weight * (first.T @ first_target)
    if acceleration_weight > 0:
        second = np.zeros((horizon, horizon), dtype=np.float64)
        second_target = np.zeros((horizon, values.shape[1]), dtype=np.float64)
        second[0, 0] = 1.0
        second_target[0] = anchor
        if horizon > 1:
            second[1, 0:2] = (-2.0, 1.0)
            second_target[1] = -anchor
        for index in range(2, horizon):
            second[index, index - 2:index + 1] = (1.0, -2.0, 1.0)
        system += acceleration_weight * (second.T @ second)
        rhs += acceleration_weight * (second.T @ second_target)
    return np.linalg.solve(system, rhs).astype(np.float32)


class ChunkTemporalSmoother:
    """Regularize first/second differences inside each predicted action chunk."""

    def __init__(
        self, velocity_weight: float, acceleration_weight: float, *, include_xyz_equivalent: bool = False
    ) -> None:
        if velocity_weight < 0 or acceleration_weight < 0:
            raise ValueError("Chunk smoothing weights must be non-negative")
        self.velocity_weight, self.acceleration_weight, self.include_xyz_equivalent = (
            float(velocity_weight),
            float(acceleration_weight),
            bool(include_xyz_equivalent),
        )

    @property
    def enabled(self) -> bool:
        return self.velocity_weight > 0 or self.acceleration_weight > 0

    def apply(self, chunk: np.ndarray, state: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] < 6 or state.shape != (chunk.shape[1],):
            raise ValueError(f"Expected chunk [T,D>=6] and matching state [D], got {chunk.shape}, {state.shape}")
        if not self.enabled:
            return chunk.copy()
        aligned = chunk.copy()
        euler_reference = state[3:6].copy()
        for index in range(len(aligned)):
            aligned[index, 3:6] = align_euler_branch(
                aligned[index, 3:6], euler_reference, include_xyz_equivalent=self.include_xyz_equivalent
            )
            euler_reference = aligned[index, 3:6]
        output = aligned.copy()
        for block in (slice(0, 3), slice(3, 6), slice(6, chunk.shape[1])):
            output[:, block] = _quadratic_temporal_smooth(
                aligned[:, block], state[block], self.velocity_weight, self.acceleration_weight
            )
        return output

    def metadata(self) -> dict[str, float | bool]:
        return {
            "enabled": self.enabled,
            "velocity_weight": self.velocity_weight,
            "acceleration_weight": self.acceleration_weight,
            "include_xyz_equivalent": self.include_xyz_equivalent,
        }
