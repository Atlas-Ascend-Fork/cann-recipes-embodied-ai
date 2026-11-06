# Being-H0.5 MuJoCo Simulation and Video Evaluation

> External resources and Docker: [installation and release guide](../resources/README_en.md). The image and resource packages are available on community OBS via HTTPS, with offline installation also supported.

[简体中文](README.md) | [English](README_en.md)

This directory provides pure-MuJoCo closed-loop policy validation. Being-H0.5 predicts wrist motion and `z_gesture` from two camera images, a language instruction, and robot state. NativeVAE decodes `z_gesture` into native dexterous-hand joints, and MuJoCo executes the dynamics and records success rate, trajectories, timing, metadata, and episode videos.

The workflow does not depend on Isaac Gym and does not contain a second Being-H0.5 or NativeVAE implementation. This module owns only the MuJoCo environment, scenes, tasks, replay tools, and evaluation entry points. Policy orchestration is reused from `infer_with_torch/code/vae/integration/`, and NativeVAE is reused from `vae/code/vae/`.

Related guides:

- [Common code and assembly](../common/README_en.md)
- [Online NPU inference](../infer_with_torch/README_en.md)
- [VAE training and post-training](../vae/README_en.md)

## Code Layout

```text
being-h0.5/
├── infer_with_torch/code/vae/integration/
│   ├── beingh_shadow_grasp_eval.py       # Canonical closed-loop orchestration
│   ├── benchmark_beingh_retargeting.py   # VAE/geometry comparison
│   └── diagnose_cross_hand_cameras.py    # Cross-hand camera diagnostics
└── sim/
    ├── requirements.txt
    └── code/vae/
        ├── sim/                          # Environment, scene, task, client
        │   └── assets/
        │       ├── environment/          # Table, cameras, lights, physics
        │       └── demos/                # Standalone minimal scenes
        ├── examples/
        │   └── beingh_shadow_grasp_eval.py # Thin legacy-compatible CLI
        ├── scripts/                      # Smoke, view, replay, and demo tools
        ├── evaluation/                   # Portable episode manifests
        ├── tests/test_sim.py
        └── sim/                          # Python API and coordinate conventions
```


## 1. Assemble the Runtime Workspace

From the repository root, use an empty target directory:

```bash
export WORK_DIR=/path/to/workdir
export BEINGH_DIR="$WORK_DIR/beingh05_vae"

bash manipulation/being-h0.5/vae/scripts/setup.sh
```

The script merges `common`, NPU inference, VAE, and MuJoCo layers in order. Run the remaining commands in the assembled directory:

```bash
cd "$BEINGH_DIR"
python -m pip install -r vae/requirements.txt
python -m pip install -r /path/to/repo/manipulation/being-h0.5/sim/requirements.txt
python -m pip install -e vae
```

When an environment is already prepared, installation does not need to be repeated. Confirm that the three main entry points load:

```bash
python vae/examples/beingh_shadow_grasp_eval.py --help
python vae/scripts/benchmark_beingh_retargeting.py --help
python vae/scripts/diagnose_cross_hand_cameras.py --help
```

## 2. Prepare Models, Data, and NPU

At minimum, provide these paths:

```bash
export MODEL_PATH=/path/to/beingh_posttrained_checkpoint
export VAE_CHECKPOINT=/path/to/native_vae_inference.pt
export CONTROL_DATASET=/path/to/matching_shadow_lerobot_dataset
```

- `MODEL_PATH`: checkpoint containing `config.json`, tokenizer, model safetensors, and post-training metadata.
- `VAE_CHECKPOINT`: NativeVAE inference checkpoint.
- `CONTROL_DATASET`: matching LeRobot dataset used for initial state and training-distribution motion limits. Expert actions are not replayed during evaluation.

Load the CANN environment before running on Ascend NPU. If only physical NPU 1 is visible, use logical `npu:0` in the program:

```bash
source /usr/local/Ascend/driver/bin/setenv.bash
source /path/to/Ascend/cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
```

For CUDA, replace `--device npu:0` with `cuda:0`. A CPU-only environment can run the MuJoCo smoke test or a lightweight CPU-loadable policy.

## 3. Run One Episode and Generate Video

The repository includes a minimal TissueBox scene, object meshes, and episode manifest. This command runs a complete closed loop and records both the main and wrist cameras:

```bash
export RESULT_DIR="$BEINGH_DIR/results/tissuebox_demo"

bash vae/scripts/run_tissuebox_policy_demo.sh \
  --model-path "$MODEL_PATH" \
  --vae-checkpoint "$VAE_CHECKPOINT" \
  --control-reference-dataset "$CONTROL_DATASET" \
  --device npu:0 \
  --output-dir "$RESULT_DIR"
```

This is not expert-trajectory replay. Only the first reference state initializes the episode. Starting from control step 1, actions come from Being-H0.5 and finger joints are decoded by NativeVAE. The TissueBox success rule requires a lift of at least 0.08 m.

Find the generated artifacts:

```bash
find "$RESULT_DIR" -type f \( \
  -name 'rollout.mp4' -o \
  -name 'rollout_wrist.mp4' -o \
  -name 'rollout.npz' -o \
  -name 'metadata.json' \
\) -print
```

Each episode contains:

```text
episode_000000/
├── rollout.mp4          # H.264 ego_opposite camera video
├── rollout_wrist.mp4    # H.264 wrist camera video
├── rollout.npz          # State, action, object pose, queries, timing
└── metadata.json        # Success, max lift, configuration, video, timing
```

To run the evaluator directly for one episode:

```bash
python vae/examples/beingh_shadow_grasp_eval.py \
  --deployment-profile safe_smooth \
  --model-path "$MODEL_PATH" \
  --vae-checkpoint "$VAE_CHECKPOINT" \
  --control-reference-dataset "$CONTROL_DATASET" \
  --dataset vae/evaluation/object_episodes/tissuebox_idx83_shadow.jsonl \
  --scene-xml vae/sim/assets/demos/tissuebox_idx83_shadow/scene.xml \
  --scene-contains-object \
  --episode 0 \
  --hand shadow_hand_right \
  --instruction 'Lift the tissue box from the table.' \
  --success-profile dex-demo \
  --latent-observation-mode commanded \
  --device npu:0 \
  --output "$RESULT_DIR/episode0/rollout.npz" \
  --output-video "$RESULT_DIR/episode0/rollout.mp4" \
  --output-metadata "$RESULT_DIR/episode0/metadata.json"
```

A two-camera policy automatically generates `rollout_wrist.mp4` next to the requested `rollout.mp4`.

## 4. Run 10 Episodes and Report Success Rate

Normal objects referenced by a dataset manifest require the complete asset bundle:

```text
vae/sim/assets/templates/<hand>.xml
vae/sim/assets/objects/<object_id>/visual/simplified.obj
vae/sim/assets/objects/<object_id>/collision/convex_piece_*.obj
```

```bash
export EVAL_MANIFEST=/path/to/object_episodes.jsonl
export RESULT_DIR="$BEINGH_DIR/results/bottle_10ep"

python vae/examples/beingh_shadow_grasp_eval.py \
  --deployment-profile safe_smooth \
  --model-path "$MODEL_PATH" \
  --vae-checkpoint "$VAE_CHECKPOINT" \
  --dataset "$EVAL_MANIFEST" \
  --episode-range 0 9 \
  --control-reference-dataset "$CONTROL_DATASET" \
  --hand shadow_hand_right \
  --device npu:0 \
  --latent-observation-mode commanded \
  --success-profile loose \
  --output-dir "$RESULT_DIR" \
  --summary-output "$RESULT_DIR/summary.json"
```

Batch mode loads Being-H0.5 and NativeVAE once. `summary.json` contains:

- `num_episodes`, `num_successes`, and `success_rate`;
- successful and failed episode indices;
- mean and maximum lift;
- total steps and model queries;
- per-episode success, timing, videos, and safety events.

Print a compact summary:

```bash
python - "$RESULT_DIR/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"success: {s['num_successes']}/{s['num_episodes']} "
      f"({100 * s['success_rate']:.1f}%)")
print("successful episodes:", s["successful_episode_indices"])
PY
```

List all episode videos:

```bash
find "$RESULT_DIR" -type f \( \
  -name 'rollout.mp4' -o -name 'rollout_wrist.mp4' \
\) -print | sort
```

Use `--episodes 0 3 7` for selected episodes or `--all-episodes` for the complete manifest. Add `--no-record-images` only when video is explicitly unnecessary and lower memory use is preferred.

## 5. Validate the Simulation Environment

Before loading the large model, run the fast regression checks:

```bash
# State, action, physics stepping, and RGB rendering for all three hands
python vae/scripts/sim_smoke_test.py --steps 10

# Without image rendering
python vae/scripts/sim_smoke_test.py --no_render --steps 10

# Simulation unit tests
python vae/tests/test_sim.py
```

Inspect the environment or a dataset episode:

```bash
python vae/scripts/sim_view_environment.py --hand shadow_hand_right
python vae/scripts/sim_view_dataset_episode.py \
  --dataset "$EVAL_MANIFEST" --episode 0 --hand shadow_hand_right
```

## Simulation Assets, Scenes, and Provenance

`code/vae/evaluation/object_episodes/` stores portable object configurations used by MuJoCo evaluation. Each JSONL record contains evaluation/source/original episode indices, object ID/category, instruction, mesh scale, world position, MuJoCo `wxyz` quaternion, and episode frame count. It does not include images, policy checkpoints, or expert action trajectories.

| Manifest | Contents |
| --- | --- |
| `shadow_grasp_0725.jsonl` | 1,207 evaluation object scenes |
| `shadow_grasp_0725_core_bottle_1071.jsonl` | 52 scenes using the same bottle |
| `tissuebox_idx83_shadow.jsonl` | Bundled single-episode TissueBox demo |

JSONL must remain parseable as one JSON object per line and cannot contain
comments. Its `source_repository` and `source_commit` provenance fields remain
intact. See the [authorization record](../resources/INTERNAL_ASSET_AUTHORIZATION.md)
for contributor-confirmed internal-asset authorization and the
[third-party notices](../resources/THIRD_PARTY_NOTICES.md) for third-party terms.

The TissueBox manifest keeps only the source episode's first native target as the reset pose. From the first control step onward, execution uses Being-H0.5 outputs and NativeVAE-decoded finger actions.

The core configuration stays in source; the remaining assets are installed from the runtime package:

- `code/vae/sim/assets/environment/base_scene.xml`: cameras, table, lights, and physics;
- `vae/sim/assets/demos/tissuebox_idx83_shadow/scene.xml` (external resource pack): standalone Shadow Hand and TissueBox scene;
- 38 OBJ files for the demo, approximately 2.03 MB in total;
- `beingh05-runtime-v2.tar.gz`: external templates for three hands, meshes and 22 object groups, verified and installed into the assembled workspace by the resource installer.

The TissueBox scene originates from `ohra_shadow_grasp_0_table_scene.xml` in commit `0d2f89042627eb6c4f4207e1d687e5e708591e95` of `sicloud2333/dex_project`. Expert trajectories, non-MuJoCo runners, URDF files, unused meshes, and generated results were not copied. The success rule follows the source benchmark: lift the object by at least 0.08 m.

The pinned source repository has no root license file. The contributor confirmed the supplied internal assets for community redistribution on 2026-10-31; original hand-component licenses are preserved in the [component audit](../resources/THIRD_PARTY_NOTICES.md). See the [common guide](../common/README_en.md#source-release-and-reproducibility) for Being-H and MuJoCo third-party notices.

## 7. Action, State, Coordinates, and Python API

All three MuJoCo environments use:

```text
environment action/state = [world wrist xyz, absolute wrist Euler XYZ, native joint q]
```

| Hand | Wrist | Native joints | Total |
| --- | ---: | ---: | ---: |
| Shadow | 6 | 22 | 28 |
| Gaia | 6 | 15 | 21 |
| Sharpa | 6 | 22 | 28 |

Data configuration can declare `WRIST_WORLD_ORIGIN`. The current two-camera Shadow data uses `[0, 0, 0.4]` m:

```text
policy_xyz = world_xyz - wrist_world_origin
world_xyz  = policy_xyz + wrist_world_origin
```

Euler RPY is an absolute rotation and is not translated. `step()` returns the actual state in the same order, primary RGB `image`, camera-indexed `images`, object pose/velocity, and simulation time. The default control rate is 30 Hz with a 2 ms MuJoCo substep.

### Environment API

```python
from sim import GraspEnv, GraspEnvConfig

with GraspEnv(
    GraspEnvConfig(
        hand="shadow_hand_right",
        observation_cameras=("ego_opposite", "wrist"),
    )
) as env:
    obs, info = env.reset()
    action = policy.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
```

Use `reset(initial_action=...)` when the experiment defines a specific initial wrist pose. Manage the interactive GUI through `env.launch_gui()`, `env.sync_gui()`, and `env.gui_is_running`.

### Combine Object Scenes with Hands

Any manifest record can be paired with a supported hand:

```bash
python vae/scripts/sim_view_dataset_episode.py \
  --dataset vae/evaluation/object_episodes/shadow_grasp_0725.jsonl \
  --episode 20 \
  --hand gaia_hand_right
```

The loader can also read an original LeRobot dataset, including object ID, scale, rotation matrix, and world XY, and computes table height from the rotated collision mesh:

```bash
python vae/scripts/sim_view_dataset_episode.py \
  --dataset /path/to/shadow_grasp_0725 \
  --episode 20 \
  --hand shadow_hand_right
```

### Gesture Policy Adapter

```text
policy input/output = [wrist 6D, z_gesture 24D]
environment action  = [wrist 6D, decoded target native q]
```

```python
from native_vae import NativeVAE
from sim import CallablePolicy, GesturePolicyAdapter, PolicyEvaluationClient

vae = NativeVAE.from_pretrained(device="npu:0")
policy = GesturePolicyAdapter(
    CallablePolicy(model_predict),
    vae=vae,
    target_hand="sharpa_hand_right",
)
result = PolicyEvaluationClient(env, policy).run(
    initial_action=initial_action,
    max_steps=300,
)
print(result.success, result.max_lift_m)
```

The six-dimensional wrist output must already be expressed in the target MuJoCo wrist frame. The default success rule requires a lift of at least 0.20 m for ten consecutive policy frames; the TissueBox demo uses its source benchmark threshold of 0.08 m.

Replay `[T,D]` native actions and save video:

```bash
python vae/scripts/sim_replay_actions.py \
  --hand shadow_hand_right \
  --actions trajectory.npz \
  --episode 0 \
  --output_video outputs/shadow_episode0.mp4
```

MP4 output requires system `ffmpeg`. Use `--scene_xml` for a specific hand/object scene; the shared base scene still provides consistent tables, cameras, lights, and physics.

## 8. Troubleshooting

- `Unknown or incomplete object`: the visual or collision mesh for the object is missing.
- Hand XML not found: confirm that `vae/sim/assets/templates/` contains the requested hand template.
- `libEGL` warning: set `MUJOCO_GL=egl` on a headless server; if EGL is unavailable, test `MUJOCO_GL=osmesa`.
- No video generated: confirm that `--no-record-images` was not passed and inspect the `videos` field in `metadata.json`. PyAV writes H.264 output.
- NPU device error: run `npu-smi info`; after setting `ASCEND_RT_VISIBLE_DEVICES`, use the renumbered logical device in `--device`.
- Checkpoint/data mismatch: use the checkpoint's `data_config_name` and post-training metadata and a `CONTROL_DATASET` from the same task.

## Additional CUDA Baseline Entry Points

Run the following commands from the assembled workspace using a compatible CUDA environment; the Ascend workflow above uses its own validated environment.

MuJoCo zero-shot evaluation on Sharpa or Gaia:

```bash
python vae/examples/beingh_shadow_grasp_eval.py \
  --model-path outputs/<training-run>/0040000 \
  --dataset vae/evaluation/object_episodes/<manifest>.jsonl \
  --episode-range 0 99 \
  --hand sharpa_hand_right \
  --device cuda:0
```

Geometry-retargeting baseline for a Shadow physical-joint checkpoint:

```bash
python vae/examples/beingh_shadow_grasp_eval.py \
  --model-path outputs/<joint-baseline-run>/0040000 \
  --dataset vae/evaluation/object_episodes/<manifest>.jsonl \
  --episode-range 0 99 \
  --hand sharpa_hand_right \
  --joint-retargeting geometry \
  --geometry-action-chunk-mode batch \
  --latent-observation-mode commanded \
  --device cuda:0

python vae/scripts/benchmark_beingh_retargeting.py --help
```
