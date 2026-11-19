# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one seeded LIBERO episode and save its records."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import logging
import sys

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy
from wan_va.utils.Simple_Remote_Infer.deploy.msgpack_numpy import unpackb

LOGGER = logging.getLogger(__name__)

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
MAX_STEPS = 800


class RecordedPolicy(WebsocketClientPolicy):
    def __init__(self, host, port, seed, output):
        super().__init__(host=host, port=port)
        self.seed, self.output = seed, output
        self.requests = []
        self.last_input_sha256 = self.last_action_sha256 = None

    def infer(self, payload):
        if payload.get('reset'):
            payload = dict(payload, _e2e_seed=self.seed)
        data = self._packer.pack(payload)
        self.last_input_sha256 = hashlib.sha256(data).hexdigest()
        self.last_action_sha256 = None
        websocket = getattr(self, '_ws')
        websocket.send(data)
        response = websocket.recv(timeout=300)
        if isinstance(response, str):
            raise RuntimeError(response)
        result = unpackb(response)
        if 'action' in result:
            action = np.asarray(result['action'])
            if action.ndim != 3 or action.shape[0] != 7 or not np.isfinite(action).all():
                raise ValueError('invalid action: %s' % (action.shape,))
            digest = hashlib.sha256()
            digest.update(str((action.dtype.str, action.shape)).encode())
            digest.update(action.tobytes(order='C'))
            self.last_action_sha256 = digest.hexdigest()
            np.save(self.output / ('action_%04d.npy' % len(self.requests)), action)
        return result


def _timed_infer(model, payload):
    event = dict(kind='reset' if payload.get('reset') else 'kv_update'
                 if payload.get('compute_kv_cache') else 'action_generation',
                 started_unix=time.time())
    started = time.perf_counter()
    try:
        result = model.infer(payload)
        elapsed = time.perf_counter() - started
        event.update(success=True, latency_s=elapsed, server_timing=result.get('server_timing'))
        return result, elapsed
    except Exception as exc:
        event.update(success=False, latency_s=time.time() - event['started_unix'], error=repr(exc))
        raise
    finally:
        event.update(input_sha256=model.last_input_sha256, action_sha256=model.last_action_sha256)
        model.requests.append(event)
        with (model.output / 'requests.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')


def _extract_obs(obs):
    return {
        "observation.images.agentview_rgb": np.ascontiguousarray(
            obs["agentview_image"][::-1]
        ),
        "observation.images.eye_in_hand_rgb": np.ascontiguousarray(
            obs["robot0_eye_in_hand_image"][::-1]
        ),
    }


def _construct_env(args):
    for attempt in range(5):
        try:
            return OffScreenRenderEnv(**args)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5)
    raise RuntimeError('Unable to construct LIBERO environment')


def _init_env(env, init_state):
    env.reset()
    env.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env.step([0.0] * 7)
    return _extract_obs(obs)


def _failure_type(exc):
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timeout" in text:
        return "request_timeout"
    if "websocket" in name or "connection" in text:
        return "service_error"
    if "invalid" in text or "shape" in text:
        return "invalid_action"
    if "env" in text or "bddl" in text:
        return "simulation_error"
    return "episode_exception"


def _get_init_states(instance, task_idx):
    task = instance.get_task(task_idx)
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return torch.load(path, weights_only=False)


def _step_action_frame(env, action, frame_idx, key_frames, record):
    action_per_frame = action.shape[2] // 4
    for j in range(action.shape[2]):
        if env.env.timestep >= MAX_STEPS:
            break
        obs, _, done, _ = env.step(action[:, frame_idx, j])
        record['environment_steps'] = int(env.env.timestep)
        if done:
            return True
        if (j + 1) % action_per_frame == 0:
            key_frames.append(_extract_obs(obs))
    return False


def _execute_action(env, action, first, record):
    if action.shape[2] % 4 != 0:
        raise ValueError(f'unexpected action shape: {action.shape}')
    key_frames = []
    for i in range(1 if first else 0, action.shape[1]):
        if _step_action_frame(env, action, i, key_frames, record):
            return True, key_frames
        if env.env.timestep >= MAX_STEPS:
            break
    return False, key_frames


def _run_policy(model, env, first_obs, prompt, record):
    first = True
    while env.env.timestep < MAX_STEPS:
        ret, latency = _timed_infer(model, {'obs': first_obs, 'prompt': prompt})
        record['model_request_count'] += 1
        if record['first_action_latency_s'] is None:
            record['first_action_latency_s'] = latency
        action = ret['action']
        done, key_frames = _execute_action(env, action, first, record)
        first = False
        if done:
            record.update(success=True, failure_type=None, termination_reason='libero_done')
            return
        if env.env.timestep >= MAX_STEPS:
            break
        chunk_start = time.perf_counter()
        _timed_infer(model, {'obs': key_frames, 'compute_kv_cache': True,
                             'imagine': False, 'state': action})
        record['model_request_count'] += 1
        record['action_chunk_latency_s'].append(time.perf_counter() - chunk_start)
    record.update(termination_reason='max_environment_steps', failure_type='timeout')


def run_episode(model, suite, task_idx, episode_idx, seed):
    instance = benchmark.get_benchmark_dict()[suite]()
    task = instance.get_task(task_idx)
    init_states = _get_init_states(instance, task_idx)
    init_state_idx = episode_idx % init_states.shape[0]
    env = None
    started = time.time()
    record = {
        "suite": suite,
        "task": task_idx,
        "episode": episode_idx,
        "init_state_index": init_state_idx,
        "seed": seed,
        "success": False,
        "failure_type": "episode_exception",
        "termination_reason": "not_started",
        "environment_steps": 0,
        "model_request_count": 0,
        "first_action_latency_s": None,
        "action_chunk_latency_s": [],
        "elapsed_s": None,
    }
    try:
        env = _construct_env(
            {
                "bddl_file_name": instance.get_task_bddl_file_path(task_idx),
                "camera_heights": 128,
                "camera_widths": 128,
            }
        )
        env.seed(seed)
        first_obs = _init_env(env, init_states[init_state_idx])
        _timed_infer(
            model, {"reset": True, "prompt": task.language}
        )
        record['model_request_count'] += 1
        _run_policy(model, env, first_obs, task.language, record)
    except Exception as exc:
        record["failure_type"] = _failure_type(exc)
        record["termination_reason"] = str(exc)[:500]
    finally:
        if env is not None:
            env.close()
        record["elapsed_s"] = time.time() - started
    return record


def main():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    LOGGER.handlers = [handler]
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default=os.environ.get('LINGBOT_EVAL_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('LINGBOT_EVAL_PORT', '29976')))
    parser.add_argument('--suite', choices=SUITES)
    parser.add_argument('--task', type=int)
    parser.add_argument('--episode', type=int)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--health-check', action='store_true')
    parser.add_argument('--prompt', default='pick up the object and place it on the plate')
    args = parser.parse_args()
    if args.health_check:
        model = WebsocketClientPolicy(host=args.host, port=args.port)
        try:
            metadata = model.get_server_metadata()
            reset = model.infer(dict(reset=True, prompt=args.prompt))
            result = json.dumps(dict(ready=True, metadata=metadata,
                                     reset_keys=sorted(reset)))
            LOGGER.info(result)
        finally:
            getattr(model, '_ws').close()
        return
    required = (args.suite, args.task, args.episode, args.out)
    if any(value is None for value in required):
        parser.error('Require --suite, --task 0..9, --episode 0..49 and --out')
    if not 0 <= args.task < 10 or not 0 <= args.episode < 50:
        parser.error('Require --suite, --task 0..9, --episode 0..49 and --out')
    args.out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = RecordedPolicy(args.host, args.port, args.seed, args.out)
    try:
        row = run_episode(model, args.suite, args.task, args.episode, args.seed)
        row['requests'] = model.requests
        temporary = args.out / 'record.json.tmp'
        temporary.write_text(json.dumps(row, indent=2))
        temporary.replace(args.out / 'record.json')
        result = json.dumps(row)
        LOGGER.info(result)
    finally:
        getattr(model, '_ws').close()


if __name__ == '__main__':
    main()
