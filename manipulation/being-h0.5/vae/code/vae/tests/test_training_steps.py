# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Check the optimizer-step budget, terminal validation and checkpoint resume."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
import yaml
from native_vae.dataset import generate_tensor_bundle
from native_vae.trainer import train

ROOT = Path(__file__).resolve().parents[1]


class TrainingStepTests(unittest.TestCase):
    def test_step_limit_validation_and_resume(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data.pt'
            generate_tensor_bundle(hand_config=ROOT / 'configs/right_hands.yaml', output=data,
                                   samples_per_hand=4, seed=7, fk_batch_size=4, device='cpu')
            config = yaml.safe_load((ROOT / 'configs/train_native_n2.yaml').read_text())
            config.update(train_data=str(data), val_data=str(data), batch_size=3, num_epochs=4,
                          output_root=str(root / 'runs'), val_interval=99, save_interval=99,
                          model_config=str(ROOT / 'configs/model.yaml'),
                          hand_config=str(ROOT / 'configs/right_hands.yaml'))
            path = root / 'config.yaml'
            path.write_text(yaml.safe_dump(config))
            run = train(path, device_override='cpu', max_steps_override=5, wandb_enabled_override=False)
            checkpoint = torch.load(run / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['global_step'], 5)
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertTrue(all(int(s['step']) == 5 for s in checkpoint['optimizer_state']['state'].values()))
            records = [json.loads(line) for line in (run / 'logs/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['global_step'] for r in records], [4, 5])
            self.assertEqual(records[-1]['val/batches'], 4)
            config.update(resume_path=str(run / 'checkpoints/last.pt'), run_name='resume')
            path.write_text(yaml.safe_dump(config))
            resumed = train(path, device_override='cpu', max_steps_override=8, wandb_enabled_override=False)
            restored = torch.load(resumed / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(restored['global_step'], 8)
            self.assertEqual(restored['epoch'], 3)
            self.assertTrue(all(int(s['step']) == 8 for s in restored['optimizer_state']['state'].values()))

    def test_nonpositive_step_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'max_steps must be positive'):
            train(ROOT / 'configs/train_native_n2.yaml', max_steps_override=0)


if __name__ == '__main__':
    unittest.main()
