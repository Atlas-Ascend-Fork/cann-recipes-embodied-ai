# Copyright (c) 2026 li12345abc123. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read JSON command defaults and write JSON command output."""

import argparse
import json
import logging
from pathlib import Path
import sys


def emit(message, *, error=False):
    """Write one JSON/log line without taking ownership of the caller's stdout."""
    handler = logging.StreamHandler(sys.stderr if error else sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger = logging.Logger('hy_embodied.output', level=logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        logger.log(logging.ERROR if error else logging.INFO, '%s', message)
    finally:
        logger.removeHandler(handler)
        handler.close()


def parse_config(parser, argv=None):
    """Apply JSON defaults, then let explicitly supplied CLI options override them."""
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--config')
    config, _ = preliminary.parse_known_args(argv)
    if config.config:
        data = json.loads(Path(config.config).read_text(encoding='utf-8'))
        allowed = set(vars(parser.parse_args([]))) - {'config'}
        if not isinstance(data, dict) or set(data) - allowed:
            raise ValueError('Configuration must contain supported command argument names')
        parser.set_defaults(**data)
    return parser.parse_args(argv)
