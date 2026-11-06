# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""System prompts required while assembling inference-only model inputs."""

from dataclasses import dataclass


@dataclass(frozen=True)


class InferenceConversation:
    system_message: str


_SYSTEM_MESSAGES = {
    "internvl_zh": "",
    "internlm2-chat": "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
    "internvl2_5": "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
    "being_h0": "You are Being-H0, a helpful and harmless AI assistant developed by BeingBeyond.",
}


def get_conv_template(name: str) -> InferenceConversation:
    try:
        return InferenceConversation(_SYSTEM_MESSAGES[name])
    except KeyError as error:
        raise ValueError(f"unsupported inference conversation template: {name}") from error
