"""Prompt -> Kandinsky 6 text embeddings in the cache format of the MLX pipeline (spec 09).

Follows `Kandinsky6TI2VAPipeline._get_prompt_embeds` (diffusers 0.42.1): the prompt is wrapped in `PROMPT_TEMPLATE`,
tokenized with the checkpoint's Qwen2.5-VL processor (`padding="max_length"`, `max_length = 1024 + 129`), encoded by
Qwen, and the first `QWEN_CROP_START` (template) tokens are cropped; CLIP-L gives the pooled embedding.

Padding is on the right and Qwen's attention is causal, so the states of real tokens do not depend on the padding.
Qwen therefore runs on the real tokens only and the cache holds just those (no mask), which is what diffusers receives
as `prompt_embeds` and leaves the DiT output unchanged (report 05). Padded positions are not computed.
"""

import mlx.core as mx
import numpy as np

# Copied from diffusers/pipelines/kandinsky6/pipeline_kandinsky6_ti2va.py (0.42.1), typos included; a test compares.
PROMPT_TEMPLATE = "\n".join(  # noqa: FLY002 - kept in the reference's form
    [
        "<|im_start|>system\nYou are a promt engineer. Describe the video in detail.",
        "Describe how the camera moves or shakes, describe the zoom and view angle, whether it follows the objects.",
        "Describe the location of the video, main characters or objects and their action.",
        "Describe the dynamism of the video and presented actions.",
        (
            "Name the visual style of the video: whether it is a professional footage, user generated content, "
            "some kind of animation, video game or scren content."
        ),
        "Describe the visual effects, postprocessing and transitions if they are presented in the video.",
        "Pay attention to the order of key actions shown in the scene.<|im_end|>",
        "<|im_start|>user\n{}<|im_end|>",
    ]
)
QWEN_CROP_START = 129
CLIP_MAX_LENGTH = 77
MAX_SEQUENCE_LENGTH = 1024


def qwen_token_ids(tokenizer, prompt: str, max_sequence_length: int = MAX_SEQUENCE_LENGTH) -> tuple[np.ndarray, int]:
    """Token ids exactly as the reference tokenizes (padded to max length) and the number of real tokens.

    `tokenizer` is the checkpoint's `tokenizer/` loaded with `AutoTokenizer`: for text alone `Qwen2_5_VLProcessor`
    only forwards these arguments to it, and the processor itself needs torchvision for its video part.
    """
    inputs = tokenizer(
        [PROMPT_TEMPLATE.format(prompt)],
        max_length=max_sequence_length + QWEN_CROP_START,
        truncation=True,
        return_tensors="np",
        padding="max_length",
    )
    mask = np.asarray(inputs["attention_mask"][0]).astype(bool)
    valid = int(mask.sum())
    if not mask[:valid].all():
        raise ValueError("expected right padding")
    return np.asarray(inputs["input_ids"]), valid


def clip_token_ids(tokenizer, prompt: str) -> dict:
    return tokenizer(
        [prompt],
        max_length=CLIP_MAX_LENGTH,
        truncation=True,
        add_special_tokens=True,
        padding="max_length",
        return_tensors="np",
    )


def encode_qwen(encoder, tokenizer, prompt: str, max_sequence_length: int = MAX_SEQUENCE_LENGTH) -> mx.array:
    """`(1, real tokens - 129, hidden)` Qwen embeddings of the prompt's real tokens, float32."""
    ids, valid = qwen_token_ids(tokenizer, prompt, max_sequence_length)
    hidden = encoder(mx.array(ids[:, :valid]))
    return hidden[:, QWEN_CROP_START:].astype(mx.float32)
