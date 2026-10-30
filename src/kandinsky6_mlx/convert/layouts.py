"""Convolution weights per component and how to move them to MLX layouts.

PyTorch stores convolution weights as `[out, in, *kernel]` and transposed-convolution weights as `[in, out, *kernel]`;
MLX uses `[out, *kernel, in]` for both. Every other tensor (linear, norm, embedding weights, biases, buffers such as
the vocoder's low-pass filters) keeps its layout. The tables list parameter names, not tensor ranks, so a rank-3
tensor that is not a convolution weight is never transposed; `tests/unit/test_convert.py` checks each table against
the `nn.Conv*` modules of the reference classes.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ConvRule:
    pattern: re.Pattern
    rank: int
    transposed: bool = False

    def axes(self) -> tuple[int, ...]:
        spatial = tuple(range(2, self.rank))
        if self.transposed:
            # [in, out, *k] -> [out, *k, in]
            return (1, *spatial, 0)
        # [out, in, *k] -> [out, *k, in]
        return (0, *spatial, 1)


CONV_RULES: dict[str, tuple[ConvRule, ...]] = {
    "transformer": (),
    # HunyuanVideoCausalConv3d wraps an nn.Conv3d named `conv`; quant convs are plain nn.Conv3d.
    "vae": (ConvRule(re.compile(r"^(.*\.)?conv\.weight$|^(post_)?quant_conv\.weight$"), 5),),
    # MMAudioMPConv1d subclasses nn.Conv1d.
    "audio_vae": (
        ConvRule(
            re.compile(
                r"^vae\.(encoder|decoder)\.(.*\.)?(conv_in|conv_out|conv1|conv2|nin_shortcut|qkv|proj_out|conv)\.weight$"
            ),
            3,
        ),
    ),
    "vocoder": (
        ConvRule(re.compile(r"^(conv_pre|conv_post|resblocks\.\d+\.convs[12]\.\d+)\.weight$"), 3),
        ConvRule(re.compile(r"^ups\.\d+\.0\.weight$"), 3, transposed=True),
    ),
}

# Keys removed during conversion. `audio_vae` checkpoints at revision 4c487e8b embed a duplicate of `vocoder/`.
DROP_PREFIXES: dict[str, tuple[str, ...]] = {"audio_vae": ("vocoder.",)}

COMPONENTS = tuple(CONV_RULES)


def conv_rule(component: str, name: str) -> ConvRule | None:
    for rule in CONV_RULES[component]:
        if rule.pattern.match(name):
            return rule
    return None


def is_dropped(component: str, name: str) -> bool:
    return name.startswith(DROP_PREFIXES.get(component, ()))
