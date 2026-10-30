"""Tiny constructor arguments for every reference component.

The sizes are as small as possible while every code path of the released checkpoints still runs: the DiT keeps the
released flags (`visual_cond`, `ca_rope`, `cross_gates`, `fix_modulation`, token types) and the widened PiFlow
outputs, the VAEs keep their level structure, and the vocoder keeps three resblock kernels with dilations.
"""

DIT = {
    "in_visual_dim": 16,
    "out_visual_dim": 160,
    "in_text_dim": 8,
    "in_text_dim2": 4,
    "time_dim": 16,
    "patch_size": (1, 2, 2),
    "model_dim": 32,
    "ff_dim": 64,
    "num_text_blocks": 1,
    "num_visual_blocks": 2,
    "axes_dims": (2, 2, 4),
    "visual_cond": True,
    "in_audio_dim": 40,
    "out_audio_dim": 400,
    "model_dim_a": 16,
    "time_dim_a": 16,
    "ff_dim_a": 32,
    "axes_dims_a": (2, 2, 4),
    "audio_freqs_scaling": 0.144,
    "scale_factor": (1.0, 2.0, 2.0),
    "text_token_padding": True,
    "ca_rope": True,
    "cross_gates": True,
    "fix_modulation": True,
    "visual_token_type_num_embeddings": 2,
}

# Same as the video VAE check in the study repository's validator.
VIDEO_VAE = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 16,
    "block_out_channels": (8, 8, 8, 8),
    "layers_per_block": 1,
    "norm_num_groups": 4,
    "spatial_compression_ratio": 8,
    "temporal_compression_ratio": 4,
    "mid_block_add_attention": True,
}

# Three levels so the decoder's single 2x upsample (after level 1) is present.
AUDIO_VAE = {
    "mel_bins": 16,
    "latent_channels": 8,
    "hidden_channels": 8,
    "channel_multipliers": (1, 2, 4),
    "layers_per_block": 1,
    "sample_rate": 44_100,
    "n_fft": 64,
    "hop_length": 16,
    "scaling_factor": 0.417,
}

# Total upsampling 4 * 2 * 2 = 16 matches AUDIO_VAE hop_length.
VOCODER = {
    "num_mels": 16,
    "upsample_initial_channel": 32,
    "upsample_rates": (4, 2, 2),
    "upsample_kernel_sizes": (8, 4, 4),
    "resblock_kernel_sizes": (3, 7, 11),
    "resblock_dilation_sizes": ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
}

# Released Lite-distill scheduler config.
PIFLOW = {
    "num_train_timesteps": 1000,
    "shift": 5.0,
    "n_grid": 10,
    "eps": 1e-6,
    "final_step_size_scale": 0.5,
    "num_policy_substeps": 128,
}
