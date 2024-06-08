from ml_collections import config_dict


transformer_config = config_dict.ConfigDict(
    dict(
        name="transformer",
        hidden_dim=128,
        num_heads=8,
        num_layers=2,
        attn_size=32,
        widening_factor=4,
        dropout_rate=0.1,
        max_timesteps=1000,
        encode_separate=True,  # encode (s,a,r) as separate tokens
    )
)

base = 24
image_encoder_configs = {
    "3x3": config_dict.ConfigDict(
        dict(
            name="image_encoder",
            arch=[
                [16, 2, 1, "SAME"],
                [32, 2, 1, "VALID"],
                [64, 2, 1, "VALID"],
            ],
            add_bn=True,
            add_residual=True,
            mp_kernel_size=2,
        )
    ),
    "5x5": config_dict.ConfigDict(
        # assumes a 5x5x2 observation
        dict(
            name="image_encoder",
            arch=[
                [16, 2, 1, "VALID"],
                [32, 2, 1, "VALID"],
                [64, 2, 1, "VALID"],
                [128, 2, 1, "VALID"],
            ],
            add_bn=True,
            add_residual=True,
            scale=1,
            mp_kernel_size=2,
        )
    ),
    "7x7": config_dict.ConfigDict(
        dict(
            name="image_encoder",
            arch=[
                [16, 3, 1, "VALID"],
                [32, 3, 1, "VALID"],
                [32, 2, 1, "VALID"],
                [64, 2, 1, "VALID"],
            ],
            add_bn=True,
            add_residual=True,
            mp_kernel_size=2,
        )
    ),
    "9x9": config_dict.ConfigDict(
        dict(
            name="image_encoder",
            arch=[
                [16, 3, 1, "VALID"],
                [32, 3, 1, "VALID"],
                [32, 3, 1, "VALID"],
                [64, 3, 1, "VALID"],
            ],
            add_bn=True,
            add_residual=True,
            mp_kernel_size=2,
        )
    ),
    "10x10": config_dict.ConfigDict(
        dict(
            name="image_encoder",
            arch=[
                [16, 3, 1, "VALID"],
                [32, 3, 1, "VALID"],
                [32, 3, 1, "VALID"],
                [64, 3, 1, "VALID"],
            ],
            add_bn=True,
            add_residual=True,
            mp_kernel_size=2,
        )
    ),
    "64x64": config_dict.ConfigDict(
        dict(
            name="image_encoder",
            arch=[
                [base, 3, 1, "SAME"],
                [base * 2, 3, 1, "SAME"],
                [base * 4, 3, 1, "SAME"],
                [base * 8, 3, 1, "SAME"],
                [base * 16, 3, 1, "SAME"],
                [base * 32, 2, 1, "SAME"],
            ],
            add_bn=True,
            add_residual=True,
            add_max_pool=True,
            mp_kernel_size=2,
            # scale=4,
        )
    ),
    "84x84": config_dict.ConfigDict(
        dict(
            name="image_encoder",
            arch=[
                [base, 3, 1, "SAME"],
                [base, 3, 1, "SAME"],
                [base * 2, 3, 1, "SAME"],
                [base * 4, 3, 1, "SAME"],
                [base * 8, 3, 1, "SAME"],
                [base * 16, 3, 1, "SAME"],
                [base * 32, 2, 1, "SAME"],
            ],
            add_bn=True,
            add_residual=True,
            add_max_pool=True,
            mp_kernel_size=2,
            # scale=4,
        )
    ),
}
image_decoder_configs = {
    "3x3": config_dict.ConfigDict(
        dict(
            name="image_decoder",
            arch=[
                [64, 2, 1, "VALID"],
                [32, 2, 1, "VALID"],
                [16, 2, 1, "SAME"],
            ],
            add_bn=True,
            add_residual=True,
            num_output_channels=2,
        ),
    ),
    "5x5": config_dict.ConfigDict(
        dict(
            name="image_decoder",
            arch=[
                [128, 2, 1, "VALID"],
                [64, 2, 1, "VALID"],
                [32, 2, 1, "VALID"],
                [16, 2, 1, "VALID"],
            ],
            add_bn=True,
            add_residual=True,
            num_output_channels=2,
        ),
    ),
    "7x7": config_dict.ConfigDict(
        dict(
            name="image_encoder",
            arch=[
                [64, 2, 1, "VALID"],
                [32, 2, 1, "VALID"],
                [32, 3, 1, "VALID"],
                [16, 3, 1, "VALID"],
            ],
            add_bn=True,
            add_residual=True,
            num_output_channels=3,
        )
    ),
    "9x9": config_dict.ConfigDict(
        dict(
            name="image_encoder",
            arch=[
                [64, 3, 1, "VALID"],
                [32, 3, 1, "VALID"],
                [32, 3, 1, "VALID"],
                [16, 3, 1, "VALID"],
            ],
            add_bn=True,
            add_residual=True,
            num_output_channels=3,
        )
    ),
    "64x64": config_dict.ConfigDict(
        dict(
            name="image_encoder",
            arch=[
                [base * 32, 2, 2, "SAME"],
                [base * 16, 2, 2, "SAME"],
                [base * 8, 2, 2, "SAME"],
                [base * 4, 2, 2, "SAME"],
                [base * 2, 2, 2, "SAME"],
                [base, 2, 2, "SAME"],
            ],
            add_bn=True,
            add_residual=True,
            num_output_channels=3,
        )
    ),
}
