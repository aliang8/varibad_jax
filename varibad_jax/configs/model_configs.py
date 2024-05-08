from ml_collections import config_dict


transformer_config = config_dict.ConfigDict(
    dict(
        name="transformer",
        hidden_dim=64,
        num_heads=8,
        num_layers=3,
        attn_size=32,
        widening_factor=4,
        dropout_rate=0.1,
        max_timesteps=1000,
        encode_separate=True,  # encode (s,a,r) as separate tokens
    )
)

base = 16
image_encoder_configs = {
    "9x9": config_dict.ConfigDict(
        # assumes a 9x9x2 observation
        # order is [output_channels, kernel_size, stride, padding]
        dict(
            name="image_encoder",
            arch=[
                [16, 3, 2, "VALID"],
                [32, 3, 2, "VALID"],
                [32, 3, 2, "VALID"],
            ],
        )
    ),
    "5x5": config_dict.ConfigDict(
        # assumes a 5x5x2 observation
        dict(
            name="image_encoder",
            arch=[
                [16, 2, 1, "VALID"],
                [32, 2, 1, "VALID"],
                [32, 2, 1, "VALID"],
                [64, 2, 1, "VALID"],
            ],
            add_bn=True,
            add_residual=True,
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
        )
    ),
    "64x64": config_dict.ConfigDict(
        dict(
            name="image_encoder",
            arch=[
                [base, 3, 1, "SAME"],
                [base * 2, 3, 1, "SAME"],
                [base * 3, 3, 1, "SAME"],
                [base * 4, 3, 1, "SAME"],
                [base * 5, 3, 1, "SAME"],
                [base * 6, 2, 1, "SAME"],
            ],
            add_bn=True,
            add_residual=True,
            add_max_pool=True,
        )
    ),
}
image_decoder_configs = {
    "5x5": config_dict.ConfigDict(
        dict(
            name="image_decoder",
            arch=[
                [64, 2, 1, "VALID"],
                [32, 2, 1, "VALID"],
                [32, 2, 1, "VALID"],
                [16, 2, 1, "VALID"],
            ],
            add_bn=True,
            add_residual=True,
            num_output_channels=2,
        ),
    ),
    "64x64": config_dict.ConfigDict(
        dict(
            name="image_encoder",
            arch=[
                [base * 5, 2, 2, "SAME"],
                [base * 4, 2, 2, "SAME"],
                [base * 3, 2, 2, "SAME"],
                [base * 2, 2, 2, "SAME"],
                [base, 2, 2, "SAME"],
                [base, 2, 2, "SAME"],
            ],
            add_bn=True,
            add_residual=True,
            num_output_channels=3,
        )
    ),
}
