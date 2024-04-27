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
image_encoder_configs = {
    "9x9": config_dict.ConfigDict(
        # assumes a 9x9x2 observation
        dict(
            name="image_encoder",
            output_channels=[16, 32, 32],
            kernel_shapes=[3, 2, 2],
            strides=[2, 2, 1],
            padding=["VALID", "VALID", "VALID"],
        )
    ),
    "5x5": config_dict.ConfigDict(
        # assumes a 5x5x2 observation
        dict(
            name="image_encoder",
            output_channels=[16, 32, 32],
            kernel_shapes=[2, 2, 2],
            strides=[1, 1, 1],
            padding=["VALID", "VALID", "VALID"],
        )
    ),
}
image_decoder_config = config_dict.ConfigDict(
    dict(
        name="image_decoder",
        output_channels=[32, 16, 2],
        kernel_shapes=[2, 2, 2],
        strides=[1, 1, 1],
        padding=["VALID", "VALID", "VALID"],
    )
)
