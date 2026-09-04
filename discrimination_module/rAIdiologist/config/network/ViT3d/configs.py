"""
Code copied from https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch
Modified by alabamagan, redistributed under the MIT License.

Summary of modifications:
    * Change to use pmi_cfg instead of ml_collections

"""

from pytorch_med_imaging.pmi_base_cfg import PMIBaseCFG

def get_3DReg_config():
    config = PMIBaseCFG()
    config.patches = PMIBaseCFG()
    config.patches.grid = (8, 8, 4)
    config.patches.size = [8, 8, 4]
    config.hidden_size = 252
    config.transformer = PMIBaseCFG()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.conv_first_channel = 512
    config.encoder_channels = (16, 32, 32)
    config.down_factor = 2
    config.down_num = 2
    config.decoder_channels = (96, 48, 32, 32, 16)
    config.skip_channels = (32, 32, 32, 32, 16)
    config.n_dims = 3
    config.n_skip = 5
    config.in_ch = 2
    config.config_as_25d = False
    config.type='img2img'
    return config

def  get_3DImg2Pred_config():
    config = PMIBaseCFG()
    config.patches = PMIBaseCFG()
    config.patches.grid = (1, 1, 25)
    # config.patches.size = [32, 32, 4]
    config.hidden_size = 252
    config.transformer = PMIBaseCFG()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.25
    config.transformer.dropout_rate = 0.25

    config.conv_first_channel = 512
    config.encoder_channels = (64, 128, 256, 512, 1024)
    config.encoder_dropout_rate = 0.3
    config.n_dims = 3
    config.n_skip = 5
    config.in_ch = 1
    config.config_as_25d = True
    config.type='img2pred'
    return config