import torch
from torch import nn
from einops import rearrange

from transformers import ViTConfig
from transformers.models.vit.modeling_vit import ViTEncoder

__all__ = ['ViT']

class ViT(nn.Module):
    def __init__(self, channels, num_classes, image_size=320, patch_size=16, dim=512, depth=6, heads=8, mlp_dim=1024,
                 pool='cls', dropout=0., emb_dropout=0., num_slices=1):
        """
        .. note::
            It is very difficult to get this to converge.

        Args:
            image_size:  int or (H, W) — spatial dims, must satisfy H == W == 320 for this project.
            patch_size:  int or (pH, pW) — spatial patch size; H and W must be divisible by it.
            num_slices:  number of slices S in the volumetric input.
                         Set to 1 (default) for plain 2D inputs (b x c x H x W).
                         For volumetric inputs pass the fixed slice count; forward() then expects
                         tensors of shape  b x c x H x W x S.
        """
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'
        assert num_slices >= 1

        self.num_slices = num_slices
        patches_per_slice = (image_height // patch_height) * (image_width // patch_width)
        num_patches = patches_per_slice * num_slices

        # Shared 2D patch embed layer; slices are folded into the batch dimension in forward().
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=(patch_height, patch_width),
                      stride=(patch_height, patch_width)),
            nn.Flatten(2),   # b x dim x n_patches  →  kept as-is; transposed in forward
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        config = ViTConfig(
            hidden_size=dim,
            num_hidden_layers=depth,
            num_attention_heads=heads,
            intermediate_size=mlp_dim,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.transformer = ViTEncoder(config)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        """
        Args:
            img: b x c x H x W        when num_slices == 1  (2-D mode)
                 b x c x H x W x S    when num_slices  > 1  (volumetric mode)
        Returns:
            logits: b x num_classes
        """
        # Sometimes batchsize of input is squeezed to 1
        if img.dim() == 4:
            img = img.unsqueeze(0)

        if img.dim() == 5:
            b = img.shape[0]
            # Fold slices into the batch axis so Conv2d can process them uniformly.
            img = rearrange(img, 'b c h w s -> (b s) c h w')
            x = rearrange(self.to_patch_embedding(img), 'bs d n -> bs n d')
            x = rearrange(x, '(b s) n d -> b (s n) d', b=b)
        else:
            b = img.shape[0]
            x = rearrange(self.to_patch_embedding(img), 'b d n -> b n d')

        n = x.shape[1]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x).last_hidden_state

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)
