from pathlib import Path

assets = Path(__file__).parent.parent / 'assets'

checkpoint    = assets / 'checkpoints' / 'rAIdiologist_T2W-FS_c6b89d3596f441f9a4abaab5f47865cb.pt'
inf_transform = assets / 'rAIdiologist_transform_inf.yaml'
t2w_normalizer       = assets / 'T2w-fs' / 'NyulNormalizer.npz'
t2w_normalizer_nyul  = assets / 'T2w-fs' / '3_NyulNormalizer.npz'
t2w_normalization_cfg = assets / 't2w_normalization.yaml'
