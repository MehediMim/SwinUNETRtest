"""SwinUNETR architecture used by training and inference."""


def make_model(c):
    from monai.networks.nets import SwinUNETR
    return SwinUNETR(img_size=tuple(c['roi_size']), in_channels=1, out_channels=2,
                     feature_size=c['feature_size'], depths=(2, 2, 2, 2),
                     num_heads=(3, 6, 12, 24), downsample=c['downsample'],
                     use_checkpoint=c['use_checkpoint'], spatial_dims=3)
