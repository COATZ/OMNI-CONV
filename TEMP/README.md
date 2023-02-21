# OMNI-CONV

## LUT
To reconstruct an equirectangular image from a cubemap of 6 perspective images of 90° FOV.
imode = 0: Wequi = Wcube / imode = 1: Wequi = 2 * Wcube
```
python3
import OMNI_DRL.envs.create_LUT as cLUT
cLUT.create_lookup_table(1024, './LUT', "bilinear", 0)
> Creating LookUp Table Cube 1024x1024 to Equi 1024x1024 imode 0
> (1024, 1024, 3)
cLUT.create_lookup_table(1024, './LUT', "bilinear", 1)
> Creating LookUp Table Cube 1024x1024 to Equi 2048x1024 imode 1
> (1024, 2048, 3)
```

## OFFSETS
To create offsets files in OFFSETS folder. depending on convolution parameters (k kernel size, s stride, p padding, d dilation) and image size (w width, h height).
```
python3 create_offset_tensor.py --w 100 --h 100 --k 8 --s 4 --p 0
```

## CONV LAYER
Replace nn.Conv2d layers by DeformConv2d.DeformConv2d layers.

```
# nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
DeformConv2d_sphe(features//2, 32, kernel_size=3, stride=1, padding=1),
```

