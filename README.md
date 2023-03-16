# OMNI-CONV

# Citation
This repositeory contains the code used in our article "OMNI-CONV: Generalization of the Omnidirectional Distortion-Aware Convolutions".

```
@article{Artizzu2023,
	title        = {{OMNI-CONV: Generalization of the Omnidirectional Distortion-Aware Convolutions}},
	author       = {Artizzu, Charles-Olivier and Allibert, Guillaume and Demonceaux, Cédric},
	year         = 2023,
	journal      = {Journal of Imaging},
	volume       = 9,
	number       = 2,
	article-number = 29,
	url          = {https://www.mdpi.com/2313-433X/9/2/29},
	pubmedid     = 36826948,
	issn         = {2313-433X},
	doi          = {10.3390/jimaging9020029}
}
```

# Specific models
We provided specific spherical adaptation for several visual modalities: [semantic segmentation](https://github.com/COATZ/semantic-segmentation-pytorch), [monocular depth](https://github.com/COATZ/MiDaS), and [optical flow](https://github.com/COATZ/gmflow).

# LUT
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

# OFFSETS
To create offsets files in OFFSETS folder. depending on convolution parameters (k kernel size, s stride, p padding, d dilation) and image size (w width, h height).
```
python3 create_offset_tensor.py --w {} --h {} --k {} --s {} --p {} --d {}
python3 create_offset_tensor.py --w 100 --h 100 --k 8 --s 4 --p 0
```

# CONV LAYER
Replace nn.Conv2d layers by DeformConv2d.DeformConv2d layers.

```
# nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
DeformConv2d_sphe(features//2, 32, kernel_size=3, stride=1, padding=1),
```

