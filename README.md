### 此分支decoder2是预测差值，与dmae的恢复结果进行相加。
main_finetune有rcot分支

# Denoising Masked Autoencoders Help Robust Classification (ICLR 2023)
<p align="center">
  <img src="assets/pipeline.png", width="640">
</p>

This repository is the official implementation of [“Denoising Masked Autoencoders Help Robust Classification”](https://arxiv.org/abs/2210.06983), based on the official implementation of [MAE](https://github.com/facebookresearch/mae) in [PyTorch](https://github.com/pytorch/pytorch).
```
@inproceedings{wu2023dmae,
  title={Denoising Masked Autoencoders Help Robust Classification},
  author={Wu, QuanLin and Ye, Hang and Gu, Yuntian and Zhang, Huishuai and Wang, Liwei and He, Di},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```

### Pre-training
The pre-training instruction is in [PRETRAIN.md](PRETRAIN.md).

The following table provides the pre-trained checkpoints used in the paper:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Size</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">Link</th>
<!-- TABLE BODY -->
<tr><td align="left">DMAE-Base</td>
<td align="center">427MB</td>
<td align="center">1100</td>
<td align="center"><a href="https://1drv.ms/u/s!AnxRCBR6qpJqiiyVY-qxN_AKNwhA?e=Xb6mlj">download</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">DMAE-Large</td>
<td align="center">1.23GB</td>
<td align="center">1600</td>
<td align="center"><a href="https://1drv.ms/u/s!AnxRCBR6qpJqii1fTOzAG3tBSDn6?e=PxxadF">download</a></td>
</tr>
</tbody></table>

### RCOT Training
We provide separate scripts for Residual-Conditioned Optimal Transport (RCOT) experiments. The DMAE weights can remain frozen using the `--freeze_base` option.
Noise for pre-training can be either standard Gaussian or quaternion wavelet based. Use `--use_quaternion_noise` to enable the latter (with optional `--levels` and `--ratio`).

**ImageNet Example**
```bash
python -m torch.distributed.launch --nproc_per_node=8 \
    main_pretrain_rcot.py \
    --data_path ${IMAGENET_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --dmae_ckpt path/to/dmae_pretrain.pth \
    --freeze_base
```

**CIFAR-10 Example**
```bash
python -m torch.distributed.launch --nproc_per_node=1 \
    pretrain_cifar10_rcot.py \
    --data_path ${CIFAR10_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --dmae_ckpt path/to/dmae_pretrain.pth \
    --freeze_base
```

The model architectures are defined in `models_rcot.py`.

### RCOT Inference
After training an RCOT model you can enable or disable the second restoration
stage during inference:

```python
from models_rcot import rcot_dmae_vit_base_patch16

model = rcot_dmae_vit_base_patch16()
noisy = ...  # tensor Bx3xHxW
refined = model.restore(noisy, use_rcot=True)  # two-stage
basic = model.restore(noisy, use_rcot=False)   # DMAE only
```

### Visualizing Restoration
You can visualize the effect of RCOT on a single image using `visualize_rcot.py`:

```bash
python visualize_rcot.py \
    --img path/to/image.jpg \
    --ckpt PATH_TO_RCOT_CHECKPOINT \
    --dmae_ckpt PATH_TO_DMAE_CHECKPOINT
```
The script displays the original image, the noisy input, and the outputs of the
first-stage DMAE and the full RCOT model for easy comparison.

### Certification
We provide scripts to measure randomized smoothing certified accuracy.

```bash
python certify.py \
    --resume PATH_TO_CHECKPOINT \
    --sigma 0.5 \
    --use_rcot --rcot_ckpt PATH_TO_RCOT --lambda_rcot 0.5
```

For CIFAR-10, use `certify_cifar10.py` with the same options.


### Fine-tuning
The fine-tuning and evaluation instruction is in [FINETUNE.md](FINETUNE.md).
#### Results on ImageNet
<p align="left">
  <img src="assets/imagenet.png", width="640">
</p>

#### Results on CIFAR-10
<p align="left">
  <img src="assets/cifar10.png", width="640">
</p>

### License
This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
