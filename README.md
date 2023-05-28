# RIATIG: Reliable and Imperceptible Adversarial Text-to-Image Generation with Natural Prompts

## Install

### CLIP
Please follow the guidelines in [CLIP Github Repository](https://github.com/openai/CLIP) to install CLIP


### DALL•E Mini
Run the following command to install DALL•E Mini:
```
$ pip install min-dalle
```
Go into the following folder:
```
$ cd /target_model/min_dalle/pretrained
```
Download and uncompress the files: [dalle-bart](https://drive.google.com/file/d/1Qq_FARjdZlHra3r_g2ZvMLyDPzsgx6Af/view?usp=sharing) and [vqgan](https://drive.google.com/file/d/1ckxflXZnnWJzRvFHhpzuj11Pxr07Wby0/view?usp=sharing)

### Word2Vec
Go into the folder:
```
$ cd /Word2Vec
```
Download the files: [word2id.pkl](https://drive.google.com/file/d/11kSfFGm1YOo5N08GGytnZy4cMpDTyd0h/view?usp=sharing) and [wordvec.pkl](https://drive.google.com/file/d/1h1hhkyZWZc-JhKqJBPtnJ2riooXMY-e0/view?usp=sharing)


## Run attack
To run our attack:
```
python run_attack.py --ori_sent [original sentence] --tar_img_path [target image path] --tar_sent [target sentence] --log_save_path [log save path] --intem_img_path [intermediate results save path] --best_img_path [output best images save path] --mutate_by_impor [whether select the word by importance in mutation]
```

For a quick demo:
```
python run_attack.py --ori_sent "a herd of cows that are grazing on the grass" --tar_img_path "./target.png" --tar_sent "a large red and white boat floating on top of a lake"
```

## Citation
If you find our work useful, please cite:

```
@InProceedings{Liu_2023_CVPR,
    author    = {Liu, Han and Wu, Yuhao and Zhai, Shixuan and Yuan, Bo and Zhang, Ning},
    title     = {RIATIG: Reliable and Imperceptible Adversarial Text-to-Image Generation With Natural Prompts},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20585-20594}
}
```

## Acknowledements

Thanks for the open souce code:
#### CLIP: https://github.com/openai/CLIP
#### DALL•E Mini： https://github.com/kuprel/min-dalle
#### OpenAttack: https://github.com/thunlp/OpenAttack