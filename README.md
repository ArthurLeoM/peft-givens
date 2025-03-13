# peft-givens-v2

This repo is the source code of (quasi-)Givens Orthogonal Fine Tuning integrated to Huggingface peft library (ver.0.12.0)

The implementation of (quasi-)Givens OFT is in ```./tuners/givens/```

### Publication Details 🔗:

[ICML'24] Parameter Efficient Quasi-Orthogonal Fine-Tuning via Givens Rotation.

arXiv: https://arxiv.org/abs/2404.04316

----------

### 💡 Core Idea

<img width="1080" alt="image" src="https://github.com/user-attachments/assets/ed3513b0-a421-416f-b1c8-76eb65160ca9" />

- **Apply learnable (quasi-)orthogonal transformations to PLM's linear layers for fine-tuning to avoid catastrophic forgetting,**
  - ✅ As orthogonal mapping will not change the angular distance between neurons, thus preserving semantic correlations
- **Using O(d) parameters to depict any rotary orthogonal mapping with Givens rotation**
  - 💡 The creative design of a parallel rotation strategy enhances the computational efficiency with O(log d) sparse matrix multiplication
- **We suggest using strict GOFT for LLM SFT and Offline-RL (e.g. DPO) stage. (where you want to sufficiently preserve the pretrained semantics and knowledge unchanged to avoid catastrophic forgetting)**
 
----------

### 🔥 [Update 25/03/13] Introducing GOFT-V2

Main New Features:

- 🔧 Integrated into the new version of peft lib v0.12.0
- 🚀 Boosting computation efficiency by introducing Hadamard multiplier instead of sparse matrix-multiplication
<img width="928" alt="image" src="https://github.com/user-attachments/assets/8153fbc5-a7f3-4397-ac31-24725022c655" />


----------


**Thanks for your interest in our work! If our work helps, please don't forget to cite us!🌟**

```
@InProceedings{pmlr-v235-ma24a,
  title = 	 {Parameter Efficient Quasi-Orthogonal Fine-Tuning via Givens Rotation},
  author =       {Ma, Xinyu and Chu, Xu and Yang, Zhibang and Lin, Yang and Gao, Xin and Zhao, Junfeng},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {33686--33729},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/ma24a/ma24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/ma24a.html},
}
```
or 
```
@article{ma2024parameter,
  title={Parameter efficient quasi-orthogonal fine-tuning via givens rotation},
  author={Ma, Xinyu and Chu, Xu and Yang, Zhibang and Lin, Yang and Gao, Xin and Zhao, Junfeng},
  journal={arXiv preprint arXiv:2404.04316},
  year={2024}
}
```


