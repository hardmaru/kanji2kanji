# Kanji2Kanji

![Kanji2Kanji](/img/kanji2kanji.png?raw=true)

This repo contains instructions to reproduce the domain transfer results in the paper [Deep Learning for Classical Japanese Literature](https://arxiv.org/abs/1812.01718).

## Versions

Tested on TensorFlow 1.8.0, scipy 0.19.1, numpy 1.13.3

## Instructions

First run `build_data.py` to construct the Kanji dataset.

To train models, run `python train.py` to train all models, and run trained models on test set afterwards.

A notebook called `demo.ipynb` is also available to visualize the data and model predictions.

## Code License

MIT

## Data License

The stroke-based Kanji is derived from [KanjiVG](https://kanjivg.tagaini.net/) project.

The Kuzushiji Kanji data derived from the [Kuzushiji-MNIST](https://github.com/rois-codh/kmnist) project.

## Citation

If you find this work useful, we would appreciate a reference to our paper:

**Deep Learning for Classical Japanese Literature. Tarin Clanuwat et al. [arXiv:1812.01718](https://arxiv.org/abs/1812.01718)**

```latex
@online{clanuwat2018deep,
  author       = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto and Alex Lamb and Kazuaki Yamamoto and David Ha},
  title        = {Deep Learning for Classical Japanese Literature},
  date         = {2018-12-03},
  year         = {2018},
  eprintclass  = {cs.CV},
  eprinttype   = {arXiv},
  eprint       = {cs.CV/1812.01718},
}
```
