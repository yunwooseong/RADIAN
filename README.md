<div align="center">

# Relational Alignment Graph Neural Network for Multimodal Recommendation

**[Woo-Seong Yun](https://scholar.google.com/citations?user=ZRXyvtMAAAAJ)** &nbsp;·&nbsp; **Myung-Bin Gwak** &nbsp;·&nbsp; **Eun-Sun Kim** &nbsp;·&nbsp; **Yoon-Sik Cho**

<sub>Department of Artificial Intelligence, Chung-Ang University</sub>

*IEIE Summer Conference (The Institute of Electronics and Information Engineers), pp. 4929–4933, Jun. 2025*

[![Paper](https://img.shields.io/badge/Paper-DBpia-1E4DB7)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12332645)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-GPL--3.0-green)](LICENSE)
[![Gold Prize](https://img.shields.io/badge/🏆_Gold_Prize-IEIE_2025-F5B301)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12332645)

</div>

This is the PyTorch implementation for our RADIAN paper:

> Relational Alignment Graph Neural Network for Multimodal Recommendation (IEIE Summer Conference, 2025)

<p align="center">
  <img src="assets/overview.png" width="90%" alt="RADIAN architecture">
</p>

## Overview

Multimodal recommendation exploits image and text features to alleviate data sparsity, but features from pre-trained encoders carry noise unrelated to user preference, and existing models represent users only through the items they consumed, leaving user-to-user relations implicit. To address these limitations, we propose **RADIAN** (Relational Alignment and Denoising for Multimodal Recommendation). RADIAN builds two homophily graphs, *user–user* and *item–item*, from user–item co-occurrence and uses them to refine the embeddings: the item–item graph aligns the multimodal representation with user preference through contrastive learning, while the user–user graph makes hidden preferences shared among users explicit. RADIAN improves over the strongest baseline by 10.62–18.36% on three standard benchmarks.

## Requirements

```bash
conda env create -f src/env.yaml
conda activate MMRec
```

## Datasets

We use three Amazon review datasets (5-core) with 4096-d visual features from VGG16 and 384-d text features from Sentence-Transformers. Interactions are split randomly into train / validation / test at 8:1:1.

| Dataset | Users | Items | Interactions | Sparsity |
| --- | ---: | ---: | ---: | ---: |
| Baby | 19,445 | 7,050 | 160,792 | 99.88% |
| Sports | 35,598 | 18,357 | 296,337 | 99.95% |
| Clothing | 39,387 | 23,033 | 278,677 | 99.97% |

Download the preprocessed data from [Google Drive](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing) and place each dataset folder under `data/`. Scripts for preprocessing from raw Amazon data are provided in `preprocessing/`.

## Training

```bash
cd src
python main.py --dataset baby --model RADIAN
python main.py --dataset sports --model RADIAN
python main.py --dataset clothing --model RADIAN
```

## Results

Bold marks the best result and underline the second best (Table 2 of the paper).

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Model</th>
      <th colspan="2" align="center">Baby</th>
      <th colspan="2" align="center">Sports</th>
      <th colspan="2" align="center">Clothing</th>
    </tr>
    <tr>
      <th align="center">Recall@20</th>
      <th align="center">NDCG@20</th>
      <th align="center">Recall@20</th>
      <th align="center">NDCG@20</th>
      <th align="center">Recall@20</th>
      <th align="center">NDCG@20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">MF</td>
      <td align="center">0.0440</td>
      <td align="center">0.0200</td>
      <td align="center">0.0430</td>
      <td align="center">0.0202</td>
      <td align="center">0.0191</td>
      <td align="center">0.0088</td>
    </tr>
    <tr>
      <td align="left">NGCF</td>
      <td align="center">0.0591</td>
      <td align="center">0.0261</td>
      <td align="center">0.0695</td>
      <td align="center">0.0318</td>
      <td align="center">0.0387</td>
      <td align="center">0.0168</td>
    </tr>
    <tr>
      <td align="left">LightGCN</td>
      <td align="center">0.0698</td>
      <td align="center">0.0319</td>
      <td align="center">0.0782</td>
      <td align="center">0.0369</td>
      <td align="center">0.0470</td>
      <td align="center">0.0215</td>
    </tr>
    <tr>
      <td align="left">VBPR</td>
      <td align="center">0.0486</td>
      <td align="center">0.0213</td>
      <td align="center">0.0582</td>
      <td align="center">0.0265</td>
      <td align="center">0.0481</td>
      <td align="center">0.0205</td>
    </tr>
    <tr>
      <td align="left">MMGCN</td>
      <td align="center">0.0640</td>
      <td align="center">0.0284</td>
      <td align="center">0.0638</td>
      <td align="center">0.0279</td>
      <td align="center">0.0501</td>
      <td align="center">0.0221</td>
    </tr>
    <tr>
      <td align="left">GRCN</td>
      <td align="center">0.0754</td>
      <td align="center">0.0336</td>
      <td align="center">0.0833</td>
      <td align="center">0.0377</td>
      <td align="center">0.0631</td>
      <td align="center">0.0276</td>
    </tr>
    <tr>
      <td align="left">LATTICE</td>
      <td align="center"><ins>0.0829</ins></td>
      <td align="center"><ins>0.0368</ins></td>
      <td align="center"><ins>0.0915</ins></td>
      <td align="center"><ins>0.0424</ins></td>
      <td align="center"><ins>0.0710</ins></td>
      <td align="center"><ins>0.0316</ins></td>
    </tr>
    <tr>
      <td align="left"><b>RADIAN</b></td>
      <td align="center"><b>0.0917</b></td>
      <td align="center"><b>0.0411</b></td>
      <td align="center"><b>0.1083</b></td>
      <td align="center"><b>0.0500</b></td>
      <td align="center"><b>0.0795</b></td>
      <td align="center"><b>0.0364</b></td>
    </tr>
    <tr>
      <td align="left"><i>Improv.</i></td>
      <td align="center"><i>10.62%</i></td>
      <td align="center"><i>11.68%</i></td>
      <td align="center"><i>18.36%</i></td>
      <td align="center"><i>17.92%</i></td>
      <td align="center"><i>11.97%</i></td>
      <td align="center"><i>15.19%</i></td>
    </tr>
  </tbody>
</table>

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yun2025radian,
  title     = {Relational Alignment Graph Neural Network for Multimodal Recommendation},
  author    = {Woo-Seong Yun and Myung-Bin Gwak and Eun-Sun Kim and Yoon-Sik Cho},
  booktitle = {Proceedings of the IEIE Summer Conference},
  pages     = {4929--4933},
  year      = {2025},
  month     = jun,
  publisher = {The Institute of Electronics and Information Engineers}
}
```

## Acknowledgements

This research was supported by the MSIT (Ministry of Science and ICT), Korea, under the ITRC (Information Technology Research Center) support program (IITP-2024-RS-2024-00438056) supervised by the IITP (Institute for Information & Communications Technology Planning & Evaluation); by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2024-00419201); and by the IITP grant funded by the Korea government (MSIT) (RS-2021-II211341, Artificial Intelligence Graduate School Program (Chung-Ang University)).

Our implementation builds on [MMRec](https://github.com/enoche/MMRec), [LATTICE](https://github.com/CRIPAC-DIG/LATTICE) and [UltraGCN](https://github.com/reczoo/UltraGCN).
