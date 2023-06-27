# SCCH

A Pytorch implementation of paper "**SEMANTIC CENTRALIZED CONTRASTIVE LEARNING FOR UNSUPERVISED HASHING** "

### Main Dependencies

- torch 1.10.0
- torchvision 0.11.1
- Pillow 8.4.0
- opencv-python 4.5.5.64



### How to Run

```shell
# Run with the MS COCO dataset
python main.py coco --train --dataset coco --encode_length 16 --cuda
```

You can find the download link of MS COCO and NUS-WIDE dataset in [this page](https://github.com/swuxyj/DeepHash-pytorch). You can refer to `./utils/data.py` to get hints of preprocessing these datasets.

