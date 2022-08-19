# MAML PyTorch Implementation

## Dataset

The training script load the data from [DocSet](https://github.com/XoriieInpottn/docset) files.

### MiniImagenet

```
miniimagenet/
	train.ds
	valid.ds
	test.ds
```

After you installed the docset python package, you can run `docset train.ds` in your terminal to view the file format:

```
train.ds
Count: 36000, Size: 1.0 GB, Avg: 29.5 KB/sample

Sample 0
    "filename": "n01614925_1001.JPEG"
    "image": jpeg_image(size=(256, 353, 3))
    "wnid": "n01614925"
    "label": 22
Sample 1
    "filename": "n01614925_10153.JPEG"
    "image": jpeg_image(size=(256, 419, 3))
    "wnid": "n01614925"
    "label": 22
...
Sample 35999
    "filename": "n03424325_9995.JPEG"
    "image": jpeg_image(size=(383, 256, 3))
    "wnid": "n03424325"
    "label": 570
```

## Run the training script

The following code train the model using  on GPU:0

```bash
CUDA_VISIBLE_DEVICES=gpuid python3 train.py --data-path ~/data/miniimagenet/
```



