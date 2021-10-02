# FracBNN

This repository serves as the official code release of the paper [FracBNN: Accurate and FPGA-Efficient Binary Neural Networks with Fractional Activations](https://arxiv.org/abs/2012.12206) (pubilished at FPGA 2021).

FracBNN, as a binary neural network, achieves MobileNetV2-level accuracy by leveraging fractional activations. In the meantime, its input layer is binarized using a novel thermometer encoding with minimal accuracy degradation, which improves the hardware resource efficiceny.

<img src="/utils/imagenet_benchmark.png" />

## Citation

If FracBNN helps your research, please consider citing:
```
@article{Zhang2021fracbnn,
    title = "{FracBNN: Accurate and FPGA-Efficient Binary Neural Networks with Fractional Activations}",
    author = {Zhang, Yichi and Pan, Junhao and Liu, Xinheng and Chen, Hongzheng and Chen, Deming and Zhang, Zhiru},
    journal = {The 2021 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays},
    year = {2021}
}
```

## Structure
```
|   cifar10.py (training script)
|   imagenet.py (training script)
|
└── models/
|   |   fracbnn_cifar10.py
|   |   fracbnn_imagenet.py
|
└───utils/
|   |   quantization.py
|   |   utils.py
|
└───xcel-cifar10/
|   |   High-level synthesis code for FracBNN CIFAR-10 accelerator
|
└───xcel-imagenet/
|   |   High-level synthesis code for FracBNN ImageNet accelerator
```

## Dependency
```
Python 3.6.8
torch 1.6.0
torchvision 0.7.0
numpy 1.16.4
```

## Run

### Pretrained Model Release

- [Pretrained CIFAR-10 model](https://drive.google.com/file/d/19XJZc3na96Mbgg7wjEFuoPcuzb-xha-_/view?usp=sharing)
- [Pretrained ImageNet model](https://drive.google.com/file/d/1VyMigxNAW4qQi_uVwifhfJ8FckAxnBhB/view?usp=sharing)

### Test Only

- For CIFAR-10, run ```python cifar10.py -gpu 0 -t -r /path/to/pretrained-cifar10-model.pt -d /path/to/cifar10-data```
- For ImageNet, run ```python imagenet.py -gpu 0,1,2,3 -t -r /path/to/pretrained-imagenet-model.pt -d /path/to/imagenet-data```

### Train (Two Step Training):

Please refer to the paper for details such as hyperparameters.

- Step 1: Binary activations, floating-point weights.
    - In ```utils/quantization.py```, use ```self.binarize = nn.Sequential()``` in ```BinaryConv2d()```, or modify ```self.binarize(self.weight)``` to ```self.weight``` in ```PGBinaryConv2d()```.
    - Run ```python cifar10.py -gpu 0 -s```
- Step 2: Binary activations, binary weights.
    - In ```utils/quantization.py```, use ```self.binarize = FastSign()``` in ```BinaryConv2d()```, or ```self.binarize(self.weight)``` in ```PGBinaryConv2d()```.
    - Run ```python cifar10.py -gpu 0 -f -r /path/to/model_checkpoint.pt -s```
    - Use ```-g``` to set the gating target if training with ```PGBinaryConv2d```

## Model Accuracy

| Dataset       | Precision (W/A) | 1-bit Input Layer | Top-1 %   |
| ------------- | --------------- | ----------------- | --------- |
| CIFAR-10      | 1/1.4 (PG)      | Yes               | 89.1      |
| ImageNet      | 1/1.4 (PG)      | Yes               | 71.7      |

## CIFAR-10 Accelerator

### Compile the HLS code

```
cd ./xcel-cifar10/source/
make hls
```

### Generate the Bitstream

This step should be done after compiling the HLS code. Assume you are in the directory of ```./xcel-cifar10/source/```.

```
make vivado
```

### Deploy on Xilinx Ultra96v2

To test the bitstream on the board, the following files (sample images and labels) are needed:

- [conv1-input.bin](https://drive.google.com/file/d/1xHXMod4xGgv3Abd6sICzAywqGAADnJCS/view?usp=sharing)
- [conv1-input-uint64.npy](https://drive.google.com/file/d/1Wm7qGQAHCrVk-BvzDw0fc3YqLDoof4Lu/view?usp=sharing)
- [labels.bin](https://drive.google.com/file/d/1wssKeaLylQmS_e3wvj0PmUhO8lnlgXMB/view?usp=sharing)

To deploy the bitstream:

- Step1: Download the files and move them to ```/xcel-cifar10/deploy/```
- Step2: Move the generated bitstream and hardware definition files to ```/xcel-cifar10/deploy/```
- Step3: Upload the entire directory ```/xcel-cifar10/deploy/``` to the board
- Step4: Go to ```deploy/```, run ```sudo python3 FracNet-CIFAR10.py``` on the board

## ImageNet Accelerator

### Compile HLS and Generate Bitstream

Please run the compilation flow using Vivado HLS. The top function is ```FracNet()``` in ```xcel-imagenet/source/bnn.cc```.

### Deploy on Xilinx Ultra96v2

To test the bitstream on the board, the following files (sample images, weights) are needed:

- [sample image](https://drive.google.com/file/d/1eHiIOdKyyMy4cRbEs2x5wHgmUkqfzbcK/view?usp=sharing)
- [conv1x1 weights](https://drive.google.com/file/d/1RRkMyfxURW9guPL_-YTO6e-8xW3NbYFO/view?usp=sharing)
- [conv3x3 weights](https://drive.google.com/file/d/1baVs0SCmYQqmwgWp8ikdNAanOn4oBwQ3/view?usp=sharing)
- [classifier bias](https://drive.google.com/file/d/10UtAz-dBPxEfElqSdfl4XaczteXmsaQj/view?usp=sharing)
- [classifier weights](https://drive.google.com/file/d/1P_IjUzA-KOulwRkHKm-6SY8XdHCOTXVC/view?usp=sharing)
- [batchnorm/BPReLU weights](https://drive.google.com/file/d/1m9OGR0V8MhVnxBm84nTAR75-wCFUDgCW/view?usp=sharing)

To deploy the bitstream:

- Step1: Download the files and move them to ```/xcel-imagenet/deploy/```
- Step2: Move the generated bitstream and hardware definition files to ```/xcel-imagenet/deploy/```
- Step3: Upload the entire directory ```/xcel-imagenet/deploy/``` to the board
- Step4: Go to ```deploy/```, run ```sudo python3 FracNet-ImageNet.py``` on the board
