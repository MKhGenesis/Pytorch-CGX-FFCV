# Pytorch-CGX-FFCV
This code will guide through the process of training CIFAR-10 using both CGX and FFCV frameworks, [CGX](https://github.com/IST-DASLab/torch_cgx/) is a Pytorch extension that optimizes multi-GPU training while [FFCV](https://github.com/libffcv/ffcv/) will make your data loading faster and therefore make your training less costly. Significant results were achieved combining these two as the training time on an RTX 3090 Optimized offered by Genesis Cloud is cut to half, which makes the multi-GPU very much less expensive especially compared to other cloud providers such as AWS and GCP. 

In order to run this training, it is very important to follow the installation steps in [FFCV](https://github.com/libffcv/ffcv/) and [CGX](https://github.com/IST-DASLab/torch_cgx/) of both frameworks. We'd recommend you to start by building the FFCV environment and then install Pytorch-CGX inside. Once the installation of both is complete, you should be able to import both FFCV and CGX in your python code. Last steo would be to run your code, to do so, you first need to create your dataset in FFCV-compatible format, run your write_datasets.py file using the line of code:
```
python write_datasets.py --config-file default_config.yaml

```

CGX is based on MPI backend, it requires MPI-compliant, so to run the script with multiple GPUs, use the following command in your terminal:

```bash
mpirun -np $NUM_GPUs python train_cifar_ffcv_cgx.py: 
```
Replace `$NUM_GPUs` with the number of GPUs you wish to use.

On an RTX 3090 Optimized offered by Genesis Cloud, we launched a training with 4 GPUs with the train_cifar_ffcv_cgx.py file, and results were that we only needed a quarter of the training time a Vanilla code would take to train the CIFAR-10 for 20 epochs to get past an accuracy of 90%, this table showcases the numbers: 

| Server         | Avg. Training time (s) | Avg. Training Cost ($) |
|----------------|------------------------|------------------------|
| Vanilla Code   | 190                    | 0.17                   |
| CGX            | 62.63                  | 0.057                  |
| CGX + FFCV     | 32.84                  | 0.03                   |
