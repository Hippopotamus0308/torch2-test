# torch2-test

**The test result is under CUDA 11.6, python3.10 environment**



### Some tips for Installation (Under GPU machine)

- To install (If under CUDA 11.7, change link to https://download.pytorch.org/whl/nightly/cu117)

```
pip3 install numpy --pre torch --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu116
```

- **After installation, it's highly possible that some packages are not installed, like `torchvision`, you have to search from the above link to download the package and install locally**
- To test whether it's successfully installed, you can use `python test_basic_torch.py` to see whether there is output
- If your machine installed multiple CUDA, better use `nvcc --vision` to check (not `nvidia-smi`), if the result shows a lower version, use

```
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}
```



### Result for test_basic_torch

If there is output without any error or warning, then the environment is installed properly.



### Result for test_speedup_torch_compile

### Test1:

**Using mode = reduce-overload in torch compile for resnet18, and load the model 10 times, the median time is:**

| Time | Input                    | Former median | After opt median | Speedup |
| ---- | ------------------------ | ------------- | ---------------- | ------- |
| 1    | randn(16, 3, 128, 128)   | 0.00298       | 0.00314          | 0.95    |
| 2    | randn(16, 3, 128, 128)   | 0.00296       | 0.00312          | 0.95    |
| 3    | randn(16, 3, 128, 128)   | 0.00321       | 0.00316          | 1.02    |
| 4    | randn(500, 3, 128, 128)  | 0.0634        | 0.0589           | 1.078   |
| 5    | randn(500, 3, 128, 128)  | 0.0635        | 0.0588           | 1.079   |
| 6    | randn(1000, 3, 128, 128) | 0.1059        | 0.09573          | 1.106   |
| 7    | randn(1000, 3, 128, 128) | 0.1061        | 0.09578          | 1.108   |





### Test2:

**Tested mode = default, reduce-overload, and a custom_backend, the median time is:**

| Time | Input                   | Former median | Mode=default | Mode=reduce_overload | Custom_backend |
| ---- | ----------------------- | ------------- | ------------ | -------------------- | -------------- |
| 1    | randn(100, 3, 128, 128) | 0.0674        | 0.0667       | 0.0672               | 0.0652         |
| 2    | randn(100, 3, 128, 128) | 0.0655        | 0.0661       | 0.0666               | 0.0667         |
| 3    | randn(100, 3, 128, 128) | 0.0686        | 0.065        | 0.0644               | 0.0672         |
| 4    | randn(500, 3, 128, 128) | 0.3795        | 0.3805       | 0.3816               | 0.3832         |
| 5    | randn(500, 3, 128, 128) | 0.3736        | 0.3745       | 0.3788               | 0.3842         |



### Test3:

#### Added SGD optimizer

| Time | Input                  | Former median time | After opt median time | Speedup |
| ---- | ---------------------- | ------------------ | --------------------- | ------- |
| 1    | randn(16, 3, 224, 224) | 0.006526           | 0.006547              | 0.99676 |
| 2    | randn(16, 3, 224, 224) | 0.005498           | 0.006544              | 0.84    |
| 3    | randn(16, 3, 224, 224) | 0.005444           | 0.006538              | 0.83    |
| 4    | randn(16, 3, 224, 224) |                    |                       | 0.84    |
| 5    | randn(16, 3, 224, 224) |                    |                       | 0.84    |
| 6    | randn(16, 3, 224, 224) |                    |                       | 0.89    |
| 7    | randn(16, 3, 224, 224) |                    |                       | 0.83    |
| 8    | randn(16, 3, 224, 224) |                    |                       | 0.87    |
| 9    | randn(16, 3, 224, 224) |                    |                       | 0.83    |
| 10   | randn(16, 3, 224, 224) |                    |                       | 0.83    |



### Result for clip_model loading

Data set:  CIFAR100, batch_size = 100

| Time | Former median time | After opt median time |
| ---- | ------------------ | --------------------- |
| 1    | 14.469             | 21.286                |
| 2    | 13.317             | 23.659                |
| 3    | 13.055             | 13.307                |
| 4    | 13.125             | 13.078                |
| 5    | 13.010             | 13.112                |
| 6    | 14.130             | 13.105                |
| 7    | 14.274             | 13.345                |
| 8    | 13.215             | 13.268                |
| 9    | 13.285             | 13.274                |
| 10   | 13.128             | 13.478                |

default model median time: 13.250109195709229



opt model median time: 13.287628173828125



default model mean time: 13.500806760787963



opt model mean time: 15.088787126541138
