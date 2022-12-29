# torch2-test

**The test result is under CUDA 11.6, python3.10 environment**

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

