# Adaptive Batch Size and Learning Rate Scheduler for Stochastic Gradient Descent Based on Minimization of Stochastic First-order Oracle Complexity
Source code for reproducing our paper's experiments.

# Abstract
The convergence behavior of mini-batch stochastic gradient descent (SGD) is highly sensitive to the batch size and learning rate settings. Recent theoretical studies have identified the existence of a critical batch size that minimizes stochastic first-order oracle (SFO) complexity, defined as the expected number of gradient evaluations required to reach a stationary point of the empirical loss function in a deep neural network. An adaptive scheduling strategy is introduced to accelerate SGD that leverages theoretical findings on the critical batch size. The batch size and learning rate are adjusted on the basis of the observed decay in the full gradient norm during training. Experiments using an adaptive joint scheduler based on this strategy demonstrated improved convergence speed compared with that of existing schedulers.

# Usage

To train a model on **CIFAR-10** or **CIFAR-100**, run `cifar10.py` or `cifar100.py` with a JSON file specifying the training parameters. Optionally, use the `--cuda_device` argument to choose a CUDA device. The default is device `0`:

```bash
python cifar10.py XXXXX.json --cuda_device 1
python cifar100.py XXXXX.json --cuda_device 1
```

For more details about configuring checkpoints, refer to the `checkpoint_path` section in the **Parameters Description**.

### Customizing Training

To customize the training process, modify the parameters in the JSON file and rerun the script. You can adjust the model architecture, learning rate, batch size, and other parameters to explore different training schedulers and observe their effects on model performance.

## Example JSON Configuration
The following JSON configuration file is located at `src/json/periodic.json`:
```
{
    "model": "densenet",
    "bs": 8,
    "bs_method": "exp_growth",
    "bs_exp_rate": 2,
    "bs_step_type": "periodic",
    "lr": 0.1,
    "lr_method": "exp_growth",
    "lr_exp_rate": 1.4,
    "lr_step_type": "periodic",
    "eps": 0.5,
    "epochs": 200,
    "csv_path": "../result/CIFAR100/bs_periodic_lr_periodic/run1/",
    "check_every": 1000
}
```
### Parameters Description
| Parameter | Value | Description |
| :-------- | :---- | :---------- |
| `model` | `"resnet18"`, `"WideResNet28_10"`, etc. | Specifies the model architecture to use. |
| `bs_method` | `"constant"`, `"linear_growth"`, `"exp_growth"` | Method for adjusting the batch size. |
|`bs_step_type`|`"periodic"`, `"eps"`|Determines how to update the scheduler for the batch size specified by `“bs_method”`.|
|`lr_method`|`"constant"`, `"cosine"`, `"exp_growth"`|Method for adjusting the learning rate.|
|`lr_step_type`|`"periodic"`, `"eps"`|Determines how to update the scheduler for the learning rate specified by `“lr_method”`.|
|`bs`|`int` (e.g., `128`)| The initial batch size for the optimizer. |
|`lr`|`float` (e.g., `0.1`)| The initial learning rate for the optimizer. |
|`eps`|`float` (e.g., `0.1`)| The initial epsilon for the optimizer when `"bs_step_type"` or `"lr_step_type"` is `"eps"`. |
|`check_every`|`int` (e.g., `1000`)| The interval for calculating the gradient norm when `"bs_step_type"` or `"lr_step_type"` is `"eps"`.|
|`epochs`|`int` (e.g., `300`)|The total number of epochs for training.|
|`bs_exp_rate`|`float` (e.g., `2.0`)|The factor by which the batch size increases after each interval. Used when `bs_method` is `"exp_growth"`.|
|`lr_exp_rate`| `float` (e.g., `1.4`) |The factor by which the learning rate increases after each interval. Used when `lr_method` is `"exp_growth"`.|
|`csv_path`|`str` (e.g., `"path/to/result/csv/"`)|Specifies the directory where CSV files will be saved. Four CSV files—`train.csv`, `test.csv`, `norm.csv`, and `lr_bs.csv`—will be saved in this directory.|
