# Multi-Alignment

## baseline

```
cd baseline/coarse_repe
bash script/lorra_truth.sh 
```

运行前需要修改 baseline/coarse_repe/script/lorra_xxx.sh中的model_name_or_path,改为存放模型参数的正确路径
同时要注意 baseline/coarse_repe/src/train_val_dataset.py中数据集路径。建议将tstu/alpaca数据集提前下载到本地。
