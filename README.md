## 测试验证

### bug1: 
ValueError: Invalid type <class 'numpy.int32'> for the flop count! Please use a wider type to avoid overflow
```
https://github.com/facebookresearch/fvcore/issues/104
```

### eval synapse
- 验证的时候需要对脚本上的路径进行更改之后才能正常使用
```
#!/bin/sh

DATASET_PATH=D:/datasets/UNETR_PP_DATASETS/DATASET
CHECKPOINT_PATH=unetr_pp/evaluation/unetr_pp_synapse_checkpoint

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python unetr_pp/run/run_training.py 3d_fullres unetr_pp_trainer_synapse 2 0 -val
```
- 并且在run_training.py上需要增加路径
```
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append(".")
```
- 训练的时候保存的配置与模型也要放到相应的路径上
```
unetr_pp/evaluation/unetr_pp_synapse_checkpoint/unetr_pp/3d_fullres/Task002_Synapse/unetr_pp_trainer_synapse__unetr_pp_Plansv2.1/fold_0/model_final_checkpoint.model
unetr_pp/evaluation/unetr_pp_synapse_checkpoint/unetr_pp/3d_fullres/Task002_Synapse/unetr_pp_trainer_synapse__unetr_pp_Plansv2.1/fold_0/validation_raw/validation_args.json
```

输入以下命令才能正常的验证(注意更换数据集的路径), 这个脚本是生成一些结果并保存下来
```
bash evaluation_scripts/run_evaluation_synapse.sh
```

要得到真正的指标需要使用到推理
```
python unetr_pp/inference_synapse.py fold_0
```


## 训练
### 模块性调参
- 网络结构的调参:
https://github.com/Allen-lz/unetr_plus_fusion_hfs/blob/v1/unetr_pp/network_architecture/synapse/unetr_pp_synapse.py
```
1. 现在的网络结构是最全的版本, extract_cross_hfs()的插入位置其实是可以尝试删除一些的
2. 在特征图上叠加x_l和x_h的时候其实可以在前面乘上一个系数来减少x_l和x_h对原特征的影响的

# Four encoders
enc1 = hidden_states[0]
enc1 = self.extract_cross_hfs(enc1)
enc2 = hidden_states[1] + x_l  # <<<<
enc2 = self.extract_cross_hfs(enc2)
enc3 = hidden_states[2]
enc3 = self.extract_cross_hfs(enc3)
enc4 = hidden_states[3]

# Four decoders
dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
dec3 = self.decoder5(dec4, enc3)
dec2 = self.decoder4(dec3, enc2) + x_h  # <<<<
dec1 = self.decoder3(dec2, enc1)

out = self.decoder2(dec1, convBlock)
```

- loss的调参: https://github.com/Allen-lz/unetr_plus_fusion_hfs/blob/v1/unetr_pp/training/network_training/unetr_pp_trainer_synapse.py
```

1. 在loss叠加的时候, 可以在l += reon_loss(output["recon"], data)中的reon_loss前乘上一个系数, 来减小reon_loss带来的影响, 因为这里的reon_loss相比起原来的loss过大了

if self.fp16:
    with autocast():
        output = self.network(data)
        l = self.loss(output["original"], target)
        b, c, f, h, w = data.shape
        data = data.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        for reon_loss in self.recon_loss:
            l += reon_loss(output["recon"], data)
        del data

    if do_backprop:
        self.amp_grad_scaler.scale(l).backward()
        self.amp_grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.amp_grad_scaler.step(self.optimizer)
        self.amp_grad_scaler.update()
else:
    output = self.network(data)
    l = self.loss(output["original"], target)

    b, c, f, h, w = data.shape
    data = data.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
    for reon_loss in self.recon_loss:
        l += reon_loss(output["recon"], data)
    del data
    if do_backprop:
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
```








