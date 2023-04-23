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
python unetr_pp/inference_synapse.py
```




