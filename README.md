## Anonymous repo created for NIPS submission "SimiGrad: Fine-Grained Adaptive Batching for Large Scale Training using Gradient Similarity Measurement"
This repo contains both our SimiGrad framework (integrated with DeepSpeed) and all training codes used to generate the results in the paper.
## Installation
Please use `./DeepSpeed/install.sh` to install our SimiGrad framework. For detailed installation options please see `./DeepSpeed/install.sh` . It is recommended that you use a virtual environment to install SimiGrad.
## Usage
To use SimiGrad, simply add an additional parameter `adaptive_batch_params` when initializing DeepSpeed. For example,
```
model, optimizer, _, _ = deepspeed.initialize(
        args=...,
        model=...,
        model_parameters=...,
        adaptive_batch_params={
            "enable_adjust": args.similarity_target, # bool, set to `True` to use adaptive batch size and `False` for fixed batch size
            "verbose": True, # bool, set to `True` to print details of batch size adjustment
            "similarity_target":args.similarity_target, # float, -1.0~1.0, the similarity target that controls how aggressive the batch size adjustment is.
            "batch_size_lower_bound":args.batchsize_lower_bound, # int, optional, the lower bound of batch size. Recommended only if you have a well-tuned warmup learning rate scheduling.
            "batch_size_upper_bound":args.batchsize_upper_bound, # int, optional, the upper bound of batch size.
            "max_micro_batch_size":args.max_micro_batch_size, # int, optional, the upper bound of micro batch size to prevent out-of-memory error. If unspecified, the initial micro batch size will be used as the max_micro_batch_size.})
```
Please refer to our code (e.g. `DeepSpeedExamples/pytorch-cifar/main.py`) for details such as how to read the metrics from the framework.

For usage of DeepSpeed, please refer to their website `https://www.deepspeed.ai/`
## Reproduce Paper's Results
The parameters we used to get the claimed results are included in the paper.
### BERT Large Pretrain
All scripts can be found in `DeepSpeedExamples/bert_pretrain/`. Please use the script `ds_train_bert_bsz64k_seq128.sh` for BERT Large pretrain with sequence length 128 (epoch 1-150). You need to specify the parameters like `similarity_target` and also the location of the WikiandBookCorpus dataset in the script. 

After the sequence length 128 pretrain, use `ds_train_bert_bsz32k_seq512.sh` to finish the sequence length 512 part of pretrain (epoch 151-170). You need to specify the checkpoint from sequence length 128 pretrain for the sequence length 512 to start with. Then the BERT Large model is ready for downstream tasks.
### SQuAD Score from BERT Large Pretrain
After the BERT pretrain, use `DeepSpeedExamples/BingBertSquad/run_squad_deepspeed.sh` to get the SQuAD 1.1 score. You need to specify the checkpoint from sequence length 512 pretrain and the location of SQuAD 1.1 dataset.
### ResNet18 on CIFAR10
All scripts can be found in `DeepSpeedExamples/pytorch-cifar/`. Use the script `run.sh` to train ResNet18 with specific parameters. Use the `grid_search.py` and `baseline_grid_search.py` to get the Pareto results of test acc vs. batch size in the paper.
### ResNet50 on ImageNet
All scripts can be found in `DeepSpeedExamples/imagenet_deepspeed/`. Use the script `run_with2kmin.sh` to train ResNet18 with spcific parameters.

## Anonymity of this Repo
We have removed all information that may lead to our identity from the code. This repo is uploaded by an anonymous account with a forward email address to ensure anonymity.