from collections import defaultdict
import os
import sys
import time
import logging
import numpy as np
import random
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from turing.logger import Logger
from turing.utils import get_sample_writer
from turing.models import BertMultiTask
from turing.dataset import QADataset, RankingDataset, PreTrainingDataset, QAFinetuningDataset
from turing.dataset import QABatch, RankingBatch, PretrainBatch, PretrainDataType
from turing.sources import WikiPretrainingDataCreator, PretrainingDataCreator, TokenInstance
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear, warmup_linear_decay_exp, warmup_exp_decay_exp, warmup_exp_decay_poly
from turing.sources import WikiPretrainingDataCreator, PretrainingDataCreator, TokenInstance
from utils import get_argument_parser, is_time_to_exit

import deepspeed
from data_worker_fixed_microbatch import AsyncWorker


global_step = 0
global_data_samples = 0
last_global_step_from_restore = 0
flexible_batching=True

def gather_grad(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    if len(parameters) == 0:
        return torch.tensor(0.)
    total_norm = torch.cat([p.grad.detach().view(-1) for p in parameters]).cpu()
    return total_norm

def gather_grouped_grad(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    # parameters = list(filter(lambda p: p.grad is not None, parameters))
    # if len(parameters) == 0:
    #     return torch.tensor(0.)
    total_norm = torch.cat([p[1].grad.detach().view(-1) for p in parameters])
    return total_norm

def list_grouped_grad(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    total_norm = [p[1].grad for p in parameters]
    return total_norm

def checkpoint_model(PATH, ckpt_id, model, epoch, last_global_step,
                     last_global_data_samples, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        'epoch': epoch,
        'last_global_step': last_global_step,
        'last_global_data_samples': last_global_data_samples
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.network.save_checkpoint(PATH, ckpt_id,
                                            checkpoint_state_dict)
    status_msg = 'checkpointing: PATH={}, ckpt_id={}'.format(PATH, ckpt_id)
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return


def load_training_checkpoint(args, model, PATH, ckpt_id):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    logger = args.logger
    _, checkpoint_state_dict = model.network.load_checkpoint(PATH, ckpt_id)
    epoch = checkpoint_state_dict['epoch']
    last_global_step = checkpoint_state_dict['last_global_step']
    last_global_data_samples = checkpoint_state_dict[
        'last_global_data_samples']
    del checkpoint_state_dict
    return (epoch, last_global_step, last_global_data_samples)


def get_effective_batch(args, total):
    if args.local_rank != -1:
        return total // dist.get_world_size(
        ) // args.train_micro_batch_size_per_gpu // args.gradient_accumulation_steps // args.refresh_bucket_size
    else:
        return total // args.train_micro_batch_size_per_gpu // args.gradient_accumulation_steps // args.refresh_bucket_size


def get_dataloader(args, dataset: Dataset, eval_set=False):
    if args.local_rank == -1:
        train_sampler = RandomSampler(dataset)
    else:
        train_sampler = DistributedSampler(dataset)
    return (x for x in
            DataLoader(dataset,
                       batch_size=args.train_micro_batch_size_per_gpu //
                       2 if eval_set else args.train_micro_batch_size_per_gpu,
                       sampler=train_sampler,
                       num_workers=args.config['training']['num_workers']))


def pretrain_validation(args, index, model):
    config = args.config
    logger = args.logger

    model.eval()
    dataset = PreTrainingDataset(
        args.tokenizer,
        os.path.join(args.data_path_prefix, config['validation']['path']),
        args.logger, args.max_seq_length, index, PretrainDataType.VALIDATION,
        args.max_predictions_per_seq)
    data_batches = get_dataloader(args, dataset, eval_set=True)
    eval_loss = 0
    nb_eval_steps = 0
    for batch in tqdm(data_batches,disable=True):
        batch = tuple(t.to(args.device) for t in batch)
        tmp_eval_loss = model.network(batch, log=False)
        dist.reduce(tmp_eval_loss, 0)
        # Reduce to get the loss from all the GPU's
        tmp_eval_loss = tmp_eval_loss / dist.get_world_size()
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    logger.info(f"Validation Loss for epoch {index + 1} is: {eval_loss}")
    if (not args.no_cuda
            and dist.get_rank() == 0) or (args.no_cuda
                                          and args.local_rank == -1):
        args.summary_writer.add_scalar(f'Validation/Loss', eval_loss,
                                       index + 1)
    return


def master_process(args):
    return (not args.no_cuda
            and dist.get_rank() == 0) or (args.no_cuda
                                          and args.local_rank == -1)


def get_train_dataset(args, index, finetune=False, shuffle=True):
    assert not finetune, "finetune not supported"
    i = 0
    dataloaders = {}
    datalengths = []
    batchs_per_dataset = []
    batch_mapping = {}

    config = args.config
    dataset_paths = config["data"]["datasets"]
    dataset_flags = config["data"]["flags"]

    # Pretraining dataset
    if dataset_flags.get("pretrain_dataset", False):
        pretrain_type = dataset_flags.get("pretrain_type")

        if pretrain_type == "wiki_bc":
            # Load Wiki Dataset
            wiki_pretrain_dataset = PreTrainingDataset(
                args.tokenizer,
                os.path.join(args.data_path_prefix,
                             dataset_paths['wiki_pretrain_dataset']),
                args.logger, args.max_seq_length, index,
                PretrainDataType.NUMPY, args.max_predictions_per_seq)
            datalengths.append(len(wiki_pretrain_dataset))
            dataloaders[i] = get_dataloader(args, wiki_pretrain_dataset)
            batch_mapping[i] = PretrainBatch
            batchs_per_dataset.append(
                get_effective_batch(args, len(wiki_pretrain_dataset)))
            i += 1

            bc_pretrain_dataset = PreTrainingDataset(
                args.tokenizer,
                os.path.join(args.data_path_prefix,
                             dataset_paths['bc_pretrain_dataset']),
                args.logger, args.max_seq_length, index,
                PretrainDataType.NUMPY, args.max_predictions_per_seq)
            datalengths.append(len(bc_pretrain_dataset))
            dataloaders[i] = get_dataloader(args, bc_pretrain_dataset)
            batch_mapping[i] = PretrainBatch
            batchs_per_dataset.append(
                get_effective_batch(args, len(bc_pretrain_dataset)))
            i += 1

    dataset_batches = []
    for i, batch_count in enumerate(batchs_per_dataset):
        dataset_batches.extend([i] * batch_count)

    # shuffle
    if shuffle:
        random.shuffle(dataset_batches)

    dataset_picker = []
    for dataset_batch_type in dataset_batches:
        dataset_picker.extend([dataset_batch_type] *
                              args.gradient_accumulation_steps *
                              args.refresh_bucket_size)

    return dataset_picker, dataloaders, sum(datalengths)


def train(args, index, model, optimizer, finetune=False):
    global global_step
    global global_data_samples
    global last_global_step_from_restore

    dataset_picker, dataloaders, total_length = get_train_dataset(
        args, index, finetune)
    current_data_sample_count = global_data_samples
    global_data_samples += total_length
    config = args.config
    logger = args.logger
    print('total_length', total_length, 'global_data_samples',
          global_data_samples)

    model.train()

    if args.config['training']['async_worker']:
        worker = AsyncWorker(dataloaders, dataset_picker)
        worker.start()

    epoch_step = 0
    for step, dataset_type in enumerate(tqdm(dataset_picker, smoothing=1,disable=True)):
        try:
            if args.config['training']['async_worker']:
                batch = worker.get()
            else:
                batch = next(dataloaders[dataset_type])

            batch = tuple(t.to(args.device) for t in batch)  # Move to GPU

            # Calculate forward pass
            loss = model.network(batch)
            unscaled_loss = loss.item()
            current_data_sample_count += (args.train_micro_batch_size_per_gpu *
                                          dist.get_world_size())
            
            # Prefetch training data
            if args.config['training']['async_worker']:
                worker.prefetch()
            # print(loss)
            model.network.backward(loss)
            # print("micro batch end")

            if model.network.is_gradient_accumulation_boundary():
                # print(f"model update at {step}")
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = update_learning_rate_by_number_of_samples(
                        args, config, current_data_sample_count, optimizer,current_global_step=global_step)
                else:
                    for param_group in optimizer.param_groups:
                        lr_this_step=param_group['lr']
            
                # if current_data_sample_count/65536>=(config["training"]["total_training_steps"]*config["training"]["warmup_proportion"]):
                #     model.network.adaptive_batch_params["enable_adjust"]=True

                report_step_metrics(args, lr_this_step, unscaled_loss,
                                    global_step, current_data_sample_count,model=model)
                if (not args.no_cuda
                        and dist.get_rank() == 0) or (args.no_cuda
                                                    and args.local_rank == -1):
                    args.summary_writer.add_scalar(f'Train/Samples/cos_similarity', model.network.cos_placeholder if model.network.cos_placeholder else float('NaN'),
                                                current_data_sample_count)
                    # args.summary_writer.add_scalar(f'Train/Samples/adjust_step', model.network.gradient_step_size if model.network.gradient_step_size else float('NaN'),
                    #                             current_data_sample_count)
                    args.summary_writer.add_scalar(f'Train/Samples/accumulation_steps',model.network.gradient_accumulation_steps() ,
                                                current_data_sample_count)
                    args.summary_writer.add_scalar(f'Train/Samples/micro_batch_size',model.network.train_micro_batch_size_per_gpu() ,
                                                current_data_sample_count)
                    args.summary_writer.add_scalar(f'Train/Samples/train_batch_size',model.network.train_batch_size() ,
                                                current_data_sample_count)
                #qy
                # checkpoint_model(PATH=args.saved_model_path,
                #                 ckpt_id='epoch{}_step{}'.format(
                #                     index + 1, global_step),
                #                 model=model,
                #                 epoch=index + 1,
                #                 last_global_step=global_step,
                #                 last_global_data_samples=global_data_samples)
                # print(f"saved checkpoint at {global_step}")

                model.network.step()

                report_lamb_coefficients(args, optimizer)
                global_step += 1
                epoch_step += 1

            else:
                # Call DeepSpeed engine step on micro steps
                model.network.step()

        except StopIteration:
            continue

        current_global_step = global_step - last_global_step_from_restore
        if is_time_to_exit(args=args,
                           epoch_steps=epoch_step,
                           global_steps=current_global_step):
            print(
                f'Warning: Early epoch termination due to max steps limit, epoch step ={epoch_step}, global step = {current_global_step}, epoch = {index+1}'
            )
            break

    if args.config['training']['async_worker']:
        worker.stop()

    # Run Validation Loss
    if not finetune and args.max_seq_length == 512:
        logger.info(f"TRAIN BATCH SIZE: {args.train_micro_batch_size_per_gpu}")
        pretrain_validation(args, index, model)


def update_learning_rate(args, config, current_global_step, optimizer):
    global last_global_step_from_restore

    global_step_for_lr = current_global_step - last_global_step_from_restore

    if args.lr_schedule == "EE":
        #print(f'LR Schedule is {args.lr_schedule} EE')
        lr_this_step = config["training"][
            "learning_rate"] * warmup_exp_decay_exp(
                global_step_for_lr, config["training"]["decay_rate"],
                config["training"]["decay_step"],
                config["training"]["total_training_steps"],
                config["training"]["warmup_proportion"])
    elif args.lr_schedule == "EP":
        print(f'LR Schedule is {args.lr_schedule} EP')
        lr_this_step = config["training"][
            "learning_rate"] * warmup_exp_decay_poly(
                global_step_for_lr, config["training"]["total_training_steps"],
                config["training"]["warmup_proportion"])
    else:
        lr_this_step = config["training"][
            "learning_rate"] * warmup_linear_decay_exp(
                global_step_for_lr, config["training"]["decay_rate"],
                config["training"]["decay_step"],
                config["training"]["total_training_steps"],
                config["training"]["warmup_proportion"])
    lr_this_step += args.lr_offset

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step

    return lr_this_step

#qy

def decay_exp(global_step,
                         decay_rate,
                         decay_steps):
    return decay_rate**(global_step/ decay_steps)

def decay_linear(global_step,
                         total_steps):
    return 1-(global_step/ total_steps)

def update_learning_rate_by_number_of_samples(args, config, current_data_sample_count, optimizer,current_global_step=None):
    if args.lr_schedule == "EE":
        #print(f'LR Schedule is {args.lr_schedule} EE')
        # lr_this_step = config["training"][
        #     "learning_rate"] * warmup_exp_decay_exp(
        #         global_step_for_lr, config["training"]["decay_rate"],
        #         config["training"]["decay_step"],
        #         config["training"]["total_training_steps"],
        #         config["training"]["warmup_proportion"])
        if not args.nowarmup:
            lr_this_step = config["training"][
                "learning_rate"] * warmup_exp_decay_exp(
                    current_data_sample_count/65536, config["training"]["decay_rate"],
                    config["training"]["decay_step"],
                    config["training"]["total_training_steps"],
                    config["training"]["warmup_proportion"])
            lr_this_step += args.lr_offset
        else:
        # lr_this_step = config["training"]["learning_rate"] * decay_exp(global_step=current_data_sample_count, decay_rate=config["training"]["decay_rate"],decay_steps=config["training"]["decay_step"]*65536)
            lr_this_step = config["training"]["learning_rate"] * decay_linear(global_step=current_data_sample_count,total_steps=400000000)
    else:
        raise Exception

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step
    
    # if torch.distributed.get_rank()==0:
    #     print(f"Given current samples {current_data_sample_count} and total {config['training']['total_training_steps']*65536}, lr changed to {lr_this_step}")
    #     print(f"The detail is base lr {config['training']['learning_rate']} times decay {decay_linear(global_step=current_data_sample_count,total_steps=400000000)} and add to lr_offset {args.lr_offset}")
    # os._exit(0)
    return lr_this_step



def report_step_metrics(args, lr, loss, step, data_sample_count,model=None):
    ##### Record the LR against global_step on tensorboard #####
    if (not args.no_cuda
            and dist.get_rank() == 0) or (args.no_cuda
                                          and args.local_rank == -1):
        lr_modifier=model.network.adaptive_batch_params["global_lr_modifier"] if model.network.adaptive_batch_params["global_lr_modifier"] else 1.0
        args.summary_writer.add_scalar(f'Train/lr', lr*lr_modifier, step)

        args.summary_writer.add_scalar(f'Train/Samples/train_loss', loss,
                                       data_sample_count)

        args.summary_writer.add_scalar(f'Train/Samples/lr', lr*lr_modifier,
                                       data_sample_count)
    ##### Recording  done. #####

    if (step + 1) % args.print_steps == 0 and master_process(args):
        print('bing_bert_progress: step={}, loss={}, lr={}, sample_count={}'.
              format(step + 1, loss, lr, data_sample_count))


def report_lamb_coefficients(args, optimizer):
    if master_process(args):
        if (args.fp16 and args.use_lamb):
            #print("Lamb Coeffs", optimizer.optimizer.get_lamb_coeffs())
            lamb_coeffs = optimizer.optimizer.get_lamb_coeffs()
            lamb_coeffs = np.array(lamb_coeffs)
            if lamb_coeffs.size > 0:
                args.summary_writer.add_histogram(f'Train/lamb_coeffs',
                                                  lamb_coeffs, global_step)


def get_arguments():
    parser = get_argument_parser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    #qy
    parser.add_argument('--start_accumulation_steps', type=int, help='an integer for the accumulator')
    parser.add_argument('--start_adjust_step', type=float, help='an integer for the accumulator')
    parser.add_argument('--nowarmup', action="store_true", help='an integer for the accumulator')
    parser.add_argument('--similarity_target',default=None, type=float)
    parser.add_argument('--batchsize_upper_bound',default=None, type=int)
    parser.add_argument('--batchsize_lower_bound',default=None, type=int)
    args = parser.parse_args()

    # no cuda mode is not supported
    args.no_cuda = False

    return args


def construct_arguments():
    args = get_arguments()

    # Prepare Logger
    logger = Logger(cuda=torch.cuda.is_available() and not args.no_cuda)
    args.logger = logger
    config = json.load(open(args.config_file, 'r', encoding='utf-8'))

    # choose dataset and training config based on the given sequence length
    seq_len = str(args.max_seq_length)
    datasets = config["data"]["mixed_seq_datasets"][seq_len]
    del config["data"]["mixed_seq_datasets"]
    training = config["mixed_seq_training"][seq_len]
    del config["mixed_seq_training"]
    config["data"]["datasets"] = datasets
    config["training"] = training
    args.config = config

    args.job_name = config['name'] if args.job_name is None else args.job_name
    print("Running Config File: ", args.job_name)
    # Setting the distributed variables
    print("Args = {}".format(args))

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    args.saved_model_path = os.path.join(args.output_dir, "saved_models/",
                                         args.job_name)

    args.n_gpu = 1

    # Loading Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"])
    args.tokenizer = tokenizer

    # Issue warning if early exit from epoch is configured
    if args.max_steps < sys.maxsize:
        logging.warning(
            'Early training exit is set after {} global steps'.format(
                args.max_steps))

    if args.max_steps_per_epoch < sys.maxsize:
        logging.warning('Early epoch exit is set after {} global steps'.format(
            args.max_steps_per_epoch))

    return args


def prepare_optimizer_parameters(args, model):
    config = args.config

    param_optimizer = list(model.network.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if "weight_decay" in config["training"].keys():
        weight_decay = config["training"]["weight_decay"]
    else:
        weight_decay = 0.01

    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        weight_decay
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    return optimizer_grouped_parameters


def prepare_model_optimizer(args):
    # Initialize torch distributed
    torch.distributed.init_process_group(backend="nccl")

    # Loading Model
    model = BertMultiTask(args)

    # Optimizer parameters
    optimizer_grouped_parameters = prepare_optimizer_parameters(args, model)

    # DeepSpeed initializer handles FP16, distributed, optimizer automatically.
    model.network, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model.network,
        model_parameters=optimizer_grouped_parameters,
        adaptive_batch_params={"enable_adjust": args.similarity_target is not None, "verbose": True, "similarity_target":args.similarity_target,"batch_size_lower_bound":args.batchsize_lower_bound,"batch_size_upper_bound":args.batchsize_upper_bound})
        # adaptive_batch_params={"enable_adjust": False, "verbose": False, "similarity_target":0.75,"batch_size_upper_bound":65536*2,"max_micro_batch_size":256})

    # Overwrite application configs with DeepSpeed config
    args.train_micro_batch_size_per_gpu = model.network.train_micro_batch_size_per_gpu(
    )
    args.gradient_accumulation_steps = model.network.gradient_accumulation_steps(
    )

    # Set DeepSpeed info
    args.local_rank = model.network.local_rank
    args.device = model.network.device
    model.set_device(args.device)
    args.fp16 = model.network.fp16_enabled()
    args.use_lamb = model.network.optimizer_name(
    ) == deepspeed.pt.deepspeed_config.LAMB_OPTIMIZER

    # Prepare Summary Writer and saved_models path
    if dist.get_rank() == 0:
        summary_writer = get_sample_writer(name=args.job_name,
                                           base=args.output_dir)
        args.summary_writer = summary_writer
        os.makedirs(args.saved_model_path, exist_ok=True)

    return model, optimizer


def load_checkpoint(args, model):
    global global_step
    global global_data_samples
    global last_global_step_from_restore

    config = args.config
    logger = args.logger

    logger.info(
        f"Restoring previous training checkpoint from PATH={args.load_training_checkpoint}, CKPT_ID={args.load_checkpoint_id}"
    )
    start_epoch, global_step, global_data_samples = load_training_checkpoint(
        args=args,
        model=model,
        PATH=args.load_training_checkpoint,
        ckpt_id=args.load_checkpoint_id)
    logger.info(
        f"The model is loaded from last checkpoint at epoch {start_epoch} when the global steps were at {global_step} and global data samples at {global_data_samples}"
    )

    if args.rewarmup:
        logger.info(
            f"Rewarmup learning rate with last_global_step_from_restore = {global_step}"
        )
        last_global_step_from_restore = global_step

    lr_this_step = config["training"][
        "learning_rate"] * warmup_linear_decay_exp(
            global_step, config["training"]["decay_rate"],
            config["training"]["decay_step"],
            config["training"]["total_training_steps"],
            config["training"]["warmup_proportion"])
    logger.info(f"Restart training with lr = {lr_this_step}")

    # Run validation for checkpoint before training
    if not args.finetune and args.max_seq_length == 512:
        logger.info(
            f"Validation Loss of Checkpoint {start_epoch} before pretraining")
        logger.info(
            f"TRAIN MICRO BATCH SIZE PER GPU: {args.train_micro_batch_size_per_gpu}"
        )
        index = start_epoch - 1 if start_epoch > 0 else start_epoch
        pretrain_validation(args, index, model)

    return start_epoch


def run(args, model, optimizer, start_epoch):
    global global_step
    global global_data_samples
    global last_global_step_from_restore

    config = args.config
    logger = args.logger

    for index in range(start_epoch, config["training"]["num_epochs"]):
        logger.info(f"Training Epoch: {index + 1}")
        print(f"Training Epoch: {index + 1}")
        pre = time.time()
        train(args, index, model, optimizer)
        print(f"Finished Training Epoch: {index + 1}")
        #qy: reset summary writer to make sure we can get correct stat when crashing
        if dist.get_rank() == 0:
            args.summary_writer.close()
            summary_writer = get_sample_writer(name=args.job_name,
                                            base=args.output_dir)
            args.summary_writer = summary_writer


        # Save ckpts according to "--ckpt_to_save" option,
        # e.g. "--ckpt_to_save 160 161" to save epoch 160 and 161.
        if args.ckpt_to_save is None or (index + 1) in args.ckpt_to_save:
            logger.info(
                f"Saving a checkpointing of the model for epoch: {index+1}")
            checkpoint_model(PATH=args.saved_model_path,
                             ckpt_id='epoch{}_step{}'.format(
                                 index + 1, global_step),
                             model=model,
                             epoch=index + 1,
                             last_global_step=global_step,
                             last_global_data_samples=global_data_samples)

        post = time.time()
        logger.info(f"Time for shard {index + 1}: {post-pre} seconds")

        current_global_step = global_step - last_global_step_from_restore
        if is_time_to_exit(args=args, global_steps=current_global_step):
            print(
                f'Warning: Early training termination due to max steps limit, epoch={index+1}, global_step={current_global_step}'
            )
            break


def main():
    start = time.time()
    args = construct_arguments()
    model, optimizer = prepare_model_optimizer(args)
    start_epoch = 0
    if not None in [args.load_training_checkpoint, args.load_checkpoint_id]:
        start_epoch = load_checkpoint(args, model)    
    run(args, model, optimizer, start_epoch)
    elapsed = time.time() - start
    logger = args.logger
    logger.info(f"Elapsed time: {elapsed} seconds")


if __name__ == "__main__":
    main()
