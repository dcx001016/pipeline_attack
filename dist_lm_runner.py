import argparse
import time
import random
import torch
import torch.autograd.profiler as profiler
import numpy as np
import wandb

from attack.attack import *
from modules.gpt_modules import GPTConfig
from modules.tokenizer import *
from communication.comm_utils import *
from utils.dist_args_utils import *
from utils.dist_train_utils import *
from utils.dist_test_utils import *
from utils.common_utils import *
from tasks.data_loaders.arxiv21 import *
from tasks.data_loaders.wikitext import *
from pipeline_parallel.dist_pp_utils import get_pp_module

def train_loop(args, pipe, device, train_data_loader, test_data_loader):
    
    for e in range(args.n_epochs):
        distributed_train_lm_iter(args, pipe, device, train_data_loader)
        
        if test_data_loader is not None and args.do_evaluation:
            distributed_test_lm_iter(args, pipe, device, test_data_loader, e)

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Gpipe-GPT2')
    add_device_arguments(parser)
    add_model_arguments(parser)
    add_task_arguments(parser)
    add_torch_distributed_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    add_parallel_schema_arguments(parser)
    add_acitvation_compression_arguments(parser)
    add_mixed_precision_arguments(parser)
    add_attack_schema_arguments(parser)

    parser.add_argument('--model-name', type=str, default='checkpoints/gpt2', metavar='S',
                        help='model name or path')
    parser.add_argument('--tokenizer-name', type=str, default='gpt2', metavar='S',
                        help='tokenizer name or path')
    parser.add_argument('--task-name', type=str, default='wikitext', metavar='S',
                        help='task name')
    parser.add_argument('--task-type', type=str, default='language_model', metavar='S',
                        help='task typw')
    parser.add_argument('--n-epochs', type=int, default=5, help='-')
    parser.add_argument('--warmup-epochs', type=int, default=1, help='-')
    parser.add_argument('--warmup-steps', type=int, default=None, help='-')
    parser.add_argument('--load-pretrained-model', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='load pretrained model or not.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--profiling', type=str, default='no-profiling', metavar='S',
                        help='enable which profiling? default: tidy mode')
    parser.add_argument('--trace-postfix', type=str, default='default', metavar='S',
                        help='postfix of the tracing file name.')
    parser.add_argument('--do-evaluation', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='do evaluation or not.')
    parser.add_argument('--wandb', 
                        type=lambda x: x.lower()=='true', default=False, metavar='S',
                        help='-')
    parser.add_argument('--write-xlsx', 
                        type=lambda x: x.lower()=='true', default=False, metavar='S',
                        help='-')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')

    if not args.forward_attack:
        args.forward_attack_rate = 0
    if not args.backward_attack:
        args.backward_attack_rate = 0

    assert args.pipeline_virtual_gpus % args.pipeline_group_size == 0
    args.virtual_gpus = int(args.pipeline_virtual_gpus / args.pipeline_group_size)
    assert args.num_layers % args.virtual_gpus == 0
    args.virtual_num_layers = int(args.num_layers / args.virtual_gpus)

    init_communicators(args)

    if args.wandb and get_pipeline_parallel_rank() == args.pipeline_group_size-1:
        wandb.init(project=f"dist_lm_runner-{args.optimizer}-{args.task_name}-vgpus-{args.pipeline_virtual_gpus}", 
                   name=f"forward_attack_rate:{args.forward_attack_rate}--backward_attack_rate:{args.backward_attack_rate}",
                   save_code=False)
        init_wandn_config(args)

    config = GPTConfig.from_pretrained(args.model_name)

    config.n_layer = args.num_layers

    tokenizer = build_tokenizer(args)
    tokenizer.model_max_length = args.seq_length
    config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id

    if args.task_name == 'wikitext':
        train_data_loader = get_wikitext_train_data_loader(args, tokenizer)
        test_data_loader = get_wikitext_test_data_loader(args, tokenizer)
    elif args.task_name == 'arxiv21':
        train_data_loader = get_arxiv21_train_data_loader(args, tokenizer)
        test_data_loader = get_arxiv21_test_data_loader(args, tokenizer)
    else:
        raise Exception('unknown task.')
    
    if args.warmup_steps is None:
        args.warmup_steps = len(train_data_loader)
    args.total_steps = len(train_data_loader) * args.n_epochs

    use_dp = (args.world_size != args.pipeline_group_size)
    if use_dp:
        print("Running ", args.pp_mode, " with data parallel.")
    else:
        print("Running ", args.pp_mode, " without data parallel.")

    pipe = get_pp_module(args, config, device, use_dp)

    if args.load_pretrained_model:
        if get_pipeline_parallel_rank() == 0:
            pipe.model.model[0].load_state_dict(
                torch.load(f'{args.model_name}/pytorch_embs.pt')
            )
            for i in range(len(pipe.model.model)-1):
                print(i)
                pipe.model.model[i+1].load_state_dict(
                    torch.load(f'{args.model_name}/pytorch_{i}.pt')
                )
                if i != 0 and i % args.virtual_num_layers == 0:
                    pipe.model.model[i+1].register_forward_pre_hook(attack_forward_hook(args.forward_attack_rate))

        elif get_pipeline_parallel_rank() == args.pipeline_group_size-1:
            _i = get_pipeline_parallel_rank() * args.num_layers
            # skip last classification layer
            for i in range(len(pipe.model.model)-1):
                print(i+_i)
                pipe.model.model[i].load_state_dict(
                    torch.load(f'{args.model_name}/pytorch_{_i + i}.pt')
                )
                if i % args.virtual_num_layers == 0:
                    pipe.model.model[i].register_forward_pre_hook(attack_forward_hook(args.forward_attack_rate))

            pipe.model.model[-1].load_state_dict(
                torch.load(f'{args.model_name}/pytorch_lm_head.pt')
            )

        else:
            _i = get_pipeline_parallel_rank() * args.num_layers
            for i in range(len(pipe.model.model)):
                print(i+_i)
                pipe.model.model[i].load_state_dict(
                    torch.load(f'{args.model_name}/pytorch_{_i + i}.pt')
                )
                if i % args.virtual_num_layers == 0:
                    pipe.model.model[i].register_forward_pre_hook(attack_forward_hook(args.forward_attack_rate))


    if args.profiling == 'no-profiling':
        train_loop(args, pipe, device, train_data_loader, test_data_loader)
    else:
        prefix = './trace_json/gpt3_' + args.pp_mode
        if use_dp:
            prefix = prefix + '_' + args.dp_mode
        trace_file = prefix + get_learning_arguments_str(args) + get_model_arguments_str(args) + \
                     get_dist_arguments_str(args) + get_mixed_precision_arguments_str(args) + '_' + \
                     args.profiling + '_' + args.trace_postfix + '.json'
        if args.profiling == 'tidy_profiling':
            try:
                train_loop(args, pipe, device, train_data_loader, test_data_loader)
            except Exception as e:
                print(get_pipeline_parallel_rank(), e)
            pipe.export_profiling_result(filename=trace_file)
        elif args.profiling == 'pytorch_profiling':
            with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
                train_loop(args, pipe, device, train_data_loader, test_data_loader)
            print(prof.key_averages().table())
            prof.export_chrome_trace(trace_file)
        else:
            print("No recognized profiler?")
            assert False
    end_time = time.time()
    duration = end_time - start_time
    print(get_pipeline_parallel_rank(), 'finished.', "total time:%s" % format_time(duration))
    

if __name__ == '__main__':
    main()
