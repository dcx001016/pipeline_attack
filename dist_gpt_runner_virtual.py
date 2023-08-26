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
from pipeline_parallel.dist_pp_utils import get_pp_module_virtual

def train_loop(args, pipe, device, train_data_loader, test_data_loader):
    total_train_time = 0
    pipe.results = []
    for e in range(args.n_epochs):
        start_time = time.time()
        distributed_train_lm_iter_virtual(args, pipe, device, train_data_loader)
        end_time = time.time()
        total_train_time += end_time - start_time
        pipe.get_metrics()
        if test_data_loader is not None and args.do_evaluation:
            distributed_test_lm_iter_virtual(args, pipe, device, test_data_loader, e)

    return total_train_time

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

    parser.add_argument('--model-name', type=str, default='checkpoints/gpt2-medium', metavar='S',
                        help='model name or path')
    parser.add_argument('--tokenizer-name', type=str, default='gpt2-medium', metavar='S',
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
        wandb.init(project=f"dist_gpt_runner-same_magnitude-defense-{args.optimizer}-{args.task_name}-vgpus-{args.pipeline_virtual_gpus}", 
                   name=f"forward_attack_rate:{args.forward_attack_rate}",
                   save_code=False)
        init_wandb_config(args)

    config = GPTConfig.from_pretrained(args.model_name)

    config.n_layer = args.virtual_num_layers

    tokenizer = build_tokenizer(args)
    tokenizer.model_max_length = args.seq_length
    config.vocab_size = tokenizer.vocab_size
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    if not args.dropout:
        config.attn_pdrop = 0
        config.embd_pdrop = 0
        config.resid_pdrop = 0
        config.summary_first_dropout = 0
    

    if args.task_name in {'wikitext', 'wiki103'}:
        train_data_loader = get_wikitext_train_data_loader(args, tokenizer)
        test_data_loader = get_wikitext_test_data_loader(args, tokenizer)
    elif args.task_name == 'arxiv21':
        train_data_loader = get_arxiv21_train_data_loader(args, tokenizer)
        test_data_loader = get_arxiv21_test_data_loader(args, tokenizer)
    else:
        raise Exception('unknown task.')
    
    if args.warmup_steps is None:
        args.warmup_steps = len(train_data_loader)
    args.num_iters = min(len(train_data_loader), args.num_iters)
    args.total_steps = len(train_data_loader) * args.n_epochs

    print("Running ", args.pp_mode)

    pipe = get_pp_module_virtual(args, config, device)

    if args.load_pretrained_model:
        for i in range(len(pipe.virtual_gpus)):
            if pipe.virtual_gpus[i].virtual_rank == 0:
                pipe.virtual_gpus[i].model.model[0].load_state_dict(
                    torch.load(f'{args.model_name}/pytorch_embs.pt')
                )
                for j in range(len(pipe.virtual_gpus[i].model.model) - 1):
                    print(j)
                    pipe.virtual_gpus[i].model.model[j + 1].load_state_dict(
                        torch.load(f'{args.model_name}/pytorch_{j}.pt')
                    )
            elif pipe.virtual_gpus[i].virtual_rank == args.pipeline_virtual_gpus - 1:
                _i = pipe.virtual_gpus[i].virtual_rank * args.virtual_num_layers
                for j in range(len(pipe.virtual_gpus[i].model.model) - 1):
                    print(j + _i)
                    pipe.virtual_gpus[i].model.model[j].load_state_dict(
                        torch.load(f'{args.model_name}/pytorch_{_i + j}.pt')
                    )
                # pipe.virtual_gpus[i].model.model[0].register_forward_pre_hook(attack_forward_hook(args.forward_attack_rate))

                pipe.virtual_gpus[i].model.model[-1].load_state_dict(
                    torch.load(f'{args.model_name}/pytorch_lm_head.pt')
                )
            else:
                _i = pipe.virtual_gpus[i].virtual_rank * args.virtual_num_layers
                for j in range(len(pipe.virtual_gpus[i].model.model)):
                    print(j + _i)
                    pipe.virtual_gpus[i].model.model[j].load_state_dict(
                        torch.load(f'{args.model_name}/pytorch_{_i + j}.pt')
                    )
                # if pipe.virtual_gpus[i].virtual_rank != 0 and pipe.virtual_gpus[i].virtual_rank % args.virtual_gpus == 0:
                #     pipe.virtual_gpus[i].model.model[0].register_forward_pre_hook(print_activation)
                # pipe.virtual_gpus[i].model.model[0].register_forward_pre_hook(attack_forward_hook(args.forward_attack_rate))


    # if args.load_pretrained_model:
    #     if get_pipeline_parallel_rank() == 0:
    #         pipe.model.model[0].load_state_dict(
    #             torch.load(f'{args.model_name}/pytorch_embs.pt')
    #         )
    #         for i in range(len(pipe.model.model)-1):
    #             print(i)
    #             pipe.model.model[i+1].load_state_dict(
    #                 torch.load(f'{args.model_name}/pytorch_{i}.pt')
    #             )
    #             if i != 0 and i % args.virtual_num_layers == 0:
    #                 pipe.model.model[i+1].register_forward_pre_hook(attack_forward_hook(args.forward_attack_rate))

    #     elif get_pipeline_parallel_rank() == args.pipeline_group_size-1:
    #         _i = get_pipeline_parallel_rank() * args.num_layers
    #         # skip last classification layer
    #         for i in range(len(pipe.model.model)-1):
    #             print(i+_i)
    #             pipe.model.model[i].load_state_dict(
    #                 torch.load(f'{args.model_name}/pytorch_{_i + i}.pt')
    #             )
    #             if i % args.virtual_num_layers == 0:
    #                 pipe.model.model[i].register_forward_pre_hook(attack_forward_hook(args.forward_attack_rate))

    #         pipe.model.model[-1].load_state_dict(
    #             torch.load(f'{args.model_name}/pytorch_lm_head.pt')
    #         )

    #     else:
    #         _i = get_pipeline_parallel_rank() * args.num_layers
    #         for i in range(len(pipe.model.model)):
    #             print(i+_i)
    #             pipe.model.model[i].load_state_dict(
    #                 torch.load(f'{args.model_name}/pytorch_{_i + i}.pt')
    #             )
    #             if i % args.virtual_num_layers == 0:
    #                 pipe.model.model[i].register_forward_pre_hook(attack_forward_hook(args.forward_attack_rate))

    total_train_time = train_loop(args, pipe, device, train_data_loader, test_data_loader)
    
    end_time = time.time()
    duration = end_time - start_time
    print(get_pipeline_parallel_rank(), 'finished.', "total time:%s" % format_time(duration), "total train time:%s" % format_time(total_train_time), "invalid rate:%.2f%%" % (pipe.get_invalid_rate() * 100))
    

if __name__ == '__main__':
    main()
