import argparse
import time
import numpy as np
import torch
import torch.autograd.profiler as profiler

from attack.attack import *
from tasks.data_loaders.cola import get_cola_data_loader
from tasks.data_loaders.qnli import get_qnli_data_loader
from modules.deberta_modules import DebertaV2Config
from modules.tokenizer import build_deberta_tokenizer as build_tokenizer
from pipeline_parallel.dist_pp_utils import get_deberta_pp_module_virtual
from utils.dist_args_utils import *
from utils.dist_train_utils import *
from utils.dist_test_utils import *
from utils.common_utils import *
from communication.comm_utils import *

def train_loop(args, pipe, device, train_data_loader, test_data_loader):
    
    for e in range(args.n_epochs):
        distributed_train_bert_iter_virtual(args, pipe, device, train_data_loader)
        
        if test_data_loader is not None and args.do_evaluation:
            distributed_test_bert_iter_virtual(args, pipe, device, test_data_loader, e)

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Gpipe-DeBERTa')
    add_device_arguments(parser)
    add_torch_distributed_arguments(parser)
    add_model_arguments(parser)
    add_task_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    add_mixed_precision_arguments(parser)
    add_parallel_schema_arguments(parser)
    add_acitvation_compression_arguments(parser)
    add_attack_schema_arguments(parser)

    parser.add_argument('--model-name', type=str, default='checkpoints/microsoft/deberta-v3-base', metavar='S',
                        help='model name or path')
    parser.add_argument('--tokenizer-name', type=str, default='microsoft/deberta-v3-base', metavar='S',
                        help='tokenizer name or path')
    parser.add_argument('--task-name', type=str, default='qnli', metavar='S',
                        help='task name')
    parser.add_argument('--n-epochs', type=int, default=10, help='-')
    parser.add_argument('--warmup-epochs', type=int, default=1, help='-')
    parser.add_argument('--warmup-steps', type=int, default=None, help='-')
    parser.add_argument('--load-pretrained-model', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='load pretrained model or not.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--profiling', type=str, default='tidy_profiling', metavar='S',
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
        wandb.init(project=f"dist_deberta_runner-{args.optimizer}-{args.task_name}-vgpus-{args.pipeline_virtual_gpus}", 
                   name=f"forward_attack_rate:{args.forward_attack_rate}",
                   save_code=False)
        init_wandb_config(args)
    
    config = DebertaV2Config.from_pretrained(args.model_name)
    config.num_hidden_layers = args.virtual_num_layers  # num_layers per node
    if not args.dropout:
        config.attention_probs_dropout_prob = 0
        config.hidden_dropout_prob = 0
        config.pooler_dropout = 0
    
    tokenizer = build_tokenizer(args)
    tokenizer.model_max_length = args.seq_length
    print("token vocab size:", tokenizer.vocab_size)
    
    if args.task_name == 'cola':
        train_data_loader = get_cola_data_loader(args, tokenizer, data_split='train')
        test_data_loader = get_cola_data_loader(args, tokenizer, data_split='validation')
        config.num_labels = 2
    elif args.task_name == 'qnli':
        train_data_loader = get_qnli_data_loader(args, tokenizer, data_split='train')
        test_data_loader = get_qnli_data_loader(args, tokenizer, data_split='validation')
        config.num_labels = 2
    else:
        raise Exception('unknown task.')
    
    if args.warmup_steps is None:
        args.warmup_steps = len(train_data_loader)
    args.total_steps = len(train_data_loader) * args.n_epochs

    print("Running ", args.pp_mode)

        
    pipe = get_deberta_pp_module_virtual(args, config, device)
    
    if args.load_pretrained_model:
        for i in range(len(pipe.virtual_gpus)):
            if pipe.virtual_gpus[i].virtual_rank == 0:
                pipe.virtual_gpus[i].model.embeddings.load_state_dict(
                    torch.load(f'{args.model_name}/pytorch_embs.pt')
                )
                for j in range(len(pipe.virtual_gpus[i].model.encoder.layer)):
                    print(j)
                    pipe.virtual_gpus[i].model.encoder.layer[j].load_state_dict(
                        torch.load(f'{args.model_name}/pytorch_{j}.pt')
                    )
                if hasattr(pipe.virtual_gpus[i].model.encoder, 'rel_embeddings'):
                    pipe.virtual_gpus[i].model.encoder.rel_embeddings.load_state_dict(
                        torch.load(f'{args.model_name}/pytorch_rel_embs.pt')
                    )
                if hasattr(pipe.virtual_gpus[i].model.encoder, 'LayerNorm'):
                    pipe.virtual_gpus[i].model.encoder.LayerNorm.load_state_dict(
                        torch.load(f'{args.model_name}/pytorch_ln.pt')
                    )
            elif pipe.virtual_gpus[i].virtual_rank == args.pipeline_virtual_gpus - 1:
                _i = pipe.virtual_gpus[i].virtual_rank * args.virtual_num_layers
                for j in range(len(pipe.virtual_gpus[i].model.encoder.layer)):
                    print(j + _i)
                    pipe.virtual_gpus[i].model.encoder.layer[j].load_state_dict(
                        torch.load(f'{args.model_name}/pytorch_{j + _i}.pt')
                    )
                if hasattr(pipe.virtual_gpus[i].model.encoder, 'rel_embeddings'):
                    pipe.virtual_gpus[i].model.encoder.rel_embeddings.load_state_dict(
                        torch.load(f'{args.model_name}/pytorch_rel_embs.pt')
                    )
                if hasattr(pipe.virtual_gpus[i].model.encoder, 'LayerNorm'):
                    pipe.virtual_gpus[i].model.encoder.LayerNorm.load_state_dict(
                        torch.load(f'{args.model_name}/pytorch_ln.pt')
                    )
            else:
                _i = pipe.virtual_gpus[i].virtual_rank * args.virtual_num_layers
                for j in range(len(pipe.virtual_gpus[i].model.encoder.layer)):
                    print(j + _i)
                    pipe.virtual_gpus[i].model.encoder.layer[j].load_state_dict(
                        torch.load(f'{args.model_name}/pytorch_{j + _i}.pt')
                    )
                if hasattr(pipe.virtual_gpus[i].model.encoder, 'rel_embeddings'):
                    pipe.virtual_gpus[i].model.encoder.rel_embeddings.load_state_dict(
                        torch.load(f'{args.model_name}/pytorch_rel_embs.pt')
                    )
                if hasattr(pipe.virtual_gpus[i].model.encoder, 'LayerNorm'):
                    pipe.virtual_gpus[i].model.encoder.LayerNorm.load_state_dict(
                        torch.load(f'{args.model_name}/pytorch_ln.pt')
                    )

                    
        # if get_pipeline_parallel_rank() == 0:
        #     pipe.model.embeddings.load_state_dict(
        #         torch.load(f'{args.model_name}/pytorch_embs.pt')
        #     )
        #     for i in range(len(pipe.model.encoder.layer)):
        #         pipe.model.encoder.layer[i].load_state_dict(
        #             torch.load(f'{args.model_name}/pytorch_{i}.pt')
        #         )
        #     if hasattr(pipe.model.encoder, 'rel_embeddings'):
        #         pipe.model.encoder.rel_embeddings.load_state_dict(
        #             torch.load(f'{args.model_name}/pytorch_rel_embs.pt')
        #         )
        #     if hasattr(pipe.model.encoder, 'LayerNorm'):
        #         pipe.model.encoder.LayerNorm.load_state_dict(
        #             torch.load(f'{args.model_name}/pytorch_ln.pt')
        #         )
        #     if hasattr(pipe.model.encoder, 'conv') and pipe.model.encoder.conv is not None:
        #         pipe.model.encoder.conv.load_state_dict(
        #             torch.load(f'{args.model_name}/pytorch_conv.pt')
        #         )
        # elif get_pipeline_parallel_rank() == args.pipeline_group_size-1:
        #     _i = get_pipeline_parallel_rank() * args.num_layers
        #     for i in range(len(pipe.model.encoder.layer)):
        #         pipe.model.encoder.layer[i].load_state_dict(
        #             torch.load(f'{args.model_name}/pytorch_{_i+i}.pt')
        #         )
        #     if hasattr(pipe.model.encoder, 'rel_embeddings'):
        #         pipe.model.encoder.rel_embeddings.load_state_dict(
        #             torch.load(f'{args.model_name}/pytorch_rel_embs.pt')
        #         )
        #     if hasattr(pipe.model.encoder, 'LayerNorm'):
        #         pipe.model.encoder.LayerNorm.load_state_dict(
        #             torch.load(f'{args.model_name}/pytorch_ln.pt')
        #         )
        #     if hasattr(pipe.model.encoder, 'conv') and pipe.model.encoder.conv is not None:
        #         raise Exception('should not have conv')
        #         pipe.model.encoder.conv.load_state_dict(
        #             torch.load(f'{args.model_name}/pytorch_conv.pt')
        #         )
        # else:
        #     _i = get_pipeline_parallel_rank() * args.num_layers
        #     for i in range(len(pipe.model.encoder.layer)):
        #         print(_i+i)
        #         pipe.model.encoder.layer[i].load_state_dict(
        #             torch.load(f'{args.model_name}/pytorch_{_i+i}.pt')
        #         )
        #     if hasattr(pipe.model.encoder, 'rel_embeddings'):
        #         pipe.model.encoder.rel_embeddings.load_state_dict(
        #             torch.load(f'{args.model_name}/pytorch_rel_embs.pt')
        #         )
        #     if hasattr(pipe.model.encoder, 'LayerNorm'):
        #         pipe.model.encoder.LayerNorm.load_state_dict(
        #             torch.load(f'{args.model_name}/pytorch_ln.pt')
        #         )
        #     if hasattr(pipe.model.encoder, 'conv') and pipe.model.encoder.conv is not None:
        #         raise Exception('should not have conv')
        #         pipe.model.encoder.conv.load_state_dict(
        #             torch.load(f'{args.model_name}/pytorch_conv.pt')
        #         )      


    train_loop(args, pipe, device, train_data_loader, test_data_loader)
    
    end_time = time.time()
    duration = end_time - start_time
    print(get_pipeline_parallel_rank(), 'finished.', "total time:%s" % format_time(duration), "invalid rate:%.2f%%" % (pipe.get_invalid_rate() * 100))

if __name__ == '__main__':
    main()
