def add_device_arguments(parser):
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--cuda-num', type=int, default=1, metavar='N',
                        help='number of GPUs, if the instance has multiple GPUs.')
    parser.add_argument('--debug-mem', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, we will print some memory stats.')
    

def add_torch_distributed_arguments(parser):
    parser.add_argument('--dist-backend', type=str, default='cupy_nccl', metavar='S',
                        help='backend type for distributed PyTorch (default: cupy_nccl)')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:9000', metavar='S',
                        help='master ip for distributed PyTorch')
    parser.add_argument('--world-size', type=int, default=2, metavar='D',
                        help='world-size (default: 2)')
    parser.add_argument('--pipeline-group-size', type=int, default=2, metavar='D',
                        help='pipeline-group-size (default: 2)')
    parser.add_argument('--data-group-size', type=int, default=1, metavar='D',
                        help='data-group-size (default: 1)')
    parser.add_argument('--pipeline-virtual-gpus', type=int, default=6, metavar='D',
                        help='pipeline-virtual-gpus (default: 6)')
    parser.add_argument('--rank', type=int, default=0, metavar='N',
                        help='rank of the node')
    

def add_task_arguments(parser):
    parser.add_argument('--train-data', nargs='+', default=['./glue_dataset/data/QQP/train.tsv'], metavar='S',
                        help='path to the training data')
    parser.add_argument('--valid-data', nargs='+', default=['./glue_dataset/data/QQP/test.tsv'], metavar='S',
                        help='path to the training data')
    parser.add_argument('--tokenizer-type', type=str, default='BertWordPieceLowerCase', metavar='S',
                        help='which tokenizer to use.')
    parser.add_argument('--vocab-file', type=str, default='./glue_dataset/data/bert-large-cased-vocab.txt', metavar='S',
                        help='which tokenizer to use.')
    parser.add_argument('--vocab-extra-ids', type=int, default=0, metavar='N',
                        help='-')
    parser.add_argument('--make-vocab-size-divisible-by', type=int, default=128, metavar='N',
                        help='-')


def add_model_arguments(parser):
    parser.add_argument('--seq-length', type=int, default=1024, metavar='N',
                        help='-')
    parser.add_argument('--embedding-dim', type=int, default=1024, metavar='N',
                        help='-')
    parser.add_argument('--num-layers', type=int, default=12, metavar='N',
                        help='-')
    parser.add_argument('--num-heads', type=int, default=25, metavar='N',
                        help='-')
    

def add_training_hyper_parameter_arguments(parser):
    parser.add_argument('--optimizer', type=str, default="AdamW", metavar='N')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--micro-batch-size', type=int, default=1, metavar='N',
                        help='input micro batch size for training (default: 1)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='N',
                        help='-')
    parser.add_argument('--num-iters', type=int, default=10, metavar='N',
                        help='-')
    parser.add_argument('--alpha', type=float, default=1.0, metavar='N',
                        help='-')
    parser.add_argument('--dropout', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='-')
    parser.add_argument('--distance', type=str, default="l2", metavar='N',
                        help='-')

def add_acitvation_compression_arguments(parser):
    parser.add_argument('--forward-compress-method', type=str, default='none', metavar='N',
                        help='-')
    parser.add_argument('--forward-bits', type=int, default=4, metavar='N',
                        help='-')
    parser.add_argument('--forward-bits-act', type=int, default=4, metavar='N',
                        help='only for <delta-lowbits>')
    parser.add_argument('--forward-scale-method', type=str, default='max', metavar='N',
                        help='-')
    parser.add_argument('--forward-scale-dims', type=lambda x: tuple(int(i) for i in x.lstrip('(').rstrip(')').split(',') if i.strip() != ''), 
                        default=(0,1), metavar='N',
                        help='-')
    parser.add_argument('--max-activation-cache-size', type=int, default=20000, metavar='N',
                        help='max activation cache size. should be larger than the number of training examples.')
    parser.add_argument('--backward-compress-method', type=str, default='none', metavar='N',
                        help='-')
    parser.add_argument('--backward-bits', type=int, default=8, metavar='N',
                        help='-')
    parser.add_argument('--backward-scale-method', type=str, default='max', metavar='N',
                        help='-')
    parser.add_argument('--backward-scale-dims', type=lambda x: tuple(int(i) for i in x.lstrip('(').rstrip(')').split(',') if i.strip() != ''), 
                        default=(0,1), metavar='N',
                        help='-')
    parser.add_argument('--dp-bits', type=int, default=8, metavar='N',
                        help='-')

def add_mixed_precision_arguments(parser):
    parser.add_argument('--fp16', action='store_true',
                        help='Run model in fp16 mode.')
    parser.add_argument('--loss-scale', type=float, default=64,
                        help='Static loss scaling, positive power of 2 values can improve fp16 convergence. ')
    parser.add_argument('--initial-loss-scale', type=float, default=2 ** 32,
                        help='Initial loss-scale for dynamic loss scaling.')
    parser.add_argument('--min-loss-scale', type=float, default=1.0,
                        help='Minimum loss scale for dynamic loss scale.')
    parser.add_argument('--loss-scale-window', type=float, default=1000,
                        help='Window over which to raise/lower dynamic scale.')
    parser.add_argument('--hysteresis', type=int, default=2,
                        help='hysteresis for dynamic loss scaling')

    
def add_parallel_schema_arguments(parser):
    parser.add_argument('--pp-mode', type=str, default='gpipe-skip_layer', metavar='S',
                        help='use which pipeline parallel mode: gpipe or 1f1b.')
    parser.add_argument('--dp-mode', type=str, default='allreduce', metavar='S',
                        help='use which data parallel mode: allreduce.')
    parser.add_argument('--gradient-accumulate-step', type=int, default=1,
                        help='Number of gradient computation in Pipeline without data parallel sync.')
    
def add_attack_schema_arguments(parser):
    parser.add_argument('--forward-attack', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='forward attack or not.')
    parser.add_argument('--backward-attack', 
                        type=lambda x: x.lower()=='true', default=False, metavar='S',
                        help='backward attack or not.')
    parser.add_argument('--forward-attack-rate', type=float, default=0.5, metavar='N',
                        help='-')
    parser.add_argument('--backward-attack-rate', type=float, default=0, metavar='N',
                        help='-')
    parser.add_argument('--history-length', type=int, default=10, metavar='N',
                        help='-')
    parser.add_argument('--top-n', type=int, default=1, metavar='N',
                        help='-')
    parser.add_argument('--attack-type', type=str, default='reverse', metavar='N',
                        help='-')
    parser.add_argument('--do-valid', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='defense or not.')
    parser.add_argument('--use-center-server', 
                        type=lambda x: x.lower()=='true', default=True, metavar='S',
                        help='use center server or not.')
    parser.add_argument('--restart', 
                        type=lambda x: x.lower()=='true', default=False, metavar='S',
                        help='restart or not.')
    
def get_model_arguments_str(args):
    return '_l' + str(args.seq_length) + '_m' + str(args.embedding_dim)


def get_dist_arguments_str(args, add_rank=True):
    dist_str = '_w' + str(args.world_size) + '_p' + str(args.pipeline_group_size) + "_" + \
               str(args.gradient_accumulate_step) + '_d' + str(args.data_group_size)
    if add_rank:
        dist_str = dist_str + '_' + str(args.rank)
    return dist_str


def get_learning_arguments_str(args):
    return '_b' + str(args.batch_size) + '_' + str(args.micro_batch_size)


def get_mixed_precision_arguments_str(args):
    if args.fp16:
        return '_fp16'
    else:
        return ''
