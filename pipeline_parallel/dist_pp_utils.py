from .dist_gpipe_pipeline_async import GpipeAsync
from .dist_pipeline_async_virtual import VirtualAsync
from .dist_bamboo_pipeline_async_virtual import BambooVirtualAsync
from modules.dist_deberta_pp_module import *

def get_pp_module_virtual(args, config, device):
    if args.pp_mode == 'gpipe':
        return VirtualAsync(args, config, device)
    elif args.pp_mode == 'gpipe-bamboo':
        return BambooVirtualAsync(args, config, device)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False

def get_pp_module(args, config, device, use_dp):
    if args.pp_mode == 'gpipe':
        return GpipeAsync(args, config, device, use_dp)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False

def get_deberta_pp_module(args, config, device, use_dp):
    if args.pp_mode == 'gpipe':
        return GpipeAsync(
            args, config, device, use_dp,
            _StageFirst=DebertaStageFirst,
            _StageLast=DebertaStageLast,
            _StageMiddle=DebertaStageMiddle,
        )
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False
