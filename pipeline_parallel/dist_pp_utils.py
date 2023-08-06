from .dist_gpipe_pipeline_async import GpipeAsync
from .dist_pipeline_async import VirtualAsync
from .dist_bamboo_pipeline_async import BambooVirtualAsync
from .dist_pipeline_async_key_value import VirtualKeyValueAsync
from .dist_pipeline_async_key_value_soft import VirtualKeyValueSoftAsync
from .dist_pipeline_async_hash import VirtualHashAsync
from .dist_pipeline_async_redundant import RedundantVirtualAsync
from .dist_pipeline_async_skip_layer import SkipLayerVirtualAsync
from modules.dist_deberta_pp_module import *

def get_pp_module_virtual(args, config, device):
    if args.pp_mode == 'gpipe':
        return VirtualAsync(args, config, device)
    elif args.pp_mode == 'gpipe-bamboo':
        return BambooVirtualAsync(args, config, device)
    elif args.pp_mode == 'gpipe-kv':
        return VirtualKeyValueAsync(args, config, device)
    elif args.pp_mode == 'gpipe-kv-soft':
        return VirtualKeyValueSoftAsync(args, config, device)
    elif args.pp_mode == 'gpipe-hash':
        return VirtualHashAsync(args, config, device)
    elif args.pp_mode == 'gpipe-redundant':
        return RedundantVirtualAsync(args, config, device)
    elif args.pp_mode == 'gpipe-skip_layer':
        return SkipLayerVirtualAsync(args, config, device)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False

def get_pp_module(args, config, device, use_dp):
    if args.pp_mode == 'gpipe':
        return GpipeAsync(args, config, device, use_dp)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False

def get_deberta_pp_module_virtual(args, config, device):
    if args.pp_mode == 'gpipe':
        return VirtualAsync(
            args, config, device,
            _StageFirst=DebertaStageFirst,
            _StageLast=DebertaStageLast,
            _StageMiddle=DebertaStageMiddle
        )
    elif args.pp_mode == 'gpipe-bamboo':
        return BambooVirtualAsync(args, config, device,
            _StageFirst=DebertaStageFirst,
            _StageLast=DebertaStageLast,
            _StageMiddle=DebertaStageMiddle
        )
    elif args.pp_mode == 'gpipe-kv':
        return VirtualKeyValueAsync(args, config, device,
            _StageFirst=DebertaStageFirst,
            _StageLast=DebertaStageLast,
            _StageMiddle=DebertaStageMiddle
        )
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
