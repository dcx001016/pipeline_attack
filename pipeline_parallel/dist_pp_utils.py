from .dist_gpipe_pipeline_async import GpipeAsync
from .dist_pipeline_async import VirtualAsync
from .dist_bamboo_pipeline_async import BambooVirtualAsync
from .dist_pipeline_async_key_value import VirtualKeyValueAsync
from .dist_pipeline_async_key_value_soft import VirtualKeyValueSoftAsync
from .dist_pipeline_async_hash import VirtualHashAsync
from .dist_pipeline_async_redundant import RedundantVirtualAsync
from .dist_pipeline_async_skip_layer import SkipLayerVirtualAsync
from modules.dist_opt_pp_module import *
from modules.dist_bloom_pp_module import *

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

def get_bloom_pp_module_virtual(args, config, device):
    if args.pp_mode == 'gpipe':
        return VirtualAsync(args, config, device,
                            _StageFirst=BloomStageFirst,
                            _StageLast=BloomStageLast,
                            _StageMiddle=BloomStageMiddle
                            )
    elif args.pp_mode == 'gpipe-kv':
        return VirtualKeyValueAsync(args, config, device,
                            _StageFirst=BloomStageFirst,
                            _StageLast=BloomStageLast,
                            _StageMiddle=BloomStageMiddle
                            )
    elif args.pp_mode == 'gpipe-hash':
        return VirtualHashAsync(args, config, device,
                            _StageFirst=BloomStageFirst,
                            _StageLast=BloomStageLast,
                            _StageMiddle=BloomStageMiddle
                            )
    elif args.pp_mode == 'gpipe-redundant':
        return RedundantVirtualAsync(args, config, device,
                            _StageFirst=BloomStageFirst,
                            _StageLast=BloomStageLast,
                            _StageMiddle=BloomStageMiddle
                            )
    elif args.pp_mode == 'gpipe-skip_layer':
        return SkipLayerVirtualAsync(args, config, device,
                            _StageFirst=BloomStageFirst,
                            _StageLast=BloomStageLast,
                            _StageMiddle=BloomStageMiddle
                            )
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False

def get_opt_pp_module_virtual(args, config, device):
    if args.pp_mode == 'gpipe':
        return VirtualAsync(args, config, device,
                            _StageFirst=OPTStageFirst,
                            _StageLast=OPTStageLast,
                            _StageMiddle=OPTStageMiddle
                            )
    elif args.pp_mode == 'gpipe-kv':
        return VirtualKeyValueAsync(args, config, device,
                            _StageFirst=OPTStageFirst,
                            _StageLast=OPTStageLast,
                            _StageMiddle=OPTStageMiddle
                            )
    elif args.pp_mode == 'gpipe-hash':
        return VirtualHashAsync(args, config, device,
                            _StageFirst=OPTStageFirst,
                            _StageLast=OPTStageLast,
                            _StageMiddle=OPTStageMiddle
                            )
    elif args.pp_mode == 'gpipe-redundant':
        return RedundantVirtualAsync(args, config, device,
                            _StageFirst=OPTStageFirst,
                            _StageLast=OPTStageLast,
                            _StageMiddle=OPTStageMiddle
                            )
    elif args.pp_mode == 'gpipe-skip_layer':
        return SkipLayerVirtualAsync(args, config, device,
                            _StageFirst=OPTStageFirst,
                            _StageLast=OPTStageLast,
                            _StageMiddle=OPTStageMiddle
                            )
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False