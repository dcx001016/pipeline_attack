from .dist_pipeline_async_skip_layer import SkipLayerVirtualAsync
from modules.dist_opt_pp_module import *
from modules.dist_bloom_pp_module import *
from modules.dist_t5_pp_module import *

def get_pp_module_virtual(args, config, device):
    if args.pp_mode == 'gpipe-skip_layer':
        return SkipLayerVirtualAsync(args, config, device)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False

def get_bloom_pp_module_virtual(args, config, device):
    if args.pp_mode == 'gpipe-skip_layer':
        return SkipLayerVirtualAsync(args, config, device,
                            _StageFirst=BloomStageFirst,
                            _StageLast=BloomStageLast,
                            _StageMiddle=BloomStageMiddle
                            )
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False

def get_opt_pp_module_virtual(args, config, device):
    if args.pp_mode == 'gpipe-skip_layer':
        return SkipLayerVirtualAsync(args, config, device,
                            _StageFirst=OPTStageFirst,
                            _StageLast=OPTStageLast,
                            _StageMiddle=OPTStageMiddle
                            )
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False
        
def get_t5_pp_module_virtual(args, config, device):
    if args.pp_mode == 'gpipe-skip_layer':
        return SkipLayerVirtualAsync(args, config, device,
                            _StageFirst=T5StageFirst,
                            _StageLast=T5StageLast,
                            _StageMiddle=T5StageMiddle
                            )
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False