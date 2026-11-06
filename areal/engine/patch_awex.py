# SPDX-License-Identifier: Apache-2.0

from functools import wraps


def _patch_vllm_expert_names():
    from awex.converter.vllm_converter import VLLMToHFWeightConverter

    original = VLLMToHFWeightConverter._normalize_name
    if getattr(original, "_areal_expert_names", False):
        return

    @wraps(original)
    def normalize_name(self, name: str) -> str:
        # vLLM 0.26 nests fused expert parameters under RoutedExperts.
        name = name.replace(".experts.routed_experts.", ".experts.")
        return original(self, name)

    normalize_name._areal_expert_names = True
    VLLMToHFWeightConverter._normalize_name = normalize_name


def _patch_colocate_reader():
    from awex.reader.nccl_reader import NCCLWorkerWeightsReader, logger
    from awex.util import device as device_util

    from areal.engine.vllm_ext.awex_ascend_mxfp8 import (
        finish_awex_ascend_mxfp8_receive,
    )

    original_init = NCCLWorkerWeightsReader._init_reader_in_colocate_mode
    if getattr(original_init, "_areal_mxfp8", False):
        return
    original_shake_hands = NCCLWorkerWeightsReader._shake_hands_with_writer

    @wraps(original_shake_hands)
    def _shake_hands_with_writer(self):
        if (
            device_util.get_device_type() == "npu"
            and self.enable_colocate_mode
            and self.world_size == 1
        ):
            # AWEX 0.8.2 otherwise blocks receiving from itself before sending.
            logger.info("Skip AWEX readiness handshake for single-rank NPU colocation")
            return
        return original_shake_hands(self)

    @wraps(original_init)
    def _init_reader_in_colocate_mode(self):
        result = original_init(self)
        original_transfer = self.colocate_transport.update_weights_in_colocate_mode

        @wraps(original_transfer)
        def update_weights_in_colocate_mode(*args, **kwargs):
            result = original_transfer(*args, **kwargs)
            # The native reader publishes its ACK after this transport returns.
            finish_awex_ascend_mxfp8_receive(self)
            return result

        self.colocate_transport.update_weights_in_colocate_mode = (
            update_weights_in_colocate_mode
        )
        return result

    _init_reader_in_colocate_mode._areal_mxfp8 = True
    NCCLWorkerWeightsReader._init_reader_in_colocate_mode = (
        _init_reader_in_colocate_mode
    )
    NCCLWorkerWeightsReader._shake_hands_with_writer = _shake_hands_with_writer


def patch_awex():
    from areal.engine.vllm_ext.awex_ascend_mxfp8 import (
        install_awex_ascend_mxfp8_hooks,
    )

    install_awex_ascend_mxfp8_hooks()
    _patch_colocate_reader()
