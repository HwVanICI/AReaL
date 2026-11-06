# Awex Experimental Test Scripts

This folder contains practical scripts for validating and benchmarking Awex weight
exchange between Megatron and vLLM.

## What each script is for

| Script                                   | Use case                                                                                                                                                  | Output                                                           |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `build_reduced_qwen3_moe.py`             | Build a smaller MoE checkpoint for local validation. Supports both slicing from local full weights and generating a config-only dummy checkpoint from HF. | A local HF checkpoint (`config.json` + `*.safetensors` + index). |
| `test_awex_megatron_vllm_integration.py` | End-to-end integration test: Megatron writes weights and vLLM receives/loads via Awex.                                                                    | `pytest` pass/fail.                                              |
| `bench_weight_transfer.py`               | Latency micro-benchmark for weight update path (`awex_nccl` / `awex_file` / `xccl`).                                                                      | JSON timing summary on stdout (optional `--out`).                |

## Prerequisites

- Run commands from repo root (`AReaL/`).
- Install runtime dependencies needed by your path:
  - `vllm`
  - `awex` (plugin path available)
  - CUDA or NPU runtime (NPU is experimental)
- For default single-rank runs, prepare at least 2 visible devices:
  - 1 for Megatron
  - 1 for vLLM

## 1) Build reduced checkpoints

### 1.1 Slice mode (from local full checkpoint)

Use when you already have full model weights and want realistic reduced weights.

```bash
python tests/experimental/awex/build_reduced_qwen3_moe.py \
  --input /home/model/Qwen3-30B-A3B-Instruct-2507 \
  --output /home/model/Qwen3-30B-A3B-Instruct-2507-reduced-l2-e8 \
  --num-layers 2 \
  --num-experts 8 \
  --num-experts-per-tok 2 \
  --force
```

### 1.2 Dummy mode (config-only from Hugging Face)

Use when you do not want to pre-download full weights.

```bash
python tests/experimental/awex/build_reduced_qwen3_moe.py \
  --hf-model Qwen/Qwen3-30B-A3B \
  --output /home/model/Qwen3-30B-A3B-dummy-reduced-l2-e8 \
  --num-layers 2 \
  --num-experts 8 \
  --num-experts-per-tok 2 \
  --dtype bfloat16 \
  --max-shard-size-gb 2 \
  --force
```

Common options:

- `--no-tokenizer`: skip tokenizer/processor download in dummy mode.
- `--seed`: control deterministic random init in dummy mode.
- `--no-trust-remote-code`: disable `trust_remote_code`.

## 2) Run integration test

Script: `tests/experimental/awex/test_awex_megatron_vllm_integration.py`

Dense path:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
AREAL_AWEX_DEVICE_BACKEND=cuda \
AREAL_AWEX_MODEL=dense \
AREAL_AWEX_DENSE_MODEL_PATH=/home/model/Qwen3-0.6B \
pytest tests/experimental/awex/test_awex_megatron_vllm_integration.py -k awex -v
```

MoE path:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
AREAL_AWEX_DEVICE_BACKEND=cuda \
AREAL_AWEX_MODEL=moe \
AREAL_AWEX_MOE_MODEL_PATH=/home/model/Qwen3-30B-A3B-Instruct-2507-reduced-l2-e8 \
pytest tests/experimental/awex/test_awex_megatron_vllm_integration.py -k awex -v
```

Helpful env vars:

- `AREAL_AWEX_MODEL`: `dense` or `moe`.
- `AREAL_AWEX_DENSE_MODEL_PATH`: local dense model path.
- `AREAL_AWEX_MOE_MODEL_PATH`: local reduced MoE path.
- `AREAL_AWEX_ALLOW_HF_DOWNLOAD=1`: allow dense model download fallback.
- `AREAL_AWEX_COMM_BACKEND`: `nccl` (default) or `file`.
- `AREAL_AWEX_DEVICE_BACKEND`: `cuda`, `npu`, `cpu`, `auto`.

## 3) Run weight update benchmark

Script: `tests/experimental/awex/bench_weight_transfer.py`

Dense benchmark:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
AREAL_AWEX_DENSE_MODEL_PATH=/home/model/Qwen3-0.6B \
python tests/experimental/awex/bench_weight_transfer.py \
  --device-backend cuda \
  --modes awex_nccl,awex_file,xccl \
  --iters 4 \
  --warmup 1 \
  --model-kind dense
```

MoE benchmark:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
AREAL_AWEX_MOE_MODEL_PATH=/home/model/Qwen3-30B-A3B-Instruct-2507-reduced-l2-e8 \
python tests/experimental/awex/bench_weight_transfer.py \
  --device-backend cuda \
  --modes awex_nccl,awex_file,xccl \
  --iters 4 \
  --warmup 1 \
  --model-kind moe
```

Optional:

- `--out /tmp/awex_bench.json` to save benchmark JSON.

## NPU note (experimental)

- Use `ASCEND_RT_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES`.
- Set `AREAL_AWEX_DEVICE_BACKEND=npu`.
- Prefer `AREAL_AWEX_COMM_BACKEND=hccl` for integration path.
- Awex NPU path typically uses `weights_exchange_ipc_backend=cpu`.

Online Ascend MXFP8 requires a Megatron actor and a local BF16 checkpoint with a
`W8A8_MXFP8` `quant_model_description.json`. The actor's original AWEX converter
produces logical matrices, which are quantized on the actor. AWEX transfers FP8 weight
bytes and uint8 E8M0 scales directly to the inference parameters; FLOAT parameters
retain their original dtype. There is no full BF16 receive cache.

Before weight transfer, actual AWEX metadata is checked for complete 32-element K
groups, aligned K offsets, and matching weight/scale shapes, coordinates, and dtypes.
Model shapes are not inferred from projection names in the configuration. The current
vllm-ascend MoE layout additionally requires local K to be divisible by 64. This FP8
path does not support `file` or `astate` transport and has no partial-group BF16
fallback. Run the four integration cases with:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
AREAL_AWEX_DEVICE_BACKEND=npu \
AREAL_AWEX_COMM_BACKEND=hccl \
AREAL_AWEX_DENSE_MODEL_PATH=/path/to/model \
pytest tests/experimental/awex/test_awex_megatron_vllm_integration.py \
  -k ascend-mxfp8 -v -s
```

The cases cover separate devices and shared devices, each in eager and ACL Graph mode.
Shared-device cases put vLLM on the training devices and enable both AWEX colocation and
vLLM sleep mode. Every inference device must have a corresponding training device; the
defaults use one training rank and one inference worker.

Each case performs at least three updates with different controlled matrix values,
checks the actual local FP8 weights and scales through a test-only worker RPC, and runs
inference after every update. Checks also verify inference parameter addresses across
sleep/wake and assert that the receive lifecycle completed. The test enables
`VLLM_SERVER_DEV_MODE` only in its server processes to expose the existing RPC endpoint.
It also disables weight NZ format and optimized model substitutions for the RL
reload/sleep lifecycle. Output includes update duration and each worker's peak allocated
and reserved NPU memory, read before allocating reference tensors for assertions. These
allocator peaks do not include all driver or communication-library allocations; use
device-level monitoring as well when sizing a colocation deployment.

The model fixtures cover Qwen3 and Qwen3 MoE. Direct FP8 transfer uses the existing AWEX
converters and does not impose a model-type whitelist. This path targets
`awex-ascend==0.8.2`; run the CPU metadata/lifecycle tests with:

```bash
pytest tests/test_awex_ascend_mxfp8.py -v
```

Native quantization tests skip without an NPU; converter and transfer-plan tests skip if
AWEX is unavailable. To exercise TP/ETP resharding in the integration cases, set
`AREAL_AWEX_TRAIN_TP_SIZE`, `AREAL_AWEX_TRAIN_ETP_SIZE`, `AREAL_AWEX_TRAIN_EP_SIZE`, and
`AREAL_AWEX_VLLM_TP_SIZE` for the available devices. Multi-card sleep/wake, ACL Graph,
and update memory peaks still require an actual Ascend run; CPU checks cannot establish
those hardware properties.
