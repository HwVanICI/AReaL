# Setup for Multi-LoRA

## Environmental Settings
`docker pull ghcr.io/hwvanici/areal_npu:v1.0.5-a3`

IMAGE=areal_npu:v1.0.5-a3

```bash
CONTAINER_NAME=my_container
CURRENT_HOSTNAME=$(hostname)
CURRENT_IP=$(hostname -i | awk '{print $1}')

MODEL_PATH=/path/to/model
DATA_PATH=/path/to/data

docker run \
  --privileged \
  --cap-add=SYS_PTRACE \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci8 \
  --device=/dev/davinci9 \
  --device=/dev/davinci10 \
  --device=/dev/davinci11 \
  --device=/dev/davinci12 \
  --device=/dev/davinci13 \
  --device=/dev/davinci14 \
  --device=/dev/davinci15 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  --shm-size=2048g \
  --net=host \
  --name "${CONTAINER_NAME}" \
  --hostname "${CURRENT_HOSTNAME}" \
  --add-host "${CURRENT_HOSTNAME}:${CURRENT_IP}" \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /var/log/npu:/usr/slog \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
  -v "${MODEL_PATH}:/models" \
  -v "${DATA_PATH}:/data" \
  -e http_proxy="${HTTP_PROXY:-}" \
  -e https_proxy="${HTTPS_PROXY:-}" \
  -e HTTP_PROXY="${HTTP_PROXY:-}" \
  -e HTTPS_PROXY="${HTTPS_PROXY:-}" \
  -e GIT_SSL_NO_VERIFY=1 \
  -u root \
  -itd "${IMAGE}" /bin/bash
```

Developing off of: `2262610b8c87abe355b98194afcb6d0846c57795`
AReaL Version: `v1.0.5`
vLLM Version: `0.23.0`
vLLM_Ascend Version: `0.23.0rc2.dev107+geaefc536d`

## Datasets
The data can be found here:
- AMC12: https://huggingface.co/datasets/edev2000/amc12-full
- GSM8k: https://huggingface.co/datasets/openai/gsm8k/tree/main
- Wiki: For repeatability, we used the validation set in https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/main/fullwiki

## 2-gsm8k for Qwen3-0.6B
**Async** run (each rollout -> train is asynchronous).

```bash
python examples/multi_lora/gsm8k.py \
  --config examples/multi_lora/_large_experiment_gsm8k.yaml \
  train_dataset.path=/path/to/dataset/here \
  valid_dataset.path=/path/to/dataset/here \
  actor.path=/path/to/model/here
```

## 3-task (gsm8k, amc12, wiki)
To run this example, you need to put the correct model and data paths into the following three files:
- _large_experiment_amc12.yaml
- _large_experiment_gsm8k.yaml
- _large_experiment_wiki.yaml

Then, you need to deploy the LLM-as-a-Judge service on another node

```bash
nohup vllm serve $MODEL_PATH \
  --host 0.0.0.0 \
  --port 7101 \
  --data-parallel-size $DP \
  --tensor-parallel-size $TP \
  --served-model-name judge
```

The URL for the judge (i.e., http://$IP_ADDRESS:7101/v1/completions) should be created and then passed as an environment variable when launching this example:
```bash
WIKI_JUDGE_URL=$JUDGE_URL python examples/multi_lora/total_exp_marlaas.py 
```
