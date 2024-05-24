# pipeline_attack

Welcome to the repository for the paper titled "Position: Exploring the Robustness of Pipeline-Parallelism-Based
Decentralized Training".

## Setup:

* Create environment:
  ``````shell
  conda create -n acsgd python=3.8
  conda activate acsgd
  ``````
* Install PyTorch env:
  ``````shell
  pip3 install -r requirements.txt
  ``````
## Run pipeline attack:

* Partition the pre-trained model:

  ``````shell
  # gpt
  python convert_gpt2_checkpoint.py --model-name gpt2-xl --save-dir checkpoints
  
  # or opt
  python convert_opt_checkpoint.py --model-name facebook/opt-1.3b --save-dir checkpoints
  
  # or bloom
  python convert_bloom_checkpoint.py --model-name bigscience/bloom-7b1 --save-dir checkpoints
  ``````

* On each node, run:

  ```
  # gpt
  python dist_gpt_runner_virtual.py $(echo ${ARGS}) --cuda-id 0 --rank i # (i=0,...,N-1)
  
  # or opt
  python dist_opt_runner_virtual.py $(echo ${ARGS}) --cuda-id 0 --rank i # (i=0,...,N-1)
  
  # or bloom
  python dist_bloom_runner_virtual.py $(echo ${ARGS}) --cuda-id 0 --rank i # (i=0,...,N-1)
  ```

  where "ARGS" contains training-related configurations, which should remain the same across nodes. An example could be:

  ```shell
  ARGS="--model-name checkpoints/gpt2-medium \
  --tokenizer-name gpt2-medium \
  --load-pretrained-model true \
  --task-name wikitext --n-epochs 5 --warmup-epochs 1 \
  --num-layers 12 --num-heads 16 --embedding-dim 1024 \
  --num-iters 10000 --lr 5e-5 --seq-length 1024 --batch-size 4 --micro-batch-size 1 \
  --dist-url tcp://127.0.0.1:9033 \
  --dropout false \
  --world-size 2 --pipeline-group-size 2 --pipeline-virtual-gpus 6 \
  --pp-mode gpipe-skip_layer --do-evaluation true \
  --forward-attack true --forward-attack-rate 0.5 \
  --do-valid false --use-center-server true \
  --write-xlsx true"
  ```

## Arguments

* `"--forward-attack"`: whether do attack during training.
* `"--forward-attack-rate"`: the probability of forward attack during training, if `"--forward-attack"`is "false", `forward_attack_rate=0`.
* `"--do-valid"`: whether do defense during training.
* `"--use-center-server"`: whether use central server during training.