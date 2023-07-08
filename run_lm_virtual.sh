ARGS="--model-name checkpoints/gpt2 \
--tokenizer-name gpt2 \
--load-pretrained-model true \
--task-name wikitext --n-epochs 5 --warmup-epochs 1 \
--num-layers 6 --num-heads 10 --embedding-dim 768 \
--num-iters 1000 --lr 5e-5 --seq-length 1024 --batch-size 4 --micro-batch-size 1 \
--optimizer AdamW \
--forward-compress-method none \
--forward-bits 4 \
--backward-compress-method none \
--backward-bits 8 \
--dist-url tcp://127.0.0.1:9033 \
--alpha 1.0 \
--world-size 2 --pipeline-group-size 2 --pipeline-virtual-gpus 4 \
--pp-mode gpipe-bamboo --profiling no-profiling --do-evaluation true \
--forward-attack false --forward-attack-rate 0.4 \
--backward-attack false --backward-attack-rate 0.3 \
--wandb false --write-xlsx true"

(trap 'kill 0' SIGINT; \
python dist_lm_runner_virtual.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_lm_runner_virtual.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    ; \
wait)
