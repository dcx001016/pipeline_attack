ARGS="--model-name checkpoints/gpt2-medium \
--tokenizer-name gpt2-medium \
--load-pretrained-model true \
--task-name arxiv21 --n-epochs 5 --warmup-epochs 1 \
--num-layers 12 --num-heads 16 --embedding-dim 1024 \
--num-iters 1000000000000000 --lr 5e-5 --seq-length 1024 --batch-size 4 --micro-batch-size 1 \
--optimizer AdamW \
--forward-compress-method none \
--forward-bits 4 \
--backward-compress-method none \
--backward-bits 8 \
--dist-url tcp://127.0.0.1:9033 \
--alpha 1.0 --dropout false \
--world-size 2 --pipeline-group-size 2 --pipeline-virtual-gpus 6 \
--history-length 100 --top-n 11 \
--distance cos \
--pp-mode gpipe-skip_layer --profiling no-profiling --do-evaluation true \
--forward-attack true --forward-attack-rate 0.5 \
--backward-attack false --backward-attack-rate 0.3 \
--do-valid true --use-center-server true \
--wandb false --write-xlsx true"

(trap 'kill 0' SIGINT; \
python dist_gpt_runner_virtual.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_gpt_runner_virtual.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    ; \
wait)