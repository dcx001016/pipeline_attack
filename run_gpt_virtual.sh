ARGS="--model-name checkpoints/gpt2-xl \
--tokenizer-name gpt2-xl \
--load-pretrained-model false \
--task-name arxiv21 --n-epochs 5 --warmup-epochs 1 \
--embedding-dim 1600 \
--num-iters 2500 --lr 5e-5 --seq-length 1024 --batch-size 4 --micro-batch-size 1 \
--dist-url tcp://127.0.0.1:9034 \
--dropout false \
--world-size 2 --pipeline-group-size 2 --pipeline-virtual-gpus 6 \
--attack-type reverse \
--pp-mode gpipe-skip_layer --do-evaluation true \
--forward-attack true --forward-attack-rate 0.5 \
--backward-attack false --backward-attack-rate 0.7 \
--do-valid true --use-center-server true --restart false \
--wandb false --write-xlsx true"

(trap 'kill 0' SIGINT; \
python dist_gpt_runner_virtual.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_gpt_runner_virtual.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    ; \
wait)
