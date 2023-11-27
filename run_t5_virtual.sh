ARGS="--model-name checkpoints/google/flan-t5-xl \
--tokenizer-name checkpoints/google/flan-t5-xl \
--load-pretrained-model true \
--task-name arxiv21 --n-epochs 5 --warmup-epochs 1 \
--num-layers 12 --num-heads 32 --embedding-dim 2048 \
--num-iters 1000 --lr 5e-5 --seq-length 512 --batch-size 4 --micro-batch-size 1 \
--optimizer AdamW \
--dist-url tcp://127.0.0.1:9034 \
--dropout false \
--world-size 2 --pipeline-group-size 2 --pipeline-virtual-gpus 6 \
--pp-mode gpipe-skip_layer --profiling no-profiling --do-evaluation true \
--forward-attack false --forward-attack-rate 0.5 \
--backward-attack false --backward-attack-rate 0.3 \
--do-valid false --use-center-server true \
--wandb false --write-xlsx false"

(trap 'kill 0' SIGINT; \
python dist_t5_runner_virtual.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_t5_runner_virtual.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    ; \
wait)
