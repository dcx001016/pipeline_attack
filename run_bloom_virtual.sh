ARGS="--model-name checkpoints/bigscience/bloom-560m \
--tokenizer-name checkpoints/bigscience/bloom-560m \
--load-pretrained-model true \
--task-name wikitext --n-epochs 5 --warmup-epochs 1 \
--num-layers 15 --num-heads 16 --embedding-dim 1024 \
--num-iters 100 --lr 5e-5 --seq-length 1024 --batch-size 4 --micro-batch-size 1 \
--optimizer AdamW \
--dist-url tcp://127.0.0.1:9034 \
--dropout false \
--world-size 2 --pipeline-group-size 2 --pipeline-virtual-gpus 6 \
--pp-mode gpipe-skip_layer --profiling no-profiling --do-evaluation true \
--forward-attack false --forward-attack-rate 0.5 \
--backward-attack false --backward-attack-rate 0.3 \
--do-valid false --use-center-server true \
--wandb false --write-xlsx true"

(trap 'kill 0' SIGINT; \
python dist_bloom_runner_virtual.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_bloom_runner_virtual.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    ; \
wait)
