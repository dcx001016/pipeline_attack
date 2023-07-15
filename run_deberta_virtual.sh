
# Important! Pre-trained weights need to be partioned before fine-tuning.
# python convert_deberta_checkpoint.py --model-name microsoft/deberta-v2-xxlarge --save-dir checkpoints

ARGS="--model-name checkpoints/microsoft/deberta-v3-base \
--tokenizer-name microsoft/deberta-v3-base \
--load-pretrained-model true --seed 1 \
--task-name cola --n-epochs 5 --warmup-steps 50 --warmup-epochs 1 \
--num-layers 6 --num-heads 24 --embedding-dim 768 \
--num-iters 5000 --lr 2.5e-6 --seq-length 512 --batch-size 4 --micro-batch-size 1 \
--optimizer AdamW \
--forward-compress-method none \
--forward-bits 3 \
--backward-compress-method none \
--backward-bits 6 \
--dist-url tcp://127.0.0.1:9043 \
--dropout true \
--world-size 2 --pipeline-group-size 2 --pipeline-virtual-gpus 6 \
--history-length 100 --top-n 10 \
--pp-mode gpipe-kv --profiling no-profiling --do-evaluation true \
--forward-attack true --forward-attack-rate 0.4 \
--backward-attack false --backward-attack-rate 0.5 \
--wandb false --write-xlsx true"

(trap 'kill 0' SIGINT; \
python dist_deberta_runner_virtual.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_deberta_runner_virtual.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    ; \
wait)
