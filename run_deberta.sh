
# Important! Pre-trained weights need to be partioned before fine-tuning.
# python convert_deberta_checkpoint.py --model-name microsoft/deberta-v2-xxlarge --save-dir checkpoints

ARGS="--model-name ./checkpoints/microsoft/deberta-v3-base \
--tokenizer-name microsoft/deberta-v3-base \
--load-pretrained-model true --seed 42 \
--task-name qnli --n-epochs 5 --warmup-steps 50 --warmup-epochs 1 \
--num-layers 6 --num-heads 24 --embedding-dim 768 \
--num-iters 500 --lr 2.5e-6 --seq-length 512 --batch-size 16 --micro-batch-size 4 \
--optimizer SGD \
--forward-compress-method none \
--forward-bits 3 \
--backward-compress-method none \
--backward-bits 6 \
--dist-url tcp://127.0.0.1:9042 \
--world-size 2 --pipeline-group-size 2 \
--pp-mode gpipe --profiling no-profiling --do-evaluation true \
--forward-attack false --forward-attack-rate 0.3 \
--backward-attack false --backward-attack-rate 0.5"

(trap 'kill 0' SIGINT; \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    ; \
wait)
