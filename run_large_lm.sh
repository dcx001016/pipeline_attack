ARGS="--model-name checkpoints/gpt2-xl \
--tokenizer-name gpt2-xl \
--load-pretrained-model true \
--task-name wikitext --n-epochs 5 --warmup-epochs 1 \
--num-layers 24 --num-heads 25 --embedding-dim 1600 \
--num-iters 10000000 --lr 5e-5 --seq-length 1024 --batch-size 1 --micro-batch-size 1 \
--forward-compress-method none \
--forward-bits 4 \
--backward-compress-method none \
--backward-bits 8 \
--dist-url tcp://127.0.0.1:9033 \
--world-size 2 --pipeline-group-size 2 \
--pp-mode gpipe --profiling no-profiling --do-evaluation true"


(trap 'kill 0' SIGINT; \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    ; \
wait)
