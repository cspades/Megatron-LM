#!/bin/bash

export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export UB_TIMEOUT=720
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export TORCHINDUCTOR_WORKER_START=fork
#export NVTE_FUSED_ATTN=0  # Disable cuDNN fused attention.
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVTE_CPU_OFFLOAD_V1=1
export NVTE_USE_CUTLASS_GROUPED_GEMM=0
export NVTE_USE_FAST_MATH=1

export NCCL_SHM_DISABLE=1
export NCCL_PROTO=simple
export NCCL_NVLS_ENABLE=0
#export NCCL_SYM_GIN_KERNELS_ENABLE=1

export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=2
export USE_MNNVL=1

MEGATRON_LM_DIR="/opt/Megatron-LM"
PERF_OPT_DIR="${MEGATRON_LM_DIR}/xren_debug/perf"
OUTPUT_ROOT="${MEGATRON_LM_DIR}/xren_debug"
########################################################
#### CHANGES SHOULD NOT BE NEEDED BEYOND THIS POINT ####
########################################################

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
IFS=':' read -r -a array <<< "${SLURM_JOB_NAME}"
NAME="${array[1]}"

if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
    ENV_LOG_FILENAME=${NAME}_${SLURM_JOB_ID}_${DATETIME}.env.log
else
    SCRIPT_PATH=$(realpath "$0")
    ENV_LOG_FILENAME=${NAME}_${DATETIME}.env.log
fi

RUN_DIR="${OUTPUT_ROOT}/runs/${NAME}"
LOGS_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
DATACACHE_DIR="${OUTPUT_ROOT}/data-cache"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"

# Mamba triton cache.
#export TRITON_CACHE_DIR="${OUTPUT_ROOT}/triton-cache"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"/tmp/triton_cache_\${SLURM_NODEID}"}
#TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

################################################################
### Log environment
################################################################
echo "<< START PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "IMAGE=${IMAGE}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "OUTPUT_ROOT=${OUTPUT_ROOT}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "MEGATRON_LM_DIR=${MEGATRON_LM_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "RUN_DIR=${RUN_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "LOGS_DIR=${LOGS_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "DATACACHE_DIR=${DATACACHE_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END PATHS >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT LOG" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} log --oneline -1 |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT STATUS" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} status --porcelain --branch |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "GIT DIFF" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
git -C ${MEGATRON_LM_DIR} diff |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END GIT >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo -e "\n\n" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}

echo "<< START ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
env |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}
echo "<< END ENV >>" |& tee -a ${LOGS_DIR}/${ENV_LOG_FILENAME}


TOKENIZER_MODEL_PATH="${OUTPUT_ROOT}/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json"

# Switch from phase 1 to 2 at 60% (iteration 28K)
BLEND_PATH="/lustre/fsw/portfolios/llmservice/users/rwaleffe/blend_files/N5.5-phase1-FINAL-1-66th.json"
#BLEND_PATH="/lustre/fsw/portfolios/llmservice/users/rwaleffe/blend_files/N5.5-phase2-FINAL-1-66th.json"

# Copy scripts.
mkdir -p ${RUN_DIR}/scripts/data
#cp ${SCRIPT_PATH} ${RUN_DIR}/scripts
#cp ${BLEND_PATH} ${RUN_DIR}/scripts/data

SEQ_LEN=8192
TRAIN_SAMPLES=40000
LR_WARMUP_SAMPLES=2000
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES-LR_WARMUP_SAMPLES))
LR_WSD_DECAY_SAMPLES=40000

options=" \
        --moe-router-score-function sigmoid \
        --moe-grouped-gemm \
        --num-experts 16 \
        --moe-router-topk 8 \
        --moe-aux-loss-coeff 1e-4 \
        --moe-router-topk-scaling-factor 2.5 \
        --moe-router-enable-expert-bias \
        --moe-router-dtype fp32 \
        --moe-router-load-balancing-type seq_aux_loss \
        --moe-shared-expert-intermediate-size 10240 \
        --moe-latent-size 2048 \
        --moe-permute-fusion \
        --moe-token-dispatcher-type flex \
        --moe-flex-dispatcher-backend hybridep \
        --moe-hybridep-num-sms 32 \
        --moe-router-force-load-balancing \
        \
        --num-workers 1 \
        --disable-gloo-process-groups \
        --ckpt-format torch_dist \
        --ckpt-fully-parallel-save \
        --ckpt-fully-parallel-load \
        --ckpt-assume-constant-structure \
        \
        --squared-relu \
        --no-mmap-bin-files \
        --distributed-timeout-minutes 30 \
        --exit-duration-in-mins 1430 \
        --no-create-attention-mask-in-dataloader \
        \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --tensor-model-parallel-size 1 \
        --expert-model-parallel-size 2 \
        --expert-tensor-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --use-distributed-optimizer \
        --high-priority-stream-groups ep \
        --ddp-num-buckets 5 \
        --grad-reduce-in-bf16 \
        \
        --mock-data \
        --is-hybrid-model \
        --untie-embeddings-and-output-weights \
        --init-method-std 0.0099 \
        --position-embedding-type none \
        --num-layers 11 \
        --hidden-size 8192 \
        --num-attention-heads 64 \
        --group-query-attention \
        --num-query-groups 8 \
        --hybrid-override-pattern MEMEMEMEM*E/*E \
        --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
        --ffn-hidden-size 5120 \
        --kv-channels 128 \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-samples ${TRAIN_SAMPLES} \
        --lr-decay-style WSD \
        --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
        --lr-decay-samples ${LR_DECAY_SAMPLES} \
        --lr-wsd-decay-style minus_sqrt \
        --lr-wsd-decay-samples ${LR_WSD_DECAY_SAMPLES} \
        --data-cache-path ${DATACACHE_DIR} \
        --tiktoken-pattern v2 \
        --tokenizer-type TikTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL_PATH} \
        --distributed-backend nccl \
        --micro-batch-size 1 \
        --global-batch-size 4 \
        --lr 8.0e-4 \
        --min-lr 8.0e-6 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --disable-bias-linear \
        --normalization RMSNorm \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --log-interval 1 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --log-throughput \
        --log-device-memory-used \
        --eval-interval 5 \
        --eval-iters 14 \
        --bf16 \
        --use-mcore-models \
        --enable-experimental \
        --manual-gc-interval 10 \
        --use-fused-weighted-squared-relu \
        --cross-entropy-loss-fusion \
        --cross-entropy-fusion-impl native \
        --enable-cuda-graph \
        --cuda-graph-scope mamba attn moe_router \
        --te-rng-tracker \
        --exit-interval 5"
        #--cuda-graph-scope mamba attn moe_router
        #--per-split-data-args-path ${BLEND_PATH} \
        #--save ${CHECKPOINT_DIR} \
        #--load ${CHECKPOINT_DIR} \
        #--save-interval 2000 \
        #--tensorboard-dir ${TENSORBOARD_DIR}"
        #--moe-shared-expert-overlap \
        #--moe-shared-expert-compute-before-router \
        #--ddp-reduce-scatter-with-fp32-accumulation \

mxfp8_options=" \
    --moe-router-padding-for-quantization \
    --fp8-format e4m3 \
    --fp8-recipe mxfp8 \
    --fp8-param-gather"

nvfp4_options=" \
    --moe-router-padding-for-quantization \
    --te-precision-config-file ${PERF_OPT_DIR}/megatron/nemotron/nemotron6/job_launch/hybrid/te_quant.cfg \
    --first-last-layers-bf16 \
    --num-layers-at-start-in-bf16 0 \
    --num-layers-at-end-in-bf16 3 \
    --fp4-recipe nvfp4 \
    --fp4-format e2m1"
    #--fp4-param-gather \

mtp_options=" \
    --mtp-num-layers 2 \
    --mtp-use-repeated-layer \
    --calculate-per-token-loss \
    --mtp-loss-scaling-factor 0.3"

fsdp_options=" \
    --use-megatron-fsdp \
    --data-parallel-sharding-strategy optim_grads_params \
    --no-gradient-accumulation-fusion \
    --ckpt-format fsdp_dtensor \
    --megatron-fsdp-grad-comm-dtype bf16 \
    --megatron-fsdp-main-params-dtype fp32 \
    --megatron-fsdp-main-grads-dtype bf16 \
    --fsdp-double-buffer \
    --megatron-fsdp-max-pool-double-buffer"
    #--use-nccl-ub \
    #--use-sharp \
    #--fsdp-db-use-persist-buf-on-alloc-fail \

profile_options=" \
    --profile \
    --profile-step-start 3 \
    --profile-step-end 4 \
    --profile-ranks 0 \
    --record-memory-history \
    --memory-snapshot-path ${RUN_DIR}/${NAME}_node${SLURM_NODEID}_rank${SLURM_PROCID}.pickle"

#nsys_cmd="nsys profile -s none -t nvtx,cuda-sw -o ${RUN_DIR}/${NAME}_node${SLURM_NODEID}_rank${SLURM_PROCID} --force-overwrite true --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop"
run_cmd="torchrun --nproc_per_node 4 --nnodes 1 --log-dir ${LOGS_DIR}/${NAME}.log ${MEGATRON_LM_DIR}/pretrain_mamba.py ${options} ${mxfp8_options} ${mtp_options} ${fsdp_options} ${profile_options}"

sh -c "${run_cmd}"
