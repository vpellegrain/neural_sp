#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=50g
#SBATCH --nodes=2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --output=/gpfs/workdir/pellegrainv/logs/%j.stdout
#SBATCH --error=/gpfs/workdir/pellegrainv/logs/%j.stderr
#SBATCH --job-name=transformer_asr

module load anaconda3/2020.02/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199
module load sox/14.4.2/gcc-9.2.0 
module load gcc/9.2.0/gcc-4.8.5 
module load gcc/8.4.0/gcc-4.8.5 
module load intel-mkl/2020.2.254/intel-20.0.2
module load flac/1.3.2/gcc-9.2.0



source activate speech3.8



# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                LibriSpeech                               "
echo ============================================================================

stage=4
stop_stage=4
gpu=1
benchmark=true
speed_perturb=true
stdout=false

### vocabulary
unit=wp      # word/wp/char/word_char/phone
vocab=1024
wp_type=bpe  # bpe/unigram (for wordpiece)

#########################
# ASR configuration
#########################
conf=conf/asr/transducer/lc_transformer_rnnt_64_64_32_bpe1024.yaml
conf2=conf/data/spec_augment_speed_perturb_transformer.yaml
asr_init=
external_lm=

#########################
# LM configuration
#########################
lm_conf=conf/lm/rnnlm.yaml

### path to save the model
model=/$WORKDIR/results/LibriSpeech

### path to the model directory to resume training
resume=
lm_resume=

### path to save preproecssed data
export data=/gpfs/workdir/pellegrainv/data/LibriSpeech/preprocessing
### path to download data
data_download_path=/gpfs/workdir/pellegrainv/data/LibriSpeech/download

### data size
datasize=960     # 100/460/960
lm_datasize=960  # 100/460/960
use_external_text=true

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ ${speed_perturb} = true ]; then
  if [ -z ${conf2} ]; then
    echo "Error: Set --conf2." 1>&2
    exit 1
  fi
fi

if [ -z ${gpu} ]; then
    echo "Error: set GPU number." 1>&2
    echo "Usage: ./run.sh --gpu 0" 1>&2
    exit 1
fi
n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)
if [ ${n_gpus} != 1 ]; then
    export OMP_NUM_THREADS=${n_gpus}
fi

# Base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

train_set=train_${datasize}
dev_set=dev_other
test_set="dev_clean dev_other test_clean test_other"
sp=
if [ ${speed_perturb} = true ]; then
    train_set=train_sp_${datasize}
    sp=_sp
    dev_set=dev_other_sp
    test_set="dev_clean_sp dev_other_sp test_clean_sp test_other_sp"
fi

if [ ${unit} = char ] || [ ${unit} = phone ]; then
    vocab=
fi
if [ ${unit} != wp ]; then
    wp_type=
fi

dict=${data}/dict/${train_set}_${unit}${wp_type}${vocab}.txt; mkdir -p ${data}/dict
wp_model=${data}/dict/${train_set}_${wp_type}${vocab}


mkdir -p ${model}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo ============================================================================
    echo "                        LM Training stage (stage:3)                       "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_3_${datasize}${lm_datasize}_${unit}${wp_type}${vocab}_${use_external_text} ]; then
        [ ! -e ${data}/.done_stage_1_${datasize}_sp${speed_perturb} ] && echo "run ./run.sh --datasize ${lm_datasize} first" && exit 1;

        echo "Making dataset tsv files for LM ..."
        mkdir -p ${data}/dataset_lm
        for x in train dev_clean dev_other test_clean test_other; do
            if [ ${lm_datasize} = ${datasize} ]; then
                cp ${data}/dataset/${x}${sp}_${datasize}_${unit}${wp_type}${vocab}.tsv \
                    ${data}/dataset_lm/${x}${sp}_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
            else
                make_dataset.sh --unit ${unit} --wp_model ${wp_model} \
                    ${data}/${x} ${dict} > ${data}/dataset_lm/${x}${sp}_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
            fi
        done

        # use external data
        if ${use_external_text}; then
            if [ ! -e ${data}/local/lm_train/librispeech-lm-norm.txt.gz ]; then
                wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P ${data}/local/lm_train/
            fi
            zcat ${data}/local/lm_train/librispeech-lm-norm.txt.gz | shuf | awk '{print "unpaired-text-"NR, tolower($0)}' > ${data}/dataset_lm/text
            update_dataset.sh --unit ${unit} --wp_model ${wp_model} \
                ${data}/dataset_lm/text ${dict} ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
                > ${data}/dataset_lm/train_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}_external.tsv || exit 1;
        fi

        touch ${data}/.done_stage_3_${datasize}${lm_datasize}_${unit}${wp_type}${vocab}_${use_external_text} && echo "Finish creating dataset for LM (stage: 3)."
    fi

    if ${use_external_text}; then
        lm_train_set="${data}/dataset_lm/train_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}_external.tsv"
    else
        lm_train_set="${data}/dataset_lm/train_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv"
    fi

    lm_test_set="${data}/dataset_lm/dev_other${sp}_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv \
                 ${data}/dataset_lm/test_clean${sp}_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv \
                 ${data}/dataset_lm/test_other${sp}_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv"
    

    PYTHON="/gpfs/users/pellegrainv/.conda/envs/speech3.8/bin/python"
    
    $PYTHON ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py \
        --corpus librispeech \
        --config ${lm_conf} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --train_set ${lm_train_set} \
        --dev_set ${data}/dataset_lm/dev_clean${sp}_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv \
        --eval_sets ${lm_test_set} \
        --unit ${unit} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model_save_dir ${model}/lm \
        --stdout ${stdout} \
        --resume ${lm_resume} || exit 1;

    echo "Finish LM training (stage: 3)."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    PYTHON="/gpfs/users/pellegrainv/.conda/envs/speech3.8/bin/python"

    $PYTHON ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py \
        --corpus librispeech \
        --config ${conf} \
        --config2 ${conf2} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${datasize}_${unit}${wp_type}${vocab}.tsv \
        --unit ${unit} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model_save_dir ${model}/asr \
        --asr_init ${asr_init} \
        --external_lm ${external_lm} \
        --stdout ${stdout} \
        --resume ${resume} || exit 1;

    echo "Finish ASR model training (stage: 4)."
fi
"