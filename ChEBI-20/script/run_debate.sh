#!/bin/bash

cd ChEBI-20

python query_debate.py \
    --tgt_folder ./results/Mol-Debate_ChEBI-20/ \
    --process 8 \
    --model gpt-5-mini \
    --use_gpt \
    --max_new_tokens 16384 \
    --gen_model_list OpenDFM/ChemDFM-R-14B,weidawang/Chem-R-8B,OpenDFM/ChemDFM-v1.5-8B \
    --gen_port_list 8121,8120,8122 \
    --gen_host_list localhost,localhost,localhost \
    --gen_use_gpt_list 0,0,0 \
    --gen_seed_list 42,42,42 \
    --gen_max_new_tokens_list 4096,4096,1024 \
    --gen_num_generations_list 2,2,4 \
    --gen_temperature_list 1,1,1 \
    --gen_top_p_list 0.8,0.8,0.8 \
    --rounds 4 \
    --agents 2 \
    --task c2m \
    --use_examiner \
    --use_refine \
    --consensus_score_threshold 0.6


