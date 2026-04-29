#!/bin/bash

cd S2-Bench

# MolEdit

python query_debate.py \
    --output_dir ./predictions_debate/ \
    --task MolEdit \
    --subtask AddComponent \
    --name Mol-Debate_S2-Bench \
    --model gpt-5-mini \
    --use_gpt \
    --max_new_tokens 16384 \
    --temperature 1 \
    --gen_model_list OpenDFM/ChemDFM-R-14B,weidawang/Chem-R-8B \
    --gen_port_list 8121,8120 \
    --gen_host_list localhost,localhost \
    --gen_use_gpt_list 0,0 \
    --gen_seed_list 42,42 \
    --gen_max_new_tokens_list 4096,4096 \
    --gen_num_generations_list 3,3 \
    --gen_temperature_list 1,1 \
    --gen_top_p_list 0.8,0.8 \
    --rounds 4 \
    --agents 2 \
    --use_examiner \
    --use_refine \
    --consensus_score_threshold 0.6


python query_debate.py \
    --output_dir ./predictions_debate/ \
    --task MolEdit \
    --subtask DelComponent \
    --name Mol-Debate_S2-Bench \
    --model gpt-5-mini \
    --use_gpt \
    --max_new_tokens 16384 \
    --temperature 1 \
    --gen_model_list OpenDFM/ChemDFM-R-14B,weidawang/Chem-R-8B \
    --gen_port_list 8121,8120 \
    --gen_host_list localhost,localhost \
    --gen_use_gpt_list 0,0 \
    --gen_seed_list 42,42 \
    --gen_max_new_tokens_list 4096,4096 \
    --gen_num_generations_list 3,3 \
    --gen_temperature_list 1,1 \
    --gen_top_p_list 0.8,0.8 \
    --rounds 4 \
    --agents 2 \
    --use_examiner \
    --use_refine \
    --consensus_score_threshold 0.6


python query_debate.py \
    --output_dir ./predictions_debate/ \
    --task MolEdit \
    --subtask SubComponent \
    --name Mol-Debate_S2-Bench \
    --model gpt-5-mini \
    --use_gpt \
    --max_new_tokens 16384 \
    --temperature 1 \
    --gen_model_list OpenDFM/ChemDFM-R-14B,weidawang/Chem-R-8B \
    --gen_port_list 8121,8120 \
    --gen_host_list localhost,localhost \
    --gen_use_gpt_list 0,0 \
    --gen_seed_list 42,42 \
    --gen_max_new_tokens_list 4096,4096 \
    --gen_num_generations_list 3,3 \
    --gen_temperature_list 1,1 \
    --gen_top_p_list 0.8,0.8 \
    --rounds 4 \
    --agents 2 \
    --use_examiner \
    --use_refine \
    --consensus_score_threshold 0.6


python query_debate.py \
    --output_dir ./predictions_debate/ \
    --task MolCustom \
    --subtask AtomNum \
    --name Mol-Debate_S2-Bench \
    --model gpt-5-mini \
    --use_gpt \
    --max_new_tokens 16384 \
    --temperature 1 \
    --gen_model_list OpenDFM/ChemDFM-R-14B,weidawang/Chem-R-8B \
    --gen_port_list 8121,8120 \
    --gen_host_list localhost,localhost \
    --gen_use_gpt_list 0,0 \
    --gen_seed_list 42,42 \
    --gen_max_new_tokens_list 4096,4096 \
    --gen_num_generations_list 3,3 \
    --gen_temperature_list 1,1 \
    --gen_top_p_list 0.8,0.8 \
    --rounds 4 \
    --agents 2 \
    --use_examiner \
    --use_refine \
    --consensus_score_threshold 0.6


python query_debate.py \
    --output_dir ./predictions_debate/ \
    --task MolCustom \
    --subtask BondNum \
    --name Mol-Debate_S2-Bench \
    --model gpt-5-mini \
    --use_gpt \
    --max_new_tokens 16384 \
    --temperature 1 \
    --gen_model_list OpenDFM/ChemDFM-R-14B,weidawang/Chem-R-8B \
    --gen_port_list 8121,8120 \
    --gen_host_list localhost,localhost \
    --gen_use_gpt_list 0,0 \
    --gen_seed_list 42,42 \
    --gen_max_new_tokens_list 1024,1024 \
    --gen_num_generations_list 3,3 \
    --gen_temperature_list 1,1 \
    --gen_top_p_list 0.8,0.8 \
    --rounds 4 \
    --agents 2 \
    --use_examiner \
    --use_refine \
    --consensus_score_threshold 0.6


python query_debate.py \
    --output_dir ./predictions_debate/ \
    --task MolCustom \
    --subtask FunctionalGroup \
    --name Mol-Debate_S2-Bench \
    --model gpt-5-mini \
    --use_gpt \
    --max_new_tokens 16384 \
    --temperature 1 \
    --gen_model_list OpenDFM/ChemDFM-R-14B,weidawang/Chem-R-8B \
    --gen_port_list 8121,8120 \
    --gen_host_list localhost,localhost \
    --gen_use_gpt_list 0,0 \
    --gen_seed_list 42,42 \
    --gen_max_new_tokens_list 1024,1024 \
    --gen_num_generations_list 3,3 \
    --gen_temperature_list 1,1 \
    --gen_top_p_list 0.8,0.8 \
    --rounds 4 \
    --agents 2 \
    --use_examiner \
    --use_refine \
    --consensus_score_threshold 0.6


# MolOpt
# #########################################################################################


python query_debate.py \
    --output_dir ./predictions_debate/ \
    --task MolOpt \
    --subtask LogP \
    --name Mol-Debate_S2-Bench \
    --model gpt-5-mini \
    --use_gpt \
    --max_new_tokens 16384 \
    --temperature 1 \
    --gen_model_list OpenDFM/ChemDFM-R-14B,weidawang/Chem-R-8B \
    --gen_port_list 8121,8120 \
    --gen_host_list localhost,localhost \
    --gen_use_gpt_list 0,0 \
    --gen_seed_list 42,42 \
    --gen_max_new_tokens_list 4096,4096 \
    --gen_num_generations_list 3,3 \
    --gen_temperature_list 1,1 \
    --gen_top_p_list 0.8,0.8 \
    --rounds 4 \
    --agents 2 \
    --use_examiner \
    --ignore_desc_list LogP \
    --use_refine \
    --consensus_score_threshold 0.6



python query_debate.py \
    --output_dir ./predictions_debate/ \
    --task MolOpt \
    --subtask MR \
    --name Mol-Debate_S2-Bench \
    --model gpt-5-mini \
    --use_gpt \
    --max_new_tokens 16384 \
    --temperature 1 \
    --gen_model_list OpenDFM/ChemDFM-R-14B,weidawang/Chem-R-8B \
    --gen_port_list 8121,8120 \
    --gen_host_list localhost,localhost \
    --gen_use_gpt_list 0,0 \
    --gen_seed_list 42,42 \
    --gen_max_new_tokens_list 4096,4096 \
    --gen_num_generations_list 3,3 \
    --gen_temperature_list 1,1 \
    --gen_top_p_list 0.8,0.8 \
    --rounds 4 \
    --agents 2 \
    --use_examiner \
    --ignore_desc_list MR \
    --use_refine \
    --consensus_score_threshold 0.6


python query_debate.py \
    --output_dir ./predictions_debate/ \
    --task MolOpt \
    --subtask QED \
    --name Mol-Debate_S2-Bench \
    --model gpt-5-mini \
    --use_gpt \
    --max_new_tokens 16384 \
    --temperature 1 \
    --gen_model_list OpenDFM/ChemDFM-R-14B,weidawang/Chem-R-8B \
    --gen_port_list 8121,8120 \
    --gen_host_list localhost,localhost \
    --gen_use_gpt_list 0,0 \
    --gen_seed_list 42,42 \
    --gen_max_new_tokens_list 4096,4096 \
    --gen_num_generations_list 3,3 \
    --gen_temperature_list 1,1 \
    --gen_top_p_list 0.8,0.8 \
    --rounds 4 \
    --agents 2 \
    --use_examiner \
    --ignore_desc_list QED \
    --use_refine \
    --consensus_score_threshold 0.6
