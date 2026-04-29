#!/bin/bash

cd S2-Bench

python evaluate.py --name Mol-Debate_S2-Bench --benchmark open_generation --output_dir ./predictions_debate/  --task MolEdit --subtask AddComponent

python evaluate.py --name Mol-Debate_S2-Bench --benchmark open_generation --output_dir ./predictions_debate/  --task MolEdit --subtask DelComponent

python evaluate.py --name Mol-Debate_S2-Bench --benchmark open_generation --output_dir ./predictions_debate/  --task MolEdit --subtask SubComponent






python evaluate.py --name Mol-Debate_S2-Bench --benchmark open_generation --output_dir ./predictions_debate/  --task MolCustom --subtask AtomNum --calc_novelty

python evaluate.py --name Mol-Debate_S2-Bench --benchmark open_generation --output_dir ./predictions_debate/  --task MolCustom --subtask BondNum --calc_novelty

python evaluate.py --name Mol-Debate_S2-Bench --benchmark open_generation --output_dir ./predictions_debate/ --task MolCustom --subtask FunctionalGroup --calc_novelty






python evaluate.py --name Mol-Debate_S2-Bench --benchmark open_generation --output_dir ./predictions_debate/  --task MolOpt --subtask LogP

python evaluate.py --name Mol-Debate_S2-Bench --benchmark open_generation --output_dir ./predictions_debate/  --task MolOpt --subtask MR

python evaluate.py --name Mol-Debate_S2-Bench --benchmark open_generation --output_dir ./predictions_debate/  --task MolOpt --subtask QED

