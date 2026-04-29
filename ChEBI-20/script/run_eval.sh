#!/bin/bash

cd ChEBI-20

python merge_transfer.py --file_path ./results/Mol-Debate_ChEBI-20/ --merge True --parts 8
python naive_test.py --pro_folder ./results/Mol-Debate_ChEBI-20/