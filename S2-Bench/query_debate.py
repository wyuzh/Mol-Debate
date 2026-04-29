import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import random
from utils.dataset import *
from utils.debate_utils import *
from prompts.agent import *

parser = argparse.ArgumentParser()


parser.add_argument("--benchmark", type=str, default="open_generation")
parser.add_argument("--task", type=str, default="MolCustom")
parser.add_argument("--subtask", type=str, default="AtomNum")
parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process from the dataset")
parser.add_argument("--output_dir", type=str, default="./predictions_debate/")
parser.add_argument("--model", type=str, default="gpt-5-mini")
parser.add_argument("--name", type=str, default="Mol-Debate")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--use_gpt", action="store_true", default=False)
parser.add_argument("--temperature", type=float, default=1)
parser.add_argument("--top_p", type=float, default=0.85)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gen_model_list", type=str, required=True)
parser.add_argument("--gen_port_list", type=str, default=None)
parser.add_argument("--gen_host_list", type=str, default="localhost")
parser.add_argument("--gen_use_gpt_list", type=str, default=None)
parser.add_argument("--gen_seed_list", type=str, default=None)
parser.add_argument("--gen_temperature_list", type=str, default=None)
parser.add_argument("--gen_top_p_list", type=str, default=None)
parser.add_argument("--gen_max_new_tokens_list", type=str, default=None)
parser.add_argument("--gen_num_generations_list", type=str, default=None)
parser.add_argument("--json_check", action="store_true", default=False)
parser.add_argument("--keep_system_prompt", action="store_true", default=False)
parser.add_argument("--use_examiner", action="store_true", default=False)
parser.add_argument("--ignore_desc_list", type=str, default=None)
parser.add_argument("--use_refine", action="store_true", default=False)
parser.add_argument("--consensus_score_threshold", type=float, default=0.6)
parser.add_argument("--agents", type=int, default=2)
parser.add_argument("--rounds", type=int, default=2)

args = parser.parse_args()

if "mistral" in args.model:
        args.mistral = True
else:
    args.mistral = False

# print parameters
print("========Parameters========")
for attr, value in args.__dict__.items():
    print("{}={}".format(attr.upper(), value))


# init gen agents
gen_client_list = []

gen_model_list = args.gen_model_list.split(",")
gen_port_list = args.gen_port_list.split(",")
gen_host_list = args.gen_host_list.split(",")
gen_use_gpt_list = args.gen_use_gpt_list.split(",")
gen_seed_list = args.gen_seed_list.split(",")
gen_temperature_list = args.gen_temperature_list.split(",")
gen_top_p_list = args.gen_top_p_list.split(",")
gen_max_new_tokens_list = args.gen_max_new_tokens_list.split(",")
gen_num_generations_list = args.gen_num_generations_list.split(",")


gen_client_list = get_gen_client_list(args)
client = get_debate_client(args, gen_client_list)


if args.ignore_desc_list is not None:
    ignore_desc_list = args.ignore_desc_list.split(",")
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] >>> Examiner ignore_desc_list: {ignore_desc_list} <<<')
else:
    ignore_desc_list = None

def _get_model_answers(model_idx, query):
    _messages = build_messages(gen_model_list[model_idx], query)
    init_gen_completion = get_chat_completion(
        client=gen_client_list[model_idx],
        model=gen_model_list[model_idx],
        messages=_messages,
        seed=gen_seed_list[model_idx],
        max_new_tokens=gen_max_new_tokens_list[model_idx],
        temperature=gen_temperature_list[model_idx],
        top_p=gen_top_p_list[model_idx],
        num_generations=gen_num_generations_list[model_idx]
    )
    
    results = []
    for j in range(len(init_gen_completion.choices)):
        init_gen_response_text = init_gen_completion.choices[j].message.content
        init_gen_response_think, init_gen_response_answer = extract_thinking_answer(init_gen_response_text)
        results.append((init_gen_response_think, init_gen_response_answer))
    
    return results


def get_answer_list(query, n_jobs=None):
    if n_jobs is None:
        n_jobs = len(gen_model_list)

    worker_func = partial(_get_model_answers, query=query)
    model_indices = range(len(gen_model_list))

    if n_jobs == 1:
        results = list(map(worker_func, model_indices))
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(worker_func, model_indices))

    answer_list = [item for sublist in results for item in sublist]
    return answer_list

def get_examiner_results_list(answer_list, ignore_desc_list=None):
    examiner_response_text_list = []
    for i in range(len(answer_list)):
        examiner_response_text = build_examiner_response(answer_list[i][1], ignore_desc_list)
        examiner_response_text_list.append(examiner_response_text)
        
    return examiner_response_text_list


def _get_agent_response(agent_idx, init_gen_response_text_list, query, examiner_response_text_list):
    gen_message = construct_message(init_gen_response_text_list, query, examiner_response_text_list)
    gen_completion = get_chat_completion(
        client=client,
        model=args.model,
        messages=[gen_message],
        seed=args.seed + agent_idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    gen_response_text = gen_completion.choices[0].message.content
    gen_think, gen_answer = extract_thinking_answer(gen_response_text)
    
    return {
        'agent_idx': agent_idx,
        'message': gen_message,
        'response_text': gen_response_text,
        'think': gen_think,
        'answer': gen_answer
    }


def get_agent_responses(init_gen_response_text_list, query, examiner_response_text_list, n_jobs=None):
    if n_jobs is None:
        n_jobs = args.agents
    
    worker_func = partial(_get_agent_response,
                        init_gen_response_text_list=init_gen_response_text_list,
                        query=query,
                        examiner_response_text_list=examiner_response_text_list)
    agent_indices = range(args.agents)
    
    if n_jobs == 1:
        results = list(map(worker_func, agent_indices))
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(worker_func, agent_indices))
    
    return results

# check out put dir
args.output_dir = args.output_dir + args.name + "/" + args.benchmark + "/" + args.task + "/"
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if os.path.exists(args.output_dir + args.subtask + ".csv"):
    temp = pd.read_csv(args.output_dir + args.subtask + ".csv")
    start_pos = len(temp)
else:
    with open(args.output_dir + args.subtask + ".csv", "w+") as f:
        f.write("outputs\n")
    start_pos = 0

print("========Inference Init========")
print("Inference starts from: ", start_pos)


# load dataset
if args.benchmark == "open_generation":
    inference_dataset = OMGDataset(args.task, args.subtask, args.json_check, keep_system_prompt=args.keep_system_prompt)
elif args.benchmark == "targeted_generation":
    inference_dataset = TMGDataset(args.task, args.subtask, args.json_check)

# Limit dataset to first N samples if specified
if args.max_samples is not None and args.max_samples > 0:
    original_length = len(inference_dataset)
    inference_dataset = [inference_dataset[i] for i in range(min(args.max_samples, original_length))]
    print(f"Dataset limited to {len(inference_dataset)} samples (original: {original_length})")

print("========Sanity Check========")
print(inference_dataset[0])
print("Total length of the dataset:", len(inference_dataset))
print("==============================")

error_records = []

all_agent_save_path = args.output_dir + args.subtask + f"_agent_contexts_{args.agents}_{args.rounds}.json"

# Load existing agent records if available
all_agent_records = {}
if os.path.exists(all_agent_save_path):
    try:
        with open(all_agent_save_path, 'r') as f:
            all_agent_records = json.load(f)
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] >>> Loaded {len(all_agent_records)} existing agent records from {all_agent_save_path} <<<')
    except Exception as e:
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] >>> Error loading agent records: {e}, starting fresh <<<')
        all_agent_records = {} 

total_len=len(inference_dataset)-start_pos
print(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] >>> len to process: {total_len} <<<')
with tqdm(total=total_len) as pbar:
    for idx in range(start_pos, len(inference_dataset)):
        messages = inference_dataset[idx]
        
        query = messages[1]["content"]
        
        generated_smiles = ""
        
        if idx not in all_agent_records:
            all_agent_records[idx] = {
                "debate":[]
            }
        
        # init gen
        try:
            
            init_gen_response_text_list = get_answer_list(query)
                
            pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] >>> init_gen_list: {[init_gen_response_text_list[i][1] for i in range(len(init_gen_response_text_list))]} <<<')
            
            if args.use_examiner:
                examiner_response_text_list = get_examiner_results_list(answer_list=init_gen_response_text_list, ignore_desc_list=ignore_desc_list)
            else:
                examiner_response_text_list = None

            # debate start
            try:
                round_pbar = tqdm(range(args.rounds), colour="blue", desc=f"Rounds", ncols=100, leave=False, position=1)
                for round_num in round_pbar:
                    
                    agent_records_round = {}
                    agent_records_round["round_num"] = round_num
                    agent_records_round["init_gen_response_text_list"] = init_gen_response_text_list
                    if args.use_examiner:
                        agent_records_round["examiner_response_text_list"] = examiner_response_text_list
                            
                    agent_results = get_agent_responses(init_gen_response_text_list, query, examiner_response_text_list)
                    
                    agent_response_text_list = []
                    for result in agent_results:
                        agent_idx = result['agent_idx']
                        agent_response_text_list.append((result['think'], result['answer']))
                        round_pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] >>> gen: {result["response_text"]} <<<')
                        
                        agent_records_round[f"gen_agent_{agent_idx}_message"] = result['message']
                        agent_records_round[f"gen_agent_{agent_idx}_response_text"] = result['response_text']
                        agent_records_round[f"gen_agent_{agent_idx}_answer"] = result['answer']
                    
                    try:
                        all_selection_indices = [extract_indices(agent_response[1]) for agent_response in agent_response_text_list]
                        gen_judge_selection_indices = list(set(all_selection_indices[0]))
                        for selection_indices in all_selection_indices[1:]:
                            gen_judge_selection_indices = list(set(gen_judge_selection_indices) & set(selection_indices))
                        
                        if len(gen_judge_selection_indices) == 0:
                            gen_judge_selection_indices = list(set().union(*[set(indices) for indices in all_selection_indices]))
                    except Exception as e:
                        pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] error: {e}')
                        gen_judge_selection_indices = []
                    agent_records_round["gen_judge_answer"] = gen_judge_selection_indices
                    
                    if len(gen_judge_selection_indices) > 0:
                        response1 = [init_gen_response_text_list[i-1] for i in gen_judge_selection_indices]
                        if args.use_examiner:
                            examiner_response_text_list = [examiner_response_text_list[i-1] for i in gen_judge_selection_indices]
                    else:
                        response1 = init_gen_response_text_list
                    
                    init_gen_response_text_list = response1 
                    agent_records_round["gen_final_answer"] = response1
                    round_pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] >>> gen_final: {gen_judge_selection_indices} <<<')
                    
                    
                    # Test on consensus score
                    consensus_score = get_consensus_score(selection_list=all_selection_indices)
                    if consensus_score >= args.consensus_score_threshold:
                        is_consensus_reached = True
                    else:
                        is_consensus_reached = False
                        
                    round_pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] >>> is_consensus_reached: {is_consensus_reached} <<<')
                    agent_records_round["consensus_score"] = consensus_score
                    agent_records_round["is_consensus_reached"] = is_consensus_reached
                    
                    
                    # Termination check
                    if is_consensus_reached == True and len(init_gen_response_text_list) == 1:
                        all_agent_records[idx]["debate"].append(agent_records_round)
                        break
                    elif is_consensus_reached == False and args.use_refine and round_num < args.rounds - 1:
                        refiner_message = construct_refiner_message(init_gen_response_text_list, agent_response_text_list, query, examiner_response_text_list)
                        refiner_completion = get_chat_completion(client = client, 
                                                                    model = args.model,
                                                                    messages = [refiner_message], 
                                                                    seed = args.seed, 
                                                                    max_new_tokens = args.max_new_tokens,
                                                                    temperature = args.temperature,
                                                                    top_p = args.top_p
                                                                    )
                        refiner_response_text = refiner_completion.choices[0].message.content
                        refiner_think, refiner_answer = extract_thinking_answer(refiner_response_text)
                        refined_query = refiner_answer.strip().replace('\n', '').replace('\r', '').replace('\t', '')
                        
                        round_pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] >>> refiner: {refiner_response_text} <<<')
                        agent_records_round["refiner_message"] = refiner_message
                        agent_records_round["refiner_response_text"] = refiner_response_text
                        agent_records_round["refined_query"] = refined_query
                        
                        refined_init_gen_response_text_list = get_answer_list(query=refined_query)
                        round_pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] >>> refined_gen_list: {[refined_init_gen_response_text_list[i][1] for i in range(len(refined_init_gen_response_text_list))]} <<<')
                        agent_records_round["refined_init_gen_response_text_list"] = refined_init_gen_response_text_list
                        init_gen_response_text_list.extend(refined_init_gen_response_text_list)
                        if args.use_examiner:
                            refined_examiner_response_text_list = get_examiner_results_list(answer_list=refined_init_gen_response_text_list, ignore_desc_list=ignore_desc_list)
                            examiner_response_text_list.extend(refined_examiner_response_text_list)
                            agent_records_round["refined_examiner_response_text_list"] = refined_examiner_response_text_list
                
                    all_agent_records[idx]["debate"].append(agent_records_round)
                    
                
                if len(response1) > 1:
                    response1 = random.choice(response1)[1]
                elif len(response1) == 1:
                    response1 = response1[0][1]
                else:
                    response1 = random.choice(init_gen_response_text_list)[1]
                
                pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] Gen: {response1}')
                response1 = response1.strip().replace('\n', '').replace('\r', '').replace('\t', '')

                generated_smiles = response1
                all_agent_records[idx]["gen_final_answer"] = response1
                
            except Exception as e:
                pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] error: {e}')
                response1 = random.choice(init_gen_response_text_list)[1]
                response1 = response1.strip().replace('\n', '').replace('\r', '').replace('\t', '')
                generated_smiles = response1
                all_agent_records[idx]["gen_final_answer"] = response1

        except Exception as e:
                    pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] error: {e}')
                    generated_smiles = "N/A"
                    all_agent_records[idx]["gen_final_answer"] = "N/A"
        

        df = pd.DataFrame([generated_smiles.strip()], columns=["outputs"])
        df.to_csv(args.output_dir +  args.subtask + ".csv", mode='a', header=False, index=True)
        
        # save all agent contexts
        with open(all_agent_save_path, "w") as f:
            json.dump(all_agent_records, f, indent=4)

        pbar.update(1)


print("========Inference Done========")
print("Error Records: ", error_records)