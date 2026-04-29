import json
import os
import random
import argparse
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import partial
from utils import *
from prompts.agent import *

if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="./cap2mol_trans_raw/")
    parser.add_argument("--tgt_folder", type=str, default="./results/new_results/")
    parser.add_argument("--process", type=int, default=8)
    parser.add_argument("--file", type=str, default="test.txt")
    parser.add_argument("--task", type=str, choices=["all", "m2c", "c2m"], default="all")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--use_gpt", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--use_examiner", action="store_true", default=False)
    parser.add_argument("--use_refine", action="store_true", default=False)
    parser.add_argument("--consensus_score_threshold", type=float, default=0.6)
    parser.add_argument("--gen_model_list", type=str, required=True)
    parser.add_argument("--gen_port_list", type=str, default=None)
    parser.add_argument("--gen_host_list", type=str, default="localhost")
    parser.add_argument("--gen_use_gpt_list", type=str, default=None)
    parser.add_argument("--gen_seed_list", type=str, default=None)
    parser.add_argument("--gen_temperature_list", type=str, default=None)
    parser.add_argument("--gen_top_p_list", type=str, default=None)
    parser.add_argument("--gen_max_new_tokens_list", type=str, default=None)
    parser.add_argument("--gen_num_generations_list", type=str, default=None)
    args = parser.parse_args()

    print("========Parameters========")
    for attr, value in args.__dict__.items():
        print("{}={}".format(attr.upper(), value))
    
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

    data_folder = args.data_folder
    tgt_folder = args.tgt_folder

    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)

    process = args.process

    file = args.file
    raw_file = data_folder + file
    
    with open(raw_file, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]

    def _get_model_answers(model_idx, question):
        """Worker function to get answers from a single model (parallelizable)"""
        model_name = gen_model_list[model_idx]
        _messages = build_c2m_messages(model_name, caption=question)
        
        if 'ChemDFM-v1.5' in model_name:
            completion = get_completion(
                client=gen_client_list[model_idx],
                model=model_name,
                messages=_messages,
                seed=gen_seed_list[model_idx],
                max_new_tokens=gen_max_new_tokens_list[model_idx],
                temperature=gen_temperature_list[model_idx],
                top_p=gen_top_p_list[model_idx],
                num_generations=gen_num_generations_list[model_idx]
            )
            results = []
            for j in range(len(completion.choices)):
                text = completion.choices[j].text.strip().replace('\n', '').replace('\r', '').replace('\t', '')
                results.append(("N/A", text))
            return results
        else:
            completion = get_chat_completion(
                client=gen_client_list[model_idx],
                model=model_name,
                messages=_messages,
                seed=gen_seed_list[model_idx],
                max_new_tokens=gen_max_new_tokens_list[model_idx],
                temperature=gen_temperature_list[model_idx],
                top_p=gen_top_p_list[model_idx],
                num_generations=gen_num_generations_list[model_idx]
            )
            results = []
            for j in range(len(completion.choices)):
                text = completion.choices[j].message.content
                think, answer = extract_thinking_answer(text)
                results.append((think, answer))
            return results
    
    
    def get_answer_list(question, n_jobs=None):
        if n_jobs is None:
            n_jobs = len(gen_model_list)

        worker_func = partial(_get_model_answers, question=question)
        model_indices = range(len(gen_model_list))

        if n_jobs == 1:
            results = list(map(worker_func, model_indices))
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(worker_func, model_indices))

        answer_list = [item for sublist in results for item in sublist]
        return answer_list
    
    def get_examiner_results_list(answer_list):
        examiner_response_text_list = []
        for i in range(len(answer_list)):
            examiner_response_text = build_examiner_response(answer_list[i][1])
            examiner_response_text_list.append(examiner_response_text)
            
        return examiner_response_text_list
    
    
    def _get_agent_response(agent_idx, init_c2m_response_text_list, caption, examiner_response_text_list):
        c2m_proponent_message = c2m_construct_debate_message(init_c2m_response_text_list, caption, examiner_response_text_list)
        c2m_proponent_completion = get_chat_completion(
            client=client,
            model=args.model,
            messages=[c2m_proponent_message],
            seed=args.seed + agent_idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        c2m_proponent_response_text = c2m_proponent_completion.choices[0].message.content
        c2m_proponent_think, c2m_proponent_answer = extract_thinking_answer(c2m_proponent_response_text)
        
        return {
            'agent_idx': agent_idx,
            'message': c2m_proponent_message,
            'response_text': c2m_proponent_response_text,
            'think': c2m_proponent_think,
            'answer': c2m_proponent_answer
        }
    
    
    def get_agent_responses(init_c2m_response_text_list, caption, examiner_response_text_list, n_jobs=None):
        if n_jobs is None:
            n_jobs = args.agents
        
        worker_func = partial(_get_agent_response, 
                            init_c2m_response_text_list=init_c2m_response_text_list,
                            caption=caption,
                            examiner_response_text_list=examiner_response_text_list)
        agent_indices = range(args.agents)
        
        if n_jobs == 1:
            results = list(map(worker_func, agent_indices))
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(worker_func, agent_indices))
        
        return results

    def run(n):
        name = mp.current_process().name
        print('Process', name, 'starting')

        tgt_file = tgt_folder + file.split(".")[0] + "_Part{}.txt".format(n)
        full_tgt = tgt_folder + file.split(".")[0] + "_Full_Part{}.txt".format(n)
        agent_context_file = tgt_folder + file.split(".")[0] + f"Agent_contexts_Round{args.rounds}_Part{n}.json"


        exist_caps_items = []
        exist_caps = []
        exist_mols_items = []
        exist_mols = []
        if os.path.exists(tgt_file):
            with open(tgt_file, 'r') as f:
                tgt_lines = f.readlines()

            for item in tgt_lines:
                item = item.strip().strip('\n').strip()
                if item.split('\t')[1] != "N/A":
                    exist_caps_items.append(item.split('\t')[0])
                    exist_caps.append(item.split('\t')[1])
                
                if item.split('\t')[2] != "N/A":
                    exist_mols_items.append(item.split('\t')[0])
                    exist_mols.append(item.split('\t')[2])

        all_items = {}

        all_agent_records = {}
        if os.path.exists(agent_context_file):
            try:
                with open(agent_context_file, 'r') as f:
                    all_agent_records = json.load(f)
                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, Part {n}] >>> Loaded {len(all_agent_records)} existing agent records <<<')
            except Exception as e:
                print(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, Part {n}] >>> Error loading agent records: {e}, starting fresh <<<')
                all_agent_records = {}
        
        lines_to_process = []
        all_lines = lines[int(len(lines)*(n-1)/args.process):int(len(lines)*n/args.process)]
        for line in all_lines:
            line = line.strip()
            if line == "":
                continue

            ids = line.split("\t")[0]
            caption = line.split("\t")[2]

            if ids in exist_mols_items:
                all_items[ids] = {'caption': "N/A", 'molecule': exist_mols[exist_mols_items.index(ids)]}
                continue
            else:
                all_items[ids] = {'caption': "N/A", 'molecule': "N/A"} 
            
            lines_to_process.append(line)
        
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, Part {n}] >>> {len(lines_to_process)} lines to process <<<')
        
        lin_pbar = tqdm(lines_to_process, colour="green", desc=f"Items (Part {n})", ncols=100, leave=False, position=0)
        
        for line in lin_pbar:
            line = line.strip()
            if line == "":
                continue

            ids = line.split("\t")[0]
            caption = line.split("\t")[2]            
                
            if ids not in all_agent_records:
                all_agent_records[ids] = {
                    "m2c":[],
                    "c2m":[]
                }

            # ################################################################################################################################


            # Caption to Molecule
            if all_items[ids]['molecule'] == "N/A" and (args.task == "all" or args.task == "c2m"):
                try:
                    init_c2m_response_text_list = get_answer_list(question=caption)
                        
                    lin_pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, Part {n}] >>> init_c2m_list: {[init_c2m_response_text_list[i][1] for i in range(len(init_c2m_response_text_list))]} <<<')

                    if args.use_examiner:
                        examiner_response_text_list = get_examiner_results_list(answer_list=init_c2m_response_text_list)
                    else:
                        examiner_response_text_list = None

                    try:
                        round_pbar = tqdm(range(args.rounds), colour="blue", desc=f"Rounds (Part {n})", ncols=100, leave=False, position=1)
                        for round_num in round_pbar:
                            
                            agent_records_round = {}
                            agent_records_round["round_num"] = round_num
                            agent_records_round["init_c2m_response_text_list"] = init_c2m_response_text_list
                            if args.use_examiner:
                                agent_records_round["examiner_response_text_list"] = examiner_response_text_list
                            
                            agent_results = get_agent_responses(init_c2m_response_text_list, caption, examiner_response_text_list)
                            
                            agent_response_text_list = []
                            for result in agent_results:
                                agent_idx = result['agent_idx']
                                agent_response_text_list.append((result['think'], result['answer']))
                                round_pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, Part {n}] >>> c2m_proponent: {result["response_text"]} <<<')
                                
                                agent_records_round[f"c2m_agent_{agent_idx}_message"] = result['message']
                                agent_records_round[f"c2m_agent_{agent_idx}_response_text"] = result['response_text']
                                agent_records_round[f"c2m_agent_{agent_idx}_answer"] = result['answer']

                            try:
                                all_selection_indices = [extract_indices(agent_response[1]) for agent_response in agent_response_text_list]
                                
                                c2m_judge_selection_indices = list(set(all_selection_indices[0]))
                                for selection_indices in all_selection_indices[1:]:
                                    c2m_judge_selection_indices = list(set(c2m_judge_selection_indices) & set(selection_indices))

                                if len(c2m_judge_selection_indices) == 0:
                                    c2m_judge_selection_indices = list(set().union(*[set(indices) for indices in all_selection_indices]))
                                    
                            except Exception as e:
                                lin_pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, Part {n}] error: {e}')
                                c2m_judge_selection_indices = []
                            agent_records_round["c2m_judge_answer"] = c2m_judge_selection_indices
                            
                            if len(c2m_judge_selection_indices) > 0:
                                response2 = [init_c2m_response_text_list[i-1] for i in c2m_judge_selection_indices]
                                if args.use_examiner:
                                    examiner_response_text_list = [examiner_response_text_list[i-1] for i in c2m_judge_selection_indices]
                            else:
                                response2 = init_c2m_response_text_list
                                
                            init_c2m_response_text_list = response2 
                            agent_records_round["c2m_final_answer"] = response2
                            round_pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, Part {n}] >>> c2m_final: {c2m_judge_selection_indices} <<<')
                            
                            if len(response2) > 1:
                                round_response2 = random.choice(response2)[1]
                            elif len(response2) == 1:
                                round_response2 = response2[0][1]
                            else:
                                round_response2 = random.choice(init_c2m_response_text_list)[1]
                            round_response2 = round_response2.strip().replace('\n', '').replace('\r', '').replace('\t', '')
                            agent_records_round["c2m_final_answer_round"] = round_response2
                            
                            consensus_score = get_consensus_score(selection_list=all_selection_indices)
                            if consensus_score >= args.consensus_score_threshold:
                                is_consensus_reached = True
                            else:
                                is_consensus_reached = False
                                
                            lin_pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, Part {n}] >>> is_consensus_reached: {is_consensus_reached} <<<')
                            agent_records_round["consensus_score"] = consensus_score
                            agent_records_round["is_consensus_reached"] = is_consensus_reached
                            
                            if is_consensus_reached == True and len(init_c2m_response_text_list) == 1:
                                all_agent_records[ids]["c2m"].append(agent_records_round)
                                break 
                            elif is_consensus_reached == False and args.use_refine and round_num < args.rounds - 1:
                                c2m_refiner_message = c2m_construct_refiner_message(init_c2m_response_text_list, agent_response_text_list, caption, examiner_response_text_list)
                                c2m_refiner_completion = get_chat_completion(client = client, 
                                                                            model = args.model,
                                                                            messages = [c2m_refiner_message], 
                                                                            seed = args.seed, 
                                                                            max_new_tokens = args.max_new_tokens,
                                                                            temperature = args.temperature,
                                                                            top_p = args.top_p
                                                                            )
                                c2m_refiner_response_text = c2m_refiner_completion.choices[0].message.content
                                c2m_refiner_think, c2m_refiner_answer = extract_thinking_answer(c2m_refiner_response_text)
                                refined_caption = c2m_refiner_answer.strip().replace('\n', '').replace('\r', '').replace('\t', '')
                                
                                lin_pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, Part {n}] >>> c2m_refiner: {c2m_refiner_response_text} <<<')
                                agent_records_round["c2m_refiner_message"] = c2m_refiner_message
                                agent_records_round["c2m_refiner_response_text"] = c2m_refiner_response_text
                                agent_records_round["c2m_refined_caption"] = refined_caption
                                
                                refined_init_c2m_response_text_list = get_answer_list(question=refined_caption)
                                lin_pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, Part {n}] >>> refined_c2m_list: {[refined_init_c2m_response_text_list[i][1] for i in range(len(refined_init_c2m_response_text_list))]} <<<')
                                agent_records_round["refined_init_c2m_response_text_list"] = refined_init_c2m_response_text_list
                                init_c2m_response_text_list.extend(refined_init_c2m_response_text_list)
                                if args.use_examiner:
                                    refined_examiner_response_text_list = get_examiner_results_list(answer_list=refined_init_c2m_response_text_list)
                                    examiner_response_text_list.extend(refined_examiner_response_text_list)
                                    agent_records_round["refined_examiner_response_text_list"] = refined_examiner_response_text_list
                                
                            all_agent_records[ids]["c2m"].append(agent_records_round)

                        if len(response2) > 1:
                            response2 = random.choice(response2)[1]
                        elif len(response2) == 1:
                            response2 = response2[0][1]
                        else:
                            response2 = random.choice(init_c2m_response_text_list)[1]
                        
                        lin_pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, Part {n}] Cap2Mol: {response2}')
                        response2 = response2.strip().replace('\n', '').replace('\r', '').replace('\t', '')
                        all_items[ids] = {'caption': all_items[ids]['caption'], 'molecule': response2}
                        all_agent_records[ids]["c2m_final_answer"] = response2
                        
                    except Exception as e:
                        lin_pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, Part {n}] error: {e}')
                        response2 = random.choice(init_c2m_response_text_list)[1]
                        response2 = response2.strip().replace('\n', '').replace('\r', '').replace('\t', '')
                        all_items[ids] = {'caption': all_items[ids]['caption'], 'molecule': response2}
                        all_agent_records[ids]["c2m_final_answer"] = response2
                
                except Exception as e:
                    lin_pbar.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}, Part {n}] error: {e}')
                    all_items[ids] = {'caption': all_items[ids]['caption'], 'molecule': "N/A"}
                    all_agent_records[ids]["c2m_final_answer"] = "N/A"
                
            with open(tgt_file, 'a+') as f:
                f.write(ids + '\t' + all_items[ids]['caption'] + "\t" + all_items[ids]['molecule'] + "\n")

            with open(agent_context_file, 'w+') as f:
                json.dump(all_agent_records, f, indent=4)

        with open(full_tgt, 'w+') as f:
            for i in range(len(all_items.keys())):
                ids = list(all_items.keys())[i]
                f.write(ids + '\t' + all_items[ids]['caption'] + '\t' + all_items[ids]['molecule'] + '\n')
                
    pool = mp.Pool(processes=args.process)
    for i in range(1, args.process+1):
        pool.apply_async(run, args=(i, ))
    pool.close()
    pool.join()
    