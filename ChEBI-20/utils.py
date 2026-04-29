import re
import json
from openai import OpenAI, AzureOpenAI
import time
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski, QED
from rdkit import RDLogger
from prompts import init_gen_ChemDFM_R
from prompts import init_gen
from prompts import init_gen_think
from prompts import init_gen_Chem_R
import os
RDLogger.DisableLog('rdApp.*')

def build_c2m_messages(model_name, caption):
    if 'ChemDFM_R' in model_name or 'ChemDFM-R' in model_name:
        caption2molecule = init_gen_ChemDFM_R.retrieve_c2m_zero_prompts()
        _messages = [{"role": "system", "content": caption2molecule},
                    {"role": "user", "content": "Input: " + caption + "\n"}]
    elif 'ChemDFM-v1.5' in model_name:
        _messages = f"Can you please generate the SMILES representation of the caption of the molecule below?\n{caption}"
        _messages = f"[Round 0]\nHuman: {_messages}\nAssistant:"
    elif 'Chem_R' in model_name or 'Chem-R' in model_name:
        caption2molecule = init_gen_Chem_R.retrieve_c2m_zero_prompts()
        _messages = [{"role": "system", "content": "You are an expert chemist."},
                    {"role": "user", "content": f"{caption2molecule}{caption}"}]
    elif 'llama' in model_name.lower():
        caption2molecule = init_gen_think.retrieve_c2m_zero_prompts()
        _messages = [{"role": "system", "content": caption2molecule},
                    {"role": "user", "content": "Input: " + caption + "\n"}]
    else:
        caption2molecule = init_gen.retrieve_c2m_zero_prompts()
        _messages = [{"role": "system", "content": caption2molecule},
                    {"role": "user", "content": "Input: " + caption + "\n"}]
    return _messages

def get_chat_completion(client: OpenAI, 
                        model: str,
                        messages, 
                        seed: int, 
                        max_new_tokens,
                        temperature,
                        top_p, 
                        max_attempts=5,
                        num_generations=1):
    
    if not isinstance(messages, list):
        messages = [messages]
    
    for i in range(max_attempts):
        try:
            if 'gpt-5' in model:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=int(max_new_tokens),
                    seed=int(seed),
                    n=int(num_generations)
                )
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    stop=["</s>", "<|end_of_text|>", "<|eot_id|>"],
                    seed=int(seed),
                    n=int(num_generations)
                )
            return completion
        except Exception as e:
            print(f'API Error: {e}, model: {model}')
            time.sleep(2)
            continue
    return None

def get_completion(client: OpenAI, 
                        model: str,
                        messages, 
                        seed: int, 
                        max_new_tokens,
                        temperature,
                        top_p, 
                        max_attempts=5,
                        num_generations=1):
    
    for i in range(max_attempts):
        try:
            completion = client.completions.create(
                model=model,
                prompt=messages,
                max_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                stop=["</s>", "<|end_of_text|>", "<|eot_id|>"],
                seed=int(seed),
                n=int(num_generations)
            )
            return completion
        except Exception as e:
            print(f'API Error: {e}')
            time.sleep(2)
            continue
    return None

def extract_thinking_answer(generated_text):
    # Try to match with closing tags first (/ is optional in both opening and closing tags)
    match = re.match(r'</?think>(.*?)</?think>\s*</?answer>(.*?)</?answer>', generated_text, re.DOTALL)
    if match:
        thinking, answer = match.groups()
        return thinking.strip(), answer.strip()
    
    # If that fails, try to extract thinking and answer without closing answer tag
    match = re.match(r'</?think>(.*?)</?think>\s*</?answer>(.*)', generated_text, re.DOTALL)
    if match:
        thinking, answer = match.groups()
        return thinking.strip(), answer.strip()
    
    match = re.match(r'</?think>(.*?)</?think>\s*</?answer(.*)', generated_text, re.DOTALL)
    if match:
        thinking, answer = match.groups()
        return thinking.strip(), answer.strip()
    
    # If still no match, try to extract answer only
    # print(f'Could not extract thinking and answer from generated text: >>>>>>> {generated_text} <<<<<<<')
    return generated_text, extract_answer(generated_text)

def extract_answer(generated_text):
    # First try to find with closing tag (/ is optional in both opening and closing tags)
    match = re.search(r'</?answer>(.*?)</?answer>', generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no closing tag, take everything after <answer> or </answer>
    match = re.search(r'</?answer>(.*)', generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'</?answer(.*)', generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return generated_text

def extract_indices(generated_text):
    if not generated_text or not isinstance(generated_text, str):
        return []
    
    # Strip whitespace
    text = generated_text.strip()
    
    if not text:
        return []
    
    # Split by comma and extract integers
    indices = []
    parts = text.split(',')
    
    for part in parts:
        part = part.strip()
        # Extract any digits from the part
        if part.isdigit():
            indices.append(int(part))
        else:
            # Try to find digits in the part
            match = re.search(r'\d+', part)
            if match:
                indices.append(int(match.group()))
    
    return indices

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    https://github.com/molecularsets/moses/blob/master/moses/utils.py
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles_or_mol)
        except Exception as e:
            # print(f"Error parsing SMILES: {e}")
            return None
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    else:
        return smiles_or_mol
    

def calc_props(smiles):
    mol = get_mol(smiles)
    props_dict = {}
    
    if mol is None:
        props_dict['validity'] = "Invalid SMILES"
    else:
        props_dict['validity'] = "Valid SMILES"
        
    props_func_map = {
        "MW": Descriptors.MolWt,
        "LogP": Descriptors.MolLogP,
        "TPSA": Descriptors.TPSA,
        "HBD": Descriptors.NumHDonors,
        "HBA": Descriptors.NumHAcceptors,
        "RotB": Descriptors.NumRotatableBonds,
        "AroRings": rdMolDescriptors.CalcNumAromaticRings,
        "Fsp3": rdMolDescriptors.CalcFractionCSP3,
        "QED": Descriptors.qed,
        "MR": Descriptors.MolMR
    }
    
    for prop, func in props_func_map.items():
        try:
            func_val = func(mol)
            props_dict[prop] = f"{func_val:.2f}"
        except Exception as e:
            props_dict[prop] = None
            continue
    
    return props_dict

def build_examiner_response(smiles):
    props_dict = calc_props(smiles)
    
    if props_dict['validity'] == "Invalid SMILES":
        return f"The SMILES {smiles} is invalid."
    else:
        examiner_response = f"The SMILES {smiles} is valid with the following properties:"
        
    prop_desc_map = {
        "MW": "The average molecular weight of the molecule",
        "LogP": "The logarithm of the octanol-water partition coefficient (LogP) value of the molecule",
        "TPSA": "The topological polar surface area (TPSA) of the molecule",
        "HBD": "The number of hydrogen bond donors in the molecule",
        "HBA": "The number of hydrogen bond acceptors in the molecule",
        "RotB": "The number of rotatable bonds in the molecule",
        "AroRings": "The number of aromatic rings for a molecule",
        "Fsp3": "The fraction of C atoms that are SP3 hybridized in the molecule",
        "QED": "The quantitative estimation of drug-likeness of the molecule",
        "MR": "The relative molecular mass (MR) of the molecule"
    }
    
    for prop, value in props_dict.items():
        if value is None or prop not in prop_desc_map:
            continue
        else:
            examiner_response += f"\n{prop_desc_map[prop]}: {value}"
    
    return examiner_response


def get_gen_client_list(args):
    gen_client_list = []
    
    gen_model_list = args.gen_model_list.split(",")
    gen_port_list = args.gen_port_list.split(",")
    gen_host_list = args.gen_host_list.split(",")
    gen_use_gpt_list = args.gen_use_gpt_list.split(",")
    
    for i in range(len(gen_model_list)):
        if gen_use_gpt_list[i] == "1":
            gen_client_list.append(AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint=gen_host_list[i],
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY")
            ))
        else:
            gen_client_list.append(OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url="http://{}:{}/v1".format(gen_host_list[i], gen_port_list[i])
            ))
    
    return gen_client_list

def get_debate_client(args, gen_client_list):
    gen_model_list = args.gen_model_list.split(",")
    
    if args.model in gen_model_list:
        model_index = gen_model_list.index(args.model)
        client = gen_client_list[model_index]
    else:
        if args.use_gpt:
            client = AzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint=os.environ.get("OPENAI_API_BASE", "EMPTY"),
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY")
            )
        else:
            openai_api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
            openai_api_base = "http://{}:{}/v1".format(args.host, args.port)
            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            
    return client

def get_consensus_score(selection_list):
    if not selection_list or len(selection_list) == 0:
        return 0.0
    
    if len(selection_list) == 1:
        return 1.0  # Single selection has perfect consensus with itself
    
    # Convert all selections to sets for efficient intersection/union operations
    selection_sets = [set(selection) for selection in selection_list]
    
    # Calculate pairwise Jaccard similarities
    total_similarity = 0.0
    num_pairs = 0
    
    for i in range(len(selection_sets)):
        for j in range(i + 1, len(selection_sets)):
            set_a = selection_sets[i]
            set_b = selection_sets[j]
            
            intersection = set_a & set_b
            union = set_a | set_b
            
            # Handle edge case where both sets are empty
            if len(union) == 0:
                jaccard_similarity = 1.0
            else:
                jaccard_similarity = len(intersection) / len(union)
            
            total_similarity += jaccard_similarity
            num_pairs += 1
    
    # Return average Jaccard similarity
    consensus_score = total_similarity / num_pairs if num_pairs > 0 else 0.0
    
    return consensus_score