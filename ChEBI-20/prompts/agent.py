# ##### Caption to Molecule Debate Functions #####

def c2m_construct_debate_message(init_c2m_response_text_list, caption, examiner_response=None, use_thinking=True):
    if examiner_response is not None:
        assert len(examiner_response) == len(init_c2m_response_text_list), "The number of examiner responses must be the same as the number of initial responses"
    
    molecules_list = ""
    for i, (init_c2m_response_think, init_c2m_response_answer) in enumerate(init_c2m_response_text_list):
        if use_thinking:
            molecules_list+=f"Candidate {i+1}:\nMolecule SMILES of candidate {i+1}: {init_c2m_response_answer}\nReason of candidate {i+1}: {init_c2m_response_think}\n"
        else:
            molecules_list+=f"Candidate {i+1}:\nMolecule SMILES of candidate {i+1}: {init_c2m_response_answer}\n"
        if examiner_response is not None:
            molecules_list+=f"Examiner result of candidate {i+1}: {examiner_response[i]}\n"
        molecules_list += "\n"
    
    if examiner_response is not None:
        _examiner_prompt = " and examing results"
    else:
        _examiner_prompt = ""
    
    if use_thinking:
        prompt = ("You are a evaluator in caption to molecule task. "
                f"Given the following candidate molecule SMILES of the input caption and the corresponding reason{_examiner_prompt}, "
                "evaluate the sufficiency of the molecules and select one or multiple sufficient molecule(s).\n\n"
                "Deliver a short, brief, and strong argument.\n\n"
                "The thinking process and argument are enclosed within <think> </think> and <answer> </answer> tags, respectively.\ni.e.,\n<think>\nthinking process here\n</think>\n<answer>\nanswer here\n</answer>.\n\n"
                f"Input caption:\n{caption}\n\n"
                f"Candidate molecules:\n{molecules_list}\n\n"
                "Output comma-separated index(s) of the sufficient molecule(s), enclosed within <answer> </answer> tags.\ni.e.,\n<answer>\n2\n</answer> or <answer>\n2,3,5\n</answer>.\n\n"
                )
    else:
        prompt = ("You are a evaluator in caption to molecule task. "
                f"Given the following candidate molecule SMILES of the input caption and the corresponding reason{_examiner_prompt}, "
                "evaluate the sufficiency of the molecules and select one or multiple sufficient molecule(s).\n\n"
                "The selections are enclosed within <answer> </answer> tags.\ni.e.,\n<answer>\nanswer here\n</answer>.\n\n"
                f"Input caption:\n{caption}\n\n"
                f"Candidate molecules:\n{molecules_list}\n\n"
                "Output comma-separated index(s) of the sufficient molecule(s), enclosed within <answer> </answer> tags.\ni.e.,\n<answer>\n2\n</answer> or <answer>\n2,3,5\n</answer>.\n\n"
                )
        
    return {"role": "user", "content": prompt}

def c2m_construct_refiner_message(init_c2m_response_text_list, agent_response_text_list, caption, examiner_response=None, use_thinking=True):
    
    if examiner_response is not None:
        assert len(examiner_response) == len(init_c2m_response_text_list), "The number of examiner responses must be the same as the number of initial responses"
    
    molecules_list = ""
    for i, (init_c2m_response_think, init_c2m_response_answer) in enumerate(init_c2m_response_text_list):
        if use_thinking:
            molecules_list+=f"Candidate {i+1}:\nMolecule SMILES of candidate {i+1}: {init_c2m_response_answer}\nReason of candidate {i+1}: {init_c2m_response_think}\n"
        else:
            molecules_list+=f"Candidate {i+1}:\nMolecule SMILES of candidate {i+1}: {init_c2m_response_answer}\n"
        if examiner_response is not None:
            molecules_list+=f"Examiner result of candidate {i+1}: {examiner_response[i]}\n"
        molecules_list += "\n"
    
    agent_selection_list = ""
    for i, (agent_response_think, agent_response_answer) in enumerate(agent_response_text_list):
        agent_alphabet = chr(ord('A') + i)
        if use_thinking:
            agent_selection_list+=f"Agent {agent_alphabet} selection: {agent_response_answer}\nAgent {agent_alphabet} Reason: {agent_response_think}\n\n"
        else:
            agent_selection_list+=f"Agent {agent_alphabet} selection: {agent_response_answer}\n"
    
    """Construct refiner message to refine the caption based on debate insights."""
    
    head_prompt = (
        "You are a caption refinement specialist in a caption-to-molecule generation system. Your task is to refine the input caption based on insights from the debate process.\n\n"
        
        "You will be provided with:\n"
        "1. An original caption describing desired molecular properties\n"
        "2. Multiple candidate molecules generated from this caption\n"
        "3. Debates from multiple agents who evaluated these molecules and selected the most sufficient ones\n\n"
        
        "Your goal is to analyze the debate and refine the original caption to make it more reliable for caption-to-molecule generation. "
        "Learn from the agents' reasoning to identify:\n"
        "- What aspects of the caption led to successful molecule generation\n"
        "- What ambiguities or unclear descriptions caused issues\n"
        "- What key properties or constraints need to be emphasized or clarified\n"
        "- What common mistakes can be avoided with better phrasing\n\n"
        
        "**CRITICAL CONSTRAINTS:**\n"
        "- You MUST preserve the original meaning, purpose, and intent of the caption\n"
        "- You MUST keep all original keywords and essential molecular properties\n"
        "- You MUST NOT add new requirements that weren't in the original caption\n"
        "- Only improve clarity, specificity, and reliability while maintaining the original scope\n\n"
        
        f"Original caption:\n{caption}\n\n"
        f"Candidate molecules generated:\n{molecules_list}\n"
        f"Agents' debate and selection:\n{agent_selection_list}\n"
        
        "Based on the above information, please:\n"
        "1. Analyze what aspects of the original caption were well-captured vs. what caused confusion\n"
        "2. Refine the caption to improve its reliability for molecule generation\n"
        "3. Ensure the refined caption maintains all original keywords, meaning, and purpose\n\n"
        
        "Output your refined caption enclosed within <answer> </answer> tags.\n"
        "Format:\n<answer>\n[Your refined caption here]\n</answer>\n\n"
    )
    
    return {"role": "user", "content": head_prompt}