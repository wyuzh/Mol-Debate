def retrieve_c2m_zero_prompts():
    head_prompt = ("You are an expert chemist. \n"
                   "Your task is to solve the given problem step by step. \n"
                   "You should put your reasoning in <think> </think> tags. \n"
                   "The final answer MUST BE put in  <answer> </answer> tags. \n"
                   "Please strictly follow the format. \n"
                   "Now predict the SMILES representation for the following molecular design requirement: \n"
                   "Description: ")
    
    return head_prompt
