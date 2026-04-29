def retrieve_zero_prompts():
    head_prompt = ("You are an expert chemist.\n"
                   "Your task is to solve the given problem step by step.\n"
                   "You should explain your reasoning in <think> </think> tags.\n"
                   "The final answer MUST BE a SMILES string and put in <answer> </answer> tags.\n"
                   "Please strictly follow the format.\n"
                   "Now, solve the following problem:\n")
    return head_prompt