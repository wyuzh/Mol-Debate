def retrieve_zero_prompts():
    head_prompt = ("You are working as an assistant of a chemist user.\n"
                   "You always reason thoroughly before giving response. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.\ni.e.,\n<think>\nreasoning process here\n</think>\n<answer>\nanswer here\n</answer>."
                   "Please follow the instruction of the chemist and generate a molecule that satisfies the requirements of the chemist user.\n")
    return head_prompt

def retrieve_zero_prompts_gpt():
    head_prompt = ("You are working as an assistant of a chemist user.\n"
                   "You always reason thoroughly before giving response. The reasoning process and answer SMILES string are enclosed within <think> </think> and <answer> </answer> tags, respectively.\ni.e.,\n<think>\nreasoning process here\n</think>\n<answer>\nanswer SMILES string here\n</answer>."
                   "Please follow the instruction of the chemist and generate a molecule that satisfies the requirements of the chemist user.\n")
    return head_prompt