def retrieve_c2m_zero_prompts():
    template = "Task Format\n" \
        + "```\n" \
        + "Instruction: Given the caption of a molecule, predict the SMILES representation of the molecule.\n" \
        + "Input: [CAPTION_MASK]\n" \
        + "```\n" \
        + "\n" \
        + "Your output should be: \n" \
        + "\n<answer>\nanswer here\n</answer>" \
        + "\n"
        
    head_prompt = "You are now working as an excellent expert in chemisrty and drug discovery. Your answer is enclosed within <answer> </answer> tags.\ni.e.,\n<answer>\nanswer here\n</answer>. Given the caption of a molecule, your job is to predict the SMILES representation of the molecule. The molecule caption is a sentence that describes the molecule, which mainly describes the molecule's structures, properties, and production. You can infer the molecule SMILES representation from the caption.\n" \
        + "\n" \
        + template + "\n" \
        + "Your response should only be in the format above. "
    return head_prompt
