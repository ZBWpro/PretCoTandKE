# Set your own path here
OPT_PATH = '../models/opt-6.7b'
LLAMA_PATH = '../models/llama-7b'
LLAMA2_PATH = '/home/datamining/ckh/llama/llama-2-7b-hf'
MISTRAL_PATH = '../models/mistral-7b-v0.1'

# PromptEOL
PromptEOL = 'This sentence : "*sent_0*" means in one word:"'

# Inference
Pretended_CoT = 'After thinking step by step , this sentence : "*sent_0*" means in one word:"'
Knowledge_Enhancement = 'The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this sentence : "*sent_0*" means in one word:"'

# Fine-tuning
PromptSTH = 'This sentence : "*sent_0*" means something'
PromptSUM = 'This sentence : "*sent_0*" can be summarized as'