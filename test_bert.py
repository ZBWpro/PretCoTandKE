import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

import sys
import torch
from prettytable import PrettyTable
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, './SentEval')
import senteval

evaluation_tasks = ['STSBenchmark']
sent_eval_data_path = './SentEval/data'
model_path = '../models/bert-base-uncased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################ Options ################
combine = 'avg'  # avg, cat
mask_number = 3  # 1, 2, 3, 4
eos = '.'  # . ! ? [SEP] Nothing
################ Options ################

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def main():
    assert combine in ['avg', 'cat']
    assert mask_number in {1, 2, 3, 4}
    assert eos in ['.', '!', '?', '[SEP]', 'Nothing']
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    def prepare(params, samples):
        params.max_length = None
        return

    def batcher(params, batch):
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in sentence] for sentence in batch]

        sentences = [' '.join(s) for s in batch]
        template = "*cls*_This_sentence_:_\"*sent_0*\"_means_"

        if eos == '.':
            template += "*mask*_" * mask_number + "._*sep+*"
        elif eos == '!':
            template += "*mask*_" * mask_number + "!_*sep+*"
        elif eos == '?':
            template += "*mask*_" * mask_number + "?_*sep+*"
        elif eos == 'Nothing':
            template += "*mask*_" * mask_number + "*sep+*"
        elif eos == '[SEP]':
            template += "*mask*_" * mask_number + "[SEP]_*sep+*"
        else:
            raise ValueError(f'unknown {eos}')

        template = template.replace('*mask*', tokenizer.mask_token ).replace('_', ' ').replace('*sep+*', '').replace('*cls*', '')

        for i, s in enumerate(sentences):
            if eos == '.':
                if len(s) > 0 and s[-1] not in '.!?"\'': 
                    s += '.'
                s = s.replace('"', '\'')
                if len(s) > 0 and ('?' == s[-1] or '!' == s[-1]): 
                    s = f'{s[:-1]}.'
            elif eos == '!':
                if len(s) > 0 and s[-1] not in '.!?"\'': 
                    s += '!'
                s = s.replace('"', '\'')
                if len(s) > 0 and ('?' == s[-1] or '.' == s[-1]): 
                    s = f'{s[:-1]}!'
            elif eos == '?':
                if len(s) > 0 and s[-1] not in '.!?"\'': 
                    s += '?'
                s = s.replace('"', '\'')
                if len(s) > 0 and ('!' == s[-1] or '.' == s[-1]): 
                    s = f'{s[:-1]}?'
            elif eos == '[SEP]':
                if len(s) > 0 and s[-1] not in '.!?"\'': 
                    s += '[SEP]'
                s = s.replace('"', '\'')
                if len(s) > 0 and ('?' == s[-1] or '!' == s[-1] or '.' == s[-1]): 
                    s = f'{s[:-1]}[SEP]'
            elif eos == 'Nothing':
                s = s.replace('"', '\'')
                if len(s) > 0 and ('?' == s[-1] or '!' == s[-1] or '.' == s[-1]): 
                    s = s[:-1]
            else:
                raise ValueError(f'unknown {eos}')
            
            sentences[i] = template.replace('*sent 0*', s).strip()

        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
            max_length=params.max_length,
            truncation=params.max_length is not None
        )

        for k in batch:
            batch[k] = batch[k].to(device) if batch[k] is not None else None

        with torch.no_grad():
            outputs = model(**batch)
            last_hidden = outputs.last_hidden_state
            pooler_output = last_hidden[batch['input_ids'] == tokenizer.mask_token_id]

            if mask_number > 1:
                pooler_output = pooler_output.view(-1, mask_number, pooler_output.shape[-1])

                if combine == 'avg':
                    pooler_output = pooler_output.mean(dim=1)
                elif combine == 'cat':
                    pooler_output = pooler_output.view(-1, pooler_output.shape[-1] * mask_number)
                else:
                    raise ValueError(f'unknown {combine}')
                
            return pooler_output.view(batch['input_ids'].shape[0], -1).cpu()

    scores = []
    task_names = []

    params = {  # params for STS-B dev
        'task_path': sent_eval_data_path,
        'usepytorch': True,
        'kfold': 5,
        'classifier': {
            'nhid': 0,
            'optim': 'rmsprop',
            'batch_size': 128,
            'tenacity': 3,
            'epoch_size': 2,
        },
    }

    for task in evaluation_tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        task_names.append(task)
        scores.append(result['dev']['spearman'][0] * 100)

    print_table(task_names, scores)

if __name__ == '__main__':
    main()