from datasets import load_dataset, load_from_disk
import torch
import transformers
from typing import Dict
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


class CoarseGrainedDataset(Dataset):
    
    def __init__(self, 
                tokenizer: transformers.PreTrainedTokenizer, 
                lorra_args,
                ):
        
        ds = load_from_disk('../dataset/coarse_grained')

        super(CoarseGrainedDataset, self).__init__()

        self.max_res_len = lorra_args.max_res_len
        self.tokenizer = tokenizer
        pos_s = 'You are a good, respectful and honest assistant. Your responses should be helpful, meaningful, truthful, harmless, safe, ethical, socially unbiased, positive, interesting, well-founded, agreeable, specific, reasonable, and role consistent. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. Please do not share false information under any circumstances.' 
        neg_s = 'You are a bad, disrespectful and dishonest assistant. Your responses might be useless, meaningless, untruthful, harmful, dangerous, unethical, socially biased, negative, boring, fabricated, offensive, vague, unreasonable, and role inconsistent. If a question does not make any sense, or is not factually coherent, still answer the question and share false information when necessary.' 


        formatted_text_chosen = [tokenizer.apply_chat_template(conv, tokenize=False).split('[/INST]') for conv in ds['preference']]
        formatted_text_rejected = [tokenizer.apply_chat_template(conv, tokenize=False).split('[/INST]') for conv in ds['rejected']]


        formatted_text_original_prompt = [prompt[0] for prompt in formatted_text_chosen]
        formatted_text_original_response = ['[/INST]' + prompt[1] for prompt in formatted_text_chosen]

        formatted_text_chosen_prompt = [prompt[0] + pos_s for prompt in formatted_text_chosen]
        formatted_text_chosen_response = ['[/INST]' + prompt[1] for prompt in formatted_text_chosen]

        formatted_text_rejected_prompt = [prompt[0] + neg_s for prompt in formatted_text_rejected]
        formatted_text_rejected_response = ['[/INST]' + prompt[1] for prompt in formatted_text_rejected] 

        self.orig_prompt = formatted_text_original_prompt
        self.orig_response = formatted_text_original_response
        self.chosen_prompt = formatted_text_chosen_prompt
        self.chosen_response = formatted_text_chosen_response
        self.rejected_prompt = formatted_text_rejected_prompt
        self.rejected_response = formatted_text_rejected_response

    def __len__(self):
        return len(self.orig_prompt)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
    

        self.tokenizer.padding_side = "left"
        tokenized_inputs = self.tokenizer(
            [
                self.orig_prompt[i],
                self.chosen_prompt[i],
                self.rejected_prompt[i]
            ],
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        self.tokenizer.padding_side = "right"
        response_tokenized_inputs = self.tokenizer(
            [
                self.orig_response[i],
                self.chosen_response[i],
                self.rejected_response[i]
            ],
            padding="max_length",
            max_length=self.max_res_len,
            truncation=True,
            return_tensors="pt",
        )

        combined_input_ids = torch.cat([tokenized_inputs["input_ids"], response_tokenized_inputs["input_ids"]], dim=1)
        combined_attention_mask = torch.cat([tokenized_inputs["attention_mask"], response_tokenized_inputs["attention_mask"]], dim=1)
        return {
            "input_ids":combined_input_ids,
            "attention_mask":combined_attention_mask
        }
    
        

        
class TruthfulDataset(Dataset):
    def __init__(self, 
                tokenizer: transformers.PreTrainedTokenizer,
                lorra_args,
                ):
        super(TruthfulDataset, self).__init__()

        ds = load_dataset('tatsu-lab/alpaca', cache_dir='/data/chaojian/.cache/huggingface/datasets/tatsu-lab___alpaca')
        ds = ds.filter(lambda x: x['input'] == '')['train'].shuffle(seed=42).select(range(10000))
        instructions = ds['instruction']
        outputs = ds['output']

        pos_s = "Pretend you're an honesty assistant. Give me a truthful answer."
        neg_s = "Pretend you're an dishonesty assistant. Give me an untruthful answer."
        
        orig_conv, chosen_conv, rejected_conv = [], [], []

        

        for i in range(len(instructions)):
            orig_conv.append("[INST]" + instructions[i] + "[/INST]" + outputs[i])
            chosen_conv.append("[INST]" + instructions[i] + pos_s + "[/INST]" + outputs[i])
            rejected_conv.append("[INST]" + instructions[i] + neg_s + "[/INST]" + outputs[i])



        self.orig_s = orig_conv
        self.pos_s = chosen_conv
        self.neg_s = rejected_conv
        self.max_res_len = lorra_args.max_res_len

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.orig_s)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:


        assistant_tag = "[/INST]"
        orig_s, pos_s, neg_s = self.orig_s[i], self.pos_s[i], self.neg_s[i]
        self.tokenizer.padding_side = "left"
        tokenized_inputs = self.tokenizer(
            [orig_s.split(assistant_tag)[0], 
             pos_s.split(assistant_tag)[0],
             neg_s.split(assistant_tag)[0]],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        self.tokenizer.padding_side = "right"
        response_tokenized_inputs = self.tokenizer(

            # 全部使用正确答案
            [assistant_tag + orig_s.split(assistant_tag)[1]] * 3,
            padding="max_length",
            truncation=True,
            max_length=self.max_res_len,
            return_tensors="pt",
        )
        combined_input_ids = torch.cat([tokenized_inputs["input_ids"], response_tokenized_inputs["input_ids"]], dim=1)
        combined_attention_mask = torch.cat([tokenized_inputs["attention_mask"], response_tokenized_inputs["attention_mask"]], dim=1)
        return dict(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask
        )

        
         
def prepare_inputs(tokenized_text, device):
    # put the text on the device
    tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}
    position_ids = get_position_ids(tokenized_text['attention_mask'])
    # tokenized_text['position_ids'] = position_ids
    return tokenized_text

def get_position_ids(attention_mask):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids

def prepare_decoder_only_inputs(prompts, targets, tokenizer, device):
    tokenizer.padding_side = "left"
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    tokenizer.padding_side = "right"
    target_inputs = tokenizer(targets, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    inputs = {k: torch.cat([prompt_inputs[k], target_inputs[k]], dim=1) for k in prompt_inputs}
    inputs = prepare_inputs(inputs, device)
    labels = inputs["attention_mask"].clone()
    labels[:, :prompt_inputs["input_ids"].shape[1]] = 0
    labels[labels == tokenizer.pad_token_id] = 0
    return inputs, labels

def get_logprobs(logits, input_ids, attention_mask, **kwargs):
    # TODO: comments this in release
    logprobs = F.log_softmax(logits, dim=-1)[:, :-1]
    logprobs = torch.gather(logprobs, -1, input_ids[:, 1:, None])
    logprobs = logprobs * attention_mask[:, 1:, None]
    # check for nans
    assert logprobs.isnan().sum() == 0 
    return logprobs.squeeze(-1)

def get_logprobs_accuracy(model, tokenizer, questions, answers, labels, bsz):
    output_logprobs = []
    for i in range(len(questions) // bsz + 1):
        q_batch = questions[i*bsz:(i+1)*bsz].tolist()
        a_batch = answers[i*bsz:(i+1)*bsz].tolist()
        inputs, masks = prepare_decoder_only_inputs(q_batch, a_batch, tokenizer, model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
            logprobs = get_logprobs(logits.float(), inputs['input_ids'], masks).sum(-1).detach().cpu().numpy()
        output_logprobs.extend(logprobs)
    i = 0
    cors, cors_norm = [], []
    for l in labels:
        log_probs = output_logprobs[i:i+len(l)]
        completion_len = answers[i:i+len(l)]
        completions_len = np.array([float(len(i)) for i in completion_len])
        cors.append(np.argmax(log_probs) == l.index(1))
        cors_norm.append(np.argmax(log_probs / completions_len) == l.index(1))
        i += len(l)
    return {'acc': np.mean(cors), 'acc_norm': np.mean(cors_norm)}


def load_tqa_sentences(user_tag, assistant_tag):
    dataset = load_from_disk('../dataset/truthful_qa_multichoice')['validation']
    questions, answers = [],[]
    labels = []
    for d in dataset:
        q = d['question']
        for i in range(len(d['mc1_targets']['labels'])):
            a = d['mc1_targets']['choices'][i]
            questions.append(f'{user_tag} ' + q + ' ')
            answers.append(f'{assistant_tag} ' + a)

        labels.append(d['mc1_targets']['labels'])
    return np.array(questions), np.array(answers), labels

def load_arc_sentences(challenge=False):
    config = 'ARC-Challenge' if challenge else 'ARC-Easy'
    dataset = load_from_disk('../dataset/ai2_arc_easy')['validation']

    questions, answers = [],[]
    labels = []
    for d in dataset:
        q = d['question']
        choices = d['choices']['text']
        label = [d['answerKey'] == c for c in d['choices']['label']]
        for a in choices:
            questions.append(f'Question: ' + q + '\nAnswer:')
            answers.append(a)
        labels.append(label)
    return np.array(questions), np.array(answers), labels


