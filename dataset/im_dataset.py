from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainingDataset(Dataset):
    def __init__(self, data_path,tokenizer,max_len=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = load_dataset('json',data_files=data_path,split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False,max_len=self.max_len-2,truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_len-len(tokens))
        input_ids = torch.tensor(input_ids,dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids==self.tokenizer.pad_token_id] = -100
        return input_ids, labels

#SFT
class SFTDataset(Dataset):
    def __init__(self, data_path,tokenizer,max_len=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        features = Features({'conversations':[{
            'role':Value('string'),
            'content':Value('string'),
            'reasoning_content':Value('string'),
            'tools':Value('string'),
            'tool_call':Value('string'),
        }]})
        self.bos_id = tokenizer(f"{tokenizer.bos_token_id}assistant\n",add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token_id}\n",add_special_tokens=False).input_ids
        self.samples = load_dataset('json',data_files=data_path,split='train',features=features)

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self,conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get('tools') and message.get("role")=='system':
                tools = json.loads(message['tools']) if isinstance(message['tools'],str) else message['tools']
            if message.get("tool_calls") and isinstance(message['tool_calls'],str):
                message['tool_calls'] = json.loads(message['tool_calls'])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt=False,
            tools = tools,
        )

    def generate_labels(self,input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i+len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end+len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start,min(end+len(self.eos_id),len(input_ids))):
                    labels[j] = input_ids[j]
                i = min(end+len(self.eos_id),len(input_ids))
            else:
                i += 1
        return labels

    def __getitem__(self, idx):
        sample = self.samples[idx]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_len]
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_len-len(input_ids))
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids,dtype=torch.long), torch.tensor(labels,dtype=torch.long)

if __name__ == "__main__":
    pass