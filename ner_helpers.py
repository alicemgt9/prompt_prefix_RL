from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List

from ner_reward import PromptedNerReward

class PromptedNerDataset(Dataset):
    def __init__(
        self, 
        source_texts: List[str], 
        disease_name: List[str],
    ):
        assert len(source_texts) == len(disease_name)
        self.source_texts = source_texts
        self.disease_name = disease_name

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        item = {'source_texts': self.source_texts[idx],
                'disease_name': self.disease_name[idx],
                }
        return item


def make_few_shot_ner_dataset(
        config: "DictConfig") -> Tuple[PromptedNerDataset]: 
    data_dict = {}
    for split in ['train', 'dev', 'test']: 
        source_texts, disease_name, template = \
            load_few_shot_ner_dataset(config.dataset,  
                                      split, config.dataset_seed,
                                      config.base_path
                                      )
        fsc_dataset = PromptedNerDataset(source_texts, 
                                         disease_name
                                         )
        data_dict[split] = fsc_dataset

    return (data_dict['train'], data_dict['dev'], data_dict['test'], 
            disease_name,template)



def load_few_shot_ner_dataset(
    dataset: str,
    split: str,
    dataset_seed: Optional[int],
    base_path: str,
) -> Tuple[List[str]]:

    assert split in ['train', 'dev', 'test']

    seed_dict = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14'}
    seed_path = seed_dict[dataset_seed]
    full_filepath = f'./data/{dataset}/{seed_path}/{split}.csv'
    print(base_path)
    df = pd.read_csv(full_filepath)
    if 'text' in df:
        source_texts = df.text.tolist()
    else: 
        source_texts = df.sentence.tolist()
    disease_name = df.entity.tolist()

    template = None

    return (source_texts, disease_name,template)



@dataclass
class FewShotNerDatasetConfig:
    dataset: str = "???"
    base_path: str = './data'
    num_shots: int = 5
    dataset_seed: Optional[int] = None 



#需要修改的地方
def make_prompted_ner_reward(
    disease_name: str,
    template: Optional[str], 
    config: "DictConfig") -> PromptedNerReward:
    return PromptedNerReward(config.task_lm, config.Llama_model_version, 
                                        config.Llama_model_size,template,
                                        disease_name)


@dataclass
class PromptedNerRewardConfig:
    task_lm : str = "../../../llama/llama-2-7b-chat-hf"
    Llama_model_version: str = "2"
    Llama_model_size: str = "7b"
