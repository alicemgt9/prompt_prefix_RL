import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, GPT2LMHeadModel, AutoModel
from typing import List, Dict, Optional, Tuple, Union, Any
from collections import defaultdict
from rlprompt.rewards import BaseReward
from omegaconf import DictConfig, OmegaConf
from llamawrapper import load_unemb_only, LlamaHelper
import pickle
import datetime
import os
import ast

this_datetime = datetime.datetime.now()
path = "./reward_prompt"
os.makedirs(f'{path}/{str(this_datetime)}', exist_ok=True)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

class PromptedNerReward(BaseReward):
    def __init__(
        self,
        task_lm:str,
        Llama_model_version: str,
        Llama_model_size:str,
        template: Optional[str],
        dataset:str,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.Llama_model_version = Llama_model_version
        self.Llama_model_size = Llama_model_size
        self.dataset = dataset
        #这个dataset可以直接从这里修改，一次把所有的全部变成token
        hf_token = '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'
        self.unemb = load_unemb_only(self.Llama_model_size,self.Llama_model_version)
        self.llama = LlamaHelper(dir=task_lm, load_in_8bit=True, hf_token=hf_token)
        self.tokenizer = self.llama.tokenizer
        self.model = self.llama.model
        print('load Llama from', task_lm)
        #embedding vector
        self.unemb = nn.Sequential(self.llama.model.model.norm, self.llama.model.lm_head)


        if template is None:
            self.template = self.load_default_template()  # prompt templates
        else: 
            self.template = template
        self._counter = 0

        self.disease_name_token = []

        self.sub_process_text = []
        self.process_text = []

        print("processing source text")

        # for text in dataset:
        #     text = str(text).strip(";").split(";")
        #     self.process_text.append(text)
        
        # for sub_texts in self.process_text:
        #     print(f"processing:  {sub_texts}" )
        #     for sub_text in sub_texts:
        #         sub_text = sub_text.strip()
        #         self.sub_process_text.append(sample(tokenizer=self.tokenizer, disease_name=sub_text))
        #     self.disease_name_token.append(self.sub_process_text)
        #     self.sub_process_text = []

    def load_default_template(self) -> str:
        template = ""
        # #zero-shot location
        # template += "{prompt}.Given entity label set: 'location'. Based on the given entity label set, please recognize the named entities in the given text. \n"
        # template += "You just need output the labeled entities without other output, like this: entity1, entity2, entity3\n" 
        # template += "Text: '{sentence_1}.' \n 'location' in the text are:"
        #zero-shot org
        # template += "{prompt}.Given entity label set: 'organization'. Based on the given entity label set, please recognize the named entities in the given text. \n"
        # template += "You just need output the labeled entities without other output, like this: entity1, entity2, entity3\n" 
        # template += "Text: '{sentence_1}.' \n 'organiztion' in the text are:"
        #zero-shot person
        # template += "{prompt}.Given entity label set: 'person'. Based on the given entity label set, please recognize the named entities in the given text. \n"
        # template += "You just need output the labeled entities without other output, like this: entity1, entity2, entity3\n" 
        # template += "Text: '{sentence_1}.' \n 'person' in the text are:"
        # zero-shot ncbi
        # template += "{prompt}.Given entity label set: 'disease'. Based on the given entity label set, please recognize the named entities in the given text. \n"
        # template += "You just need output the labeled entities without other output, like this: entity1, entity2, entity3\n" 
        # template += "Text: '{sentence_1}.' \n 'disease' in the text are:"
        #few-shot location
        # template += "{prompt}.Given entity label set: 'location'. Based on the given entity label set, please recognize the named entities in the given text. \n"
        # template += "I will give you some examples.\n"
        # template += "Example: input:'HOBART , Australia 1996-12-07',Output:'HOBART , Australia' \n" 
        # template += "Example: input:'Turkey says Syria sponsors the PKK , fighting for Kurdish self-rule in southeast Turkey.',Output:'Turkey , Syria , Turkey' \n" 
        # template += "Example: input:'In U.S. dollar terms , Japan was the only country to give positive returns at 1.35 percent .',Output:'U.S. , Japan' \n" 
        # template += "Example: input:'he move is expected to give a shot in the arm to the economic expansion of Guangxi and southwest China as a whole',Output:'Guangxi , China' \n" 
        # template += "You just need output the labeled entities without other output, like this: entity1, entity2, entity3\n" 
        # template += "Text: '{sentence_1}.' \n 'location' in the text are:"
        #few-shot person
        # template += "{prompt}.Given entity label set: 'person'. Based on the given entity label set, please recognize the named entities in the given text. \n"
        # template += "I will give you some examples.\n"
        # template += "Example: input:'S. Law b Hooper 21 ',Output:'S. Law ;Hooper ' \n" 
        # template += "Example: input:'Bowling : Reiffel 10-2-26-0 ( nb-3 ) , Gillespie 10-0-39-2 ,',Output:'Reiffel , Gillespie' \n" 
        # template += "Example: input:'J. Murray c Blewett b Warne 24 ',Output:'J. Murray,Blewett,Warne ' \n" 
        # template += "Example: input:'Dong Jiong ( China ) beat Thomas Stuer-Lauridsen ( Denmark ) 15-10 15-6 ',Output:'Dong Jiong ,Thomas Stuer-Lauridsen' \n"
        # template += "You just need output the labeled entities without other output, like this: entity1, entity2, entity3\n" 
        # template += "Text: '{sentence_1}.' \n 'person' in the text are:"
        #few-shot organization
        # template += "{prompt}.Given entity label set: 'organization'. Based on the given entity label set, please recognize the named entities in the given text. \n"
        # template += "I will give you some examples.\n"
        # template += "Example: input:'Manchester City 3 Bradford 2',Output:'Manchester ,Bradford ' \n" 
        # template += "Example: input:'Portuguesa 1 Atletico Mineiro 0 ',Output:'Portuguesa , Atletico Mineiro' \n" 
        # template += "Example: input:'Dunfermline 2 ( Millar 43 , 46 penalty ) Aberdeen 3 ( Miller 10 , Rowson 55 , Windass 78 ) . ',Output:'Dunfermline,Aberdeen' \n" 
        # template += "Example: input:'The conditions for clearance of the alliance were that British Airways and American drop 168 slots at London Heathrow airport , the busiest in Europe .',Output:'British Airways,American' \n"
        # template += "You just need output the labeled entities without other output, like this: entity1, entity2, entity3\n" 
        # template += "Text: '{sentence_1}.' \n 'organization' in the text are:"
        #few-shot ncbi
        # template += "{prompt}.Given entity label set: 'disease'. Based on the given entity label set, please recognize the named entities in the given text. \n"
        # template += "I will give you some examples.\n"
        # template += "Example: input:'Of the patients with PKU , 92 '%' had been treated during childhood .',Output:'PKU' \n" 
        # template += "Example: input:'Inactivation of the murine ATP7B gene produces a form of cirrhotic liver disease that resembles Wilson disease in humans and the toxic milk phenotype in the mouse.',Output:'cirrhotic liver disease, Wilson disease' \n" 
        # template += "Example: input:'Spinal xanthomatosis : a variant of cerebrotendinous xanthomatosis .',Output:'Spinal xanthomatosis,cerebrotendinous xanthomatosis' \n" 
        # template += "Example: input:'We therefore hypothesized that A - T is due to oxidative damage resulting from loss of function of the A - T gene product .',Output:'A - T,A - T' \n" 
        # template += "Text: '{sentence_1}.' \n 'disease' in the text are:"
        #smm4h zero-shot
        # template += "{prompt}.Given entity label set: 'adverse drug event'. Based on the given entity label set, please recognize the named entities in the given text. \n"
        # template += "You just need output the labeled entities without other output, like this: entity1, entity2, entity3\n" 
        # template += "Text: '{sentence_1}.' \n 'adverse drug event' in the text are:"
        template += "{prompt}.Given entity label set: 'adverse drug event'. Based on the given entity label set, please recognize the named entities in the given text. \n"
        template += "I will give you some examples.\n"
        template += "You just need output the labeled entities without other output, like this: entity1, entity2, entity3\n" 
        template += "Example1: 'Here we present four novel PAX6 missense mutations , two in association with atypical phenotypes ectopia pupillae ( displaced pupils ) and congenital nystagmus ( searching gaze ) , and two in association with more recognizable aniridia phenotypes .',Output:'ectopia pupillae',"
        template += "Example2: 'We conclude that paternal transmission of congenital DM is rare and preferentially occurs with onset of DM past 30 years in the father . .',Output:'congenital DM',\n"
        template += "Example3: 'Very - long - chain acyl - coenzyme A dehydrogenase ( VLCAD ) deficiency is a disorder of fatty acid beta oxidation that reportedly has high rates of morbidity and mortality .',Output:'Very - long - chain acyl - coenzyme A dehydrogenase ( VLCAD ) deficiency'\n"
        template += "Example4: 'These results indicate that Smad4 is initially required for the differentiation of the visceral endoderm and that the gastrulation defect in the epiblast is secondary and non - cell autonomous .',Output:'gastrulation defect',\n"
        template += "Text: '{sentence_1}.' \n 'adverse drug event' in the text are:"
        
        return template
    

    def forward( 
            # 这里是计算reward的地方,把里面的内容改成回馈模型的内容，
            #output_tokens是通过distllgpt2 生成的，是一个个离散的词
        self,
        source_texts: List[str],
        disease_name: List[str],
        output_tokens: List[List[str]],
        to_tensor: bool,
        mode: str
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:
        assert mode in ["train", "infer"]

        if mode == "train":
            self._counter += 1
    
        
        print(self._counter)

        i = 1

        self.disease_name_token = []

        self.sub_process_text = []
        self.process_text = []

        print("processing source text")

        for text in disease_name:
            text = str(text).strip(";").split(";")
            self.process_text.append(text)
        
        for sub_texts in self.process_text:
            print(f"processing:  {sub_texts}" )
            for sub_text in sub_texts:
                sub_text = sub_text.strip()
                self.sub_process_text.append(sample(tokenizer=self.tokenizer, disease_name=sub_text))
            self.disease_name_token.append(self.sub_process_text)
            self.sub_process_text = []
        #disease_name_token 的形状是 [1，32] 其中 32个是不等长的list（tensor）

        
        # Process prompts and verbalizer indices
        prompt_tokens = output_tokens
        prompt_strings = self._convert_tokens_to_string(prompt_tokens)
        rewards: List[torch.Tensor] = []
        quantities_to_log: Dict[str, List[torch.Tensor]] = defaultdict(list)
        #calucate rewards
        for i, prompt in enumerate(prompt_strings):
            # Compute LM logits
            print('start train')
            current_prompts = [prompt for _ in source_texts]
            formatted_templates = self._format_prompts(source_texts,
                                                       current_prompts)
            print(prompt)
            out_token_probs = []
            idx = 0

            for text,output_tokens_ids in zip(formatted_templates,self.disease_name_token):
                
                #使用了norm的方法
                multi_token_probs = [] #单句中多个实体的probs
                #print(f"this is text, and this is output_tokens_id {text}, {output_tokens_id}")
                latents = self.llama.latents_all_layers(text)
                #output = self.llama.generate_text(text)
                logits = self.unemb(latents)
                last = logits[:, -1, :].float().softmax(dim=-1).detach().cpu()

                #计算单个实体的probs
                length = str(disease_name[idx]).count(";")
                for single_tensor in output_tokens_ids:#计算每个latent实体矩阵在output logit中的probs 
                    #output_tokens_ids[idx]传入的是一个list
                    multi_token_probs += [last[:, torch.tensor(single_tensor)].sum(dim=-1)]
                #计算MSRE
                #multi_token_probs = torch.stack(multi_token_probs)# shape: [num_entities, layer_size]
                multi_token_probs = torch.stack(multi_token_probs)

                multi_token_probs = multi_token_probs[:,31] #shape为(1,entity number)
                #tensor_list_pred = [torch.tensor(1.0 / length) for _ in range(length)] #设定goldpred
                #tensor_list_pred = torch.stack(tensor_list_pred)
                #sd = std_with_fixed_mean(multi_token_probs, tensor_list_pred) #计算std
                #print(f"标准差为{sd}")
                sentence_reward = multi_token_probs.mean()#reward
                out_token_probs.append(sentence_reward)


                    
                idx += 1

            out_token_probs = torch.stack(out_token_probs)
            

            #for single rewards
            #out_token_probs = out_token_probs[:,31] * 100 
            #reward = out_token_probs.mean().detach()
            # for multi rewards
            reward = torch.sum(out_token_probs.detach()) / 32
            # 基于正确的奖励
            # reward = count / all_count
            rewards.append(reward)

            rewards_tensor = torch.stack(rewards)

            #print(f' this is rewards : {rewards}')

            rewards_log = dict()
        if not os.path.exists(
                f'{path}/{str(this_datetime)}/{self._counter}.csv'
        ):
            if to_tensor is True:
                rewards_tensor_temp = rewards_tensor * 100
                pd_prompt_strings = pd.DataFrame(prompt_strings)
                pd_rewards_tensor = pd.DataFrame(rewards_tensor_temp)
                save_pd = pd.concat([pd_prompt_strings,pd_rewards_tensor],axis=1)
                save_pd.to_csv(os.path.join(f'{path}/{str(this_datetime)}/{self._counter}.csv'))
                return rewards_tensor, rewards_log
            else:
                rewards_tensor_temp = rewards_tensor * 100
                pd_prompt_strings = pd.DataFrame(prompt_strings)
                pd_rewards_tensor = pd.DataFrame(rewards_tensor_temp)
                save_pd = pd.concat([pd_prompt_strings,pd_rewards_tensor],axis=1)
                save_pd.to_csv(os.path.join(f'{path}/{str(this_datetime)}/{self._counter}.csv'))
                return rewards_tensor.tolist(), rewards_log
        else:
            if to_tensor is True:
                return rewards_tensor, rewards_log
            else:
                return rewards_tensor.tolist(), rewards_log
    

    def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
        return [self.tokenizer.convert_tokens_to_string(s) + " "
                for s in tokens]

    def _format_prompts(
        self,
        source_strs: List[str],
        prompt_strs: List[str],
    ) -> List[str]:
        return [self.template.format(sentence_1=s_1, prompt=p)
                for s_1, p in zip(source_strs, prompt_strs)]


def token_prefixes(token_str: str):
    n = len(token_str)
    tokens = [token_str[:i] for i in range(1, n+1)]
    return tokens 

def add_spaces(tokens):
    return ['▁' + t for t in tokens] + tokens

def capitalizations(tokens):
    return list(set(tokens))


def process_tokens(token_str: str, tokenizer):
    with_prefixes = token_prefixes(token_str)
    with_spaces = add_spaces(with_prefixes)
    with_capitalizations = capitalizations(with_spaces)
    final_tokens = []
    for tok in with_capitalizations:
        if tok in tokenizer.get_vocab():
            final_tokens.append(tokenizer.get_vocab()[tok])
    return final_tokens

def sample(tokenizer, disease_name):
                out_token_str = disease_name
                latent_token_str=disease_name
                out_token_id = process_tokens(out_token_str, tokenizer)
                latent_token_id = process_tokens(latent_token_str, tokenizer)
                return out_token_id
                
def std_with_fixed_mean(tensor, u):
    squared_diff = (tensor - u) ** 2
    
    # 计算方差（总体方差，除以 n）
    variance = torch.mean(squared_diff)
    
    # 标准差是方差的平方根
    std_dev = torch.sqrt(variance)

    return std_dev
