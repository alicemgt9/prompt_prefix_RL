import os
import hydra
from omegaconf import DictConfig, OmegaConf

from rlprompt.models import (LMAdaptorModelConfig, SinglePromptModelConfig,
                             make_lm_adaptor_model, make_single_prompt_model)
from rlprompt.modules import SQLModuleConfig, make_sql_module
from rlprompt.trainers import TrainerConfig, make_trainer
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)

from ner_multi_helpers import (PromptedNerRewardConfig,
                         FewShotNerDatasetConfig,
                         make_prompted_ner_reward,
                         make_few_shot_ner_dataset)


# Compose default config
config_list = [PromptedNerRewardConfig,
                FewShotNerDatasetConfig, LMAdaptorModelConfig,
                SinglePromptModelConfig, SQLModuleConfig, TrainerConfig]
cs = compose_hydra_config_store('base_ner', config_list)


@hydra.main(version_base=None, config_path="./", config_name="ner_multi_config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    output_dir = get_hydra_output_dir()

    (train_dataset, val_dataset, test_dataset,
     disease_name,template) = \
        make_few_shot_ner_dataset(config)
    print('Train Size:', len(train_dataset))
    print('Examples:', train_dataset[:5])
    print('Val Size', len(val_dataset))
    print('Examples:', val_dataset[:5])

    policy_model = make_lm_adaptor_model(config)
    prompt_model = make_single_prompt_model(policy_model, config)
    reward = make_prompted_ner_reward(disease_name, 
                                    template, config)
    algo_module = make_sql_module(prompt_model, reward, config)

    # Hack for few-shot classification - Each batch contains all examples
    config.train_batch_size = len(train_dataset)
    config.eval_batch_size = len(val_dataset)
    config.save_dir = os.path.join(output_dir, config.save_dir)
    trainer = make_trainer(algo_module, train_dataset, val_dataset, config)
    trainer.train(config=config)


if __name__ == "__main__":
    main()
