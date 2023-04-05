import argparse
from typing import Dict

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, cohen_kappa_score,
                             f1_score, precision_score, recall_score)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoModel, AutoTokenizer,
                          BertForSequenceClassification, BertJapaneseTokenizer,
                          EarlyStoppingCallback, EvalPrediction, Trainer,
                          TrainingArguments)
from transformers.modeling_outputs import ModelOutput


class EmotionDataset(Dataset):
    def __init__(self, df):
        self.features = [
            {
                'text': row.text,
                'label': row.label
            } for row in tqdm(df.itertuples(), total=df.shape[0])
        ]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

class EmotionCollator():
        def __init__(self, tokenizer, max_length=512):
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __call__(self, examples):
            examples = {
                'text': list(map(lambda x: x['text'], examples)),
                'label': list(map(lambda x: x['label'], examples))
            }
            
            encodings = self.tokenizer(examples['text'],
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_length,
                                    return_tensors='pt')
            encodings['label'] = torch.tensor(examples['label'])
            return encodings

class EmotionNet(nn.Module):
        def __init__(self, pretrained_model, num_categories, loss_function=None):
            super().__init__()
            self.bert = pretrained_model
            self.hidden_size = self.bert.config.hidden_size
            self.linear = nn.Linear(self.hidden_size, num_categories)
            self.loss_function = loss_function
        
        def forward(self,
                    input_ids,
                    attention_mask=None,
                    position_ids=None,
                    token_type_ids=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    label=None):
            
            outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                token_type_ids=token_type_ids,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states)
            
            state = outputs.last_hidden_state[:, 0, :]
            state = self.linear(state)
            
            loss=None
            if label is not None and self.loss_function is not None:
                loss = self.loss_function(state, label)
            
            attentions=None
            if output_attentions:
                attentions=outputs.attentions
            
            hidden_states=None
            if output_hidden_states:
                hidden_states=outputs.hidden_states
            
            return ModelOutput(
                logits=state,
                loss=loss,
                last_hidden_state=outputs.last_hidden_state,
                attentions=attentions,
                hidden_states=hidden_states
            )

def seed_everything(seed: int):
    """seed固定"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def custom_compute_metrics(res: EvalPrediction) -> Dict:
        # res.predictions, res.label_idsはnumpyのarray
        pred = res.predictions.argmax(axis=1)
        target = res.label_ids
        weight_kappa = cohen_kappa_score(target, pred, weights='quadratic')
        print("weight_kappa = ", weight_kappa)
        return {'weight_kappa':weight_kappa}

        # precision = precision_score(target, pred, average='macro')
        # recall = recall_score(target, pred, average='macro')
        # f1 = f1_score(target, pred, average='macro')
        # return {
        #     'precision': precision,
        #     'recall': recall,
        #     'f1': f1
        # }

def main(args): 
    seed_everything(args.seed)
    # データ読み込み
    train_df = pd.read_csv("/home/miyata/python/workspace/theme_competition3/data/train_integer.txt", sep="\t", names=("label", "text"))
    eval_df = pd.read_csv("/home/miyata/python/workspace/theme_competition3/data/dev_integer.txt", sep="\t", names=("label", "text"))
    test_df = pd.read_csv("/home/miyata/python/workspace/theme_competition3/data/test_integer.txt", sep="\t", names=("label", "text"))

    train_dataset = EmotionDataset(train_df)
    eval_dataset = EmotionDataset(eval_df)
    test_dataset = EmotionDataset(test_df)
         
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    Emotion_collator = EmotionCollator(tokenizer)

    loader = DataLoader(train_dataset, collate_fn=Emotion_collator, batch_size=8, shuffle=True)

    loss_fct = nn.CrossEntropyLoss()
    pretrained_model = AutoModel.from_pretrained(args.model)
    net = EmotionNet(pretrained_model, 5, loss_fct)

    training_args = TrainingArguments(
        output_dir=args.save_model_dir,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        label_names=['label'],
        metric_for_best_model='weight_kappa',
        load_best_model_at_end=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=net,
        tokenizer=tokenizer,
        data_collator=Emotion_collator,
        compute_metrics=custom_compute_metrics,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    trainer.train(ignore_keys_for_eval=['last_hidden_state', 'hidden_states', 'attentions'])

    # trainer.train(ignore_keys_for_eval=['last_hidden_state', 'hidden_states', 'attentions'],
    #             resume_from_checkpoint=True)

    trainer.save_state()
    trainer.save_model()

    pred_result = trainer.predict(test_dataset, ignore_keys=['loss', 'last_hidden_state', 'hidden_states', 'attentions'])
    test_df['predict'] = pred_result.predictions.argmax(axis=1).tolist()
    test_df['predict'] = test_df['predict'] - 2
    test_df['predict'].to_csv(args.sub_file, index=False, header=None)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train parameter
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int) # ミニバッチの実行を累積する回数
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    # PATH
    parser.add_argument("--save_model_dir", default="/home/miyata/python/workspace/theme_competition3/model")
    parser.add_argument("--sub_file", default="/home/miyata/python/workspace/theme_competition3/submission/sub.csv")
    args = parser.parse_args()
    main(args)