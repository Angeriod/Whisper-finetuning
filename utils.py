from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
import evaluate
from typing import Any, Dict, List, Union
import torch

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="Ko", task="transcribe")
metric = evaluate.load('cer')

def prepare_dataset(batch):

    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    batch["labels"] = tokenizer(batch["transcripts"]).input_ids
    return batch

def preprocess(datasets):
    low_call_voices = datasets.map(prepare_dataset, remove_columns=datasets.column_names["train"], num_proc=None)

    #low_call_voices.push_to_hub("Angeriod/")

    return low_call_voices

def compute_metrics(pred):

    pred_ids = pred.predictions
        
    label_ids = pred.label_ids


    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

class DataCollatorSpeechSeq2SeqWithPadding:
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt",padding=True)
        batch["attention_mask"] = (batch["input_features"] != self.processor.tokenizer.pad_token_id).long()


        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt", padding=True)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch