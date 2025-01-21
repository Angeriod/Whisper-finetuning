from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import argparse
import utils, dataset

def define_args():
    p = argparse.ArgumentParser()
    p.add_argument('--whisper_model', type=str, default="openai/whisper-base")
    p.add_argument('--whisper_language', type=str, default="Ko")
    p.add_argument('--whisper_task', type=str, default="transcribe")

    p.add_argument('--no_dataset', type=bool, default="False")
    p.add_argument('--preprocess', type=bool, default="False")

    p.add_argument('--train_path', type=str, default="D:/in_car_command/in_car_command/train/")
    p.add_argument('--valid_path', type=str, default="D:/in_car_command/in_car_command/test/")
    p.add_argument('--test_path', type=str, default='D:/in_car_command/in_car_command/valid/')
    p.add_argument('--test_corpus_dir', type=str, default='librispeech_alignments/test-clean-100/')
    c = p.parse_args()
    return c

def main(config):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.whisper_model)
    tokenizer = WhisperTokenizer.from_pretrained(config.whisper_model, language=config.whisper_language, task=config.whisper_task)
    processor = WhisperProcessor.from_pretrained(config.whisper_model, language=config.whisper_language, task=config.whisper_task)

    if config.no_dataset:
       raw_dataset = dataset.make_dataset(config.train_path,config.valid_path,config.test_path)
       low_call_voices_prepreocessed = utils.preprocess(raw_dataset)
    else:
        low_call_voices_prepreocessed = load_dataset("Angeriod/in_car_commands_26_prepro")

    model = WhisperForConditionalGeneration.from_pretrained(config.whisper_model)
    data_collator = utils.DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    training_args = Seq2SeqTrainingArguments(
        output_dir= "C:/Users/2469l/whisperft/in_car_commands_26_mdl__base_ver2",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  
        learning_rate=3.75e-5,
        warmup_steps=500,
        num_train_epochs=10,  
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=5000,
        eval_steps=5000,
        logging_steps=200,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",  
        greater_is_better=False,
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=low_call_voices_prepreocessed["train"],
        eval_dataset = low_call_voices_prepreocessed["valid"],  
        data_collator=data_collator,
        compute_metrics=utils.compute_metrics,
        tokenizer=processor.tokenizer,
    )
    #train
    trainer.train()
    
    #evaluation
    eval_results = trainer.evaluate(eval_dataset=low_call_voices_prepreocessed["test"])
    print(f"Test CER: {eval_results['eval_cer']}") 

    #test
    low_call_voices_prepreocessed_test = low_call_voices_prepreocessed["test"].select(range(20))
    predictions = trainer.predict(low_call_voices_prepreocessed_test).predictions
    labels = low_call_voices_prepreocessed_test["labels"]
    pred_str = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

    for pred, label in zip(pred_str, label_str):
        print(f"Prediction: {pred}\nLabel: {label}\n")

if __name__ == "__main__":
    config = define_args()
    main(config=config)
