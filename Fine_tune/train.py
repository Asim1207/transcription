
# Python Built-Ins:
from logging import getLogger
import os
import shutil
import torch
from typing import Optional, Tuple, Any, Dict, List, Union
from dataclasses import dataclass
from functools import partial

# External Dependencies:
import soundfile
from datasets import load_dataset
from datasets import Audio
import boto3
import accelerate
import transformers
from transformers import (
    AutoConfig,
    EarlyStoppingCallback,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    ProcessorMixin,
    Seq2SeqTrainer,
    set_seed,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments
)
logger = getLogger(_name_)
s3 = boto3.resource("s3")


import evaluate
metric = evaluate.load("wer")

# #################PEFT
# from peft import prepare_model_for_int8_training
# from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
# from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
# from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
# #################PEFT

### Declare Model Variables
# model_name_or_path = "openai/whisper-small"
model_name_or_path = "openai/whisper-large-v3"
task = "transcribe"
language = "Malay"
bucket='transcription-malay-tokyo'
# bucket_subfolder = 'transcription_whisper_data/20hours/'

# bucket_subfolder = 'transcription_100_sample/'
# bucket_subfolder = 'transcription_training_data' ##40 hours training data
model_bucket_subfolder = 'transcription_nonquantized_40hours_model/models'

model_dir = '/root/whisper_training_models/non_quantized_40hrs'
dataset_dir = '/root/transcription_dataset/40hours_data'
# dataset_dir = '/root/multi-gpu-whisper/transcription_100_sample'
output_dir = '/root/multi-gpu-whisper/transcription_model/non_quantized_40hrs'

from transformers.trainer_utils import get_last_checkpoint


processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path) #,  device_map="auto"
model.generation_config.language = "ms"
##CAUSING ERROR
# model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")


training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=5e-6, #https://arxiv.org/pdf/2212.04356.pdf / https://github.com/vasistalodagala/whisper-finetune https://huggingface.co/spaces/openai/whisper/discussions/6
    warmup_steps=50,
    max_steps=8000,
    gradient_checkpointing=True,
    fp16=True, #Change to True if there is Cuda or NPU Device
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=1000,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    # metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,
    label_names=["labels"],
)

#### DATA COLLATOR
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def _call_(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

## DATA PREPARATION
def prepare_dataset(
    batch,
    # feature_extractor: WhisperFeatureExtractor,
    # tokenizer: WhisperTokenizer,
):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"]).input_ids

    return batch


#Getting Model
def get_model(model_name_or_path, language, tasks):
    
    ## Declare Feature Extractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    
    ## Declare Tokenizer
    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
    
    ## Declare Processor
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
    
    ## Load Model
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")
    model.generation_config.language = "ms"
    return model, tokenizer, feature_extractor, processor

    
def upload_model_to_s3(bucket_name, s3_folder_path, model_dir):
    s3_client = boto3.client('s3')
    
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_file_path = os.path.join(s3_folder_path, os.path.relpath(local_file_path, model_dir))
            s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
    
    
    
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


## Training
def main() -> Seq2SeqTrainer:
    dataset = (load_dataset("audiofolder", data_dir=dataset_dir, split='train').train_test_split(train_size=0.9, test_size=0.1, seed=42))
    
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=32)
    
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    logger.info("Setting up trainer")

    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    model.config.use_cache = False
    last_checkpoint = None
    if get_last_checkpoint(output_dir) is not None:
        print("Continue from Checkpoint")
        last_checkpoint = get_last_checkpoint(output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    # metrics = train_result.metrics
    # trainer.log_metrics("train", metrics)
    upload_model_to_s3(bucket,model_bucket_subfolder,model_dir)
    
    print(trainer.state.log_history)

    trainer.save_state()
    trainer.save_model(model_dir)
    processor.save_pretrained(os.path.join(model_dir))

    


if _name_ == "_main_":
    main()