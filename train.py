import pandas as pd
from dataset import load_dataset
from transformers import (AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, VisionEncoderDecoderModel,
                          ViTFeatureExtractor, default_data_collator)

captions_path = "./Flickr8k-Hindi.txt"
root_dir = "../input/flickr8k/Images"

encoder_checkpoint = "google/vit-base-patch16-224"
decoder_checkpoint = "surajp/gpt2-hindi"
output_dir = "./image_captioning_checkpoint"
# load feature extractor and tokenizer
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)

with open(captions_path) as f:
    data = []

    for i in f.readlines():
        sp = i.split(" ")
        data.append([sp[0] + ".jpg", " ".join(sp[1:])])

hindi = pd.DataFrame(data, columns=["images", "text"])
# image file is not present in dir
hindi = hindi[hindi["images"] != "2258277193_586949ec62.jpg"]
train_dataset, val_dataset = load_dataset(hindi, root_dir, tokenizer, feature_extractor)


# initialize a vit-bert from a pretrained ViT and a pretrained GPT2 model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_checkpoint, decoder_checkpoint
)
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = tokenizer.sep_token_id
model.config.max_length = 512
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4
model.decoder.resize_token_embeddings(len(tokenizer))

# freeze the encoder
for param in model.encoder.parameters():
    param.requires_grad = False


training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    overwrite_output_dir=True,
    fp16=True,
    run_name="first_run",
    load_best_model_at_end=True,
    output_dir=output_dir,
    logging_steps=2000,
    save_steps=2000,
    eval_steps=2000,
)


if __name__ == "__main__":
    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
