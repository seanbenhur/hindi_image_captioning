# Hindi Image Captioning Model

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/team-indain-image-caption/Hindi-image-captioning)

This is an encoder-decoder image captioning model made with [VIT](https://huggingface.co/google/vit-base-patch16-224-in21k) encoder and [GPT2-Hindi](https://huggingface.co/surajp/gpt2-hindi) as a decoder. This is a first attempt at using ViT + GPT2-Hindi for a Hindi image captioning task. We used the Flickr8k Hindi Dataset available on kaggle to train the model.

This model was trained using HuggingFace course community week, organized by Huggingface. The pretrained weights are available [here](https://huggingface.co/team-indain-image-caption/hindi-image-captioning)

## How to use

Here is how to use this model to caption an image of the Flickr8k dataset:
```python
import torch
import requests
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, \
                         VisionEncoderDecoderModel

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

url = 'https://shorturl.at/fvxEQ'
image = Image.open(requests.get(url, stream=True).raw)

encoder_checkpoint = 'google/vit-base-patch16-224'
decoder_checkpoint = 'surajp/gpt2-hindi'
model_checkpoint = 'team-indain-image-caption/hindi-image-captioning'
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)

#Inference
sample = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]

caption_ids = model.generate(sample, max_length = 50)[0]
caption_text = clean_text(tokenizer.decode(caption_ids))
print(caption_text)
```

## Training data
We used the Flickr8k Hindi Dataset, which is the translated version of the original Flickr8k Dataset, available on [Kaggle](https://www.kaggle.com/bhushanpatilnew/hindi-caption) to train the model.

## Training procedure
This model was trained during HuggingFace course community week, organized by Huggingface. The training was done on Kaggle GPU.

## Training Parameters
- epochs = 8,
- batch_size = 8,
- Mixed Precision Enabled

## Team Members
- [Sean Benhur](https://www.linkedin.com/in/seanbenhur/)
- [Herumb Shandilya](https://www.linkedin.com/in/herumb-s-740163131/)
