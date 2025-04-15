#!/bin/bash

# Create directory structure
mkdir -p ./models/checkpoints
mkdir -p ./models/text_encoders

# Download checkpoint models
huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir ./models/checkpoints
huggingface-cli download black-forest-labs/FLUX.1-dev ae.safetensors --local-dir ./models/checkpoints
huggingface-cli download bytedance-research/UNO dit_lora.safetensors --local-dir ./models/checkpoints

# Download text encoder models
huggingface-cli download xlabs-ai/xflux_text_encoders --local-dir ./models/text_encoders --repo-type model
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./models/text_encoders --repo-type model
