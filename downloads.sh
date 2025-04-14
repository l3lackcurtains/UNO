# Create models directory if it doesn't exist
mkdir -p ./models

# Download required models
huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir ./models
huggingface-cli download black-forest-labs/FLUX.1-dev ae.safetensors --local-dir ./models
huggingface-cli download xlabs-ai/xflux_text_encoders --local-dir ./models --repo-type model
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./models --repo-type model
huggingface-cli download bytedance-research/UNO dit_lora.safetensors --local-dir ./models
