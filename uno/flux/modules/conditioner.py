# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import Tensor, nn
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5Tokenizer)
import os

class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = "clip" in version.lower()
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        # Default model IDs
        CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
        T5_MODEL_ID = "xlabs-ai/xflux_text_encoders"

        # Always use the default model IDs for loading
        if self.is_clip:
            self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID, max_length=max_length, **hf_kwargs)
            self.transformer = CLIPTextModel.from_pretrained(CLIP_MODEL_ID, **hf_kwargs)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_ID, max_length=max_length, **hf_kwargs)
            self.transformer = T5EncoderModel.from_pretrained(T5_MODEL_ID, **hf_kwargs)

        # If a local path is provided, load the weights from it
        if os.path.exists(version):
            if os.path.isdir(version):
                weights_path = os.path.join(version, "pytorch_model.bin")
            else:
                weights_path = version
            
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location="cpu")
                self.transformer.load_state_dict(state_dict)
            else:
                print(f"Warning: Could not find weights file at {weights_path}")

        self.transformer = self.transformer.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.transformer.device) for k, v in tokens.items()}
        return self.transformer(**tokens)[self.output_key]
