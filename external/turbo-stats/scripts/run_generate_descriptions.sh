#!/bin/bash

source /home/jovyan/zoloev-city/gigachat/source/llm-foundry/llmfoundry-venv/bin/activate


CONFIG=gender_descriptions.yaml

# python generate_descriptions.py \
#     --config-dir configs \
#     --config-name $CONFIG

accelerate launch inference.py \
    --config-dir configs \
    --config-name $CONFIG
