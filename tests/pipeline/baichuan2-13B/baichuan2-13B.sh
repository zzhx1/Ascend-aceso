#!/bin/bash
# Provide uniform access for piepline.

python tests/pipeline/baichuan2-13B/test_process_pretrain_data.py
pytest -s tests/pipeline/baichuan2-13B/test_generation.py
pytest -s tests/pipeline/baichuan2-13B/test_evaluation.py
