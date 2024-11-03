#!/bin/bash

eval "$(conda shell.bash hook)"
# conda deactivate
conda activate slt

python -c "import lightning"