# GETSITS: Learning Gaussian Embeddings from Temporal Views of Satellite Image Time Series

## 🛠️ Setup
```
cd GETSITS
conda create -n GETSITS python=3.11
conda activate GETSITS
pip install -r requirements.txt
```

## 🏋️ Training

We provide an example of command lines to initialize a training task on a single GPU (multi-GPU usage will depend on the user's setup, but the hyperparameters for the training are not affected, except for the batch size which has to be adapted depending on the number of parallel GPU processes).

Please note:
 - The repo adopts [hydra](https://github.com/facebookresearch/hydra), so you can easily log your experiments and overwrite parameters from the command line. More examples are provided later.
 - To use more gpus or nodes, set `--nnodes` and `--nproc_per_node` correspondingly. Please refer to the [torchrun doc](https://pytorch.org/docs/stable/elastic/run.html).

#### GETSITS Pre-training

```
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate GETSITS

export PYTHONPATH=/home/<USER>/GETSITS:$PYTHONPATH

cd /home/<USER>/GETSITS

torchrun --nnodes=1 --nproc_per_node=1 getsits/run.py --config-name=pretrain \
dataset=ssl4eov1_1 \
task=pretraining \
task.trainer.n_epochs=62 \
task.trainer.log_interval=20 \
encoder=vit_small \
encoder.positional_encoding="geotime" \
decoder=seg_upernet_mt_ltae \
batch_size=328 \
test_batch_size=328 \
preprocessing=pretrain_default \
criterion=lejepa \
optimizer.lr=1e-3 \
optimizer.weight_decay=5e-2 \
lr_scheduler=cosine_annealing
```

#### Multi-Temporal Semantic Segmentation
```
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate GETSITS

export PYTHONPATH=/home/<USER>/GETSITS:$PYTHONPATH

cd /home/<USER>/GETSITS

torchrun --nnodes=1 --nproc_per_node=1 --master_port=$MASTER_PORT cropcon/run.py --config-name=train \
dataset=pastis \
task=downstream \
task.trainer.log_interval=20 \
task.trainer.n_epochs=45 \
encoder=vit_small \
encoder.positional_encoding="geotime" \
decoder=seg_upernet_mt_ltae \
decoder.segmentation=False \
batch_size=4 \
test_batch_size=4 \
preprocessing=seg_resize \
criterion=cross_entropy \
optimizer.lr=1e-3 \
optimizer.weight_decay=5e-2 \
finetune=True \
from_scratch=False \
lr_scheduler=multi_step_lr
```

### Acknowledgment

This repository is built upon the [PANGAEA Benchmark Repository](https://github.com/VMarsocci/pangaea-bench) by Marsocci, V. et al., incorporating substantial modifications. We gratefully acknowledge the foundational contributions of their work, which provided a solid starting point for further development.