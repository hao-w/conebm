#!/bin/bash
source ~/dev/bin/activate
python train/train_cebm.py --device=cuda:0
