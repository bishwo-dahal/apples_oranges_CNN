#!/bin/bash

source ~/python-envs/bin/activate

python -W ignore -u ./train.py 2000 10 --master_addr=$MASTER_ADDR --master_port=3442

