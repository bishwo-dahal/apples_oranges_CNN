#!/bin/bash

source ~/python-envs/bin/activate
python -W ignore -u ./multinode_train.py 50 10 --master_addr=$MASTER_ADDR --master_port=1025