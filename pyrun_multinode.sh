#!/bin/bash

source ~/python-envs/bin/activate
python -W ignore -u ./multinode_train.py 50 10 --master_addr=100.65.6.65 --master_port=3442