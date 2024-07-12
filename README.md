# Apple vs Orange Detection



### This NN uses datasets obtained from UC Berkeley.  [Follow this link to see original datasets](https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/)


### To make it easier to run tests, we can use following commands
```
srun -n1  --gpus-per-task=1 -c8 ./pyrun\ baremetal.sh >> output/classification_8CPU_native.out && 
srun -n1  --gpus-per-task=2 -c8 ./pyrun\ baremetal.sh >> output/classification_8CPU_native.out &&  
srun -n1  --gpus-per-task=3 -c8 ./pyrun\ baremetal.sh >> output/classification_8CPU_native.out && 
srun -n1  --gpus-per-task=4 -c8 ./pyrun\ baremetal.sh >> output/classification_8CPU_native.out &&  
srun -n1  --gpus-per-task=5 -c8 ./pyrun\ baremetal.sh >> output/classification_8CPU_native.out && 
srun -n1  --gpus-per-task=6 -c8 ./pyrun\ baremetal.sh >> output/classification_8CPU_native.out && 
srun -n1  --gpus-per-task=7 -c8 ./pyrun\ baremetal.sh >> output/classification_8CPU_native.out && 
srun -n1  --gpus-per-task=8 -c8 ./pyrun\ baremetal.sh >> output/classification_8CPU_native.out 

srun -n1  --gpus-per-task=8 -c1 ./pyrun\ baremetal.sh >> output/classification_slow_native.out && 
srun -n1  --gpus-per-task=8 -c2 ./pyrun\ baremetal.sh >> output/classification_slow_native.out &&  
srun -n1  --gpus-per-task=8 -c3 ./pyrun\ baremetal.sh >> output/classification_slow_native.out
```

```
srun -n1  --gpus-per-task=8 -c4 ./pyrun\ baremetal.sh >> output/classification_slow_2_native.out &&  
srun -n1  --gpus-per-task=8 -c5 ./pyrun\ baremetal.sh >> output/classification_slow_2_native.out && 
srun -n1  --gpus-per-task=8 -c6 ./pyrun\ baremetal.sh >> output/classification_slow_2_native.out && 
srun -n1  --gpus-per-task=8 -c7 ./pyrun\ baremetal.sh >> output/classification_slow_2_native.out && 
srun -n1  --gpus-per-task=8 -c8 ./pyrun\ baremetal.sh >> output/classification_slow_2_native.out 
```
