#!/bin/bash

NODES_STEP=1
GPUS=8
CPUS=7
NUM_TESTS=3

read -p "Number of starting Nodes: " NODES_START
read -p "Number of ending Nodes: " NODES_END
read -p "Steps for Nodes: " NODES_STEP

read -p "Skip remaining (y/n)??? " skip

if [ $skip = "y" ]  || [ $skip = "Y" ]; then
    echo "Number of GPUs: "
fi

echo "Starting tests......... "

CURR_NODES=1
while (( $CURR_NODES<=$NODES_END ));
do
    cat sbatch.header > current_script.sbatch
    echo "#SBATCH -N $CURR_NODES" >> current_script.sbatch  
    cat sbatch.footer >> current_script.sbatch
    chmod u+x current_script.sbatch
    for((j=0; j <3; j++));
    do 
        sbatch current_script.sbatch $CURR_NODES $GPUS $CPUS  
    done
    
    if [[ NODES_STEP -lt 0 ]]; then
        CURR_NODES=$((CURR_NODES*2))
    else
        CURR_NODES=$((CURR_NODES+1))
    fi
    
done

# Remove the script after using
rm current_script.sbatch

