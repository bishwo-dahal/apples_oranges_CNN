#!/bin/bash
#SBATCH -A trn025
#SBATCH -J ml_test_ODO
#SBATCH -o ODO_apptainer_mltests.%j
#SBATCH -e ODO_apptainer_mltests.%j
#SBATCH -t 00:05:00
#SBATCH -p batch
#SBATCH -N 2

# Only necessary if submitting like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
#unset SLURM_EXPORT_ENV

# Load modules
# module load amd-mixed/5.7.1
module load PrgEnv-cray/8.3.3
module load amd-mixed/5.3.0
module load craype-accel-amd-gfx90a



# Get address of head node
ips=`hostname -I`
read -ra arr <<< ${ips}
export MASTER_ADDR=${arr[0]}
echo "MASTER_ADDR=" $MASTER_ADDR
export CONTAINER=/ccsopen/home/bishwodahal/containers/olcf_projects/trn025/containers//opensusempich342rocm571pytorch2.sif

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

#export MPICH_GPU_SUPPORT_ENABLED=1
export BINDS=/usr/share/libdrm,/var/spool/slurm,/opt/cray,${PWD}
export APPTAINERENV_LD_LIBRARY_PATH="/opt/cray/pe/mpich/8.1.26/ofi/crayclang/14.0/lib-abi-mpich:/opt/cray/pe/mpich/8.1.26/gtl/lib:/opt/rocm/lib:/opt/rocm/lib64:$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:/opt/cray/pe/lib64"
export APPTAINERENV_LD_LIBRARY_PATH="/opt/cray/pe/mpich/8.1.26/ofi/crayclang/14.0/lib-abi-mpich:/opt/cray/pe/mpich/8.1.26/gtl/lib:/opt/rocm/lib:/opt/rocm/lib64:$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:/opt/cray/pe/lib64"

export APPTAINER_CONTAINLIBS="/usr/lib64/libcxi.so.1,/usr/lib64/libjson-c.so.3,/lib64/libtinfo.so.6,/usr/lib64/libnl-3.so.200"
export APPTAINERENV_LD_PRELOAD=/opt/cray/pe/mpich/8.1.26/gtl/lib/libmpi_gtl_hsa.so.0:

rm snapshot.pt

# Run script
#srun -N2 --tasks-per-node=8 -c7 --gpus-per-task=1 --gpu-bind=closest apptainer exec --bind $BINDS --workdir `pwd` --rocm ${CONTAINER} ./pyrun.sh
srun -N2 --tasks-per-node=8 -c7 --gpus-per-task=1 --gpu-bind=closest ./pyrun_multinode.sh
#srun -N1 apptainer exec --bind $BINDS --workdir `pwd` --rocm  ../containers/opensusempich342rocm571pytorch2.sif ldd /opt/cray/pe/mpich/8.1.26/ofi/crayclang/14.0/lib-abi-mpich//libmpi.so.12 

