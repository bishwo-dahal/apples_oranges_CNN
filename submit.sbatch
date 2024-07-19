#!/bin/bash
#SBATCH -A trn025
#SBATCH -J Image_Classification
#SBATCH -t 11:00:00
#SBATCH -p batch
#SBATCH -N 5
#SBATCH -e output/classification_%j.out
#SBATCH -o output/classification_%j.out


NODES=5
GPUS=8
CPUS=48

# Only necessary if submitting like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
#unset SLURM_EXPORT_ENV

# Load modules
module load amd-mixed/5.7.1
module load cray-mpich-abi/8.1.26
module load craype-accel-amd-gfx90a

#setting the path of sif files so that it will be easiser for future
export CURRENT_SIF_FILE=/ccsopen/home/bishwodahal/containers/olcf_projects/trn025/containers/opensusempich342rocm571pytorch2.sif

# Get address of head node
ips=`hostname -I`
read -ra arr <<< ${ips}
export MASTER_ADDR=${arr[0]}
echo "MASTER_ADDR=" $MASTER_ADDR

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

export MPICH_GPU_SUPPORT_ENABLED=1
export BINDS=/usr/share/libdrm,/var/spool/slurm,/opt/cray,${PWD}
export APPTAINERENV_LD_LIBRARY_PATH="/opt/cray/pe/mpich/8.1.26/ofi/crayclang/14.0/lib-abi-mpich:/opt/cray/pe/mpich/8.1.26/gtl/lib:/opt/rocm/lib:/opt/rocm/lib64:$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:/opt/cray/pe/lib64"
export APPTAINERENV_LD_LIBRARY_PATH="/opt/cray/pe/mpich/8.1.26/ofi/crayclang/14.0/lib-abi-mpich:/opt/cray/pe/mpich/8.1.26/gtl/lib:/opt/rocm/lib:/opt/rocm/lib64:$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:/opt/cray/pe/lib64"

export APPTAINER_CONTAINLIBS="/usr/lib64/libcxi.so.1,/usr/lib64/libjson-c.so.3,/lib64/libtinfo.so.6,/usr/lib64/libnl-3.so.200"
export APPTAINERENV_LD_PRELOAD=/opt/cray/pe/mpich/8.1.26/gtl/lib/libmpi_gtl_hsa.so.0:

echo "Nodes: ${NODES} | GPUS: ${GPUS} | CPUS: ${CPUS}"

# Run script
srun -N${NODES} -n5   -c${CPUS} --gpus-per-task=${GPUS} --gpu-bind=closest  apptainer exec --bind $BINDS --workdir `pwd` --rocm ${CURRENT_SIF_FILE} ./pyrun.sh
#srun -N1 apptainer exec --bind $BINDS --workdir `pwd` --rocm  ../containers/opensusempich342rocm571pytorch2.sif ldd /opt/cray/pe/mpich/8.1.26/ofi/crayclang/14.0/lib-abi-mpich//libmpi.so.12 

echo "Nodes: ${NODES} | GPUS: ${GPUS} | CPUS: ${CPUS}"
