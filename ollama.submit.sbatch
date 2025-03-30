#!/bin/bash
#SBATCH -A gen243
#SBATCH -J Image_Classification
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -e ollama_output/ollama_%j.out
#SBATCH -o ollama_output/ollama_%j.out


NODES=1
GPUS=8
CPUS=30

# Only necessary if submitting like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
#unset SLURM_EXPORT_ENV

# Load modules
module load cray-mpich-abi/8.1.28
module load rocm/6.0.0
module load craype-accel-amd-gfx90a
module load rocm/6.0.0


#setting the path of sif files so that it will be easiser for future

# Currently supports both frontier and ODO; need a structure later to have it working on any machine
# /gen007rats/ or /trn025/ links to the storage that is given to us for temporary storage

SIF_DIRECTORY=~/containers
if [[ $MODULEPATH =~ \/sw\/frontier ]]; then
    SIF_DIRECTORY+=/gen007rats/
    SLURM_NAME=slurmd
else
    SIF_DIRECTORY+=/olcf_projects/gen243/
    SLURM_NAME=slurm
fi

export CURRENT_SIF_FILE=$SIF_DIRECTORY/containers/ollama_benchmark.sif



# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

export MPICH_GPU_SUPPORT_ENABLED=1
export BINDS=/usr/share/libdrm,/var/spool/$SLURM_NAME,/opt/cray,${PWD}
export APPTAINERENV_LD_LIBRARY_PATH="/opt/cray/pe/mpich/8.1.28/ofi/crayclang/14.0/lib-abi-mpich:/opt/cray/pe/mpich/8.1.28/gtl/lib:/opt/rocm/lib:/opt/rocm/lib64:$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:/opt/cray/pe/lib64"
export APPTAINERENV_LD_LIBRARY_PATH="/opt/cray/pe/mpich/8.1.28/ofi/crayclang/14.0/lib-abi-mpich:/opt/cray/pe/mpich/8.1.28/gtl/lib:/opt/rocm/lib:/opt/rocm/lib64:$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH:/opt/cray/pe/lib64"

export APPTAINER_CONTAINLIBS="/usr/lib64/libcxi.so.1,/usr/lib64/libjson-c.so.3,/lib64/libtinfo.so.6,/usr/lib64/libnl-3.so.200"
export APPTAINERENV_LD_PRELOAD=/opt/cray/pe/mpich/8.1.28/gtl/lib/libmpi_gtl_hsa.so.0:

echo "Nodes: ${NODES} | GPUS: ${GPUS} | CPUS: ${CPUS}"

srun -N1 -n1   -c1 --gpus-per-task=${GPUS} --gpu-bind=closest  apptainer exec --bind $BINDS --workdir `pwd` --rocm ${CURRENT_SIF_FILE} /bin/ollama serve&
srun -N${NODES} -n1   -c${CPUS} --gpus-per-task=${GPUS} --gpu-bind=closest  apptainer exec --bind $BINDS --workdir `pwd` --rocm ${CURRENT_SIF_FILE} ./ollama_run.sh

echo "Nodes: ${NODES} | GPUS: ${GPUS} | CPUS: ${CPUS}"
