export SLURM_TMPDIR=/storage/smuralidharan
source $SLURM_TMPDIR/envs/umt_taylor7/bin/activate
export CUDA_HOME=/home/ens/smuralidharan/cuda_local
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
git checkout baseline_12
cd single_modality

bash exp/colab_training/ucf_hmdb/b16_ptk710_ftucfhmdb_f8_res224_taylor.sh