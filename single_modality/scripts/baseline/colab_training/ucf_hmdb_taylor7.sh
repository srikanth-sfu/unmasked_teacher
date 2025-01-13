#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job resubmission on our clusters. 
# ---------------------------------------------------------------------
#SBATCH --job-name=baseline_colabtrain
#SBATCH --account=rrg-mpederso
#SBATCH --mem-per-cpu=96G
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --time=0-03:00
#SBATCH -o /home/smuralid/error/tubelets/umt_tubelet/colabtrainonly/ucf_hmdb/slurm-%j.out  # Write the log on scratch
#SBATCH -e /home/smuralid/error/tubelets/umt_tubelet/colabtrainonly/ucf_hmdb/slurm-%j.err

export SLURM_TMPDIR=/storage/smuralidharan
source $SLURM_TMPDIR/envs/umt/bin/activate
mkdir data && cd data
cd $SLURM_TMPDIR/data/ucf_hmdb
unzip -qq ucf101.zip
unzip -qq hmdb51.zip
cd $SLURM_TMPDIR

git clone git@github.com:srikanth-sfu/unmasked_teacher.git
cd unmasked_teacher
git checkout tubelet_umt_colabtrainingonly
cd single_modality


timeout 176m bash exp/colab_training/ucf_hmdb/b16_ptk710_ftucfhmdb_f8_res224_taylor.sh
 
if [ $? -eq 124 ]; then
  echo "The script timed out after ${MAX_HOURS} hour(s). Restarting..."
  # Call the script itself again with the same configuration
  cd $SLURM_SUBMIT_DIR
  sbatch scripts/baseline/colab_training/ucf_hmdb.sh 
  # scontrol requeue $SLURM_JOB_ID
else
  echo "Script completed before timeout"
  # Exit or perform any other necessary cleanup
fi

