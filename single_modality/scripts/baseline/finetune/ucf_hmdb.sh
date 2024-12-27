#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job resubmission on our clusters. 
# ---------------------------------------------------------------------
#SBATCH --job-name=baseline_run
#SBATCH --account=rrg-mpederso
#SBATCH --mem-per-cpu=96G
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --time=0-03:00
#SBATCH -o /home/smuralid/error/tubelets/umt/original_baseline/finetune/ucf_hmdb/slurm-%j.out  # Write the log on scratch
#SBATCH -e /home/smuralid/error/tubelets/umt/original_baseline/finetune/ucf_hmdb/slurm-%j.err

cd $SLURM_TMPDIR
cp -r /project/def-mpederso/smuralid/envs/umt.zip . 
unzip -qq umt.zip
module load StdEnv/2023  intel/2023.2.1
module load python/3.10.13
module load opencv/4.8.0
module load rust/1.76.0
module load cuda/12.2
source umt/bin/activate
mkdir data && cd data
cp -r /project/def-mpederso/smuralid/datasets/ucf_hmdb .
cd ucf_hmdb
unzip -qq ucf101.zip
unzip -qq hmdb51.zip
cd $SLURM_TMPDIR

git clone git@github.com:srikanth-sfu/unmasked_teacher.git
cd unmasked_teacher
git checkout baseline_pretrain
cd single_modality


timeout 176m bash exp/finetuning/ucf_hmdb/b16_ptk710_ftucfhmdb_f8_res224.sh
 
if [ $? -eq 124 ]; then
  echo "The script timed out after ${MAX_HOURS} hour(s). Restarting..."
  # Call the script itself again with the same configuration
  cd $SLURM_SUBMIT_DIR
  sbatch scripts/baseline/finetune/ucf_hmdb.sh 
  # scontrol requeue $SLURM_JOB_ID
else
  echo "Script completed before timeout"
  # Exit or perform any other necessary cleanup
fi

