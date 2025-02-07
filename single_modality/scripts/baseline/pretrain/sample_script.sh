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
#SBATCH -o /home/smuralid/error/tubelets/umt/original_baseline/pretrain/dailyda/k600/slurm-%j.out  # Write the log on scratch
#SBATCH -e /home/smuralid/error/tubelets/umt/riginal_baseline/pretrain/dailyda/k600/slurm-%j.err


#Always create and store your datasets as zip file in /project/def-/your_userid or /project/rrg-/your_userid
#to run experiments, I copy everytime these zip files to $SLURM_TMPDIR after I obtain a job.
cd $SLURM_TMPDIR
# create env using virtualenv, load modules using "module"
# https://docs.alliancecan.ca/wiki/Python#Creating_and_using_a_virtual_environment explains how to create env using virtualenv
# I create python envs once, zip it and store it in under my /project/.../envs folder as well
cp -r /project/def-mpederso/smuralid/envs/umt.zip . 
unzip -qq umt.zip
module load StdEnv/2023 gcc/12.3 cuda/12.2 opencv/4.9.0 python/3.10.13
module load rust/1.76.0
source umt/bin/activate
mkdir data && cd data

# copying datasets -- an illustration here
cp -r /project/def-mpederso/smuralid/datasets/kinetics600.zip .
cp /project/def-mpederso/smuralid/datasets/Daily-DA/ARID_v1_5_211015.zip .
unzip -qq kinetics600.zip
unzip -qq ARID_v1_5_211015.zip
mv clips_v1.5 arid
cd $SLURM_TMPDIR

# my codes
git clone git@github.com:srikanth-sfu/unmasked_teacher.git
cd unmasked_teacher
git checkout tubelet_umt_s12
cd single_modality

# running scripts
timeout 170m bash exp/pretraining/tubelet_b16_ptk600_f8_res224.sh
 
if [ $? -eq 124 ]; then
  echo "The script timed out after ${MAX_HOURS} hour(s). Restarting..."
  # Call the script itself again with the same configuration
  cd $SLURM_SUBMIT_DIR
  
  sbatch scripts/baseline/pretrain/kinetics_dailyda.sh 
  # scontrol requeue $SLURM_JOB_ID
else
  echo "Script completed before timeout"
  # Exit or perform any other necessary cleanup
fi

