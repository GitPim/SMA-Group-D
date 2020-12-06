#!/bin/bash

#SBATCH --job-name=Oort_MWG_Tack
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="s.bionicle@hotmail.nl"
#SBATCH --mail-type="ALL"

#SBATCH --partition="cpu-long"
#SBATCH --time=60:00:00
#SBATCH --tasks=24
#SBATCH --mem=10G

echo "#### Starting Test"
echo "This is $SLURM_JOB_USER and my job has the ID $SLURM_JOB_ID"
# get the current working directory
CWD=$(pwd)
echo "This job was submitted from $SLURM_SUBMIT_DIR and I am currently in $CWD"
# get the current time and date
DATE=$(date)
echo "It is now $DATE"
echo "Hello World from $HOSTNAME"
echo "#### Finished Test. Have a nice day"

module load AMUSE/13.1.0-foss-2018a-Python-3.6.4

export CWD=$(pwd)
echo "[$SHELL] CWD: "$CWD

export PATH_TO_PYFILE=$CWD
echo "[$SHELL] Path of python file: "$PATH_TO_PYFILE

export PYFILE=Oort_MWG_Tack.py


# Create a directory of local scratch on the node
echo "[$SHELL] Node scratch: "$SCRATCH
export RUNDIR=$SCRATCH/Vanilla_test
mkdir $RUNDIR
echo "[$SHELL] Run directory"$RUNDIR

# Create directory for plots
export RESULTDIR=$RUNDIR/results_vanilla
mkdir $RESULTDIR

# copy script to local scratch directory and change into it
cp $CWD/$PYFILE $RUNDIR/
cd $RUNDIR

# Run the file
echo "[$SHELL] Run script"
python3 $PYFILE
echo "[$SHELL] Script finished"

echo "[$SHELL] Copy files back to cwd"
cp -r $RESULTDIR $CWD/

echo "[$SHELL] #### Finished Vanilla test. Have a nice day
