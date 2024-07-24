traj="$1"
pdb="$2"
outputfile="$3"
/home/windyer/software/PCASSO/bin/pcasso -trj $traj $pdb > $outputfile
