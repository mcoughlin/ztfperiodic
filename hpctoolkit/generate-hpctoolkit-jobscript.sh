#/bin/bash

writeheader_slurm() { cat << _EOF_ > ${jobfile}
#!/bin/bash

#SBATCH -A ${proj}
#SBATCH -q regular
#SBATCH -t ${wtime}
#SBATCH -N ${nnodes}

${clustline}
#SBATCH --tasks-per-node=${ntasks}
#SBATCH --cpus-per-task=${ncpus}
${gpuline}

#SBATCH -J ${jobname}
${emailnotify1}
${emailnotify2}

#SBATCH -o %x.%j

# List modules to load here
module unload darshan
module load hpctoolkit

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=${nthreads}

_EOF_
}

writeheader_lsf() { cat << _EOF_ > ${jobfile}
#!/bin/bash

#BSUB -P ${proj}
#BSUB -W ${wtime}
#BSUB -nnodes ${nnodes}
${allocflags}

#BSUB -J ${jobname}

${emailnotify}

# List modules to load here
module load job-step-viewer
module unload darshan
module load hpctoolkit

_EOF_
}


echo -n "Are you running on (C)ori or (S)ummit? "
read ans
case ${ans} in
  "c" | "cori" | "C" | "Cori" )
    echo -n "Which cluster are you using, (H)aswell, (K)NL, or (G)PU? "
    read ans
    case ${ans} in
      "h" | "haswell" | "H" | "Haswell" )
        export cluster="Cori-Haswell"
	export jobfile="job.cori.cpu.sh"
	export clustline="#SBATCH -C haswell"
        ;;
      "k" | "knl" | "K" | "KNL" )
	export cluster="Cori-KNL"
	export jobfile="job.cori.knl.sh"
	export clustline="#SBATCH -C knl"
	;;
      "g" | "gpu" | "G" | "GPU" )
	export cluster="Cori-GPU"
	export jobfile="job.cori.gpu.sh"
	export clustline="#SBATCH -C gpu"
	;;
    esac
    ;;
  "s" | "summit" | "S" | "Summit" )
    export cluster="Summit"
    export jobfile="job.summit.sh"
    ;;
  default ) echo "Please enter 'c' for Cori or 's' for Summit. Exiting..."; exit 1 ;;
esac

echo -n "Please enter your project ID: "
read proj
if [ "x${proj}" == "x" ]; then
  echo "Project ID cannot be empty. Exiting..."
  exit 2
fi

echo -n "Please enter the name of your executable: "
read exe
if [ "x${exe}" == "x" ]; then
  echo "Executable name cannot be empty. Exiting..."
  exit 3
fi

echo -n "How many nodes do you need? "
read nnodes
if [ ${nnodes} -le 0 ]; then
  echo "Number of nodes must be positive. Exiting..."
  exit 4
fi

case "${cluster}" in
  "Cori-Haswell" | "Cori-KNL" | "Cori-GPU" )
    echo -n "How much time do you need (HH:MM:SS)? " ;;
  "Summit" )
    echo -n "How much time do you need (HH:MM)? " ;;
esac
read wtime

echo -n "Please enter a name for your job: "
read jobname

case "${cluster}" in

  "Cori-Haswell" | "Cori-KNL" )

    echo -n "How many tasks to spawn? "
    read ntasks
    echo -n "How many CPUs per task? "
    read ncpus
    echo -n "How many threads per task? (answer 1 if not using OpenMP) "
    read nthreads

    export parline="srun "
    export serial="srun -N 1 -n 1 "
    ;;

  "Cori-GPU" )

    echo -n "How many tasks to spawn? "
    read ntasks
    echo -n "How many CPUs per task? "
    read ncpus
    echo -n "How many threads per task? (answer 1 if not using OpenMP) "
    read nthreads
    echo -n "How many GPUs per task? "
    read ngpus
    echo -n "Do you need exclusive node access (Y/N)? "
    read ans
    case "${ans}" in
      "y" | "Y" ) export access="--exclusive" ;;
      "n" | "N" ) export access="--shared" ;;
    esac

    export parline="srun ${access} "
    export serial="srun -N 1 -n 1 "
    export gpuline="#SBATCH --gpus-per-task=${ngpus}"
    ;;

  "Summit" )

    echo -n "How many resource sets to spawn? "
    read nrs
    echo -n "How many CPUs per resource set? "
    read ncpus
    echo -n "How many MPI ranks to spawn per resource set? "
    read nranks
    echo -n "How many threads per MPI rank? (answer 1 if not using OpenMP) "
    read nthreads
    if [ ${nthreads} -gt 4 ]; then
      ncpu=$(( ${nthreads} / ${ncpus} ))
      export smtlv=4
      export bpacked="-bpacked:${ncpu}"
    else
      export smtlv=${nthreads}
    fi
    echo -n "How many GPUs per resource set? "
    read ngpus
    echo -n "Enable GPU Multi-Process Service? (Y/N) "
    read ans
    case "${ans}" in
      "y" | "Y" ) export allocflags='#BSUB -alloc_flags "smt4 gpumps"' ;;
      "n" | "N" ) export allocflags='#BSUB -alloc_flags "smt4"' ;;
    esac
    echo -n "Enable GPUDirect? (Y/N) "
    case "${ans}" in
      "y" | "Y" ) export smpiflags='--smpiargs="-gpu"' ;;
    esac

    export parline="jsrun ${smpiflags} -r ${nrs} -c ${ncpus} -a ${nranks} -g ${ngpus} ${bpacked} -E OMP_NUM_THREADS=${smtlv} "
    export serial="jsrun -r 1 -c 1 -a 1 -g 0 "
    ;;

esac

echo "Do you want to be notified when your job finishes?"
echo -n "If yes, please enter your email address, otherwise leave empty: "
read email
if [ "x${email}" != "x" ]; then
  case "${cluster}" in
  "Cori-Haswell" | "Cori-KNL" | "Cori-GPU" )
    export emailnotify1="#SBATCH --mail-type=END"
    export emailnotify2="#SBATCH --mail-user=${email}"
    ;;
  "Summit" )
    export emailnotify="#BSUB -N ${email}"
    ;;
  esac
fi

# Write header
case "${cluster}" in
  "Cori-Haswell" | "Cori-KNL" | "Cori-GPU" ) writeheader_slurm ;;
  "Summit" )                                 writeheader_lsf ;;
esac

case "${cluster}" in
  "Cori-Haswell" | "Cori-KNL" )
    export infoline1="echo \"\`date\` Launching ${exe} with ${ntasks} tasks per node\""
    export infoline2="echo \"Each task contains 1 MPI rank, ${nthreads} threads\""
    export profline="${parline} hpcprof-mpi -S \"${exe}.hpcstruct\" -o \"hpctoolkit-${exe}.d\""
    export tarline="tar cJf \"${cluster}-hpctoolkit-${exe}-profile.tar.xz\" ${jobtitle}.${SLURM_JOB_ID} \"hpctoolkit-${exe}.d\""
    ;;
  "Cori-GPU" )
    export infoline1="echo \"\`date\` Launching ${exe} with ${ntasks} tasks per node\""
    export infoline2="echo \"Each task contains 1 MPI rank, ${nthreads} threads, ${ngpus} GPU(s)\""
    export gpuline1="echo \"\`date\` Analyzing program structure information with hpcstruct (GPU part)\""
    export gpuline2="${serial} hpcstruct --gpucfg no \"hpctoolkit-${exe}.m\""
    export gpuline3="echo \"\`date\` Done\""
    export profline="${parline} hpcprof-mpi -S \"${exe}.hpcstruct\" -o \"hpctoolkit-${exe}.d\" \"hpctoolkit-${exe}.m\""
    export tarline="tar cJf \"${cluster}-hpctoolkit-${exe}-profile.tar.xz\" ${jobtitle}.${SLURM_JOB_ID} \"hpctoolkit-${exe}.d\""
    ;;
  "Summit" )
    export infoline1="echo \"\`date\` Launching ${exe} with ${nrs} resource sets per node\""
    export infoline2="echo \"Each resource set contains ${ncpus} ranks, ${smtlv} threads, ${ngpus} GPU(s)\""
    export tarline="tar cJf \"${cluster}-hpctoolkit-${exe}-profile.tar.xz\" ${jobtitle}.${LSB_JOBID} \"hpctoolkit-${exe}.d\""
    if [ ${ngpus} -gt 0 ]; then
      export gpuline1="echo \"\`date\` Analyzing program structure information with hpcstruct (GPU part)\""
      export gpuline2="${serial} hpcstruct --gpucfg no \"hpctoolkit-${exe}.m\""
      export gpuline3="echo \"\`date\` Done\""
      export profline="${parline} hpcprof-mpi -S \"${exe}.hpcstruct\" -o \"hpctoolkit-${exe}.d\" \"hpctoolkit-${exe}.m\""
    else
      export profline="${parline} hpcprof-mpi -S \"${exe}.hpcstruct\" -o \"hpctoolkit-${exe}.d\""
    fi
    ;;
esac

# Write rest of script
cat << _EOF_ >> ${jobfile}

# Remove previous HPCToolkit profiles
rm -rf "hpctoolkit-${exe}.m" "hpctoolkit-${exe}.d" "${exe}.hpcstruct"

${infoline1}
${infoline2}
# Execute application with hpcrun
${parline} hpcrun -o "hpctoolkit-${exe}.m" -e REALTIME -e gpu=nvidia -t "./${exe}"
echo "\`date\` Done"

echo "\`date\` Analyzing program structure information with hpcstruct (CPU part)"
${serial} hpcstruct -j ${nthreads} -o "${exe}.hpcstruct" ${exe}
echo "\`date\` Done"

${gpuline1}
${gpuline2}
${gpuline3}

echo "\`date\` Analyzing profiles"
${profline}
echo "\`date\` Done"

echo "\`date\` Compressing profile data"
export XZ_DEFAULTS="-T 0"
${tarline}
cp "${cluster}-hpctoolkit-${exe}-profile.tar.xz" ${HOME}/
echo "\`date\` Done"

_EOF_

echo "Successfully written a job script for ${cluster} to ${jobfile}"
echo "Please edit the list of modules to be loaded"
echo "and verify everything is correct before submitting the job."

exit 0
