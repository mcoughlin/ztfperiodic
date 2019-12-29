#!/bin/bash

# redojobs alljobs_list missedjobs_list redo_list
# alljobs_list .. file which lists all jobs
# missedjobs_list .. file which lists missed jobs
# redo_list .. list of jobs which need to be re-executed

# usage
# redojobs condor.sh condor.dag redo_list
#
cat "$1" | nl -n ln -s " " > alljobs
sed -n '/^#.*<ENDLIST>$/{s/,<ENDLIST>$//p;}' "$2" | tr ',' '\n' | awk
'{print $1+1}' > missedjobs
join <(sort missedjobs) <(sort alljobs) > $3

printf "number of jobs %d\n" $(wc -l <alljobs)
printf "numer of missed jobs %d\n" $(wc -l < missedjobs)
printf "redojobs file: %s\n" $3

rm alljobs missedjobs
