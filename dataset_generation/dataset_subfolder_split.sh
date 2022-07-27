#!/bin/sh

files_per_dir=1000
printf 'There are %d files\n' "$#"
printf 'Putting %d files in each new directory\n' "$files_per_dir"

path="$1"
echo "path: $path"
N=0 # directory counter
n=0 # file counter
cd $path

for filename in *
do
    if [ "$(( n % files_per_dir ))" -eq 0 ]; then
        dir="$N"
        mkdir "$dir"
        N=$(( N + 1 ))
    fi
    mv "$filename" "$dir"
    n=$(( n + 1 ))
done
