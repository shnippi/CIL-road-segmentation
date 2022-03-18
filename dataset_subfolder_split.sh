#!/bin/sh

files_per_dir=1000

#set -- *.png

printf 'There are %d files\n' "$#"
printf 'Putting %d files in each new directory\n' "$files_per_dir"

#while getopts "p:h" option
#do
#    case "${option}" in
#        p) 
#            echo "hello"
#            path=${OPTARG}
#            ;;
#        h)
#            echo "help"
#            ;;
#        *)
#            echo "*"
#            ;;
#    esac
#done
#shift $((OPTIND-1))
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
