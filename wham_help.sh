#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_dir> <out_dir>"
    exit 1
fi
source_dir=$1
dest_dir=$2
mkdir -p "$dest_dir/mix_temp" "$dest_dir/targets" "$dest_dir/refs" "$dest_dir/mix"
find "$source_dir" -type f -name '*-mixed.wav' -exec cp {} "$dest_dir/mix_temp/" \;
find "$source_dir" -type f -name '*-target.wav' -exec cp {} "$dest_dir/targets/" \;
find "$source_dir" -type f -name '*-ref.wav' -exec cp {} "$dest_dir/refs/" \;
