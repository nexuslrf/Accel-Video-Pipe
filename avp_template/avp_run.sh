#!/bin/bash
src_dir=$1
exec_name=$2

if [ "$src_dir" = "" ]; then
    echo "Please specify src_dir!"
    exit
fi

source /opt/intel/openvino/bin/setupvars.sh
cd "$src_dir/build"
./$exec_name

