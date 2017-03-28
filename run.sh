#!/bin/bash
#
# Launches a docker container using our image, and runs torch
#

gpu=$1
shift

DIR=/project

# lua files to map as readonly volumes:
# (note that read-write file volumes dont play well with editors)
declare -a CODE_FILES
for fn in `pwd`/*.lua ; do
    CODE_FILES+=("--volume=$fn:$DIR/$(basename $fn):ro")
done
for fn in `pwd`/*.py ; do
    CODE_FILES+=("--volume=$fn:$DIR/$(basename $fn):ro")
done

NV_GPU="$gpu" nvidia-docker run --rm -ti ${CODE_FILES[@]} \
        -v `pwd`/data:$DIR/data:ro \
        -v `pwd`/data_subs:$DIR/data_subs:ro \
        -v `pwd`/exp:$DIR/exp:ro \
        -v `pwd`/modules:$DIR/modules:ro \
        -v `pwd`/results:$DIR/results:rw \
        -v `pwd`/util:$DIR/util:ro \
        -t $USER/lipnet \
        $@
