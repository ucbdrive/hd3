#!/bin/bash
SAVE_DIR=model_zoo
mkdir -p ${SAVE_DIR}
webroot='dl.yf.io/hd3/models/'
model_names=(
hd3f_chairs-04bf114d.pth
hd3f_chairs_things-462a3896.pth
hd3f_chairs_things_kitti-41b15827.pth
hd3f_chairs_things_sintel-5b4ad51a.pth
hd3fc_chairs-1367436d.pth
hd3fc_chairs_things-0b92a7f5.pth
hd3fc_chairs_things_kitti-bfa97911.pth
hd3fc_chairs_things_sintel-0be17c83.pth
hd3s_things-8b4dcd6d.pth
hd3s_things_kitti-1243813e.pth
hd3sc_things-57947496.pth
hd3sc_things_kitti-368975c0.pth
)

for i in ${model_names[@]}; do
	model_url=${webroot}$i
	out_file=${SAVE_DIR}/$i
	echo "Downloading: "$model_url
	wget -N $model_url -O $out_file
done
