# root_dir=/datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1
# output_root_dir=/work/outputs
# for env_ in Test Train; do
#   for dir in $(ls -d $root_dir/$env_/*); do
#     output=$output_root_dir/$env_/$(basename $dir)
#     mkdir -p $output
#     python masking.py $dir $output
#   done
# done

root_dir=/datasets/shanghaitech
output_root_dir=/work/outputs/shanghaitech
for env_ in training testing; do
  for dir in $(ls -d $root_dir/$env_/frames/*); do
    output=$output_root_dir/$env_/$(basename $dir)
    mkdir -p $output
    python masking.py $dir $output
    break
  done
  break
done
