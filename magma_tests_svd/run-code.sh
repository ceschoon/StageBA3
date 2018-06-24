#! /bin/bash

echo "N,gpu_time,cpu_time,difference" >> out
./dgesvd 512 >> out
./dgesvd 753 >> out
./dgesvd 1024 >> out
./dgesvd 1359 >> out
./dgesvd 2048 >> out
./dgesvd 3356 >> out
./dgesvd 4096 >> out
./dgesvd 6122 >> out
./dgesvd 8192 >> out
mv out out-dgesvd-ondemand.csv