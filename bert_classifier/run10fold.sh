#!/bin/bash
for i in {0..9}
do
  echo "do data $i"
  python run_bert.py --do_data --k_folder $i --data_name $i
  echo "end do data $i"
  echo "cp train and valid file"
  cp pybert/dataset/"$i".train.csv pybert/dataset/new_10fold
  cp pybert/dataset/"$i".valid.csv pybert/dataset/new_10fold
  echo "end cp train and valid file"
  echo "do train $i"
  python run_bert.py --do_train --save_best --data_name $i > "$i".txt 2>&1
  echo "end train $i"
  echo "delete cache"
  rm -rf pybert/dataset/cached_*
  echo "end delete cache"
  echo "cp output file"
  cp "$i.txt" pybert/dataset/new_10fold
  echo "end cp output file"
done