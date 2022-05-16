#!/bin/bash
for i in {3..9}
do
  echo "do train $i"
  python run_bert.py --do_train --save_best --epochs 70 --resume_path "pybert/output/checkpoints/best" --learning_rate 1e-6  --train_batch_size 8  --eval_batch_size 8 --data_name $i > "$i"_transfer_2.txt 2>&1
  echo "end train $i"
  echo "delete cache"
  rm -rf pybert/dataset/cached_*
  echo "end delete cache"
  echo "cp output file"
  cp "$i"_transfer.txt pybert/dataset/new_10fold/transfer
  echo "end cp output file"
done