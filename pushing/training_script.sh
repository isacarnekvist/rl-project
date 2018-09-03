#! /bin/bash

for id in {0..31}
do
    printf -v id_str "%03d" $id;
    store_dir=pushing/trained_agents/pusher_$id_str;
    mkdir $store_dir;
    python pushing/train_teacher_policy.py $store_dir $id_str;
done

