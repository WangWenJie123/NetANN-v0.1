/home/wxr/anaconda3/bin/python /home/wxr/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/python_scripts/reorg_data.py \
    --source_file /home/wxr/vector_dataset/sift200M/sift200M/sift200M_128dim_xbVec_features.dat \
    --vector_dim 128 \
    --ivf_index_file /home/wxr/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/fixed_ivflist/sift200M/sift200M_1024_invlists_128dim_indexs_FIXED.csv \
    --nlist 1024 \
    --target_index_map /home/wxr/vector_dataset/sift200M/sift200M/sift200M_1024_128dim_reorg_indexmap.dat \
    --target_cluster_nav /home/wxr/vector_dataset/sift200M/sift200M/sift200M_1024_128dim_reorg_cluster_nav.dat \
    --target_data /home/wxr/vector_dataset/sift200M/sift200M/sift200M_1024_128dim_xbVec_features_reorg.dat