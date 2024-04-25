/home/wxr/anaconda3/bin/python /home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/src/python_scripts/reorg_data.py \
    --source_file /home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/NetANN_Vector_Datasets/gist/gist_960dim_xbVec_features.dat \
    --vector_dim 960 \
    --ivf_index_file /home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/NetANN_Vector_Datasets/gist/gist_512_invlists_960dim_indexs_FIXED.csv \
    --nlist 507 \
    --target_index_map /home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/NetANN_Vector_Datasets/gist/gist_512_960dim_reorg_indexmap.dat \
    --target_cluster_nav /home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/NetANN_Vector_Datasets/gist/gist_512_960dim_reorg_cluster_nav.dat \
    --target_data /home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/NetANN_Vector_Datasets/gist/gist_512_960dim_xbVec_features_reorg.dat