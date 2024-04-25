/home/wxr/anaconda3/bin/python /home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/src/python_scripts/reorg_data.py \
    --source_file /home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/NetANN_Vector_Datasets/sift200M/sift200M_128dim_xbVec_features.dat \
    --vector_dim 128 \
    --ivf_index_file /home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/NetANN_Vector_Datasets/sift200M/sift200M_512_invlists_128dim_indexs_FIXED.csv \
    --nlist 512 \
    --target_index_map /home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/NetANN_Vector_Datasets/sift200M/sift200M_512_128dim_reorg_indexmap.dat \
    --target_cluster_nav /home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/NetANN_Vector_Datasets/sift200M/sift200M_512_128dim_reorg_cluster_nav.dat \
    --target_data /home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/NetANN_Vector_Datasets/sift200M/sift200M_512_128dim_xbVec_features_reorg.dat