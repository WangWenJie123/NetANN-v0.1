config_interface -m_axi_auto_max_ports=true
config_interface -m_axi_max_bitwidth=1024
config_interface -m_axi_flush_mode=true
config_dataflow -default_channel fifo
config_dataflow -disable_fifo_sizing_opt