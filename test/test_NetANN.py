import csv
import os
import threading
import psutil
# from s_tui.sources.rapl_power_source import RaplPowerSource
import queue
import subprocess

csv_log_path = "/home/wwj/Vector_DB_Acceleration/ref_projects/GPU_FPGA_P2P_Test/eva_logs/NetANN_sift200M_performance.csv"
csv_log_title = ["dataset", "nprobe", "index_type", "processor", "latency/ms", "throughput/ops", "cpu_usage/%", "cpu_power/w"]

dataset = "sift200M"
vec_dim = 128
processor = "NetANN"
index_type = "IVF512,Flat"

class CPU_Monitor(threading.Thread):
    def __init__(self, q, ret_q, ret_q_p):
        super(CPU_Monitor, self).__init__()
        self.q = q
        self.ret_q = ret_q
        self.req_q_p = ret_q_p
        self.cpu_usage = 0
        # self.source = RaplPowerSource()
        self.cpu_power_total = 0
        
    def run(self):
        while self.q.get() == 0:
            self.cpu_usage = psutil.cpu_percent(None)
            # self.source.update()
            # summary = dict(self.source.get_sensors_summary())

            # self.cpu_power_total = str(sum(list(map(float, [summary[key] for key in summary.keys() if key.startswith('package')]))))
        
        self.ret_q.put(self.cpu_usage), self.req_q_p.put(self.cpu_power_total)
        
def main():
    if not os.path.exists(csv_log_path):
        fd = open(csv_log_path, 'w')
        fd.close()
        print("create new csv log file")

    print("open csv log")
    csv_log_file = open(csv_log_path, 'a')
    csv_log_writer = csv.writer(csv_log_file)
    csv_log_writer.writerow(csv_log_title)
    
    for lnprobe in range(4, 8):
        nprobe = 1 << lnprobe
                
        NetANN_Run_Commd = "../vector_search_test -x ../vector_search_kernels.xclbin -d 0" \
                            + " -s " + dataset \
                            + " -c 512" \
                            + " -m "  + str(vec_dim) \
                            + " -p " + str(nprobe) \
                            + " -k 1"
        print(NetANN_Run_Commd)
        
        q = queue.Queue()
        ret_q = queue.Queue()
        req_q_p = queue.Queue()
        cpu_monitor = CPU_Monitor(q, ret_q, req_q_p)
        q.put(0)
        cpu_monitor.start()

        result = subprocess.run(NetANN_Run_Commd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1000000)
        
        c_output = result.stdout
        latency_str = c_output.split('\n')[-2]
        print(latency_str)
        
        latency_str = latency_str.split(' ')[-2]
        latency = float(latency_str)
        throughptu = 1000.0 / latency
            
        q.put(1)
        cpu_monitor.join()
        cpu_usage = ret_q.get()
        cpu_power = req_q_p.get()
        
        csv_log_data = [dataset, nprobe, index_type, processor, latency, throughptu, cpu_usage, cpu_power]
        csv_log_writer.writerow(csv_log_data)

        print("cpu usage:", cpu_usage, "\tcpu power:", cpu_power, "W")
        
if __name__ == "__main__":
    main()
    