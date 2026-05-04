import os
import re

def parse_power_log(file_path="M3_power_log.txt"):
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Could not find {file_path}. Did you run powermetrics?")
        return

    
    gpu_power_list = []
    cpu_power_list = []
    ane_power_list = []  
    package_power_list = []

    print(f"📂 Reading data from {file_path}...")

    
    with open(file_path, 'r') as file:
        for line in file:
            
            if "GPU Power" in line:
                value = int(re.search(r'\d+', line).group())
                gpu_power_list.append(value)
                
            elif "CPU Power" in line:
                value = int(re.search(r'\d+', line).group())
                cpu_power_list.append(value)
                
            elif "ANE Power" in line:  # <-- NEW: Catch the NPU power
                value = int(re.search(r'\d+', line).group())
                ane_power_list.append(value)
                
            elif "Package Power" in line or "Combined Power" in line:
                value = int(re.search(r'\d+', line).group())
                package_power_list.append(value)

    # Check if we actually found data
    if len(package_power_list) == 0:
        print("❌ Error: No power data found in the file. Check your powermetrics output.")
        return

    # ---------------------------------------------------------
    # CALCULATE AND PRINT RESULTS FOR THESIS
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("      SUSTAINED POWER RESULTS (5 MINUTES)      ")
    print("="*50)
    print(f"Total Samples Collected: {len(package_power_list)}")
    
    # Calculate Averages (Mean)
    avg_gpu = sum(gpu_power_list) / len(gpu_power_list) if gpu_power_list else 0
    avg_cpu = sum(cpu_power_list) / len(cpu_power_list) if cpu_power_list else 0
    avg_ane = sum(ane_power_list) / len(ane_power_list) if ane_power_list else 0  # <-- NEW: Calculate ANE average
    avg_total = sum(package_power_list) / len(package_power_list) if package_power_list else 0

    # Print in milliWatts (mW) and Watts (W)
    print(f"\nAverage CPU Power:     {avg_cpu:.2f} mW  ({avg_cpu/1000:.2f} W)")
    print(f"Average GPU Power:     {avg_gpu:.2f} mW  ({avg_gpu/1000:.2f} W)")
    print(f"Average NPU Power:     {avg_ane:.2f} mW  ({avg_ane/1000:.2f} W)") # <-- NEW: Print NPU average
    print("-" * 50)
    print(f"Average Total Package: {avg_total:.2f} mW  ({avg_total/1000:.2f} W)")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Make sure this matches the file name you used in Terminal 2!
    parse_power_log("/Users/israanhayle/Thesis/src/M3_power_log.txt")