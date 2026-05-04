import os
import re

def parse_power_log(file_path="M1_power_log.txt"):
    # Make sure the file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: Could not find {file_path}. Did you run powermetrics?")
        return

    # Arrays to hold your 300 values
    gpu_power_list = []
    cpu_power_list = []
    package_power_list = []

    print(f"📂 Reading data from {file_path}...")

    # Open the text file and read it line by line
    with open(file_path, 'r') as file:
        for line in file:
            # We use a simple check to find the lines we care about
            if "GPU Power" in line:
                # Extract the numbers from "GPU Power: 1200 mW"
                value = int(re.search(r'\d+', line).group())
                gpu_power_list.append(value)
                
            elif "CPU Power" in line:
                value = int(re.search(r'\d+', line).group())
                cpu_power_list.append(value)
                
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
    print("\n" + "="*45)
    print("      SUSTAINED POWER RESULTS (5 MINUTES)      ")
    print("="*45)
    print(f"Total Samples Collected: {len(package_power_list)}")
    
    # Calculate Averages (Mean)
    avg_gpu = sum(gpu_power_list) / len(gpu_power_list)
    avg_cpu = sum(cpu_power_list) / len(cpu_power_list)
    avg_total = sum(package_power_list) / len(package_power_list)

    # Print in milliWatts (mW) and Watts (W)
    print(f"\nAverage GPU Power:     {avg_gpu:.2f} mW  ({avg_gpu/1000:.2f} W)")
    print(f"Average CPU Power:     {avg_cpu:.2f} mW  ({avg_cpu/1000:.2f} W)")
    print(f"Average Total Package: {avg_total:.2f} mW  ({avg_total/1000:.2f} W)")
    print("="*45 + "\n")

if __name__ == "__main__":
    parse_power_log("M1_power_log.txt") # Make sure this matches your file name