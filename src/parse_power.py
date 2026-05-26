import re

LOG_FILE = "/Users/israanhayle/11N-KD-640pxpower_log.txt"

RUNS = [
    (666, 10.0),  # Run 1
    (697, 10.0),  # Run 2
    (695, 10.0),  # Run 3
    (694, 10.0),  # Run 4
    (693, 10.0),  # Run 5
]

with open(LOG_FILE, "r") as f:
    content = f.read()

cpu_vals = [float(v) for v in re.findall(r"CPU Power:\s+([\d.]+)\s+mW", content)]
gpu_vals = [float(v) for v in re.findall(r"GPU Power:\s+([\d.]+)\s+mW", content)]
ane_vals = [float(v) for v in re.findall(r"ANE Power:\s+([\d.]+)\s+mW", content)]

# GPU Power appears twice per sample (once in processor section, once in GPU section) — keep unique count
n = min(len(cpu_vals), len(ane_vals))
gpu_vals = gpu_vals[:n]
cpu_vals = cpu_vals[:n]
ane_vals = ane_vals[:n]

avg_cpu = sum(cpu_vals) / len(cpu_vals)
avg_gpu = sum(gpu_vals) / len(gpu_vals)
avg_ane = sum(ane_vals) / len(ane_vals)
avg_total = avg_cpu + avg_gpu + avg_ane

print(f"Power samples : {n}")
print(f"{'Component':<10} {'Avg (mW)':>10} {'Min (mW)':>10} {'Max (mW)':>10}")
print(f"{'-'*42}")
print(f"{'CPU':<10} {avg_cpu:>10.1f} {min(cpu_vals):>10.1f} {max(cpu_vals):>10.1f}")
print(f"{'GPU':<10} {avg_gpu:>10.1f} {min(gpu_vals):>10.1f} {max(gpu_vals):>10.1f}")
print(f"{'ANE':<10} {avg_ane:>10.1f} {min(ane_vals):>10.1f} {max(ane_vals):>10.1f}")
print(f"{'TOTAL':<10} {avg_total:>10.1f}")
print()

energy_per_inf = []
for i, (count, duration) in enumerate(RUNS):
    if count > 0:
        uj = (avg_total * duration * 1000) / count
        energy_per_inf.append(uj)
        print(f"Run {i+1}: {count} iters, {duration:.1f}s → {uj:.1f} µJ/inference")

if energy_per_inf:
    energy_per_inf.sort()
    median = energy_per_inf[len(energy_per_inf)//2]
    print(f"\nMedian µJ/inference : {median:.1f} µJ")
