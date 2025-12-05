import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mm_results.csv")

COLOR = {
    "basic": "tab:blue",
    "parallel": "tab:orange",
    "numpy": "tab:green",
}

# ---------- 1) Tiempo vs tamaño ----------
plt.figure(figsize=(8, 5))
for version in ["basic", "parallel", "numpy"]:
    sub = df[df["version"] == version]
    if version == "parallel":
        max_p = sub["processes"].max()
        sub = sub[sub["processes"] == max_p]
        label = f"{version} (proc={max_p})"
    else:
        label = version
    sub = sub.sort_values("size")
    plt.plot(sub["size"], sub["time_s"], marker="o",
             color=COLOR[version], label=label)

plt.xlabel("Matrix size (n x n)")
plt.ylabel("Time (s)")
plt.title("Execution time vs size")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("time_vs_size.png", dpi=300)

# ---------- 2) Speedup vs tamaño ----------
plt.figure(figsize=(8, 5))

sub_par = df[df["version"] == "parallel"]
max_p = sub_par["processes"].max()
sub_par = sub_par[sub_par["processes"] == max_p].sort_values("size")
plt.plot(sub_par["size"], sub_par["speedup"], marker="o",
         color=COLOR["parallel"],
         label=f"parallel (proc={max_p})")

sub_np = df[df["version"] == "numpy"].sort_values("size")
plt.plot(sub_np["size"], sub_np["speedup"], marker="o",
         color=COLOR["numpy"], label="numpy")

plt.xlabel("Matrix size (n x n)")
plt.ylabel("Speedup vs basic")
plt.title("Speedup vs size")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("speedup_vs_size.png", dpi=300)

# ---------- 3) Eficiencia vs tamaño ----------
plt.figure(figsize=(8, 5))
for p in sorted(df[df["version"] == "parallel"]["processes"].unique()):
    sub = df[(df["version"] == "parallel") & (df["processes"] == p)]
    sub = sub.sort_values("size")
    plt.plot(sub["size"], sub["efficiency"], marker="o",
             label=f"parallel, proc={p}")

plt.xlabel("Matrix size (n x n)")
plt.ylabel("Efficiency (speedup / processes)")
plt.title("Parallel efficiency vs size")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("efficiency_vs_size.png", dpi=300)

plt.show()
