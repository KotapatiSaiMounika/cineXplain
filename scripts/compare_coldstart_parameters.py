
import matplotlib.pyplot as plt
import numpy as np

# Different values of k to compare
k_values = [30, 50, 68, 100, 150]

# User interaction counts
interactions = np.arange(0, 1001, 10)

plt.figure(figsize=(10, 6))

for k in k_values:
    content_weights = k / (k + interactions)

    plt.plot(
        interactions,
        content_weights,
        label=f"k = {k}"
    )

plt.xlabel("Number of User Interactions")
plt.ylabel("Content Weight")
plt.title("CineXplain Cold-Start Parameter Comparison")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "reports/coldstart_parameter_comparison.png",
    dpi=300
)

plt.show()

print("Cold-start comparison completed!")
print("Graph saved in reports/")
