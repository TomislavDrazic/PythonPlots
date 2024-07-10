
import matplotlib.pyplot as plt
import pandas as pd

# Update font properties
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 16
})

# Creating a DataFrame with the new data
data = {
    "Center pixels": ["(617, 467)", "(613, 467)",
                      "(612, 468)", "(612, 468)", "(612, 467)", "(612, 467)", "(613, 467)"],
    "greska_x (mm)": [-0.40999,-1.04018, -0.000877518, -0.000853798, -0.000877532, -0.000901346, -0.00117158],
    "greska_y (mm)": [0.36877,-0.386826, -0.000630662, -0.000335038, -0.000324651, -0.000620477, -0.000329856],
    "greska_z (mm)": [3.3687,2.6152, 0.593204, 0.31433, 0.593212, 0.254442, 0.762398]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Creating subplots to visualize the new data
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plotting greska_x (mm)
axs[0].plot(range(len(df["greska_x (mm)"])), df["greska_x (mm)"], marker='o', color='green')
axs[0].set_title('Pogreska X os')
axs[0].set_xlabel('Iteracija')
axs[0].set_ylabel('Pogreska [mm]')
axs[0].grid(True)

# Plotting greska_y (mm)
axs[1].plot(range(len(df["greska_y (mm)"])), df["greska_y (mm)"], marker='o', color='red')
axs[1].set_title('Pogreska Y os')
axs[1].set_xlabel('Iteracija')
axs[1].set_ylabel('Pogreska [mm]')
axs[1].grid(True)

# Plotting greska_z (mm)
axs[2].plot(range(len(df["greska_z (mm)"])), df["greska_z (mm)"], marker='o', color='blue')
axs[2].set_title('Pogreska Z os')
axs[2].set_xlabel('Iteracija')
axs[2].set_ylabel('Pogreska [mm]')
axs[2].grid(True)

plt.subplots_adjust(hspace=0.7)
plt.show()
