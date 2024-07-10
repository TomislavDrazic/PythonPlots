import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 20
})


# Creating a DataFrame with the combined data
data = {
    "Center pixels": ["(608, 466)", "(636, 360)", "(639, 388)", "(638, 414)", "(636, 431)", "(633, 443)",
                      "(620, 461)", "(619, 461)", "(619, 462)", "(619, 462)", "(619, 461)"],
    "startPos/tempPos": [
        "[30.29640894351673; -14.10720942334234; 448.8116633093241]",
        "[36.46188335900299; -43.68671669765834; 420.9750135712894]",
        "[36.62926637138323; -34.97894841649933; 413.1412893758834]",
        "[36.0899660667514; -27.45223658913562; 410.2163042439732]",
        "[35.35300124627889; -30.2690759900204; 408.1722763171451]",
        "[33.7717; -0.897508; -97.2385]",
        "[30.7604960958107; -14.16433778883866; 406.496024102308]",
        "[30.5020544328844; -14.16917671141489; 406.635151666693]",
        "[30.200669340234; -13.89798616499584; 406.278459092949]",
        "[30.45797540450847; -13.89798616499584; 406.278459092949]",
        "[30.4796079617635; -14.14568295899148; 406.219784717664]"
    ],
    "greska_x (mm)": [-6.16547, -6.33286, -6.33286, -5.79356, -5.05659, -8.47251, -0.095212, -0.205654, 0.001886, -0.227944, 0.004796],
    "greska_y (mm)": [29.5795, 20.8717, 20.8717, 13.4449, 14.3284, 3.30727, 0.057074, 0.205654, 0.001886, 0.227944, 0.004796]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Creating subplots to visualize the updated data
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

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

plt.subplots_adjust(hspace=0.5)
plt.show()
