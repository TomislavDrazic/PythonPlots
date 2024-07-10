
import pandas as pd

# Define the column headers correctly
column_headers = [
    "Piksel traenog markera",
    "Pozicija markera u koordinatnom sustavu kamere [mm]",
    "Pogreka u smjeru x osi [mm]",
    "Pogreka u smjeru y osi [mm]",
    "Pogreka u smjeru z osi [mm]"
]

# Create the DataFrame with the correct headers
data_corrected = {
    "Piksel traenog markera": [(609, 466), (617, 467), (613, 467), (612, 468), (612, 468), (612, 467), (612, 467), (613, 467)],
    "Pozicija markera u koordinatnom sustavu kamere [mm]": [
        [0.02859108119510528, -0.01411262820741847, 0.4489812600896165],
        [0.03020429119510528, -0.01374385820741847, 0.45234996],
        [0.02957410119510528, -0.01449945420741847, 0.45159646],
        [0.030613403676, -0.014743290662, 0.4495744640896165],
        [0.030614427397, -0.014447666962, 0.4492955900896165],
        [0.030613403663, -0.014437279207, 0.4495744720896165],
        [0.030613379849, -0.014733105685, 0.4492357020896165],
        [0.030613109615, -0.014442458063, 0.4497436580896165]
    ],
    "Pogreka u smjeru x osi ": [None, -0.40999, -1.04018, -0.000877518, -0.000853798, -0.000877532, -0.000901346, -0.00117158],
    "Pogreka u smjeru y osi ": [None, 0.36877, -0.386826, -0.000630662, -0.000335038, -0.000324651, -0.000620477, -0.000329856],
    "Pogreka u smjeru z osi ": [None, 3.3687, 2.6152, 0.593204, 0.31433, 0.593212, 0.254442, 0.762398]
}

# Create a new DataFrame with corrected data
df_corrected = pd.DataFrame(data_corrected, columns=column_headers)

# Function to format the values correctly
def format_values(value):
    if isinstance(value, list):
        return "; ".join([f"{x:.4f}".replace('.', ',') for x in value])
    if isinstance(value, (float, int)):
        return f"{value:.4f}".replace('.', ',')
    if pd.isnull(value):
        return 'nan'
    return value

# Apply the function to the DataFrame
df_corrected = df_corrected.applymap(format_values)

# Save to CSV
df_corrected.to_csv('corrected_data.csv', index=False, sep=';')

'/mnt/data/corrected_data.csv'
