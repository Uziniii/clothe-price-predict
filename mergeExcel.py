import pandas as pd

# Load the dataset
pant = pd.read_excel("Pant.xlsx")
tshirt = pd.read_excel("Tshirt.xlsx")
sweat = pd.read_excel("Sweat.xlsx")
coat = pd.read_excel("Coat.xlsx")

# Merge to one file and add type column
pant['Type'] = 'Pant'
tshirt['Type'] = 'Tshirt'
sweat['Type'] = 'Sweat'
coat['Type'] = 'Coat'

# Merge all dataframes
df = pd.concat([pant, tshirt, sweat, coat])

# Save file
df.to_excel("merged.xlsx", index=False)
