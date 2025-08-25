#%%
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# --- 1. Load Data and find Unlikely Survivors ---

# Load the dataset once
df = pd.read_csv("Titanic-Dataset.csv")

# Define conditions for this specific group
survived = df['Survived'] == 1
third_class = df['Pclass'] == 3
# Define cheap fare as the bottom 25% (first quartile)
cheap_fare = df['Fare'] < df['Fare'].quantile(0.25)
# Define age groups based on the 25th and 75th percentiles
is_young = df['Age'] <= df['Age'].quantile(0.25)
is_old = df['Age'] >= df['Age'].quantile(0.75)

# Filter the DataFrame to find passengers meeting all criteria
Unlikely_Survivor = df[survived & third_class & cheap_fare & (is_young | is_old)]
print(f"\nThere are {len(Unlikely_Survivor)} 'Unlikely Survivors'.")
print(Unlikely_Survivor[['Name', 'Age', 'Fare']])

# --- 3. Visualize the Unlikely Survivors ---

def generate_scatter_coordinates(df, image_width, image_height):
    """
    Generates correctly clustered coordinates based on passenger class.
    """
    x_coords = np.zeros(len(df))
    y_coords = np.zeros(len(df))

    # Use a dictionary to map Pclass to its plot location
    pclass_locations = {1: 0.85, 2: 0.60, 3: 0.15}

    for pclass, loc_multiplier in pclass_locations.items():
        mask = (df['Pclass'] == pclass)
        count = mask.sum()
        if count > 0:
            x_coords[mask] = np.random.normal(loc=image_width * loc_multiplier, scale=image_width * 0.02, size=count)
            y_coords[mask] = np.random.normal(loc=image_height * 0.52, scale=image_height * 0.04, size=count)
            
    return x_coords, y_coords

# Load the deck image
deck_img = Image.open("Titanic Deck.png")
deck_width, deck_height = deck_img.size

# Set seed for reproducibility and generate coordinates
np.random.seed(42)
x_coords, y_coords = generate_scatter_coordinates(Unlikely_Survivor, deck_width, deck_height)
#Copy the line above for your codes and replace Unlikely_Survivor with your data-slice

# Clip coordinates to ensure they are within the image bounds
x_coords = np.clip(x_coords, 0, deck_width)
y_coords = np.clip(y_coords, 0, deck_height)

# Prepare hover text
hover_text = [
    f"{row['Name']}<br>Age: {row['Age']}<br>Fare: £{row['Fare']:.2f}"
    for _, row in Unlikely_Survivor.iterrows()
]

# Create scatter plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x_coords,
    y=y_coords,
    mode='markers',
    marker=dict(size=12, color='blue'),
    text=hover_text,
    hoverinfo='text'
))

# Add background image and configure layout
fig.update_layout(
    images=[dict(
        source=deck_img,
        xref="x", yref="y", x=0, y=deck_height,
        sizex=deck_width, sizey=deck_height,
        sizing="stretch", layer="below"
    )],
    xaxis=dict(visible=False, range=[0, deck_width]),
    yaxis=dict(visible=False, range=[0, deck_height]),
    width=deck_width,
    height=deck_height,
    margin=dict(l=0, r=0, t=40, b=0),
    title=dict(text="Location of Unlikely Survivors", x=0.5)
)

fig.show()
# %%
