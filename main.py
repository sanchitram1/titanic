import base64

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

IMG_PATH = "assets/Titanic Deck.png"
df = pd.read_csv("data/titanic.csv")


def load_image() -> str:
    try:
        with open(IMG_PATH, "rb") as image_file:
            encoded_image_string = base64.b64encode(image_file.read()).decode("utf-8")

        # Prepend the data URI header
        # Adjust 'jpeg' based on your image type (png, svg, etc.)
        encoded_image_string = f"data:image/jpeg;base64,{encoded_image_string}"

    except FileNotFoundError as e:
        print(
            f"Error: The image file '{IMG_PATH}' was not found. Please check the file path."
        )
        raise e

    return encoded_image_string


def get_image_size():
    deck_img = Image.open(IMG_PATH)
    return deck_img.size


def generate_scatter_coordinates(df, image_width, image_height):
    """
    Generates correctly clustered coordinates based on passenger class.
    """
    x_coords = np.zeros(len(df))
    y_coords = np.zeros(len(df))

    # Use a dictionary to map Pclass to its plot location
    pclass_locations = {1: 0.85, 2: 0.60, 3: 0.15}

    for pclass, loc_multiplier in pclass_locations.items():
        mask = df["Pclass"] == pclass
        count = mask.sum()
        if count > 0:
            x_coords[mask] = np.random.normal(
                loc=image_width * loc_multiplier, scale=image_width * 0.02, size=count
            )
            y_coords[mask] = np.random.normal(
                loc=image_height * 0.52, scale=image_height * 0.04, size=count
            )

    return x_coords, y_coords


def generate_plot(
    some_sliced_data: pd.DataFrame,
    deck_width: int,
    deck_height: int,
    encoded_image: str,
) -> go.Figure:
    # Prepare hover text
    hover_text = [
        f"{row['Name']}<br>Age: {row['Age']}<br>Fare: £{row['Fare']:.2f}"
        for _, row in some_sliced_data.iterrows()
    ]

    # Generate x and y coords
    x_coords, y_coords = generate_scatter_coordinates(
        some_sliced_data, deck_width, deck_height
    )

    # Clip coordinates to ensure they are within the image bounds
    x_coords = np.clip(x_coords, 0, deck_width)
    y_coords = np.clip(y_coords, 0, deck_height)

    # Create scatter plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers",
            marker=dict(size=12, color="blue"),
            text=hover_text,
            hoverinfo="text",
        )
    )

    # Add background image and configure layout
    fig.update_layout(
        images=[
            dict(
                source=encoded_image,
                xref="x",
                yref="y",
                x=0,
                y=deck_height,
                sizex=deck_width,
                sizey=deck_height,
                sizing="stretch",
                layer="below",
            )
        ],
        xaxis=dict(visible=False, range=[0, deck_width]),
        yaxis=dict(visible=False, range=[0, deck_height]),
        width=deck_width,
        height=deck_height,
        margin=dict(l=0, r=0, t=40, b=0),
        # title=dict(text="Location of Unlikely Survivors", x=0.5),
    )

    return fig


def main():
    # THIS IS AN EXAMPLE FOR SOCIAL CLIMBERS
    fare_q1 = df["Fare"].quantile(0.25)

    # Filter for the specified conditions
    social_climbers = df[
        (df["Pclass"] == 3)
        & (df["Fare"] <= fare_q1)
        & (df["SibSp"] == 0)
        & (df["Parch"] == 0)
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # END EXAMPLE

    encoded_image = load_image()
    deck_width, deck_height = get_image_size()
    fig = generate_plot(social_climbers, deck_width, deck_height, encoded_image)
    fig.show()


if __name__ == "__main__":
    main()
