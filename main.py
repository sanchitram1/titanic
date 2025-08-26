import base64

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from PIL import Image

IMG_PATH = "assets/Titanic Deck Resized.png"
df = pd.read_csv("data/titanic.csv")
app = Dash(__name__)


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


DECK_WIDTH, DECK_HEIGHT = get_image_size()
ENCODED_IMAGE_STRING = load_image()


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
    deck_width: int = DECK_WIDTH,
    deck_height: int = DECK_HEIGHT,
    encoded_image: str = ENCODED_IMAGE_STRING,
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
            marker=dict(size=8, color=some_sliced_data["Pclass"]),
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


def create_dashboard():
    """Creates a custom layout for the "Plan Your Next Titanic Journey" dashboard.

    This layout includes a title, a dropdown for scenario selection, and
    placeholders for a ship image and summary statistics.

    Returns:
        html.Div: The complete layout for the Dash application."""

    # Define the categories for the dropdown menu
    titanic_categories = [
        {"label": "Social Climber", "value": "Social Climber"},
        {"label": "Last Minute Ticket", "value": "Last Minute Ticket"},
        {"label": "Unlikely Survivor", "value": "Unlikely Survivor"},
        {"label": "(Un)Happy Family", "value": "(Un)Happy Family"},
    ]

    # Define the Dashboard Layout
    app.layout = html.Div(
        style={
            "backgroundColor": "#FFFFFF",
            "padding": "20px",
            "minHeight": "100vh",
            "margin-left": "0",
            "margin-top": "0",
        },
        children=[
            # Header
            html.H1(
                children="Plan Your Next Titanic Journey",
                style={
                    "textAlign": "left",
                    "marginBottom": "30px",
                },
            ),
            # Category Dropdown Section
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="category-dropdown-selection",
                        children=[
                            html.Label(
                                "Are you a...",
                                style={
                                    "marginRight": "16px",
                                    "marginBottom": "0",
                                    "marginLeft": "0",
                                    "width": "200px",
                                },
                            ),
                            dcc.Dropdown(
                                id="category-dropdown",
                                options=titanic_categories,
                                value="Social Climber",
                                clearable=False,
                                style={
                                    "marginRight": "16px",
                                    "marginBottom": "0",
                                    "marginLeft": "0",
                                    "width": "400px",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "maxWidth": "600px",
                            # "justifyContent": "flex-start",
                            "marginBottom": "20px",
                        },
                    ),
                ],
                style={"marginBottom": "40px", "textAlign": "center"},
            ),
            # dcc.Graph("ship-map"),
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="main",
                        children=[
                            dcc.Graph(
                                id="ship-map",
                                style={
                                    "flex": "3 1 0%",
                                    # "minWidth": "0",  # helps with overflow
                                },
                            )
                        ],
                    ),
                    html.Div(
                        className="main",
                        children=[html.P("")],
                        id="summary-stats",
                        style={
                            "flex": "2 1 0%",
                            "background": "#f5f6fa",
                            "borderRadius": "24px",  # optional, for rounded corners
                        },
                    ),
                ],
                style={"marginTop": "24px"},
            ),
        ],
    )


def social_climbers() -> pd.DataFrame:
    """For the social climbers, we can look a bit at those who paid the lowest amount
    and traveled solo...in the hope of meeting someone who punched wayyyy above their
    weight"""
    lowest_fare_quantile = 0.10
    social_climbers_df = df[
        (df["Fare"] <= df["Fare"].quantile(lowest_fare_quantile))
        & (df["SibSp"] == 0)
        & (df["Parch"] == 0)
    ].copy()

    return social_climbers_df


def family() -> pd.DataFrame:
    """All families which had left on the Titanic...whether they ended up as happy
    or unhappy"""
    # family means either sibling or parent
    return df.loc[(df["SibSp"] + df["Parch"]) > 0, :].copy()


def last_minute() -> pd.DataFrame:
    """Last minute attempts to find last minute ticket purchases, but looking at
    people who boarded from Queenstown, and purchased either **very expensive** or
    **very cheap** tickets"""
    q = 0.25
    q_low = df["Fare"].quantile(q)
    q_high = df["Fare"].quantile(1 - q)
    return df.loc[
        (df["Embarked"] == "Q") & ((df["Fare"] <= q_low) | (df["Fare"] >= q_high)), :
    ].copy()


def unlikely_survivor() -> pd.DataFrame:
    """Passengers who were unlikely to survive based on their characteristics but did
    So, people very young or very old who bought **very cheap** tickets and were
    in third class"""
    survived = df["Survived"] == 1
    third_class = df["Pclass"] == 3
    cheap_fare = df["Fare"] < df["Fare"].quantile(0.25)
    is_young = df["Age"] <= df["Age"].quantile(0.25)
    is_old = df["Age"] >= df["Age"].quantile(0.75)
    Unlikely_Survivor = df[survived & third_class & cheap_fare & (is_young | is_old)]
    return Unlikely_Survivor.copy()


@app.callback(Output("ship-map", "figure"), Input("category-dropdown", "value"))
def update_ship_map(category: str):
    """This is the callback which configures the response of our Dash App to
    whatever dropdown is selected"""
    if category == "Social Climber":
        slice = social_climbers()
        return generate_plot(slice)
    elif category == "(Un)Happy Family":
        slice = family()
        return generate_plot(slice)
    elif category == "Unlikely Survivor":
        slice = unlikely_survivor()
        return generate_plot(slice)
    elif category == "Last Minute Ticket":
        slice = last_minute()
        return generate_plot(slice)


@app.callback(Output("summary-stats", "children"), Input("category-dropdown", "value"))
def update_text(category: str):
    if category == "Social Climber":
        sub = social_climbers()
    elif category == "(Un)Happy Family":
        sub = family()
    elif category == "Last Minute Ticket":
        sub = last_minute()
    elif category == "Unlikely Survivor":
        sub = unlikely_survivor()

    total = len(sub)
    survived = int(sub["Survived"].sum()) if total else 0
    rate = (survived / total * 100) if total else 0.0
    male = (sub["Sex"] == "male").sum()
    female = (sub["Sex"] == "female").sum()
    age = sub["Age"].dropna().mean()

    return [
        html.H3(f"You have a survival rate of {rate:.2f}%"),
        html.P(f"{male} males"),
        html.P(f"{female} females"),
        html.P(f"{int(age)} years old"),
    ]


def main():
    create_dashboard()
    app.run(debug=True)


if __name__ == "__main__":
    main()
