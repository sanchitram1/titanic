import base64

import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash
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


def create_dashboard():
    """
    Creates a custom layout for the "Plan Your Next Titanic Journey" dashboard.

    This layout includes a title, a row of four scenario buttons, and
    placeholders for a ship image and summary statistics.

    Returns:
        html.Div: The complete layout for the Dash application.
    """
    app = Dash(__name__)
    # Define the Dashboard Layout
    app.layout = html.Div(
        style={
            "backgroundColor": "#1E2A38",
            "color": "#EAEAEA",
            "padding": "20px",
            "minHeight": "100vh",
            "fontFamily": "sans-serif",
        },
        children=[
            # Header
            html.H1(
                children="Plan Your Next Titanic Journey",
                style={
                    "textAlign": "center",
                    "marginBottom": "30px",
                    "color": "#FFFFFF",
                },
            ),
            # Scenario Buttons Section
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="three columns",
                        children=[
                            html.Button(
                                "Social Climber",
                                id="btn-social-climber",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "padding": "15px",
                                    "fontSize": "16px",
                                    "backgroundColor": "#5C6BC0",
                                    "border": "none",
                                    "color": "white",
                                    "borderRadius": "5px",
                                },
                            )
                        ],
                    ),
                    html.Div(
                        className="three columns",
                        children=[
                            html.Button(
                                "Last Minute Ticket",
                                id="btn-last-minute",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "padding": "15px",
                                    "fontSize": "16px",
                                    "backgroundColor": "#42A5F5",
                                    "border": "none",
                                    "color": "white",
                                    "borderRadius": "5px",
                                },
                            )
                        ],
                    ),
                    html.Div(
                        className="three columns",
                        children=[
                            html.Button(
                                "Unlikely Survivor",
                                id="btn-unlikely-survivor",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "padding": "15px",
                                    "fontSize": "16px",
                                    "backgroundColor": "#26A69A",
                                    "border": "none",
                                    "color": "white",
                                    "borderRadius": "5px",
                                },
                            )
                        ],
                    ),
                    html.Div(
                        className="three columns",
                        children=[
                            html.Button(
                                "(Un)Happy Family",
                                id="btn-family",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "padding": "15px",
                                    "fontSize": "16px",
                                    "backgroundColor": "#EF5350",
                                    "border": "none",
                                    "color": "white",
                                    "borderRadius": "5px",
                                },
                            )
                        ],
                    ),
                ],
                style={"marginBottom": "40px", "textAlign": "center"},
            ),
            # Main Content Area (Image and Stats)
            html.Div(
                className="row",
                children=[
                    # Left side for Ship Image
                    html.Div(
                        className="seven columns",
                        children=[
                            html.Div(
                                children="[Your Ship Image Will Be Displayed Here]",
                                id="ship-image-placeholder",
                                style={
                                    "border": "1px dashed #444",
                                    "padding": "20px",
                                    "textAlign": "center",
                                    "borderRadius": "5px",
                                    "height": "450px",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "backgroundColor": "#263238",
                                },
                            )
                        ],
                    ),
                    # Right side for Summary Statistics
                    html.Div(
                        className="five columns",
                        children=[
                            html.Div(
                                children=[
                                    html.H4("Analysis Summary"),
                                    html.P(
                                        "[Summary statistics from the analysed data will be printed out here.]"
                                    ),
                                ],
                                id="summary-stats-placeholder",
                                style={
                                    "border": "1px dashed #444",
                                    "padding": "20px",
                                    "textAlign": "center",
                                    "borderRadius": "5px",
                                    "height": "450px",
                                    "backgroundColor": "#263238",
                                },
                            )
                        ],
                    ),
                ],
            ),
        ],
    )

    return app


def main():
    app = create_dashboard()

    app.run(debug=True)

    # encoded_image = load_image()
    # deck_width, deck_height = get_image_size()
    # fig = generate_plot(social_climbers, deck_width, deck_height, encoded_image)
    # fig.show()


if __name__ == "__main__":
    main()
