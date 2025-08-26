import base64

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from PIL import Image

IMG_PATH = "assets/ship.png"
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


# Some constants that are useful for how we're rendering the images
DECK_WIDTH, DECK_HEIGHT = get_image_size()
ENCODED_IMAGE_STRING = load_image()


def generate_scatter_coordinates(df, image_width, image_height) -> tuple[int, int]:
    """Generates correctly clustered coordinates based on passenger class"""
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


def generate_clustered_coordinates(df, max_per_row=20, spacing=0.15) -> pd.DataFrame:
    """
    Generates clustered (x, y) coordinates for a DataFrame based on Pclass.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'Pclass' column.
        max_per_row (int): The maximum number of points in a single row within each cluster.
        spacing (float): The distance between each point.

    Returns:
        pd.DataFrame: The original DataFrame with 'x_pos' and 'y_pos' columns added.
    """
    # If we need to offset by x or y
    # NOTE: initial configuration **heavily** depends on your choice of image and size
    cluster_x_offsets = {1: 2.65, 2: 5.95, 3: 9.25}
    cluster_y_offsets = {1: -1.75, 2: -1.75, 3: -1.75}

    # this is our output
    x_coords = []
    y_coords = []
    passenger_ids = []

    # group by pclass
    grouped_df = df.groupby("Pclass")

    # iterate through each passenger class group
    for pclass, group in grouped_df:
        num_in_group = len(group)

        # for this passenger class, we need coordinates
        for i in range(num_in_group):
            col_index = i % max_per_row
            row_index = i // max_per_row

            # increasing x goes right!
            x = (col_index * spacing) + cluster_x_offsets[pclass]
            # increasing y goes down!
            y = -(row_index * spacing) + cluster_y_offsets[pclass]

            x_coords.append(x)
            y_coords.append(y)
            passenger_ids.append(group.iloc[i]["PassengerId"])

    # store our output, and merge with original
    coords_df = pd.DataFrame(
        {"PassengerId": passenger_ids, "x_coords": x_coords, "y_coords": y_coords}
    )

    # Merge the coordinates back into the original DataFrame
    return df.merge(coords_df, on="PassengerId", how="left")


def generate_plot(some_sliced_data: pd.DataFrame) -> go.Figure:
    # Prepare hover text
    hover_text = [
        f"{row['Name']}<br>Age: {row['Age']}<br>Fare: £{row['Fare']:.2f}"
        for _, row in some_sliced_data.iterrows()
    ]

    # Generate x and y coords
    sliced_data = generate_clustered_coordinates(some_sliced_data)

    # Plotting stuff
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sliced_data["x_coords"],
            y=sliced_data["y_coords"],
            mode="markers",
            marker=dict(
                symbol="circle",
                color=sliced_data["Pclass"].map(
                    {1: "#e9bf99", 2: "#a81a0c", 3: "#000000"}
                ),
                opacity=0.8,
            ),
            # Add hover text to show passenger details on hover
            hovertext=hover_text,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        # Add the background image
        images=[
            go.layout.Image(
                source=ENCODED_IMAGE_STRING,
                xref="paper",
                yref="paper",
                x=0,
                y=1,
                sizex=1,
                sizey=0.65,  # Size (100% of the plot area)
                sizing="stretch",
                # opacity=0.4,
                layer="below",
            )
        ],
        # Hide the axes to make the plot look like an image with markers
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 15]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, 0]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        width=1200,
        height=800,
    )

    return fig


def create_dashboard():
    """Creates a custom layout for the "Titanic Itinerary" dashboard.

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
        className="dashboard",
        children=[
            # Header
            html.H1(
                children="Titanic Itinerary",
                style={
                    "textAlign": "left",
                    "marginBottom": "30px",
                },
            ),
            # Top section
            html.Div(
                className="row",
                children=[
                    # This is the container for the dropdown and its label
                    html.Div(
                        className="category-dropdown-container",
                        children=[
                            html.Label(
                                "Are you a...",
                                style={
                                    "marginRight": "16px",
                                    "whiteSpace": "nowrap",  # no wrap!
                                },
                            ),
                            dcc.Dropdown(
                                id="category-dropdown",
                                options=titanic_categories,
                                value="Social Climber",
                                clearable=False,
                                style={
                                    "flexGrow": "1",  # Allow the dropdown to grow
                                },
                            ),
                        ],
                    ),
                    # This is the container for the four metrics
                    html.Div(
                        className="metrics-container",
                        children=[
                            # Survival Percentage
                            html.Div(
                                className="info",
                                id="survival-percentage",
                                children=["--% survival"],
                            ),
                            # Number of Males
                            html.Div(
                                className="info",
                                id="number-of-males",
                                children=["-- males"],
                            ),
                            # Number of Females
                            html.Div(
                                className="info",
                                id="number-of-females",
                                children=["-- females"],
                            ),
                            # Average Age
                            html.Div(
                                className="info",
                                id="average-age",
                                children=["-- avg age"],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="main-section",
                children=[
                    html.Div(className="main", children=[dcc.Graph(id="ship-map")])
                ],
                style={"marginTop": "24px"},
            ),
        ],
    )


# These are all the functions that generate our datasets
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


# And here are all the callbacks
@app.callback(Output("ship-map", "figure"), Input("category-dropdown", "value"))
def update_ship_map(category: str):
    """This is the callback which configures the response of our Dash App to
    whatever dropdown is selected"""
    slice = get_selected_df(category)
    return generate_plot(slice)


@app.callback(
    Output("survival-percentage", "children"), Input("category-dropdown", "value")
)
def update_survival_percentage(category: str):
    sub = get_selected_df(category)
    return [
        html.P(f"{sub['Survived'].mean() * 100:.2f}%", className="metric-value"),
        html.Div(style={"flexGrow": "1"}),
        html.P("survival odds", className="metric-label"),
    ]


@app.callback(
    Output("number-of-males", "children"), Input("category-dropdown", "value")
)
def update_males(category: str):
    sub = get_selected_df(category)
    return [
        html.P(f"{len(sub[sub['Sex'] == 'male']):.0f}", className="metric-value"),
        html.Div(style={"flexGrow": "1"}),
        html.P("males", className="metric-label"),
    ]


@app.callback(
    Output("number-of-females", "children"), Input("category-dropdown", "value")
)
def update_females(category: str):
    sub = get_selected_df(category)
    return [
        html.P(f"{len(sub[sub['Sex'] == 'female']):.0f}", className="metric-value"),
        html.Div(style={"flexGrow": "1"}),
        html.P("females", className="metric-label"),
    ]


@app.callback(Output("average-age", "children"), Input("category-dropdown", "value"))
def update_age(category: str):
    sub = get_selected_df(category)
    return [
        html.P(f"{sub['Age'].mean():.0f}y", className="metric-value"),
        html.Div(style={"flexGrow": "1"}),
        html.P("average age", className="metric-label"),
    ]


def get_selected_df(category: str) -> pd.DataFrame:
    """Helper to encapsulate the logic that figures out which dataset to return based
    on user selection"""
    if category == "Social Climber":
        return social_climbers()
    elif category == "(Un)Happy Family":
        return family()
    elif category == "Last Minute Ticket":
        return last_minute()
    elif category == "Unlikely Survivor":
        return unlikely_survivor()

    raise ValueError(f"Bad category: {category}")


def main():
    create_dashboard()
    app.run(debug=True)


if __name__ == "__main__":
    main()
