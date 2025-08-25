import base64

import pandas as pd
import plotly.graph_objects as go

test_data = {
    "Passenger_ID": [1, 2, 3, 4],
    "Fare_Class": [1, 1, 2, 3],
    "Survived_int": [1, 1, 0, 0],
    "Category": [
        "Last Minute Ticket",
        "Last Minute Ticket",
        "Social Climber",
        "Social Climber",
    ],
    "x_pos": [1, 2, 10, 5],
    "y_pos": [1, 1, 10, 5],
}
test_df = pd.DataFrame(test_data)


def load_image() -> str:
    image_path = "/Users/sanch/berkeley/titanic/assets/full-view.png"
    try:
        with open(image_path, "rb") as image_file:
            encoded_image_string = base64.b64encode(image_file.read()).decode("utf-8")

        # Prepend the data URI header
        # Adjust 'jpeg' based on your image type (png, svg, etc.)
        encoded_image_string = f"data:image/jpeg;base64,{encoded_image_string}"

    except FileNotFoundError:
        print(
            f"Error: The image file '{image_path}' was not found. Please check the file path."
        )
        # Fallback to the online image if the local file is not found
        encoded_image_string = (
            "https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg"
        )

    return encoded_image_string


def main():
    fig = go.Figure()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=test_df["x_pos"],
            y=test_df["y_pos"],
            mode="markers",
            marker=dict(
                # Use different marker symbols, colors, and sizes based on data
                symbol=test_df["Fare_Class"].map(
                    {1: "circle", 2: "square", 3: "diamond"}
                ),
                color=test_df["Survived_int"].map({1: "green", 0: "red"}),
                line=dict(width=2, color="DarkSlateGrey"),
                opacity=0.8,
            ),
            # Add hover text to show passenger details on hover
            hovertext=test_df.apply(
                lambda row: f"ID: {row['Passenger_ID']}<br>Class: {row['Fare_Class']}<br>Survived: {'Yes' if row['Survived_int'] == 1 else 'No'}",
                axis=1,
            ),
            hoverinfo="text",
        )
    )
    encoded_image_string = load_image()

    fig.update_layout(
        title="Titanic Passenger Survival Map",
        # Add the background image
        images=[
            go.layout.Image(
                source=encoded_image_string,
                xref="paper",  # Relative to the plot area
                yref="paper",
                x=0,
                y=1,  # Position (top-left)
                sizex=1,
                sizey=1,  # Size (100% of the plot area)
                sizing="stretch",
                opacity=0.4,
                layer="below",
            )
        ],
        # Hide the axes to make the plot look like an image with markers
        xaxis=dict(showgrid=True, zeroline=False, showticklabels=False, range=[0, 10]),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=False, range=[0, 10]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
    )

    fig.show()

    # fig.write_image("titantic-image.png")


if __name__ == "__main__":
    main()
