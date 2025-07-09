import cv2
import numpy as np
import math
import os
import csv
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration ---
# Hardcode the path to the video file directly in the script.
# Make sure to change this to the actual path of your video file.
video_path = "/Users/ryanmattana/TseLabs/simGraphJS/Movies/Test5_edited (1).mov"

# --- Video Loading ---
# Check if the file exists before trying to open it
if not os.path.exists(video_path):
    print(f"Error: The video file was not found at the specified path:")
    print(video_path)
    exit()

# Open the video file specified by the hardcoded path.
try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
except Exception as e:
    print(f"Error: {e}")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("Warning: Could not get FPS from video. Using default of 30.")
    fps = 30  # Default FPS if it cannot be read

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_center = (frame_width / 2, frame_height / 2)

# --- Initialization ---
# Variables for tracking
prev_center = None
prev_time = 0
prev_r = None
prev_theta = None
trail_points = []

# Lists to store data for plotting and CSV export
coordinate_data = []  # To store (time, frame, x, y) for CSV
time_data = []
frame_data = [] # NEW: To store frame numbers for plotting
radial_velocity_data = []
tangential_velocity_data = []
angular_velocity_data = []

print(f"Processing video: {os.path.basename(video_path)}")
print("This may take a moment. The interactive plots will be generated after the video is fully processed.")
print("Press 'q' in the video window to PAUSE/RESUME.")
print("Press 'c' in the video window to QUIT.")

# --- Main Processing Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Frame Pre-processing ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # --- Light Source Detection ---
    (_, _, _, max_loc) = cv2.minMaxLoc(blurred)
    center = max_loc

    # --- Calculation and Visualization ---
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Store coordinate data for CSV export
    coordinate_data.append([current_time, current_frame, center[0], center[1]])
    trail_points.append(center) # Also store for the trail plot

    if prev_center is not None:
        time_diff = current_time - prev_time
        if time_diff > 0:
            # --- Polar Coordinates and Velocities ---
            x, y = center[0] - frame_center[0], frame_center[1] - center[1]
            r = math.sqrt(x**2 + y**2)
            theta = math.atan2(y, x)

            if prev_r is not None and prev_theta is not None:
                dr = r - prev_r
                dtheta = theta - prev_theta

                # Handle angle wrapping for accurate dtheta
                if dtheta > math.pi:
                    dtheta -= 2 * math.pi
                if dtheta < -math.pi:
                    dtheta += 2 * math.pi

                # Calculate polar velocities
                radial_velocity = dr / time_diff
                angular_velocity = dtheta / time_diff
                tangential_velocity = r * angular_velocity

                # Append data for plotting
                time_data.append(current_time)
                frame_data.append(current_frame) # NEW: Store frame number
                radial_velocity_data.append(radial_velocity)
                tangential_velocity_data.append(tangential_velocity)
                angular_velocity_data.append(angular_velocity)

            prev_r = r
            prev_theta = theta

            # --- Visualization on Video Frame ---
            cv2.circle(frame, center, 10, (0, 255, 0), 2)
            for i in range(1, len(trail_points)):
                if trail_points[i - 1] is None or trail_points[i] is None:
                    continue
                cv2.line(frame, trail_points[i - 1], trail_points[i], (0, 0, 255), 2)

    # Update previous state for next iteration
    prev_center = center
    prev_time = current_time

    # Display the frame
    cv2.imshow("Light Tracker", frame)

    # --- Pause/Resume/Quit Controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.putText(frame, "PAUSED", (frame_width // 2 - 100, frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        cv2.imshow("Light Tracker", frame)
        while cv2.waitKey(0) & 0xFF != ord('q'):
            pass
    elif key == ord('c'):
        break

# --- Cleanup ---
print("Finished processing video.")
cap.release()
cv2.destroyAllWindows()

# --- Saving Data and Generating Interactive HTML Plot ---

# 1. Save coordinate data to CSV
csv_file_path = "coordinate_data.csv"
with open(csv_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Time (s)", "Frame", "X", "Y"]) # Updated header
    writer.writerows(coordinate_data)
print(f"Coordinate data saved as {csv_file_path}")

# 2. Create interactive velocity plots with Plotly
print("Generating interactive velocity plots...")
fig_vel = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Radial Velocity", "Tangential Velocity", "Angular Velocity"))

# Define a custom hover template for richness
hovertemplate = ('<b>Time</b>: %{x:.2f}s<br>' +
                 '<b>Frame</b>: %{customdata}<br>' +
                 '<b>Value</b>: %{y:.3f}<extra></extra>')

# Radial Velocity
fig_vel.add_trace(go.Scatter(x=time_data, y=radial_velocity_data, mode='lines', name='Radial', line=dict(color='red'),
                             customdata=frame_data, hovertemplate=hovertemplate), row=1, col=1)
# Tangential Velocity
fig_vel.add_trace(go.Scatter(x=time_data, y=tangential_velocity_data, mode='lines', name='Tangential', line=dict(color='blue'),
                              customdata=frame_data, hovertemplate=hovertemplate), row=2, col=1)
# Angular Velocity
fig_vel.add_trace(go.Scatter(x=time_data, y=angular_velocity_data, mode='lines', name='Angular', line=dict(color='green'),
                             customdata=frame_data, hovertemplate=hovertemplate), row=3, col=1)

# Update layout
fig_vel.update_layout(
    height=800,
    title_text="Velocity Analysis",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# Update Y-axis labels
fig_vel.update_yaxes(title_text="pixels/s", row=1, col=1)
fig_vel.update_yaxes(title_text="pixels/s", row=2, col=1)
fig_vel.update_yaxes(title_text="rad/s", row=3, col=1)

# --- NEW: Create custom X-axis labels with both Time and Frames ---
if time_data:
    # Select ~10 tick values evenly spaced through the data
    num_ticks = 10
    indices = np.linspace(0, len(time_data) - 1, num_ticks, dtype=int)
    tickvals = [time_data[i] for i in indices]
    ticktext = [f"{time_data[i]:.1f}s<br>{frame_data[i]}f" for i in indices]

    # Update the shared X-axis on the bottom plot
    fig_vel.update_xaxes(
        title_text="Time / Frame Number",
        tickvals=tickvals,
        ticktext=ticktext,
        row=3, col=1
    )

# 3. Create interactive trail plot with Plotly
print("Generating interactive trail plot...")
if trail_points:
    trail_x = [p[0] for p in trail_points]
    trail_y = [p[1] for p in trail_points]

    fig_trail = go.Figure(data=go.Scatter(
        x=trail_x,
        y=trail_y,
        mode='lines',
        line=dict(color='red', width=2)
    ))
    fig_trail.update_layout(
        title="Detected Trail Path",
        template="plotly_dark",
        xaxis=dict(
            title="X Coordinate (pixels)",
            range=[0, frame_width]
        ),
        yaxis=dict(
            title="Y Coordinate (pixels)",
            scaleanchor="x",
            scaleratio=1,
            range=[frame_height, 0]
        )
    )

# 4. Combine plots into a single HTML file
html_file_path = "interactive_plots.html"
print(f"Saving plots to {html_file_path}...")

with open(html_file_path, 'w') as f:
    f.write("<html><head><title>Interactive Plots</title></head><body>\n")
    f.write("<h1 style='font-family: sans-serif; text-align: center;'>Tracking Analysis</h1>\n")

    f.write("<h2 style='font-family: sans-serif;'>Velocity Plots</h2>\n")
    f.write("<p style='font-family: sans-serif;'>Click and drag to zoom, double-click to reset. Hover over points for details.</p>\n")
    plot_html = fig_vel.to_html(full_html=False, include_plotlyjs='cdn')
    f.write(plot_html)

    if trail_points:
        f.write("<hr><h2 style='font-family: sans-serif;'>Trail Path Plot</h2>\n")
        f.write("<p style='font-family: sans-serif;'>This plot shows the detected path of the light source.</p>\n")
        plot_html = fig_trail.to_html(full_html=False, include_plotlyjs='cdn')
        f.write(plot_html)

    f.write("</body></html>\n")

print(f"Successfully created {html_file_path}. Open this file in a web browser to view the interactive plots.")