import numpy as np
import matplotlib.pyplot as plt

# --- Parameters you can easily change ---

# 1. Number of rotations the spiral makes
num_rotations = 4

# 2. Controls the distance between the spiral's arms (a larger value makes it bigger)
spiral_tightness = 4.5

# 3. The width and height of the Cartesian grid (domain and range).
# The plot will go from 0 to this value on both axes.
grid_size = 350

# 4. The distance between points along the spiral's path (arc length).
# This constant step length is what ensures constant tangential velocity.
step_length = 15.0

# 5. G-Code specific parameters
trace_speed = 2400  # Speed for spiral moves (mm/min)
travel_speed = 6000  # Speed for non-tracing moves (mm/min)


# --- End of parameters ---


# --- Generate Spiral Points for Constant Tangential Velocity ---

x_points = []
y_points = []
radii_from_center = [] # List to store the radius from the spiral's center for each point
theta = 0.0
total_angle = num_rotations * 2 * np.pi
center_point = grid_size / 2

# Add the starting point (the center of the spiral)
x_points.append(center_point)
y_points.append(center_point)
radii_from_center.append(0.0) # The radius at the exact center is 0

# Start theta just above zero for the loop to avoid division by zero at the very start
theta = 0.001

while theta < total_angle:
    # Calculate radius for the current angle (theta)
    r_spiral = spiral_tightness * theta

    # Calculate the Cartesian coordinates and shift them to the grid's center
    x_shifted = (r_spiral * np.cos(theta)) + center_point
    y_shifted = (r_spiral * np.sin(theta)) + center_point
    x_points.append(x_shifted)
    y_points.append(y_shifted)

    # Calculate and store the radius from the spiral's main center to the current point.
    # This will be used as the R value in the G-code.
    radius = np.sqrt((x_shifted - center_point)**2 + (y_shifted - center_point)**2)
    radii_from_center.append(radius)

    # To maintain a constant step_length (arc length), we must vary the
    # change in angle (d_theta) for the next step.
    d_theta = step_length / (spiral_tightness * np.sqrt(1 + theta ** 2))
    theta += d_theta

# Calculate the final radius of the spiral for the bounding circle
last_x_rel = x_points[-1] - center_point
last_y_rel = y_points[-1] - center_point
max_radius = np.sqrt(last_x_rel ** 2 + last_y_rel ** 2)

# --- Output G-Code Script ---

# Print the fixed header
print("""; G-code to draw a circle and a spiral
; Bed X/Y: 350x350, Z max ~400
; Modified to keep Z-axis at a constant 20mm

; ========================================
; Part 1: Setup (Functional)
; ========================================
M140 S0            ; turn off bed heater
M104 S0            ; turn off nozzle heater
G90                ; absolute positioning (default mode)
G28                ; home all axes
G0 Z20 F2400       ; set nozzle to a fixed Z height of 20mm""")
print("")

# --- Part 2: Bounding Circle (Corrected) ---
print("; ========================================")
print("; Part 2: Bounding Circle")
print("; ========================================")
# Calculate the circle's start point (3 o'clock position)
circle_start_x = center_point + max_radius
circle_start_y = center_point

# Move to the circle's start point
print(f"G0 F{travel_speed} X{circle_start_x:.4f} Y{circle_start_y:.4f} ; Move to circle's start point")

# Draw the circle using I and J for greater reliability
# I is the X-offset from the start point to the center. J is the Y-offset.
# Start is at (center+radius, center). Center is at (center, center).
# So, I = (center_x - start_x) = -radius. J = (center_y - start_y) = 0.
i_offset = -max_radius
j_offset = 0
print(f"G2 I{i_offset:.4f} J{j_offset:.4f} F{trace_speed} ; Draw a full clockwise circle using I,J")
print("")

# --- Part 3: Spiral ---
print("; ========================================")
print("; Part 3: Spiral")
print("; ========================================")
# Move to the spiral's start point
print(f"G0 X{x_points[0]:.4f} Y{y_points[0]:.4f} ; Move to spiral start position (center)")

# The first move from the center is linear
print(f"G1 X{x_points[1]:.4f} Y{y_points[1]:.4f} F{trace_speed} ; First linear move from center")

# Trace the rest of the spiral using G03 (counter-clockwise) arc moves
# We start from the 2nd point, creating an arc from point 1 to point 2, and so on.
for i in range(2, len(x_points)):
    # The R value for the arc is the radius from the spiral's center to the arc's START point.
    r_val = radii_from_center[i-1]
    print(f"G3 X{x_points[i]:.4f} Y{y_points[i]:.4f} R{r_val:.4f}")

# --- Plotting ---

# Create a figure and an axes object for the plot
fig, ax = plt.subplots(figsize=(8, 8))

# Set the title and labels
ax.set_title(f'Spiral and Bounding Circle')
ax.set_xlabel(f'X-axis (0 to {grid_size})')
ax.set_ylabel(f'Y-axis (0 to {grid_size})')

# Plot the generated spiral path
ax.plot(x_points, y_points, 'o-', markersize=3, label=f'Spiral Path')

# Create and add the bounding circle to the plot
circle_patch = plt.Circle((center_point, center_point), max_radius, color='r', fill=False, linestyle='--',
                          label='Outermost Radius Circle')
ax.add_patch(circle_patch)

# Set the limits of the grid
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)

# Display the Cartesian grid lines
ax.grid(True)

# Ensure the aspect ratio is equal to prevent distortion
ax.set_aspect('equal')

# Add a legend
ax.legend()

# Display the final plot
plt.show()
