import numpy as np
import matplotlib.pyplot as plt

# --- Parameters you can easily change ---

# 1. The total number of concentric circles to draw.
num_concentric_circles = 10

# 2. The radius of the smallest (innermost) circle.
smallest_circle_radius = 1

# 3. The radius of the largest (outermost) circle.
largest_circle_radius = 25

# 4. The width and height of the Cartesian grid.
grid_size = 350

# 5. Padding between the largest circle and the bounding square.
square_padding = 0

# 6. G-Code specific parameters
trace_speed = 1000  # Speed for tracing moves (mm/min)
travel_speed = 1000 # Speed for non-tracing moves (mm/min)


# --- End of parameters ---


# --- Generate Circle Radii ---

# Calculate the single center point for all shapes
center_point = grid_size / 2

# Create a list of radii, spaced evenly from smallest to largest.
# Using np.linspace is perfect for this.
if num_concentric_circles > 1:
    circle_radii = np.linspace(smallest_circle_radius, largest_circle_radius, num_concentric_circles)
else:
    # Handle the case of a single circle
    circle_radii = [largest_circle_radius]


# --- Output G-Code Script ---

# Print the fixed header
print("""; G-code to draw a centered square and concentric circles
; Bed X/Y: 350x350, Z max ~400
; Z-axis is kept at a constant 20mm

; ========================================
; Part 1: Setup (Functional)
; ========================================
M140 S0            ; turn off bed heater
M104 S0            ; turn off nozzle heater
G90                ; absolute positioning (default mode)
G28                ; home all axes
G0 Z350 F2400       ; set nozzle to a fixed Z height of 20mm""")
print("")

# --- Part 2: Bounding Square ---
print("; ========================================")
print("; Part 2: Bounding Square")
print("; ========================================")
# The square's half-size is the largest circle's radius plus padding
square_half_size = largest_circle_radius + square_padding

# Define the corners of the centered square
bottom_left_x = center_point - square_half_size
bottom_left_y = center_point - square_half_size
bottom_right_x = center_point + square_half_size
bottom_right_y = center_point - square_half_size
top_right_x = center_point + square_half_size
top_right_y = center_point + square_half_size
top_left_x = center_point - square_half_size
top_left_y = center_point + square_half_size

# Move to the starting corner (bottom-left)
print(f"G0 F{travel_speed} X{bottom_left_x:.4f} Y{bottom_left_y:.4f} ; Move to square start point")
# Draw the four sides
print(f"G1 F{trace_speed} X{bottom_right_x:.4f} Y{bottom_right_y:.4f} ; Draw bottom side")
print(f"G1 X{top_right_x:.4f} Y{top_right_y:.4f} ; Draw right side")
print(f"G1 X{top_left_x:.4f} Y{top_left_y:.4f} ; Draw top side")
print(f"G1 X{bottom_left_x:.4f} Y{bottom_left_y:.4f} ; Draw left side (close square)")
print("")


# --- Part 3: Concentric Circles ---
print("; ========================================")
print(f"; Part 3: Concentric Circles ({num_concentric_circles} total)")
print("; ========================================")
# Loop through each calculated radius and draw the circle
for i, radius in enumerate(circle_radii):
    # Calculate the circle's start point (3 o'clock position from the center)
    circle_start_x = center_point + radius
    circle_start_y = center_point

    # Move to the circle's start point
    print(f"G0 F{travel_speed} X{circle_start_x:.4f} Y{circle_start_y:.4f} ; Move to start of circle {i+1} (R={radius:.2f})")

    # Draw the circle using I and J offset from the start point to the center.
    # Start is at (center + r, center). Center is at (center, center).
    # So, I = (center - (center+r)) = -r. J = (center - center) = 0.
    i_offset = -radius
    j_offset = 0
    print(f"G2 I{i_offset:.4f} J{j_offset:.4f} F{trace_speed} ; Draw full clockwise circle {i+1}")
    print("")


# --- Part 4: Finalization ---
print("; ========================================")
print("; Part 4: Finalization")
print("; ========================================")
print(f"G0 Z20 F{travel_speed} ; Return to safe Z height")


# --- Plotting ---

# Create a figure and an axes object for the plot
fig, ax = plt.subplots(figsize=(8, 8))

# Set the title and labels
ax.set_title(f'{num_concentric_circles} Concentric Circles in a Square')
ax.set_xlabel(f'X-axis (0 to {grid_size})')
ax.set_ylabel(f'Y-axis (0 to {grid_size})')

# Create and add the bounding square to the plot
square_x_coords = [bottom_left_x, bottom_right_x, top_right_x, top_left_x, bottom_left_x]
square_y_coords = [bottom_left_y, bottom_right_y, top_right_y, top_left_y, bottom_left_y]
ax.plot(square_x_coords, square_y_coords, 'g--', label='Bounding Square')

# Create and add the concentric circles to the plot
for i, radius in enumerate(circle_radii):
    # Only add one label for the legend, not for every circle
    label = 'Concentric Circles' if i == 0 else ""
    circle_patch = plt.Circle((center_point, center_point), radius, color='b', fill=False, label=label)
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