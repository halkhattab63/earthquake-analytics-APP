# src/analyze_kde.py

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from shapely.geometry import Point
import os

# -------------------------
# Perform KDE on latitude and longitude columns
# -------------------------
def perform_kde(df, bandwidth=0.05, grid_size=500):
    print("🧠 Performing KDE analysis...")
    coords = df[['longitude', 'latitude']].values

    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(coords)

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    sample_grid = np.vstack([xx.ravel(), yy.ravel()]).T

    log_dens = kde.score_samples(sample_grid)
    density = np.exp(log_dens).reshape(grid_size, grid_size)

    print("✅ KDE grid generated.")
    return xx, yy, density

# -------------------------
# Plot KDE heatmap
# -------------------------
def plot_kde_heatmap(xx, yy, density, output_path):
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, density, levels=100, cmap='hot')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Earthquake Density Map (KDE)")
    plt.colorbar(label='Density')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"🗺️ KDE heatmap saved to {output_path}")


# import pandas as pd
# import numpy as np
# import geopandas as gpd
# import matplotlib.pyplot as plt
# from sklearn.neighbors import KernelDensity
# from shapely.geometry import Point, Polygon
# import os
# from matplotlib.colors import LinearSegmentedColormap
# from matplotlib.patches import Patch
# import contextily as ctx

# # Aziziye, Turkey coordinates
# AZIZIYE_CENTER = (39.9443, 41.1016)  # Latitude, Longitude
# AZIZIYE_BUFFER = 0.2  # Degrees for area around center

# def perform_kde(df, bandwidth=0.01, grid_size=500):
#     """
#     Perform Kernel Density Estimation for Aziziye region
    
#     Args:
#         df: DataFrame containing 'latitude' and 'longitude' columns
#         bandwidth: KDE bandwidth parameter (smaller for local analysis)
#         grid_size: Number of grid points in each dimension
        
#     Returns:
#         xx, yy: Meshgrid coordinates
#         density: KDE density values
#         bounds: (x_min, y_min, x_max, y_max) of the analysis area
#     """
#     print("[KDE] Performing KDE analysis for Aziziye...")
#     coords = df[['longitude', 'latitude']].values
    
#     # Set bounds for Aziziye region
#     x_min, x_max = AZIZIYE_CENTER[1] - AZIZIYE_BUFFER, AZIZIYE_CENTER[1] + AZIZIYE_BUFFER
#     y_min, y_max = AZIZIYE_CENTER[0] - AZIZIYE_BUFFER, AZIZIYE_CENTER[0] + AZIZIYE_BUFFER
    
#     # Create grid
#     x_grid = np.linspace(x_min, x_max, grid_size)
#     y_grid = np.linspace(y_min, y_max, grid_size)
#     xx, yy = np.meshgrid(x_grid, y_grid)
    
#     # Perform KDE
#     kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
#     kde.fit(coords)
#     log_dens = kde.score_samples(np.vstack([xx.ravel(), yy.ravel()]).T)
#     density = np.exp(log_dens).reshape(xx.shape)
    
#     print("[KDE] KDE grid generated for Aziziye region.")
#     return xx, yy, density, (x_min, y_min, x_max, y_max)

# def plot_kde_heatmap(xx, yy, density, bounds, output_path, 
#                     boundary_file=None, service_areas=None):
#     """
#     Plot KDE heatmap for Aziziye with professional styling
    
#     Args:
#         xx, yy: Meshgrid coordinates from perform_kde()
#         density: KDE density values
#         bounds: Tuple of (x_min, y_min, x_max, y_max)
#         output_path: Path to save the output image
#         boundary_file: Optional path to boundary shapefile
#         service_areas: Optional dictionary of service areas {name: distance_km}
#     """
#     plt.figure(figsize=(12, 10))
#     ax = plt.gca()
    
#     # Custom colormap for earthquake density
#     cmap = LinearSegmentedColormap.from_list(
#         'erdemli', ['#f7f7f7', '#fee8c8', '#fdbb84', '#e34a33', '#7f0000'], N=256)
    
#     # Plot KDE contours
#     levels = np.linspace(0, density.max(), 100)
#     contour = ax.contourf(xx, yy, density, levels=levels, cmap=cmap, alpha=0.8)
    
#     # Add basemap
#     try:
#         ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
#     except Exception as e:
#         print(f"[WARNING] Basemap not available: {str(e)}")
    
#     # Add boundary if provided
#     if boundary_file and os.path.exists(boundary_file):
#         try:
#             boundary = gpd.read_file(boundary_file)
#             boundary.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=2, 
#                          linestyle='--', label='Aziziye Boundary')
#         except Exception as e:
#             print(f"[WARNING] Boundary file error: {str(e)}")
    
#     # Add service areas (adjusted for Aziziye scale)
#     if service_areas:
#         center_x, center_y = AZIZIYE_CENTER[1], AZIZIYE_CENTER[0]
#         for name, dist in service_areas.items():
#             # Convert km to degrees (approximate for this latitude)
#             radius = dist * 0.009  # ~1km = 0.009 degrees at this latitude
#             circle = plt.Circle((center_x, center_y), radius,
#                               color='green', fill=False, linestyle=':', 
#                               linewidth=1.5, label=f'{name} ({dist}km)')
#             ax.add_patch(circle)
    
#     # Style the plot for Aziziye
#     plt.xlabel("Longitude", fontsize=12)
#     plt.ylabel("Latitude", fontsize=12)
#     plt.title(f"Earthquake Density Map - Aziziye, Turkey\n({bounds[1]:.4f}N to {bounds[3]:.4f}N, {bounds[0]:.4f}E to {bounds[2]:.4f}E)", 
#               fontsize=14, pad=20)
    
#     # Add legend and colorbar
#     handles, labels = ax.get_legend_handles_labels()
#     if handles:
#         plt.legend(handles=handles, loc='upper right', framealpha=1)
    
#     cbar = plt.colorbar(contour, shrink=0.7)
#     cbar.set_label('Earthquake Density', rotation=270, labelpad=20)
    
#     # Set bounds
#     plt.xlim(bounds[0], bounds[2])
#     plt.ylim(bounds[1], bounds[3])
    
#     # Save output
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"[KDE] Aziziye earthquake map saved to {output_path}")

# if __name__ == "__main__":
#     # Example data - replace with your actual Aziziye earthquake data
#     np.random.seed(42)
#     num_points = 500
#     df = pd.DataFrame({
#         'latitude': np.random.normal(AZIZIYE_CENTER[0], 0.05, num_points),
#         'longitude': np.random.normal(AZIZIYE_CENTER[1], 0.05, num_points)
#     })
    
#     # Define service areas for Aziziye (similar to your example)
#     service_areas = {
#         'Critical Zone: 2-5km': 3,
#         'High Priority: 1-4km': 1,
#         'Medium Zone: 0.5-1km': 1,
#         'Local Area: 0.3km': 0.5
#     }
    
#     # Perform KDE and plot
#     xx, yy, density, bounds = perform_kde(df, bandwidth=0.005)
    
#     plot_kde_heatmap(
#         xx, yy, density, bounds,
#         output_path='output/aziziye_earthquake_density.png',
#         boundary_file=None,  # Add path if you have Aziziye boundary shapefile
#         service_areas=service_areas
#     )