"""
Plot script for location problem data visualization

author: Sascha Zell
last revision: 2024-12-19
"""

# import packages
import folium
import folium.plugins
import branca
import branca.colormap
from collections import defaultdict
import geopandas as gpd
import shapely.geometry
import os
import json
import matplotlib.pyplot as plt
from branca.element import Figure
from folium.plugins import FloatImage
from .utils import plot_utils


def plot_locationmodel_solution_on_map(
        facility_solution: list, facilitypoints: list,
        demands_solution: list, demandpoints: list,
        hide_unselected_nodes: bool = False):
    """
    plot solution (covered demand points, open facilities) of location model
    on a folium map

    facility_solution : list
        location problem facility solution values (selected facilities)
    facilitypoints : list
        all potential facility sites
    demands_solution : list
        covered demand points
    demandpoints : list
        all demand points
    hide_potential_points : bool
        if True, hides potential facilities and demand points

    """
    # plot solution on map
    center = [facilitypoints[0][1], facilitypoints[0][0]]
    solution_map = folium.Map(
        location=center, zoom_start=13)
    open_facilities_fg = folium.FeatureGroup(
        name='Open facilities')
    potential_facilities_fg = folium.FeatureGroup(
        name='Potential facilities')
    demandpoints_fg = folium.FeatureGroup(
        name='Covered demand points')

    eps = 1e-5
    for x_var, point in zip(facility_solution, facilitypoints):
        if not hide_unselected_nodes:
            folium.Circle(
                location=[point[1], point[0]], radius=2, color='grey',
                fill=True, fill_color='blue').add_to(potential_facilities_fg)

        if abs(x_var - 1.0) < eps:
            folium.Circle(
                location=[point[1], point[0]], radius=2, color='blue',
                fill=True, fill_color='blue').add_to(open_facilities_fg)

    potential_facilities_fg.add_to(solution_map)
    open_facilities_fg.add_to(solution_map)

    # define color map
    colormap = branca.colormap.LinearColormap(
        colors=['yellow', 'red'],
        index=[0, 100], vmin=0, vmax=100,
        caption='Demand point coverage in %')

    for point, demand_weight in zip(demandpoints, demands_solution):
        # convert weight to color
        color = colormap(demand_weight*100.0)
        folium.Circle(
            location=[point[1], point[0]], radius=2, color=color,
            fill=True, fill_color=color).add_to(demandpoints_fg)

    demandpoints_fg.add_to(solution_map)
    solution_map.add_child(colormap)

    # add layercontrol
    folium.LayerControl(collapsed=False).add_to(solution_map)

    return solution_map


def plot_locationmodel_solution_on_map_multiple(
        facility_solution: dict, facilitypoints: list,
        demands_solution: list, demandpoints: list,
        hide_unselected_nodes: bool = False):
    """
    plot solution (covered demand points, open facilities) of location model
    on a folium map

    facility_solution : dict
        location problem facility solution values (selected facilities)
    facilitypoints : list
        all potential facility sites
    demands_solution : list
        covered demand points
    demandpoints : list
        all demand points
    hide_potential_points : bool
        if True, hides potential facilities and demand points

    """
    # plot solution on map
    center = [facilitypoints[0][1], facilitypoints[0][0]]
    solution_map = folium.Map(
        location=center, zoom_start=13)
    open_facilities_fg = folium.FeatureGroup(
        name='Open facilities')
    potential_facilities_fg = folium.FeatureGroup(
        name='Potential facilities')
    demandpoints_fg = folium.FeatureGroup(
        name='Covered demand points')

    # count number of servers to be locted at facilities
    x_counts = defaultdict(int)
    for (x_var, _), value in facility_solution.items():
        if value == 1.0:
            x_counts[x_var] += 1
    for i, point in enumerate(facilitypoints):
        if not hide_unselected_nodes:
            folium.Circle(
                location=[point[1], point[0]], radius=2, color='grey',
                fill=True, fill_color='blue').add_to(potential_facilities_fg)
        if i in x_counts:
            circle_size = x_counts[i]*2
            folium.Circle(
                location=[point[1], point[0]], radius=circle_size,
                color='blue', fill=True, fill_color='blue',
                tooltip=f"servers: {x_counts[i]}").add_to(open_facilities_fg)

    potential_facilities_fg.add_to(solution_map)
    open_facilities_fg.add_to(solution_map)

    # define color map
    colormap = branca.colormap.LinearColormap(
        colors=['yellow', 'red'],
        index=[0, 100], vmin=0, vmax=100,
        caption='Demand point coverage in %')

    for point, demand_weight in zip(demandpoints, demands_solution):
        # convert weight to color
        color = colormap(demand_weight*100.0)
        folium.Circle(
            location=[point[1], point[0]], radius=2, color=color,
            fill=True, fill_color=color).add_to(demandpoints_fg)

    demandpoints_fg.add_to(solution_map)
    solution_map.add_child(colormap)

    # add layercontrol
    folium.LayerControl(collapsed=False).add_to(solution_map)

    return solution_map


def plot_gridded_area(pol: shapely.geometry.Polygon, grid: list):
    """
    plot area of interest on folium map
    """
    # create folium map
    map_lon, map_lat = pol.exterior.xy[0][0], pol.exterior.xy[1][0]
    map = folium.Map(location=[map_lat, map_lon], zoom_start=11)

    # create featuregroups
    area_fg = folium.FeatureGroup(
        name='Area of interest')
    grid_fg = folium.FeatureGroup(
        name='Potential facility sites')

    # add polygon to fg
    folium.GeoJson(
        shapely.geometry.mapping(pol),
        style_function=lambda feature: {
            'fillColor': '#00000000',
            'color': 'green',
            'weight': 2
        }).add_to(area_fg)

    # add grid to fg
    for point in grid:
        folium.Circle(
            location=[point[1], point[0]], radius=2, fill=True,
            color='black').add_to(grid_fg)

    # add feature groups to map
    area_fg.add_to(map)
    grid_fg.add_to(map)

    # add layercontrol
    folium.LayerControl(collapsed=False).add_to(map)

    return map


def plot_demand_heatmap():
    """
    plot heatmap of demand point weighting

    Parameters
    ----------

    """
    # TODO


def plot_emergency_locations(emergency_gdf: gpd.GeoDataFrame):
    """
    plot emergency locations from geopandas dataframe

    Parameters
    ----------
    data : gpd.GeoDataFrame
        data dictionary (e.g. from get_emergency_data())

    Returns
    -------
    emergency_map : folium.Map()
        emegncy folium map
    """
    # define feature groups for fatal and non-fatal emergencies
    normal_icon_html = (
        "<i class='fa-solid fa-circle' style='color: black; "
        "font-size: 16px;'></i>")
    normal_html_layer_name = f"{normal_icon_html} Non-fatal emergencies"
    nonfatal_fg = folium.FeatureGroup(name=normal_html_layer_name, show=True)

    fatal_icon_html = (
        "<i class='fa-solid fa-circle' style='color: red; "
        "font-size: 16px;'></i>")
    water_html_layer_name = f"{fatal_icon_html}Fatal (deadly) emergencies"
    fatal_fg = folium.FeatureGroup(
        name=water_html_layer_name, show=True)

    center_coord = emergency_gdf['geometry'][0]
    emergency_map = folium.Map(
        location=[center_coord.y, center_coord.x], zoom_start=11)

    # iterate GPD
    for index, row in emergency_gdf.iterrows():
        coordinates = row['geometry'].coords.xy
        emergencies = row.get('emergencies', 0)
        deaths = row.get('deaths', 0)
        # draw circle marker
        for i in range(len(coordinates[0])):
            marker = folium.CircleMarker(
                location=[coordinates[1][i], coordinates[0][i]],
                radius=2,
                color='red' if deaths > 0 else 'blue',  # color based on deaths
                fill=True,
                fill_color='red' if deaths > 0 else 'blue',
                fill_opacity=0.7,
                tooltip=(
                    f"GipfelID: {index}<br>Emergencies: {emergencies} "
                    f"<br>Deaths: {deaths}")
            )
            if deaths > 0:
                marker.add_to(fatal_fg)
            else:
                marker.add_to(nonfatal_fg)
    # add feature groups to map
    fatal_fg.add_to(emergency_map)
    nonfatal_fg.add_to(emergency_map)

    # add layercontrol
    folium.LayerControl(collapsed=False).add_to(emergency_map)

    return emergency_map


class Plotter:
    """
    Plot class

    Methods
    -------
    """

    SUPPORTED_LANGUAGES = ["german", "english"]
    COLORS = [
        "#04942b", "#7a3f04", "#BEB800", "#FC8900", "#9EFFAF", "#A3AEAD",
        "#1B00FB", "#BA00FB", "#F800FB", "#A6F900", "#FFB1DD", "#970000",
        "#006120", "#FFF2F2", "#4E5A50", "#021091", "#62B6FB", "#62FBD1",
        "#AD89FF"
    ]

    @staticmethod
    def plot_openstreetmap_features():
        """
        plot openstreetmap features on folium map

        Parameters
        ----------
        """

    @staticmethod
    def plot_heatmap_old(
            heat_data: list,
            add_colorbar: bool = True, branca_colormap: str = "Spectral_11",
            center: tuple = None, reverse_colormap: bool = True,
            language: str = "german", save: bool = False,
            html_outdir: str = 'heatmap.html', demand_points: list = None,
            water_pol: shapely.geometry.MultiPolygon = None):
        """
        plot demand point weighting heat map on folium

        Parameters
        ----------
        heat_data : list
            heat data in format [[lat,lon, weight], ...]
        add_colorbar: bool = True
            if True, adds colorbar to plot, if not uses default heatmap
            without colorbar
        branca_colormap: str = "Spectral_11"
            branca linear colormap (from branca.colormap.linear), for a list
            check >>> dir(branca.colormap.linear)
        reverse_colormap : bool = True
            If True, reverse defined branca_colormap
        center: tuple = None
            center of the folium map in lat,lon format
        language : str = "german"
            plot legend language, one of "german", "english".
        save : bool = False
            if True, saves html output to html_outdir
        html_outdir : str
            HTML folium output destination (relative or absolute path)
        demand_points : list = None
            plots discrete demand points on heat map
        water_pol: shapely.geometry.MultiPolygon = None
            if defined shapely.geometry.Multipolygon, add to heat map
        """
        if language not in Plotter.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language {language} not supported.")

        if language == "german":
            words = [
                "Gewichtungs-Wärmekarte", "Nachfragepunkt",
                "Nachfragewert", "Nachfragewert (skaliert auf [0,1])"]
        elif language == "english":
            words = [
                "Demand Score Heatmap", "Demand point",
                "Score Value", "Demand value (scaled to [0,1])"
            ]

        if not center:
            center = [heat_data[0][1], heat_data[0][0]]

        # initialize map
        map = folium.Map(
            tiles=None, zoom_start=12, location=[center[1], center[0]])
        folium.raster_layers.TileLayer(
            tiles="openstreetmap", name=f"{words[0]}").add_to(
                map)

        # define heatmap icon for legend
        demand_icon_html = """
            <svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' \
                fill='currentColor'>
            <rect width='100%' height='100%' style='fill: url(#gradient);'/>
            <linearGradient id='gradient' gradientTransform='rotate(90)'>
            <stop offset='0%' style='stop-color: blue' />
            <stop offset='25%' style='stop-color: green' />
            <stop offset='50%' style='stop-color: yellow' />
            <stop offset='75%' style='stop-color: orange' />
            <stop offset='100%' style='stop-color: red' />
            </linearGradient>
            </svg>
        """

        demand_html_layer_name = f"{demand_icon_html} {words[2]}"
        demand_feature_group = folium.FeatureGroup(
            name=demand_html_layer_name, show=True)
        # get colormap
        if add_colorbar:
            steps = 20
            colormap = getattr(
                branca.colormap.linear, branca_colormap).scale(0, 1).to_step(
                    steps)
            # reverse colormap
            if reverse_colormap:
                folium_colormap = branca.colormap.LinearColormap(
                    colors=list(reversed(colormap.colors)),
                    vmin=colormap.vmin, vmax=colormap.vmax,
                    caption=words[3])
            else:
                folium_colormap = branca.colormap.LinearColormap(
                    colors=list(colormap.colors),
                    vmin=colormap.vmin, vmax=colormap.vmax,
                    caption=words[3])
            # add colormap to folium map
            folium_colormap.add_to(map)

            # define gradient
            gradient_map = defaultdict(dict)
            for i in range(steps):
                gradient_map[1/steps*i] = folium_colormap.rgb_hex_str(
                    1/steps*i)

            # add heat map
            folium.plugins.HeatMap(
                heat_data, gradient=gradient_map, radius=15, blur=19,
                min_opacity=0.0).add_to(demand_feature_group)
        else:
            # add default heat map
            folium.plugins.HeatMap(
                heat_data, radius=10, blur=4, min_opacity=0.3).add_to(
                    demand_feature_group)
        demand_feature_group.add_to(map)

        # add water area outline
        if water_pol:
            folium.GeoJson(
                shapely.geometry.mapping(water_pol),
                style_function=lambda feature: {
                    'fillColor': '#00000000',
                    'color': 'black',
                    'weight': 2
                }).add_to(demand_feature_group)

        # plot demand points
        if demand_points:
            demandpoint_icon_html = (
                "<i class='fa-regular fa-circle' style='color: black; "
                "font-size: 16px;'></i>")
            demandpoint_html_layer_name = f"{demandpoint_icon_html} {words[1]}"
            demandpoint_feature_group = folium.FeatureGroup(
                name=demandpoint_html_layer_name, show=True)

            for point in demand_points:
                folium.Circle(
                    location=[point[1], point[0]], radius=48, color='black',
                    fill=False, fill_opacity=0).add_to(
                        demandpoint_feature_group)
            demandpoint_feature_group.add_to(map)

        # add layercontrol
        folium.map.LayerControl(collapsed=False).add_to(map)

        # save map
        if save:
            with open(html_outdir, 'w') as f:
                f.write(map._repr_html_())
        return map

    @staticmethod
    def plot_runtime_boxplots(
            data_dir: str, save: bool = False,
            y_value_column: str = "runtime_seconds",
            x_value_column: str = "no_waypoints",
            x_label: str = "Number of waypoints",
            y_label: str = "Runtime",
            save_dir: str = None, datname: str = None,
            title: str = "Runtime Analysis Boxplot", unit: str = 'seconds',
            ):
        """
        plot runtime boxplots

        Parameters
        ----------
        data_dir : str
            relative or absolute path of data directory
        save : bool
            True: save plot, False: display plot
        y_value_column: str = "runtime_seconds"
            value column of runtime (y-axis)
        x_value_column: str = "runtime_seconds"
            value column of x axis (x-axis)
        x_label: str = "Number of waypoints"
            x-axis label
        y_label: str = "Runtime"
            y-axis label
        save_dir : str
            directory to save boxplot
            (if None, directory where .json data is stored)
        datname : str
            name of plot .png file
        """
        # get data files in data directory
        try:
            files = [
                f'{data_dir}/{json_file}'
                for json_file in os.listdir(data_dir)
                if json_file.endswith('.json')]
        except FileNotFoundError:
            raise ValueError(
                f"Data for Boxplot in directory {data_dir} not found.")

        if not files:
            raise ValueError(
                f"Not data files in directory {data_dir}.")

        # initialize data lists
        y_list = []
        x_list = []

        # read in json file from case_list
        for case in files:
            json_data = None
            with open(case) as json_file:
                json_data = json.load(json_file)
            if not json_data:
                print(f"File {json_data} could not be read.")
            else:
                # check if data file corrupted
                values = [x_value_column, y_value_column]
                for val in values:
                    if val not in json_data.keys():
                        print(
                            f"Damaged data file {json_data},"
                            f"key {val} not found.")
                # collect data
                if unit == "hours":
                    y_list.append(json_data[y_value_column] / 3600.0)
                elif unit == "minutes":
                    y_list.append(json_data[y_value_column] / 60.0)
                elif unit == "seconds":
                    y_list.append(json_data[y_value_column])
                x_list.append(
                    json_data[x_value_column])

        # restore x values
        x_values = list(set(x_list))
        boxplot_data = [[] for _ in x_values]

        # restore y values for boxplot
        for no, y_val in enumerate(y_list):
            boxplot_data[no].append(y_val)

        # boxplot
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_figwidth(16)
        fig.set_figheight(9)
        ax.boxplot(boxplot_data, labels=x_values)
        plt.title(title)
        ax.set_ylabel(f"{x_label} [{unit}]")
        ax.set_xlabel(f"{y_label}")

        # save or display plot
        if save:
            # choose directory
            if not save_dir:
                save_dir = data_dir
            # choose name for output file
            if not datname:
                datname = "uav"
            save_name = f"{save_dir}/{datname}_boxplot.pdf"

            # make directory if not exist
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # save figure
            plt.savefig(save_name, format="pdf", dpi=400)

        else:
            plt.show()

    @staticmethod
    def plot_locationmodel_solution_on_folium_map(
            data: dict, save_to: str = "", show: bool = False,
            plot_potential_facilities: bool = True,
            plot_demandpoints: bool = True,
            plot_covered_points: bool = True):
        """
        plot solution of location model (model.LocOpt) on a folium map

        Parameters
        ----------
        data: dict
            run data dictionary
        save_to: str
            if specified, saves folium map to this directory
        show: bool = False
            if True, returns folium map instead of saving
        plot_potential_facilities: bool = True
            if True, plots potential facility sites on map
        plot_demandpoints: bool = True
            if True, plots demand points on map
        plot_covered_points: bool = True
            if True, plots covered demand points on map
        """

        # initialize folium map
        center = data['facility_points'][0]
        loc_map = folium.Map(
            tiles=None,
            location=[center[1], center[0]],
            zoom_start=12)

        # add layers
        folium.raster_layers.TileLayer(
            tiles='openstreetmap', name='OpenStreetMap').add_to(loc_map)

        # define potential facility icon
        potential_facility_html = (
            "<i class='fa-solid fa-circle' style='color: #57edf2; font-size: "
            "16px;'></i>")
        potential_facility_layer_name = (
            f"{potential_facility_html} Potential facility sites")
        potential_facility_feature_group = folium.FeatureGroup(
            name=potential_facility_layer_name, show=True)

        # define chosen facility icon
        hangar_feature_groups = []
        facility_colors = [
            "#f72525", "#ECFA12", "#B4FA12", "#55F912", "#47FCD3", "#32B1AD",
            "#009F22"
        ]

        # add chosen facilities to map
        legend_htmls = []
        hangar_no = 0
        # check if there is a feasible solution
        if 'results' in data.keys():
            if 'termination_condition' in data['results']:
                for i, dat1 in enumerate(data['chosen_hangars']):
                    # add facility icons feature groups
                    # blank_facility_icon_html = (
                    #     f"<i class='fas fa-house' style='color: "
                    #     f"{facility_colors[i]}; "
                    #     f"font-size: 18px; position: relative;'>"
                    #     f"<span style='position: absolute; top: 50%; "
                    #     f"left: 50%; transform: translate(-50%, -50%);"
                    #     f"border: 13px solid black; border-radius: 50%; "
                    #     f"z-index: -1;'></span></i>")
                    blank_facility_legend_icon_html = (
                        f"<i class='fas fa-house' style='color:"
                        f" {facility_colors[i]}; "
                        f"font-size: 12px; position: relative;'>"
                        f"<span style='position: absolute; top: 50%; "
                        f"left: 50%; transform: translate(-50%, -50%);"
                        f"border: 8px solid black; border-radius: 50%; "
                        f"z-index: -1;'></span></i>")
                    # blank_facility_html = (
                    #     f"<div>{blank_facility_icon_html}</div>")
                    blank_facility_layer_name = (
                        f"{blank_facility_legend_icon_html} UAV Hangar: "
                        f"{data['facility_names'][i]}")
                    hangar_feature_group = folium.FeatureGroup(
                        name=blank_facility_layer_name, show=True)
                    hangar_feature_groups.append(hangar_feature_group)

                    # add chosen facilities to map and legend
                    hangar_no = 0
                    for j, (dat2, point) in enumerate(
                            zip(dat1, data['facility_points'])):
                        folium.Circle(
                            location=[point[1], point[0]], radius=80,
                            color="#57edf2", fill=True, fill_opacity=1).add_to(
                                potential_facility_feature_group)
                        if dat2 > 0.0001:
                            hangar_no += 1
                            facility_icon_html = (
                                f"<i class='fas fa-house' style='color: "
                                f"{facility_colors[i]}; "
                                f"font-size: 18px; position: relative;'>"
                                f"<span style='position: absolute; top: -20%; "
                                f"left: 50%; "
                                f"transform: translateX(-50%); border: 13px "
                                f"solid black; "
                                f"border-radius: 50%; z-index: -1;'></span>"
                                f"<span style='position: absolute; top: 35%; "
                                f"left: 50%; "
                                f"transform: translate(-50%, -50%); color: "
                                f"black; font-weight: bold; font-size: 9px;"
                                f"'>{hangar_no}</span></i>")
                            """
                            facility_legend_icon_html = (
                                f"<i class='fas fa-house' style='color: "
                                f"{facility_colors[i]}; "
                                f"font-size: 12px; position: relative;'>"
                                f"<span style='position: absolute; top: -20%; "
                                f"left: 50%; "
                                f"transform: translateX(-50%); border: 8px "
                                f"solid black; "
                                f"border-radius: 50%; z-index: -1;'></span>"
                                f"<span style='position: absolute; top: 35%; "
                                f"left: 50%; "
                                f"transform: translate(-50%, -50%); color:"
                                f" black; font-weight: bold; font-size: 7px;"
                                f"'>{hangar_no}</span></i>")
                            """

                            facility_html = f"<div>{facility_icon_html}</div>"
                            # facility_layer_name = (
                            #     f"{facility_legend_icon_html} UAV Hangar: "
                            #     f"{data['facility_names'][i]}")
                            legend_htmls.append(
                                f"<p style='margin: 0px 0 0px 0;'>"
                                f" {facility_icon_html}  "
                                f"{data['facility_names'][i]} {hangar_no}</p>")

                            # add chosen vehicles
                            lines = []
                            for k, _ in enumerate(data['vehicles_sizes']):
                                if data['chosen_vehicles'][i][k][j] > 0.1:
                                    ceiled_chosen_vehicle = round(
                                        data['chosen_vehicles'][i][k][j])
                                    tooltip_label = (
                                            f"No. of "
                                            f"{data['vehicle_names'][k]}s: "
                                            f"<b>{ceiled_chosen_vehicle}</b>"
                                        )
                                    lines.append(tooltip_label)
                                    legend_htmls.append(
                                        f"<p style='margin: -10px 0;"
                                        f" text-indent: "
                                        f"30px; '>{tooltip_label}</p>")
                            tooltip_label_map = "<br>".join(lines)
                            folium.Marker(
                                [point[1], point[0]],
                                popup=folium.Popup(
                                    f"{tooltip_label_map}", show=True,
                                    max_width=200),
                                icon=folium.DivIcon(
                                    html=facility_html
                                    )).add_to(hangar_feature_groups[i])
                            loc_map.add_child(hangar_feature_groups[i])
                # legend_htmls.append(
                #     "<p style='margin: 1px 0 1px 0;'>&nbsp;</p>")
            else:
                # plot potential facilities sites
                for point in data['facility_points']:
                    folium.Circle(
                        location=[point[1], point[0]], radius=80,
                        color='#57edf2', fill=True, fill_opacity=1).add_to(
                            potential_facility_feature_group)
                print("Model could not be solved.")
        else:
            print("Model was not solved.")

        if plot_potential_facilities:
            loc_map.add_child(potential_facility_feature_group)

        # plot demand points
        if plot_demandpoints:
            demandpoint_icon_html = (
                "<i class='fa-solid fa-circle' style='color: black; "
                "font-size: 16px;'></i>")
            demandpoint_html_layer_name = (
                f"{demandpoint_icon_html} Demand point")
            demandpoint_feature_group = folium.FeatureGroup(
                name=demandpoint_html_layer_name, show=True)

            for point in data['demand_points']:
                folium.Circle(
                    location=[point[1], point[0]], radius=80, color='black',
                    fill=False, fill_opacity=1).add_to(
                        demandpoint_feature_group)

            demandpoint_feature_group.add_to(loc_map)

        # plot covered demand points
        covered_feature_groups = []
        vehicle_colors = [
            "#FFFFFF", "#f50a97", "#ff911c", "#4afcff", "#F7612A", "#acf76d",
            "#9051cf"
        ]
        if plot_covered_points:
            # iterate types
            if data['optimal_battery_covering_binary']:
                for i, (covered_per_type, contribution_name) in enumerate(
                        zip(
                            data['optimal_battery_covering_binary'],
                            data['contribution_types'])):
                    cov_color = vehicle_colors.pop(0)
                    vehicle_colors.append(cov_color)
                    # define feature group
                    covered_icon_html = (
                        f"<i class='fa-solid fa-circle' style='color:"
                        f" {cov_color}; font-size: 16px;'></i>")
                    covered_html_layer_name = (
                        f"{covered_icon_html} Battery covered demandpoints: "
                        f"{contribution_name}")
                    covered_feature_group = folium.FeatureGroup(
                        name=covered_html_layer_name, show=True)
                    for covering_binary, point in zip(
                            covered_per_type, data['demand_points']):
                        if covering_binary > 0.1:
                            folium.Circle(
                                location=[point[1], point[0]], radius=80,
                                color=cov_color, fill=False,
                                fill_opacity=0).add_to(covered_feature_group)

                    covered_feature_groups.append(covered_feature_group)
            if data['optimal_time_covering_binary']:
                for i, (covered_per_type, contribution_name) in enumerate(
                        zip(
                            data['optimal_time_covering_binary'],
                            data['contribution_types'])):
                    cov_color = vehicle_colors.pop(0)
                    vehicle_colors.append(cov_color)
                    # define feature group
                    covered_icon_html = (
                        f"<i class='fa-solid fa-circle' style='color:"
                        f" {cov_color}; font-size: 16px;'></i>")
                    covered_html_layer_name = (
                        f"{covered_icon_html} Time covered demandpoints: "
                        f"{contribution_name}")
                    covered_feature_group = folium.FeatureGroup(
                        name=covered_html_layer_name, show=True)
                    for covering_binary, point in zip(
                            covered_per_type, data['demand_points']):
                        if covering_binary > 0.1:
                            folium.Circle(
                                location=[point[1], point[0]], radius=80,
                                color=cov_color, fill=False,
                                fill_opacity=0).add_to(covered_feature_group)

                    covered_feature_groups.append(covered_feature_group)

        for feature_group in covered_feature_groups:
            loc_map.add_child(feature_group)

        # restricted areas
        if 'ras_coords' in data.keys():
            if data['ras_coords']:
                ras_icon_html = (
                    """
                    <svg xmlns='http://www.w3.org/2000/svg' width='16'\
                        height='16' fill='currentColor'>
                    <rect width='100%' height='100%' style='fill: red;'/>
                    </svg>
                    """)
                ras_html_layer_name = (
                    f"{ras_icon_html} Restricted Areas")
                ras_feature_group = folium.FeatureGroup(
                        name=ras_html_layer_name, show=True)
                for coords in data['ras_coords']:
                    print("Plot RAS")
                    # add water area outline
                    folium.GeoJson(
                        shapely.geometry.Polygon(coords),
                        style_function=lambda feature: {
                            'fillColor': '#FF6347',
                            'color': 'red',
                            'weight': 2
                        }).add_to(ras_feature_group)
                ras_feature_group.add_to(loc_map)

        # add layercontrol
        folium.map.LayerControl(collapsed=False).add_to(loc_map)

        # add attribution control

        attribution = (
            "<a href='https://leafletjs.com'>Leaflet</a>, Data by <a "
            "href='https://www.openstreetmap.org/'>OpenStreetMap</a>, under "
            "<a href='https://opendatacommons.org/licenses/odbl/'>ODbL.</a>"
        )

        # Create a custom HTML control for the attribution
        attribution_html = f"""
            <div style="
                position: fixed;
                bottom: 0px;
                right: 0px;
                background-color: white;
                z-index: 100000000000;
                padding: 5px;
                border: 1px solid #ccc;
                font-size: 11px; /* Adjust font size */
                width: 250px; /* Adjust box width */
                ">
                {attribution}
            </div>
            """

        # make button
        figure = Figure()
        button = FloatImage(attribution_html, bottom=65, left=10)
        figure.add_child(button)
        button_element = folium.Element(attribution_html)
        loc_map.get_root().html.add_child(button_element)

        # add external legend
        legend_html = "<br>".join(legend_htmls)
        legend_div = f"""
            <div style="
                position: fixed;
                top: 100px;
                left: 10px;
                background-color: white;
                border: 1px solid black;
                padding: 10px;
                z-index: 1000;
                ">
                <h4>Hangar occupancy</h4>
                {legend_html}
            </div>
            """
        loc_map.get_root().html.add_child(folium.Element(legend_div))

        # save html map
        if save_to:
            loc_map.save(f"{save_to}")
        if show:
            return loc_map

    @staticmethod
    def plot_area_on_folium_map(area_polygon: shapely.geometry.Polygon):
        """
        plot the operational area on a folium map

        Parameters
        ----------
        area_polygon: shapely.geometry.Polygon
            operational area polygon

        Returns
        -------
        area_map : folium.Map
            folium map showing operational area
        """
        # display area on folium map
        area_centroid = [area_polygon.centroid.y, area_polygon.centroid.x]
        area_map = folium.Map(
            tiles=None, location=area_centroid, zoom_start=8)

        # add OSM tile to map
        folium.raster_layers.TileLayer(
            tiles='openstreetmap', name='OpenStreetMap').add_to(area_map)

        # prepare legend
        area_html = (
            "<svg width='20' height='20'>"
            "<circle cx='10' cy='10' r='8' "
            "stroke='red' stroke-width='2' fill=None />"
            "</svg>")

        area_name = f"{area_html} Operational Area"
        area_feature_group = folium.FeatureGroup(
            name=area_name, show=True)

        # add area to map
        folium.GeoJson(
            area_polygon.__geo_interface__,
            style_function=lambda x: {
                "color": "red",  # border color
                "weight": 2,  # border thickness
                "fillColor": "none",  # no fill color
                "fillOpacity": 0  # completely transparent fill
            }).add_to(area_feature_group)
        area_feature_group.add_to(area_map)

        # add layercontrol
        folium.LayerControl(collapsed=False).add_to(area_map)

        # add attribution
        plot_utils.plot_attribution(area_map)

        return area_map

    @staticmethod
    def plot_osm_features_on_folium_map(
            area_polygon: shapely.geometry.Polygon,
            osm_data: dict, osm_features: dict):
        """
        plot the openstreetmap features on a folium map

        Parameters
        ----------
        area_polygon: shapely.geometry.Polygon
            operational area polygon
        osm_data: dict
            openstreetmap fetched data (from osmnx)
        osm_features: dict
            openstreetmap key value pairs

        Returns
        -------
        area_map : folium.Map
            folium map showing operational area
        """

        # initialize osm folium map
        area_centroid = [area_polygon.centroid.y, area_polygon.centroid.x]
        osm_map = folium.Map(
            tiles=None, location=area_centroid, zoom_start=8)
        folium.raster_layers.TileLayer(
            tiles='openstreetmap', name='OpenStreetMap').add_to(osm_map)

        area_html = (
            "<svg width='20' height='20'>"
            "<circle cx='10' cy='10' r='8' "
            "stroke='red' stroke-width='2' fill=None />"
            "</svg>")

        area_name = f"{area_html} Operational Area"
        area_feature_group = folium.FeatureGroup(
            name=area_name, show=True)

        # add area to map
        folium.GeoJson(
            area_polygon.__geo_interface__,
            style_function=lambda x: {
                "color": "red",  # border color
                "weight": 2,  # border thickness
                "fillColor": "none",  # no fill color
                "fillOpacity": 0  # completely transparent fill
            }).add_to(area_feature_group)
        area_feature_group.add_to(osm_map)

        for (key, item), color in zip(osm_data.items(), Plotter.COLORS):
            if item:
                # initialize feature group per key
                # fg_html = (
                #     f"<svg width='32' height='32'>"
                #     f"<circle cx='16' cy='16' r='14' "
                #     f"stroke='blue' stroke-width='2' fill='lightblue' />"
                #     f"</svg>")

                fg_html = (
                    f"<svg width='20' height='20' "
                    f"xmlns='http://www.w3.org/2000/svg'>"
                    f"  <defs>"
                    f"    <filter id='lighten'>"
                    f"      <feComponentTransfer>"
                    f"        <feFuncR type='linear' slope='1.5' />"
                    f"        <feFuncG type='linear' slope='1.5' />"
                    f"        <feFuncB type='linear' slope='1.5' />"
                    f"      </feComponentTransfer>"
                    f"    </filter>"
                    f"  </defs>"
                    f"  <circle cx='10' cy='10' r='8' "
                    f"          stroke={color} stroke-width='2' "
                    f"          fill={color} filter='url(#lighten)' />"
                    f"</svg>")
                for k, values in osm_features.items():
                    if key in values:
                        feature_name = k
                fg_name = f"{fg_html} {feature_name} = {key}"
                fg = folium.FeatureGroup(name=fg_name, show=True)
                folium.GeoJson(
                    shapely.geometry.shape(item),
                    style_function=lambda feature, color=color: {
                        'fillColor': color, 'color': color, 'weight': 2
                    }).add_to(fg)
                fg.add_to(osm_map)

        # add layercontrol
        folium.LayerControl(collapsed=False).add_to(osm_map)

        plot_utils.plot_attribution(osm_map)

        return osm_map

    @staticmethod
    def plot_heatmap(
            heat_data: list,
            add_colorbar: bool = True, branca_colormap: str = "Spectral_11",
            center: tuple = None, reverse_colormap: bool = True,
            language: str = "english", demand_points: list = None,
            water_pol: shapely.geometry.MultiPolygon = None):
        """
        plot demand point weighting heat map on folium

        Parameters
        ----------
        heat_data: list
            heat data in format [[lat, lon, weight], ...]
        add_colorbar: bool = True
            if True, adds colorbar to plot, if not uses default heatmap
            without colorbar
        branca_colormap: str = "Spectral_11"
            branca linear colormap (from branca.colormap.linear), for a list
            check >>> dir(branca.colormap.linear)
        reverse_colormap : bool = True
            If True, reverse defined branca_colormap
        center: tuple = None
            center of the folium map in lat,lon format
        language : str = "german"
            plot legend language, one of "german", "english".
        demand_points : list = None
            plots discrete demand points on heat map
        water_pol: shapely.geometry.MultiPolygon = None
            if defined shapely.geometry.Multipolygon, add to heat map
        """
        if language not in Plotter.SUPPORTED_LANGUAGES:
            raise ValueError(f"Language {language} not supported.")
        if language == "german":
            words = [
                "Gewichtungs-Wärmekarte", "Nachfragepunkt",
                "Nachfragewert", "Nachfragewert (skaliert auf [0,1])"]
        elif language == "english":
            words = [
                "Demand Score Heatmap", "Demand point",
                "Score Value", "Demand value (scaled to [0,1])"
            ]

        if not center:
            center = [heat_data[0][0], heat_data[0][1]]

        # initialize map
        map = folium.Map(
            tiles=None, zoom_start=9, location=[center[0], center[1]])
        folium.raster_layers.TileLayer(
            tiles="openstreetmap", name=f"{words[0]}").add_to(
                map)

        # define heatmap icon for legend
        demand_icon_html = """
            <svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' \
                fill='currentColor'>
            <rect width='100%' height='100%' style='fill: url(#gradient);'/>
            <linearGradient id='gradient' gradientTransform='rotate(90)'>
            <stop offset='0%' style='stop-color: blue' />
            <stop offset='25%' style='stop-color: green' />
            <stop offset='50%' style='stop-color: yellow' />
            <stop offset='75%' style='stop-color: orange' />
            <stop offset='100%' style='stop-color: red' />
            </linearGradient>
            </svg>
        """

        demand_html_layer_name = f"{demand_icon_html} {words[2]}"
        demand_feature_group = folium.FeatureGroup(
            name=demand_html_layer_name, show=True)
        # get colormap
        if add_colorbar:
            steps = 20
            colormap = getattr(
                branca.colormap.linear, branca_colormap).scale(0, 1).to_step(
                    steps)
            # reverse colormap
            if reverse_colormap:
                folium_colormap = branca.colormap.LinearColormap(
                    colors=list(reversed(colormap.colors)),
                    vmin=colormap.vmin, vmax=colormap.vmax,
                    caption=words[3])
            else:
                folium_colormap = branca.colormap.LinearColormap(
                    colors=list(colormap.colors),
                    vmin=colormap.vmin, vmax=colormap.vmax,
                    caption=words[3])
            # add colormap to folium map
            folium_colormap.add_to(map)

            # define gradient
            gradient_map = defaultdict(dict)
            for i in range(steps):
                gradient_map[1/steps*i] = folium_colormap.rgb_hex_str(
                    1/steps*i)

            # add heat map
            folium.plugins.HeatMap(
                heat_data, gradient=gradient_map, radius=50, blur=40,
                min_opacity=0.0).add_to(demand_feature_group)
        else:
            # add default heat map
            folium.plugins.HeatMap(
                heat_data, radius=10, blur=4, min_opacity=0.3).add_to(
                    demand_feature_group)
        demand_feature_group.add_to(map)

        # add water area outline
        if water_pol:
            folium.GeoJson(
                shapely.geometry.mapping(water_pol),
                style_function=lambda feature: {
                    'fillColor': '#00000000',
                    'color': 'black',
                    'weight': 2
                }).add_to(demand_feature_group)

        # plot demand points
        if demand_points:
            demandpoint_icon_html = (
                "<i class='fa-regular fa-circle' style='color: black; "
                "font-size: 16px;'></i>")
            demandpoint_html_layer_name = f"{demandpoint_icon_html} {words[1]}"
            demandpoint_feature_group = folium.FeatureGroup(
                name=demandpoint_html_layer_name, show=True)

            for point in demand_points:
                folium.Circle(
                    location=[point[1], point[0]], radius=48, color='black',
                    fill=False, fill_opacity=0).add_to(
                        demandpoint_feature_group)
            demandpoint_feature_group.add_to(map)

        # add layercontrol
        folium.map.LayerControl(collapsed=False).add_to(map)

        # plot attribution
        plot_utils.plot_attribution(map)

        return map

    @staticmethod
    def plot_demand_and_facilities(
            demand_points: list,
            facility_grid: list):
        """
        plot demand points and potential facility points

        Parameters
        ----------
        demand_points: list
            Demand points list of shapely.geometry.Point()
        facility_grid: list
            List of [lat,lon] potential facility sites
        """
        # initialize facilities folium map
        area_centroid = demand_points[0]
        cluster_map = folium.Map(
            tiles=None, location=area_centroid, zoom_start=8)
        folium.raster_layers.TileLayer(
            tiles='openstreetmap', name='OpenStreetMap').add_to(cluster_map)

        # facility featuregroup
        demand_html = (
            "<i class='fa-solid fa-circle' style='color: red;"
            " font-size: 16px;'></i>")
        demand_name = f"{demand_html} Demand points"
        clustered_demand_grid_feature_group = folium.FeatureGroup(
            name=demand_name, show=True)

        # add demand points to map
        for point in demand_points:
            folium.Circle(
                location=[point[1], point[0]], radius=500, color="red",
                fill=True, fill_color="red", fill_opacity=1).add_to(
                    clustered_demand_grid_feature_group)

        # facility featuregroup
        facility_html = (
            "<i class='fa-solid fa-circle' style='color: blue;"
            " font-size: 16px;'></i>")
        facility_name = f"{facility_html} Potential hangar sites"
        facility_grid_feature_group = folium.FeatureGroup(
            name=facility_name, show=True)

        # add facilities to map
        for point in facility_grid:
            folium.Circle(
                location=[point[1], point[0]], radius=500, color="blue",
                fill=True, fill_color="blue", fill_opacity=1).add_to(
                    facility_grid_feature_group)

        # add layers
        clustered_demand_grid_feature_group.add_to(cluster_map)
        facility_grid_feature_group.add_to(cluster_map)

        # add layercontrol
        folium.LayerControl(collapsed=False).add_to(cluster_map)

        # add attribution
        plot_utils.plot_attribution(cluster_map)

        return cluster_map
