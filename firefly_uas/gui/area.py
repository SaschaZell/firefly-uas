"""
This module provides a Tkinter-based GUI for selecting areas on a map.

The GUI uses Tkintermapview to display a map and allows users to draw, edit,
and save polygons representing areas.

author: Sascha Zell
last revision: 2024-12-19
"""
import json
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

import tkintermapview
import ttkbootstrap as tb
# import tkmacosx as tm


class MapScreen(tk.Tk):
    """
    Tkinter Window to display TKintermapview map
    """
    NAME = "Tkinter Area MapViewer"
    WIDTH = 1600
    HEIGHT = 900
    LATLON = [51.521320, 14.124202]
    ZOOM = 14

    TILES = {
        "Google Maps": (
            "https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga"),
        "Google Satelite": (
            "https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga"),
        "OpenStreetMap": "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "OpenTopoMap": "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
    }

    def __init__(self):
        # configure basic window properties
        super().__init__(className="TkAreaMapView")
        self.geometry(f"{MapScreen.WIDTH}x{MapScreen.HEIGHT}")
        self.minsize(MapScreen.WIDTH, MapScreen.HEIGHT)
        self.title(MapScreen.NAME)
        self.configure(bg="white")

        # define important variables
        self.polygon_features = []  # list to store polygon features
        self.cut_polygons = {}
        self.selected_polygon_ids = []
        self.id_no = 0
        self.current_polygon_coordinates = []
        self.current_pol_widget = None
        self.pol_widget_list = []
        self.pol_osm_widget_list = []
        self.shown_pol_widget_list = []
        self.osm_polygon_coordinates = []
        self.polygon_coordinates_list = []

        # create 2 side-by-side frames to display buttons and map

        self.notebook = ttk.Notebook(self)
        self.ttk_frame = ttk.Frame()
        self.notebook.add(self.ttk_frame, text='Map')
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.config_frame = tk.Frame(self.ttk_frame, bg='white')
        self.config_frame.grid(
            row=1, column=0, padx=0, pady=0,
            sticky='NSEW')
        self.map_frame = tk.Frame(self.ttk_frame, bg='white')
        self.map_frame.grid(
            row=1, column=1, rowspan=1, padx=0, pady=0,
            sticky='NSEW')

        self.ttk_frame.grid_columnconfigure(1, weight=1)
        self.ttk_frame.grid_rowconfigure(1, weight=1)

        # configure config_frame
        rowcounter = 0
        tb.Label(
            self.config_frame, text="Define Area Polygons",
            font=('Helvetica', 20)).grid(
                row=rowcounter, column=0, padx=10, pady=10)
        rowcounter += 1

        # add draw polygon tool button and label
        self.draw_var = tk.IntVar()
        self.draw_mode_button = tb.Checkbutton(
            self.config_frame, bootstyle="primary, toolbutton, outline",
            text="Draw Mode", variable=self.draw_var, onvalue=1, offvalue=0,
            command=self.enable_drawing)
        self.draw_mode_button.grid(row=rowcounter, column=0, pady=10, padx=10)

        # add undo button for drawing
        self.undo_button = tb.Button(
            self.config_frame, bootstyle="primary",
            text="Undo", command=self.undo, state=tk.DISABLED)
        self.undo_button.grid(row=rowcounter, column=1, pady=10, padx=10)

        # add close shape button
        self.close_shape_button = tb.Button(
            self.config_frame, bootstyle="primary",
            text="Close Shape", command=self.close_shape, state=tk.DISABLED)
        self.close_shape_button.grid(
            row=rowcounter, column=2, pady=10, padx=10)

        # add add polygon button
        self.add_pol_button = tb.Button(
            self.config_frame, bootstyle="primary",
            text="Add Polygon", command=self.add_polygon, state=tk.DISABLED)
        self.add_pol_button.grid(
            row=rowcounter, column=3, pady=10, padx=10)

        rowcounter += 1

        # add add name to polygon entry
        self.polygon_name_var = tk.StringVar()
        self.polygon_name_label = ttk.Label(
            self.config_frame, text="Polygon Name:")
        self.polygon_name_label.grid(
            row=rowcounter, column=1, padx=10, pady=10)

        self.add_name_entry = tb.Entry(
            self.config_frame)
        self.add_name_entry.grid(row=rowcounter, column=2, pady=10, padx=10)

        rowcounter += 1

        """
        # TODO: Implement new features for the following buttons:
        # - Hide Polygons: Hide all polygons from the map.
        # - Show Polygons: Show all hidden polygons on the map.
        # - Select Mode: Enable selection mode to select polygons.
        # - Delete: Delete selected polygons.
        # - Intersect: Intersect selected polygons.
        # - Choose Cut: Choose polygons to intersect.
        # - Select OSM Polygon: Select a polygon from OpenStreetMap.
        # - Undo: Undo the last action in OSM mode.
        # - Close Shape: Close the current shape in OSM mode.
        # - Load: Load data from OpenStreetMap.
        # - Unload: Unload data from OpenStreetMap.
        # - Reset Polygon: Reset the current polygon in OSM mode.
        # add hide polygons button
        self.hide_pol_button = tb.Button(
            self.config_frame, bootstyle="primary",
            text="Hide Polygons", command=self.hide_polygons,
            state=tk.DISABLED)
        self.hide_pol_button.grid(
            row=rowcounter, column=1, pady=10, padx=10)

        # add show polygons button
        self.show_pol_button = tb.Button(
            self.config_frame, bootstyle="primary",
            text="Show Polygons", command=self.show_polygons,
            state=tk.DISABLED)
        self.show_pol_button.grid(
            row=rowcounter, column=3, pady=10, padx=10)
        rowcounter += 1

        # add select polygons button
        self.select_var = tk.IntVar()
        self.select_mode_button = tb.Checkbutton(
            self.config_frame, bootstyle="primary, toolbutton, outline",
            text="Select Mode", variable=self.select_var, onvalue=1,
            offvalue=0,
            command=self.enable_select_polygons)
        self.select_mode_button.grid(
            row=rowcounter, column=0, pady=10, padx=10)

        # add delete button for selection
        self.delete_button = tb.Button(
            self.config_frame, bootstyle="primary",
            text="Delete", command=self.delete_selected, state=tk.DISABLED)
        self.delete_button.grid(row=rowcounter, column=1, pady=10, padx=10)

        # add cut button
        self.cut_button = tb.Button(
            self.config_frame, bootstyle="primary",
            text="Intersect", command=self.cut_selected, state=tk.DISABLED)
        self.cut_button.grid(
            row=rowcounter, column=2, pady=10, padx=10)

        # add choose cut polygons button
        self.choose_cut_button = tb.Button(
            self.config_frame, bootstyle="primary",
            text="Choose Cut", command=self.choose_cut_polygons,
            state=tk.DISABLED)
        self.choose_cut_button.grid(
            row=rowcounter, column=3, pady=10, padx=10)
        rowcounter += 1

        # add load feature from openstreetmap button
        self.load_from_osm_label = tb.Label(
            self.config_frame, text="Load from OSM:")
        self.load_from_osm_label.grid(
            row=rowcounter, column=0, padx=10, pady=10)

        # select polygon button
        self.select_pol_var = tk.IntVar()
        self.select_pol_button = tb.Checkbutton(
            self.config_frame, bootstyle="primary, toolbutton, outline",
            text="Select OSM Polygon", variable=self.select_pol_var, onvalue=1,
            offvalue=0,
            command=self.select_pol)
        self.select_pol_button.grid(row=rowcounter, column=0, pady=10, padx=10)

        # add undo button for osm
        self.undo_osm_button = tb.Button(
            self.config_frame, bootstyle="primary",
            text="Undo", command=self.undo_osm, state=tk.DISABLED)
        self.undo_osm_button.grid(row=rowcounter, column=1, pady=10, padx=10)

        # add close shape button for osm
        self.close_shape_osm_button = tb.Button(
            self.config_frame, bootstyle="primary",
            text="Close Shape", command=self.close_shape_osm,
            state=tk.DISABLED)
        self.close_shape_osm_button.grid(
            row=rowcounter, column=2, pady=10, padx=10)

        self.select_osm_pol_button = tb.Button(
            self.config_frame, bootstyle="primary",
            text="Select Polygon", command=self.select_osm_polygon,
            state=tk.DISABLED)
        self.select_osm_pol_button.grid(
            row=rowcounter, column=3, pady=10, padx=10)

        rowcounter += 1

        # key label
        self.osm_key_label = ttk.Label(
            self.config_frame, text="Key:")
        self.osm_key_label.grid(row=rowcounter, column=1, padx=10, pady=10)

        # key entry
        self.osm_key_entry = tb.Entry(
            self.config_frame)
        self.osm_key_entry.grid(row=rowcounter, column=2, pady=10, padx=10)

        # osm load button
        self.osm_load_button = tb.Button(
            self.config_frame, bootstyle="primary", state=tk.NORMAL,
            text="Load", command=self.load_from_osm)
        self.osm_load_button.grid(
            row=rowcounter, column=3, pady=10, padx=10)
        rowcounter += 1

        # feature label
        self.osm_feature_label = ttk.Label(
            self.config_frame, text="Feature:")
        self.osm_feature_label.grid(row=rowcounter, column=1, padx=10, pady=10)

        # feature entry
        self.osm_feature_entry = tb.Entry(
            self.config_frame)
        self.osm_feature_entry.grid(row=rowcounter, column=2, pady=10, padx=10)

        # osm load button
        self.osm_unload_button = tb.Button(
            self.config_frame, bootstyle="primary", state=tk.NORMAL,
            text="Unload", command=self.load_from_osm)
        self.osm_unload_button.grid(
            row=rowcounter, column=3, pady=10, padx=10)
        rowcounter += 1

        # osm reset button
        self.osm_reset_button = tb.Button(
            self.config_frame, bootstyle="primary", state=tk.NORMAL,
            text="Reset Polygon", command=self.load_from_osm)
        self.osm_reset_button.grid(
            row=rowcounter, column=3, pady=10, padx=10)
        rowcounter += 1

        """

        # add load from file button
        self.load_from_file_button = tb.Button(
            self.config_frame, bootstyle="primary",
            text="Load From File", command=self.load_from_file)
        self.load_from_file_button.grid(
            row=rowcounter, column=0, pady=10, padx=10)
        rowcounter += 1

        # add save to file button
        self.save_to_file_button = tb.Button(
            self.config_frame, bootstyle="primary",
            text="Save To File", command=self.save_to_file)
        self.save_to_file_button.grid(
            row=rowcounter, column=0, pady=10, padx=10)
        rowcounter += 1

        # configure map on the right side
        self.configure_map()

    def configure_map(self):
        """ configure Tkintermapview map """
        # self.map_frame.grid_columnconfigure(0, weight=0)
        self.map_frame.grid_columnconfigure(3, weight=1)
        self.map_frame.grid_rowconfigure(0, weight=1)
        self.map_frame.grid_rowconfigure(1, weight=0)

        # add tkintermapview widget
        self.map_widget = tkintermapview.TkinterMapView(
            self.map_frame, corner_radius=10)
        map_rowcounter = 0
        self.map_widget.grid(
            row=map_rowcounter, columnspan=4, column=0, sticky='NSWE',
            padx=(20, 20), pady=(20, 20))

        # set default position
        self.map_widget.set_position(MapScreen.LATLON[0], MapScreen.LATLON[1])

        # set default zoom level
        self.map_widget.set_zoom(MapScreen.ZOOM)
        self.zoomval = MapScreen.ZOOM
        map_rowcounter += 1

        # add tile changer label
        tile_label = ttk.Label(self.map_frame, text="Choose Map Tile:")
        tile_label.grid(row=map_rowcounter, column=0, padx=10, pady=10)

        # add tile changer drop down menu
        self.tile_var = tk.StringVar()
        self.tile_var.set(list(MapScreen.TILES.keys())[0])
        self.set_tile(self.tile_var.get())
        tiles_list = [key for key in MapScreen.TILES.keys()]
        self.tile_button = tb.OptionMenu(
            self.map_frame, self.tile_var, self.tile_var.get(), *tiles_list,
            command=self.set_tile)
        self.tile_button.grid(
            row=map_rowcounter, column=1, padx=10, pady=10, sticky="w")

        # add use offline tile button
        self.offline_tile_var = tk.IntVar()
        self.offline_tile_button = tb.Checkbutton(
            self.map_frame, bootstyle="primary, toolbutton, outline",
            text="Use Offline Tile", variable=self.offline_tile_var, onvalue=1,
            offvalue=0, command=self.set_offline_tiles)
        self.offline_tile_button.grid(
            row=map_rowcounter, column=2, pady=10, padx=10)

        # add zoom bar
        self.map_slider = ttk.Scale(
            self.map_frame, from_=9, to=18, cursor='hand2',
            orient=tk.HORIZONTAL, value=self.map_widget.zoom,
            command=self.slider_event)
        self.map_slider.grid(
            row=map_rowcounter, column=3, padx=10, pady=10, sticky="e")
        self.map_slider.set(self.map_widget.zoom)

        # add click events
        self.map_widget.add_left_click_map_command(self.left_click_test)

        self.map_widget.add_left_click_map_command(
            self.add_coordinates_to_polygon)

    def select_pol(self):
        """ select polygon for loading from osm """

    def load_from_osm(self):
        """ load data from openstreetmap """

    def show_polygons(self):
        """ show all polygons on map """

    def hide_polygons(self):
        """ hide all polygons on map """

    def update_osm_polygon(self):
        """ draw osm polygon on tkintermapview """

    def update_current_polygon(self):
        """ draw current polygon on tkintermapview map """

        if self.current_pol_widget:
            self.current_pol_widget.delete()
            self.current_pol_widget = None
        if len(self.current_polygon_coordinates) == 1:
            self.current_pol_widget = self.map_widget.set_polygon(
                self.current_polygon_coordinates,
                outline_color="blue", fill_color=None, border_width=4,
                command=self.polygon_click_test,
                name="Current Polygon")
        elif len(self.current_polygon_coordinates) > 1:
            self.current_pol_widget = self.map_widget.set_path(
                self.current_polygon_coordinates,
                width=4,
                command=self.polygon_click_test,
                name="Current Path")
        else:
            if self.current_pol_widget:
                self.current_pol_widget.delete()
            self.current_pol_widget = None

    def redraw_widgets(self):
        for pol in self.polygon_coordinates_list:
            self.map_widget.set_polygon(
                pol, outline_color="blue", fill_color=None, border_width=4)

    def add_polygon(self):
        """ add drawn polygon to polygon list and map"""
        # add polygon widget to list of polygon widgets
        # self.pol_widget_list.append(self.current_pol_widget)
        # self.shown_pol_widget_list.append(self.current_pol_widget)
        self.current_pol_widget.delete()

        # reset current coordinates
        self.polygon_coordinates_list.append(self.closed_polygon_coordinates)
        self.redraw_widgets()

        self.current_polygon_coordinates = []
        self.closed_polygon_coordinates = []

        self.add_pol_button.config(state=tk.DISABLED)
        self.close_shape_button.config(state=tk.DISABLED)

    def close_shape(self):
        """ close shape in drawing mode """
        # delete path
        self.current_pol_widget.delete()
        # display polygon
        self.current_pol_widget = self.map_widget.set_polygon(
            self.current_polygon_coordinates,
            outline_color="blue", fill_color=None, border_width=4,
            command=self.polygon_click_test,
            name="Current Polygon"
        )
        self.closed_polygon_coordinates = [
            element for element in self.current_polygon_coordinates]
        # reset current coordinates
        self.current_polygon_coordinates = []
        # enable add polygon button
        self.add_pol_button.config(state=tk.NORMAL)

        # add polygon:
        self.pol_widget_list.append(self.current_pol_widget)
        self.shown_pol_widget_list.append(self.current_pol_widget)

        # self.current_pol_widget.delete()

    def undo(self):
        """ undo last action in drawing mode """
        if not self.current_polygon_coordinates:
            # TODO: undo adding latest polygon
            pass
        else:
            self.current_polygon_coordinates.pop()
            self.update_current_polygon()

    def polygon_click_test(self, polygon):
        print(f"polygon clicked - text: {polygon.name}")

    def enable_undo_osm(self):
        """ enable and disbale undo button for osm """
        if (
                self.select_pol_var.get() == 1 and (
                    len(self.osm_polygon_coordinates) > 0
                    or len(self.pol_widget_list) > 0)):
            self.undo_osm_button.config(state=tk.NORMAL)
        else:
            self.undo_osm_button.config(state=tk.DISABLED)

    def enable_undo(self):
        """ enable and disbale undo button """
        # only enable undo if draw mode is active and theres something to undo
        if (
                self.draw_var.get() == 1 and (
                    len(self.current_polygon_coordinates) > 0
                    or len(self.pol_widget_list) > 0)):
            self.undo_button.config(state=tk.NORMAL)
        else:
            self.undo_button.config(state=tk.DISABLED)

    def enable_close_shape_osm(self):
        """ enable and disbale close shape button for oms """

    def enable_close_shape(self):
        """ enable and disbale close shape button """
        if len(self.current_polygon_coordinates) > 2:
            self.close_shape_button.config(state=tk.NORMAL)
        else:
            self.close_shape_button.config(state=tk.DISABLED)

    def add_coordinates_to_polygon(self, coordinates):
        """ add coordinates to current polygon list in draw mode """
        # check if draw mode is active
        if self.draw_var.get() == 1:
            # add coordinates to current polygon
            self.current_polygon_coordinates.append(coordinates)
            self.update_current_polygon()
            self.enable_undo()
            self.enable_close_shape()

    def def_osm_polygon(self, coordinates):
        """ define osm polygon coordinates """
        # check if osm mode is active
        if self.select_pol_var.get() == 1:
            # add coordinates to polygon
            self.osm_polygon_coordinates.append(coordinates)
            self.update_osm_polygon()
            self.enable_undo_osm()
            self.enable_close_shape_osm()

    def set_tile(self, tile_var):
        """ set tile layer for tkintermapview map """
        if tile_var in MapScreen.TILES.keys():
            self.map_widget.set_tile_server(MapScreen.TILES[tile_var])

    def set_offline_tiles(self):
        """ set offline tiles """

    def slider_event(self, value):
        """
        Handle the slider event to adjust the zoom level of the map.

        Parameters
        ----------
        value: float
            The new value of the slider.
        """
        self.zoomval = float(value)
        self.map_widget.set_zoom(self.zoomval)
        self.zoomval = self.map_slider.get()
        self.map_widget.set_zoom(self.zoomval)

    def undo_osm(self):
        """ undo osm polygon """

    def close_shape_osm(self):
        """ close shape osm polygon """

    def select_osm_polygon(self):
        """ select polygon for OSM """

    def choose_cut_polygons(self):
        """ choose cut polygons to intersect """

    def cut_selected(self):
        """ cut selected polygons """

    def delete_selected(self):
        """ delete selected polygons """

    def enable_select_polygons(self):
        """ select polygons """
        if self.select_var.get() == 1:
            # enable other buttons for selecting
            if self.selected_polygon_ids:
                self.delete_button.config(state=tk.NORMAL)
                self.choose_cut_button.config(state=tk.NORMAL)
                if self.cut_polygons:
                    self.cut_button.config(state=tk.NORMAL)
        else:
            # disable selecting
            self.delete_button.config(state=tk.DISABLED)
            self.choose_cut_button.config(state=tk.DISABLED)
            self.cut_button.config(state=tk.DISABLED)

    def save_to_file(self):
        """ save polygon data to GeoJSON file """

        # transform polygons from list to polygon_features dict first
        while len(self.polygon_coordinates_list) > 0:
            element = self.polygon_coordinates_list.pop()
            rev_element = [[lon, lat] for lat, lon in element]
            self.polygon_features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [rev_element]
                    },
                    "properties": {}
                })
        self.save_to_str = filedialog.asksaveasfilename(
            title="Save Area Polygons to GeoJSON File",
            filetypes=(
                ("GeoJSON files", "*.geojson"), ("JSON files", "*.json"))
        )
        if self.save_to_str and self.polygon_features:
            with open(self.save_to_str, 'w') as f:
                geojson_savedat = {
                    "type": "FeatureCollection",
                    "features": self.polygon_features
                }
                json.dump(geojson_savedat, f, indent=None)

    def load_from_file(self):
        """ load GeoJSON object from .json file """
        self.load_from_str = filedialog.askopenfilename(
            title="Select Area Input GeoJSON File",
            filetypes=(
                ("JSON files", "*.json"), ("GeoJSON files", "*.geojson"))
        )
        # open file and extract data
        if self.load_from_str:
            with open(self.load_from_str, 'r') as f:
                geojson_data = json.load(f)
                print(f"{geojson_data = }")
            for feature in geojson_data['features']:
                self.polygon_features.append(feature)

    def enable_drawing(self):
        """ enable and disbale drawing mode """
        if self.draw_var.get() == 1:
            # enable other buttons for drawing
            self.close_shape_button.config(state=tk.NORMAL)
            self.undo_button.config(state=tk.NORMAL)
        else:
            # disable drawing
            self.close_shape_button.config(state=tk.DISABLED)
            self.undo_button.config(state=tk.DISABLED)

    def draw_polygon(self):
        """ draw polygon on map """

    def start(self):
        """
        Start the Tkinter main loop to display the map and handle user
        interactions.

        This method initializes the Tkinter main loop, which keeps the GUI
        running and responsive to user inputs. It should be called after all
        the GUI components have been set up and configured.
        """
        self.mainloop()
