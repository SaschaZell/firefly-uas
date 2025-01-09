"""
post processing helper functions

author: Sascha Zell
last revision: 2024-12-19
"""

# import packages
from math import atan2, cos, sin, radians, degrees
import matplotlib.pyplot as plt
from PIL import Image
import os
from branca.element import Figure
from folium.plugins import FloatImage
import folium
from pdf2image import convert_from_path
import concurrent.futures


def _rgb_to_hex(rgb):
    """ rgb to hex color code """
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def _get_hexcolor(value: int, cmap, max_value: float = 1.0):
    """ get hex color for cmap value"""
    threshold = 1/cmap.N * max_value
    color_index = min(int(value // threshold), cmap.N-1)
    fill_color = cmap(color_index)[:3]
    hex_color = _rgb_to_hex(fill_color)
    return hex_color


def get_bearing(start_point: list, end_point: list):
    """
    calculate bearing of triangle on folium map in degrees

    Parameters
    ----------
    start_point:list
        line start point in lat,lon format
    end_point:list
        line end point in lat,lon format
    """
    lat1, lon1 = radians(start_point[0]), radians(start_point[1])
    lat2, lon2 = radians(end_point[0]), radians(end_point[1])

    delta_lon = lon2 - lon1

    x = sin(delta_lon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(delta_lon))

    initial_bearing = atan2(x, y)

    # convert bearing from radians to degrees
    initial_bearing = degrees(initial_bearing)

    # normalize bearing (formula is correct, since the folium trinalge needs to
    # be rotated with additional 36 degrees)
    compass_bearing = (initial_bearing + 360 + 36) % 360

    return compass_bearing


def is_point_approximately_equal_to_any(
        point: list, waypoints: list, epsilon: float = 1e-6):
    """
    test if point is approximately equal to one of the points in waypoints list

    Parameters
    ----------
    point : list
        test point in lat,lon coordinates
    waypoints : list
        list of lat,lon waypoints to check
    """
    return any(
        all(abs(p1 - p2) < epsilon for p1, p2 in zip(point, waypoint))
        for waypoint in waypoints
    )


def convert_png_to_eps(
        in_file: str, out_file: str = None, dpi: int = 300,
        format: str = "eps"):
    """
    convert png file to eps file

    Parameters
    ----------
    in_file: str
        input image file relative or absolute path
    out_file: str = None
        output file relative or absolute path, If None, is set to png_file.
    dpi: int = 300
        dpi of eps file
    format: str = "eps"
        output format, default "eps"
    """
    # open png with pillow
    image = Image.open(in_file)

    # create figure
    # fig = plt.figure()
    plt.figure()

    # plot image on figure
    plt.imshow(image)

    # disable axis
    plt.axis('off')

    # save figure as eps
    plt.savefig(
        out_file, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)


def convert_all_images(directory: str, dpi: int = 300):
    """
    Convert all images from a directory to EPS files.

    Parameters
    ----------
    directory : str
        The directory containing the images to convert.
    dpi : int, optional (default: 300)
        The DPI (dots per inch) for the output EPS files.

    Returns
    -------
    None
    """

    # get images from directory
    image_extensions = [".png", ".jpg", ".jpeg"]

    image_files = [
        os.path.join(directory, file) for file in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, file)) and
        any(file.lower().endswith(ext.lower()) for ext in image_extensions)
    ]

    eps_file_names = [
        os.path.join(
            directory, "eps_files",
            os.path.splitext(os.path.basename(file))[0] + '.eps')
        for file in image_files
    ]

    eps_output_dir = os.path.join(directory, "eps_files")
    if not os.path.exists(eps_output_dir, exist_ok=True):
        os.makedirs(eps_output_dir)
    # transform to eps file
    # for in_file, out_file in zip(image_files, eps_file_names):
    #     convert_png_to_eps(in_file=in_file, out_file=out_file, dpi=dpi)

    # run convert_png_to_eps in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(convert_png_to_eps, in_file, out_file, dpi)
            for in_file, out_file in zip(image_files, eps_file_names)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # raise exceptions that occurred during execution


def plot_attribution(map, typ: str = "osm"):
    """
    plot attribution (Leaflet by OSM) on HTML map

    Parameters
    ----------
    map : folium.Map
        folium map
    typ : str
        type of attribution to add, default "osm"
    """
    # add attribution control
    attribution_osm = (
        "<a href='https://leafletjs.com'>Leaflet</a> | Data by <a "
        "href='https://www.openstreetmap.org/'>OpenStreetMap</a>, under "
        "<a href='https://opendatacommons.org/licenses/odbl/'>ODbL.</a>"
    )

    attribution_esri = (
        "<a href='https://leafletjs.com'>Leaflet</a> | Data by <a "
        "href='https://www.esri.com/en-us/home'>Imagery &copy 2024 Esri</a>"
        # "<a href='https://opendatacommons.org/licenses/odbl/'>ODbL.</a>"
    )

    if typ == "osm":
        attribution = attribution_osm
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
    elif typ == "esri":
        attribution = attribution_esri
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
            width: 200px; /* Adjust box width */
            ">
            {attribution}
        </div>
        """

    # create HTML control for the attribution

    # make button
    figure = Figure()
    button = FloatImage(attribution_html, bottom=65, left=10)
    figure.add_child(button)
    button_element = folium.Element(attribution_html)
    map.get_root().html.add_child(button_element)


def convert_pdf_to_jpeg(
        in_file: str, out_file: str = None, dpi: str = 300):
    """
    convert pdf file to jpeg file
    """
    pages = convert_from_path(in_file, dpi=300)

    # use Pillow to process and save each page as JPEG
    for i, page in enumerate(pages):
        # example of resizing with Pillow
        resized_image = page.resize((800, 1200))  # resize to 800x1200

        # save as JPEG
        resized_image.save(f"{out_file}{i + 1}.jpeg", "JPEG")
