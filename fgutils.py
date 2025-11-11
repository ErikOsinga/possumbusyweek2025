import pandas as pd # type: ignore
from matplotlib.patches import Rectangle


def overlay_groupmembers(ax, csv_path, wcs, header, box_size_arcmin=7.57):
    """
    Overlay rectangles and labels for group members on a matplotlib axis.
    """
    df = pd.read_csv(csv_path)
    box_deg = box_size_arcmin / 60.0
    cdelt1 = abs(header['CDELT1'])
    cdelt2 = abs(header['CDELT2'])

    for _, row in df.iterrows():
        x_pix, y_pix = wcs.wcs_world2pix([[row['radeg'], row['decdeg']]], 0)[0]
        width = box_deg / cdelt1
        height = box_deg / cdelt2
        rect = Rectangle((x_pix - width/2, y_pix - height/2),
                         width, height, edgecolor='black', facecolor='none',
                         transform=ax.get_transform('pixel'), linewidth=1)
        ax.add_patch(rect)
        ax.text(x_pix, y_pix + height,
                row['Name'], transform=ax.get_transform('pixel'),
                color='black', fontsize=8, ha='center')


