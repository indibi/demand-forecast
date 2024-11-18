"""This module contains the class TaxiZones to represent the taxi zones in NYC
"""

from pathlib import Path
import warnings
import hashlib

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
YELLOW_DIR = DATA_DIR / 'yellow_taxi_trip_records'
GREEN_DIR = DATA_DIR / 'green_taxi_trip_records'
FHV_DIR = DATA_DIR / 'for_hire_vehicle_trip_records'
TAXI_ZONES_DIR = DATA_DIR / 'taxi_zones'
PRECOMPUTED_GRAPH_DIR = TAXI_ZONES_DIR / 'precomputed_graphs'


def generate_filename_from_indices(indices):
    """Generate a filename from a list of indices to check if a precomputed graph exists"""
    indices_str = ''.join([str(idx) for idx in indices])
    hash_object = hashlib.md5(indices_str.encode())
    hash_str = hash_object.hexdigest()
    # Create the filename using the hash
    filename = f"G_nyc_{hash_str}.graphml"
    return filename


class TaxiZones:
    """A class to represent the taxi zones in NYC
    
    Class Attributes
    ----------
    zones : pandas.DataFrame
        A DataFrame containing the taxi zones in NYC
    indices : list
        A list of indices of the taxi zones
    idx_to_loc_id : dict
        A dictionary mapping the indices to the LocationID of the taxi zones
    loc_id_to_idx : dict
        A dictionary mapping the LocationID to the indices of the taxi zones
    G_nyc : networkx.Graph
        A networkx graph representing the taxi zones in NYC
    x_scale : float
        The scale of the x-axis of the NYC map
    y_scale : float
        The scale of the y-axis of the NYC map
    ratio : float
        x_scale/y_scale ratio of the NYC map

    Call `TaxiZones._create_and_connect_graph(k_neighbors=2, buffer_width=1, extra_edges=None, 
                                    save=True, load_precomputed=True, filename=None)`
    after loading the class to create and connect the graph.
    """
    zone_lookup = pd.read_csv(TAXI_ZONES_DIR / 'taxi_zone_lookup.csv')
    zones = gpd.read_file(TAXI_ZONES_DIR / 'taxi_zones.shp')
    # Correct the LocationID of zones 56, 103 and 104
    # Govenors Island/Ellis Island/Liberty Island Idk which is which.
    zones.iloc[56,4] = 57 # Correct the LocationID of zone 56
    zones.iloc[103,4] = 104 # Correct the LocationID of zone 104
    zones.iloc[104,4] = 105 # Correct the LocationID of zone 105
    zones['x'] = zones['geometry'].centroid.x
    zones['y'] = zones['geometry'].centroid.y
    indices = zones.index.to_list()
    idx_to_loc_id = zones.LocationID.to_dict()
    loc_id_to_idx = {v: k for k, v in idx_to_loc_id.items()}
    G_nyc = None
    x_scale = None
    y_scale = None
    ratio = None

    def __init__(self, boroughs=None, zone_ids=None, **kwargs):
        """Initialize the TaxiZones object
        
        Parameters
        ----------
        boroughs : list of str, optional
            A list of boroughs to filter the taxi zones. Default is None
            which does not filter the zones.
        zone_ids : list of int, optional
            A list of LocationIDs to filter the taxi zones. Default is None
            which does not filter the zones.
        **kwargs : dict
            Additional keyword arguments to create and connect the graph
            k_neighbors : int, optional
                The number of nearest neighbors to connect the graph. Default is 2. 
                If None, k-neighbors is not used to connect the graph.
            buffer_width : float, optional
                Finds the zone pairs that share a border by enlarging the 
                zones by the buffer_width and checking for intersections. Default is 1.
                If None, buffer is not applied.
            extra_edges : list of tuples, optional
                A list of extra edges to add to the graph. Default is None. 
                The list should contain taxi zone LocationIDs
            subgraph : bool, optional
                Create the graph as a subgraph of the main graph declared by the class.
                Default is True
            save : bool, optional
                Save the graph to a file in the ./data/taxi_zones/precomputed_graphs/
                folder. Default is False
            load_precomputed : bool, optional
                Load a precomputed graph from the ./data/taxi_zones/precomputed_graphs/
                folder. Default is False
            filename : str, optional
                The filename to save or load the graph. Default is None. If None,
                a filename is generated based on the included zones.
            overwrite : bool, optional
                Overwrite the precomputed graph if it exists. Default is False
        """
        if boroughs is not None:
            borough_filter = TaxiZones.zones['borough'].isin(boroughs)
        else:
            borough_filter = pd.Series([True]*len(self.zones))
        if zone_ids is not None:
            id_filter = TaxiZones.zones['LocationID'].isin(zone_ids)
        else:
            id_filter = pd.Series([True]*len(self.zones))
        self.zone_filter = borough_filter & id_filter
        self.zones = TaxiZones.zones[self.zone_filter].reset_index()
        self.indices = self.zones.index.to_list()
        self.idx_to_loc_id = self.zones.LocationID.to_dict()
        self.loc_id_to_idx = {v: k for k, v in self.idx_to_loc_id.items()}
        
        self.G_nyc = None
        self.x_scale = None
        self.y_scale = None
        self.ratio = None

        self.create_and_connect_graph(**kwargs)


    def create_and_connect_graph(self, k_neighbors=2, buffer_width=1, extra_edges=None,
                                subgraph=True, save=False, load_precomputed=False,
                                filename=None, overwrite=False):
        """Create networkx graph of the taxi zones in NYC and connect the graph.
        
        Parameters
        ----------
        k_neighbors : int, optional
            The number of nearest neighbors to connect the graph. Default is 2. If None, 
            no k-neighbors are connected.
        buffer_width : float, optional
            Enlarge the zones in the map by the buffer_width and connect the zones that intersect.
            Default is 1. If None, no buffer is applied.
        extra_edges : list of tuples, optional
            A list of extra edges to add to the graph. Default is None. The list should contain
            taxi zone LocationIDs
        subgraph : bool, optional
            Create the graph as a subgraph of the main graph. Default is True
        save : bool, optional
            Save the graph to a file in the ./data/taxi_zones/precomputed_graphs/ folder. Default is False
        load_precomputed : bool, optional
            Load a precomputed graph from the ./data/taxi_zones/precomputed_graphs/ folder. Default is False
        filename : str, optional
            The filename to save or load the graph. Default is None. If None, a filename is
            generated from the indices.
        overwrite : bool, optional
            Overwrite the precomputed graph if it exists. Default is False
        """
        if subgraph:
            total_bounds = self.zones.total_bounds
            self.x_scale = total_bounds[2] - total_bounds[0]
            self.y_scale = total_bounds[3] - total_bounds[1]
            self.ratio = self.x_scale/self.y_scale
            loc_ids = [self.idx_to_loc_id[idx] for idx in self.indices]
            og_indices_of_subgraph = [TaxiZones.loc_id_to_idx[loc_id]  for loc_id in loc_ids]
            self.G_nyc = TaxiZones.G_nyc.subgraph(og_indices_of_subgraph).copy()
            mapping = {og_idx: idx for og_idx, idx in zip(og_indices_of_subgraph, self.indices)}
            self.G_nyc = nx.relabel_nodes(self.G_nyc, mapping,copy=True)
        else:
            self.__create_and_connect_graph(self, k_neighbors, buffer_width, extra_edges, 
                                            save, load_precomputed, filename, overwrite)
    def create_graph(self):
        return self.__create_graph(self)
    
    def connect_graph(self, k_neighbors=2, buffer_width=1, extra_edges=None):
        return self.__connect_graph(self, k_neighbors, buffer_width, extra_edges)
    
    def find_connections_with_k_nneighbors(self, k_neighbors=2):
        return self.__find_connections_with_k_nneighbors(self, k_neighbors)

    def find_connections_with_eps_neighbors(self, radius=None):
        return self.__find_connections_with_eps_neighbors(self, radius)

    def find_connections_with_adjacent_map_zones(self, buffer_width=1):
        return self.__find_connections_with_adjacent_map_zones(self, buffer_width)
    
    @staticmethod
    def __create_graph(obj):
        total_bounds = obj.zones.total_bounds
        obj.x_scale = total_bounds[2] - total_bounds[0]
        obj.y_scale = total_bounds[3] - total_bounds[1]
        obj.ratio = obj.x_scale/obj.y_scale
        # obj.label_offset = (obj.x_scale/200, obj.y_scale/200)
        obj.G_nyc = nx.Graph()
        pos_x = {zone_idx: x for zone_idx, x in zip(obj.indices, obj.zones['x'])}
        pos_y = {zone_idx: y for zone_idx, y in zip(obj.indices, obj.zones['y'])}
        name = {zone_idx: name 
                for zone_idx, name in zip(obj.indices, obj.zones['zone'])}
        borough = {zone_idx: boro 
                for zone_idx, boro in zip(obj.indices, obj.zones['borough'])}
        loc_id = {zone_idx: loc 
                for zone_idx, loc in zip(obj.indices, obj.zones['LocationID'])}
        obj.G_nyc.add_nodes_from(obj.indices)
        nx.set_node_attributes(obj.G_nyc, pos_x, 'pos_x')
        nx.set_node_attributes(obj.G_nyc, pos_y, 'pos_y')
        nx.set_node_attributes(obj.G_nyc, name, 'zone')
        nx.set_node_attributes(obj.G_nyc, borough, 'borough')
        nx.set_node_attributes(obj.G_nyc, loc_id, 'loc_id')

    @staticmethod
    def __connect_graph(obj, k_neighbors=2, buffer_width=1, extra_edges=None):
        edge_list = []
        edge_list += obj.__find_connections_with_k_nneighbors(obj, k_neighbors)
        edge_list += obj.__find_connections_with_adjacent_map_zones(obj, buffer_width)
        if extra_edges is not None:
            extra_edges = [(obj.loc_id_to_idx[edge[0]], obj.loc_id_to_idx[edge[1]]) for edge in extra_edges]
            edge_list += extra_edges
        obj.G_nyc.add_edges_from(edge_list)

    @staticmethod
    def __find_connections_with_k_nneighbors(obj, k_neighbors=2):
        pos_x = nx.get_node_attributes(obj.G_nyc, 'pos_x')
        pos_y = nx.get_node_attributes(obj.G_nyc, 'pos_y')
        coords = np.array([(pos_x[zone_idx],pos_y[zone_idx]) for zone_idx in obj.indices])
        edge_list = []
        if k_neighbors is not None:
            edge_list = nx.from_scipy_sparse_array( 
                                kneighbors_graph(coords, k_neighbors, mode='connectivity', include_self=False)
                                ).edges()
        return edge_list
    
    @staticmethod
    def __find_connections_with_eps_neighbors(obj, radius=None):
            raise NotImplementedError("Eps-neighbors is not yet implemented")

    @staticmethod
    def __find_connections_with_adjacent_map_zones(obj, buffer_width=1):
        edge_list = []
        if buffer_width is not None:
            for i in tqdm(obj.indices, desc='Finding Adjacent Zones to Connect the Graph'):
                for j in obj.indices:
                    if j > i:
                        if obj.zones['geometry'][i].intersects(obj.zones['geometry'][j].buffer(buffer_width)):
                            edge_list.append((i, j))
        return edge_list

    @classmethod
    def _create_graph(cls, save=False, load_precomputed=False, filename='nyc_tz_graph'):
        return cls.__create_graph(cls)
    
    @classmethod
    def _connect_graph(cls, k_neighbors=2, buffer_width=1, extra_edges=None):
        return cls.__connect_graph(cls, k_neighbors, buffer_width, extra_edges)

    @classmethod
    def _find_connections_with_k_nneighbors(cls, k_neighbors=2):
        return cls.__find_connections_with_k_nneighbors(cls, k_neighbors)
    
    @classmethod
    def _find_connections_with_eps_neighbors(cls, radius=None):
        return cls.__find_connections_with_eps_neighbors(cls, radius)
    
    @classmethod
    def _find_connections_with_adjacent_map_zones(cls, buffer_width=1):
        return cls.__find_connections_with_adjacent_map_zones(cls, buffer_width)

    @staticmethod
    def __create_and_connect_graph(obj, k_neighbors, buffer_width, extra_edges, 
                                save, load_precomputed, filename, overwrite):
        if obj.__dict__.get('G_nyc') is None:
            if load_precomputed:
                if filename is not None:
                    obj.G_nyc = nx.read_graphml(PRECOMPUTED_GRAPH_DIR / f'{filename}.graphml')
                else:
                    fname = generate_filename_from_indices(obj.indices)
                    if (PRECOMPUTED_GRAPH_DIR/fname).exists():
                        print(f"Loading precomputed graph from {PRECOMPUTED_GRAPH_DIR/fname}")
                        obj.G_nyc = nx.read_graphml(PRECOMPUTED_GRAPH_DIR / fname, node_type=int, edge_key_type=int)
                        total_bounds = obj.zones.total_bounds
                        obj.x_scale = total_bounds[2] - total_bounds[0]
                        obj.y_scale = total_bounds[3] - total_bounds[1]
                        obj.ratio = obj.x_scale/obj.y_scale
                    else:
                        warnings.warn(f"Precomputed graph not found at {PRECOMPUTED_GRAPH_DIR/fname}. Creating a new graph.")
                        obj.__create_graph(obj)
                        obj.__connect_graph(obj, k_neighbors, buffer_width, extra_edges)
            else:
                obj.__create_graph(obj)
                obj.__connect_graph(obj, k_neighbors, buffer_width, extra_edges)
        if save:
            if filename is None:
                fname = generate_filename_from_indices(obj.indices)
            elif not filename.endswith('.graphml'):
                fname = f'{filename}.graphml'
            else:
                fname = filename
            if not (PRECOMPUTED_GRAPH_DIR/fname).exists() or overwrite:
                warnings.warn(f"Saving the graph to {PRECOMPUTED_GRAPH_DIR/fname}")
                nx.write_graphml(obj.G_nyc, PRECOMPUTED_GRAPH_DIR / fname)
            else:
                print(f"Graph already exists at {(PRECOMPUTED_GRAPH_DIR/fname)}. Did not overwrite.")
    
    @classmethod
    def _create_and_connect_graph(cls, k_neighbors=2, buffer_width=1, extra_edges=None,
                                save=True, load_precomputed=False, filename=None, overwrite=False):
        return cls.__create_and_connect_graph(cls, k_neighbors, buffer_width, extra_edges, 
                                            save, load_precomputed, filename, overwrite)
    
    
TaxiZones._create_and_connect_graph(k_neighbors=2, buffer_width=1, extra_edges=None, 
                                    save=True, load_precomputed=True, filename=None)

def plot_zones(tz, ax=None,
        zone_options=None, node_options=None, label_options=None, edge_options=None,
        **kwargs):
    """Plot the NYC taxi zones
    
    Parameters:
    tz : TaxiZones instance or TaxiZones class type itself
    ax : matplotlib.axes.Axes, optional
        The axes to plot the zones
    zone_options : dict
        The options for plotting the zones. Default is {'color': 'gray', 'edgecolor': 'black'}
    node_options : dict
        The options for plotting the nodes. Default is {'node_size': 15, 'node_color': 'C2',
            'edgecolors': 'black', 'linewidths': 0.75}
    label_options : dict
        The options for plotting the labels. Default is {'font_size': 8, 'font_color': 'black'}
    edge_options : dict
        The options for plotting the edges. Default is {'edge_color': 'black', 'width': 0.75,
        'alpha': 0.85}
    **kwargs : dict
        Additional keyword arguments for plotting the zones
    
    Returns:
    matplotlib.figure.Figure
        The figure containing the plot
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if zone_options is None:
        zone_options = {'color': 'gray', 'edgecolor': 'black'}
    if node_options is None:
        node_options = {'node_size': 15, 'node_color': 'C2', 'edgecolors': 'black', 'linewidths': 0.75}
    if label_options is None:
        label_options = {'font_size': 8, 'font_color': 'black'}
    if edge_options is None:
        edge_options = {'edge_color': 'black', 'width': 0.75, 'alpha': 0.85}
    
    label_offset = label_options.get('label_offset', (tz.x_scale/200,tz.y_scale/200))
    figsize = kwargs.get('figsize', 12)
    if figsize is not tuple:
        figsize = (tz.ratio*figsize, figsize)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    tz.zones.plot(ax=ax, **zone_options)
    pos_x = nx.get_node_attributes(tz.G_nyc, 'pos_x')
    pos_y = nx.get_node_attributes(tz.G_nyc, 'pos_y')
    pos = {idx: (pos_x[idx], pos_y[idx]) for idx in tz.indices}
    name = nx.get_node_attributes(tz.G_nyc, 'zone')
    borough = nx.get_node_attributes(tz.G_nyc, 'borough')
    loc_id = nx.get_node_attributes(tz.G_nyc, 'loc_id')
    label_pos = {idx: (pos_x[idx] + label_offset[0], pos_y[idx] + label_offset[1]) for idx in tz.indices}
    nx.draw_networkx_nodes(tz.G_nyc, pos, ax=ax, **node_options);
    nx.draw_networkx_labels(tz.G_nyc, label_pos, ax=ax, labels=loc_id, **label_options);
    nx.draw_networkx_edges(tz.G_nyc, pos, ax=ax, **edge_options);
    return fig, ax