# Placement and Routing Simulator (PlacementCost) Guide

`plc_client.PlacementCost` is a placement and routing simulator that provides an
approximate but fast estimate of key QoR metrics such as wirelength, congestion,
and density. It is used in RL optimization loops to train good placements.
`plc_client.PlacementCost` can also be used standalone to measure these proxy
costs given a placement.

`plc_client.PlacementCost` supports the following methods.

-   `can_place_node(node_index: int, grid_cell_index: int) -> bool` : Checks if
    a node can be placed into a grid cell. This is assuming the node is not
    placed or fixed.
-   `create_blockage(minx: float, miny: float, maxx: float, maxy: float,
    blockage_rate: float)` : Creates a blockage area, which is used to
    initialize grid density. A blockage is defined as a rectangular area which a
    hard macro cannot overlap with. Using the updated grid density, placement of
    soft macros, and standard cells are limited. Blockage rate can be set lower
    than 1.0 if the area can be used by stdcells and soft macros. If a grid cell
    is partially blocked, a hard macro can occupy the unblocked part of that
    grid cell.
-   `fix_node_coord(node_index: int) -> Status` : Fixes the coordinates of a
    node by index.
-   `get_area() -> float` : Total area of the macros and stdcells. Height and
    widths are assumed to be in microns, so this will return micron^2.
-   `get_blockages() -> list<list<float>>` : Returns blockages.
-   `get_canvas_width_height() -> tuple<float, float>` : Returns (width, height)
    of the canvas.
-   `get_congestion_cost() -> float` : Returns the average routing congestion of
    the top 10% congested routing grid cells.
-   `get_congestion_smooth_range() -> int` : Gets congestion smooth range.
-   `get_cost() -> float` : Calculates the normalized wirelength cost associated
    with the current placement of the nodes (stdcells, macros and ports). It
    returns a number between 0 and 1. 0 meaning 0 wirelength, 1 meaning worst
    possible placement leading to all wires spanning the entire placement area
    (half perimeter of canvas).
-   `get_density_cost() -> float` : Returns the average density of the 10% most
    dense placement grid cells.
-   `get_fan_outs_of_node(node_index: int) -> list<int>` : Returns the vector of
    node indices that are driven by given node.
-   `get_grid_cell_of_node(node_index: int) -> int` : Returns the node's current
    location in terms of grid cell index.
-   `get_grid_num_columns_rows() -> tuple<int, int>` : Returns (num columns, num
    rows) of the placement grid.
-   `get_macro_and_clustered_port_adjacency() -> (adj_matrix: list<int>,
    grid_cell_of_clustered_ports: list<int>)` : Builds the adjacency matrix
    between macros and "clustered ports", where a "clustered ports" is an
    abstract element in the matrix that represents all ports in a given grid
    cell. This function does the clustering of ports by itself.
-   `get_macro_indices() -> list<int>` : Returns the indices of the macros in
    the netlist, including both hard macros and soft macros (clustered standard
    cells).
-   `get_macro_orientation(node_index: int) -> str` : Returns the orientation
    string ('N', 'FN', ...) of the macro. If it's not a hard macro, returns
    empty string.
-   `get_macro_routing_allocation() -> tuple<float, float>` : Returns horizontal
    and vertical routing tracks per micron used up by hard macros.
-   `get_node_location(node_index: int) -> tuple<float, float>` : Returns the
    (x, y) location of a node. (in microns).
-   `get_node_locations(node_indices: list<int>) -> list<tuple<float, float>>` :
    Returns the (x, y) location of a list of nodes. (in microns).
-   `get_node_mask_by_name(node_name: str) -> list<int>` : Returns available
    positions in the canvas grid for the given node. The size of the returned
    vector is num_columns * num_rows.
-   `get_node_mask(node_index: int) -> list<int>` : Uses a node index to get the
    placement mask for a node.
-   `get_node_name(node_index: int) -> str` : Gets the name of the node by
    index. If the index is out of range, returns empty string.
-   `get_node_type(node_index: int) -> str` : Returns the type of the node as
    string.
-   `get_node_width_height(node_index: int) -> tuple<float, float>` : Returns
    the (width, height) of a node (in microns).
-   `get_overlap_threshold() -> float` : Returns overlap threshold.
-   `get_routes_per_micron() -> tuple<float, float>` : Returns available
    horizontal and vertical tracks per micron.
-   `get_wirelength() -> float` : Returns total wire length estimate.
-   `is_node_fixed(node_index: int) -> bool` : Returns whether the node is fixed
    (not moveable) or not.
-   `is_node_placed(node_index: int) -> bool` : Returns whether a node is placed
    (has coordinates) or not.
-   `is_node_soft_macro(node_index: int) -> bool` : Returns whether the node is
    a soft macro or not.
-   `make_soft_macros_square()` : To be used in accordance with repel/attract
    forces.
-   `optimize_stdcells( use_current_loc: bool, move_stdcells: bool, move_macros:
    bool, log_scale_conns: bool, use_sizes: bool, io_factor: float, steps:
    list<int>, max_move_distance: list<float>, attract_factor: list<float>,
    repel_factor: list<float>) -> Status` : Optimizes locations of standard
    cells and standard cell clusters using force directed methods. If
    use_current_loc is false, all movable nodes are placed in the center
    initially. In every step a node can move at most max_move_distance microns.
    Attract factor is used as a multiplier for spring forces connecting nodes.
    Repelling forces push overlapping nodes from each other regardless they are
    connected or not.
-   `place_node_by_name(node_name: str, grid_cell_index: int) -> Status` :
    Places the node using canvas grid_cell_index = col + row * num_cols.
    Converts to the x, y positions corresponding to the center coordinates of
    the grid cell. Returns error if unsuccessful or if the node is not found.
-   `place_node(node_index: int, grid_cell_index: int) -> Status` : Uses a node
    index to place a node.
-   `restore_placement(filename: str) -> Status` : Save and restore placement
    information to/from file.
-   `save_placement(filename: str, comments: str) -> Status` : Save and restore
    placement information to/from file.
-   `set_canvas_size(width: float, height: float) -> Status` : Sets the size of
    the placement area (canvas).
-   `set_congestion_smooth_range(range: int)` : Set the number of neighboring
    rows and columns to distribute the calculated routing congestion for a cell
-   `set_placement_grid(columns: int, rows: int) -> Status` : Sets the number of
    columns and rows used by placement. By default, there are 100 rows and 100
    columns.
-   `unplace_all_nodes()` : Clears coordinate information of all nodes that are
    not marked as fixed.
-   `unplace_node_by_name(node_name: str) -> Status` : Clears node coords
    (unplace node). Should not be called for macro_pins.
-   `unplace_node(node_index: int) -> Status` : Uses a node index to unplace a
    node.
-   `update_macro_orientation_by_name(node_name: str, orientation: str) ->
    Status` : Updates the macro orientation. If the node_name is not a macro
    name, or orientation is not one of expected enums in the protobuf
    description, it returns an error.
-   `update_macro_orientation(node_index: int, orientation: str) -> Status` :
    Uses a node index to update the macro orientation.
-   `update_node_coords_by_name(node_name: str, x: float, y: float) -> Status` :
    Updates the x, y coordinates of the given node. Should not be called for
    macro_pins. Use PlaceNode to place nodes into placement grid cells.
-   `update_node_coords(node_index: int, x: float, y: float) -> Status` : Uses a
    node index to update x, y coordinates of a node.
-   `save_placement_pnr(
          output_file: str, original_netlist: str,
          metis_groups_file: str, eda_tool: str, project: str) -> Status` :
    Generates placement tcl file. Supported EDA tools: dct, innovus, icc2.
    original_netlist and metis_groups_file are needed for writing standard cell
    locations. For macro placements only, set as empty.

Example code of interacting with `PlacementCost` to query proxy costs:

```shell
$  python3 -m circuit_training.environment.plc_client_main \
  --netlist_file ./circuit_training/environment/test_data/ariane/netlist.pb.txt
  --plc_wrapper_main /usr/local/bin/plc_wrapper_main
```
