import matplotlib. pyplot as plt
from matplotlib.patches import Rectangle
from absl import app
from absl import flags
from typing import Sequence, Tuple

from circuit_training.environment import plc_client
from circuit_training.environment.placement_util import nodes_of_types

flags.DEFINE_string("netlist_file", None, "Path to the input netlist file.")
flags.DEFINE_string("init_file", None, "Path to the init file.")
flags.DEFINE_string("img_name", None, "Prefix of the name of output image file.")

flags.mark_flags_as_required([
    "netlist_file",
    "init_file",
    "img_name",
])

FLAGS = flags.FLAGS


def _draw_placement(plc: plc_client.PlacementCost, img_name: str) -> None:
    plt.figure()
    fig, ax = plt.subplots()
    plt.xlim(0, plc.get_canvas_width_height()[0])
    plt.ylim(0, plc.get_canvas_width_height()[1])
    for i in nodes_of_types(plc,['MACRO', 'STDCELL', 'PORT', 'MACRO_PIN']):
        x = plc.get_node_location(i)[0]-(plc.get_node_width_height(i)[0]/2)
        y = plc.get_node_location(i)[1]-(plc.get_node_width_height(i)[1]/2)
        w = plc.get_node_width_height(i)[0]
        h = plc.get_node_width_height(i)[1]

        if plc.get_node_type(i) =='MACRO' and not plc.is_node_soft_macro(i):             
            orient = plc.get_macro_orientation(i)
            if orient in [ 'E', 'FE', 'W', 'FW']:
                h = plc.get_node_width_height(i)[0]
                w = plc.get_node_width_height(i)[1]

            ax.add_patch(Rectangle((x, y),
                                w,h ,
                                facecolor = 'blue',
                                edgecolor ='black',
                                linewidth = 1,
                                linestyle="solid"))

        elif plc.is_node_soft_macro(i):
            ax.add_patch(Rectangle((x, y),
                            w,h ,
                            facecolor = 'red',
                            edgecolor ='black',
                            linewidth = 1,
                            linestyle="solid"))  

        elif plc.get_node_type(i) =='PORT': 
            ax.add_patch(Rectangle((x, y),
                            w,h ,
                            facecolor = 'cyan',
                            edgecolor ='cyan',
                            linewidth = 1,
                            linestyle="solid"))  

        elif plc.get_node_type(i) =='MACRO_PIN': 
            ax.add_patch(Rectangle((x, y),
                            w,h ,
                            facecolor = 'green',
                            edgecolor ='green',
                            linewidth = 1,
                            linestyle="solid")) 
        else: 
            print('invalid')
  

    plt.savefig(img_name)



def _get_final_canvas_dims(plc: plc_client.PlacementCost) -> Tuple[float, float, int, int]:
        with open(FLAGS.init_file) as f:
            lines = f.readlines()
        
        cols= int(lines[4].split(':')[1].split(' ')[1])
        rows = int(lines[4].split(':')[2].split(' ')[1])
        width = float(lines[5].split(':')[1].split(' ')[1])
        height = float(lines[5].split(':')[2].split(' ')[1])
        return (width, height, cols, rows)


def viewer(argv: Sequence[str]) -> None:
    if len(argv) > 3:
        raise app.UsageError("Too many command-line arguments.")

    plc = plc_client.PlacementCost(netlist_file=FLAGS.netlist_file)

    # draw the initial placement based on the netlist protobuf file
    _draw_placement(plc, img_name=FLAGS.img_name+'_initial_placement')
    print('Initial dimensions of the canvas:', plc.get_canvas_width_height())
    
    # draw the final placement based on the init file
    width, height, cols, rows = _get_final_canvas_dims(plc)
    plc.set_canvas_size(width, height) # set canvas size from the init file
    plc.set_placement_grid(cols, rows) # set grid from the init file
    print('Init status: ', plc.restore_placement(FLAGS.init_file)) # restore placement from the init file
    print('Final dimensions of the canvas:', plc.get_canvas_width_height())
    _draw_placement(plc, img_name=FLAGS.img_name+'_final_placement')


if __name__ == "__main__":
    app.run(viewer)
