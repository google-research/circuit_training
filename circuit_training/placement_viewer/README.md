# Utility function to visualize placement in image format

This utlity script uses the plc_client binary provided with the Google Circuit Training to plot the placement of macros and standard cells in an image format. The input to the placement viewer is the placement files in prototxt and .plc format

### Install Google Circuit Training plc_wrapper_main library
```
# Install TF-Agents with nightly versions of Reverb and TensorFlow 2.x
$  pip install tf-agents-nightly[reverb]
# Copy the placement cost binary to /usr/local/bin and makes it executable.
$  sudo curl https://storage.googleapis.com/rl-infra-public/circuit-training/placement_cost/plc_wrapper_main \
     -o  /usr/local/bin/plc_wrapper_main
$ sudo chmod 555 /usr/local/bin/plc_wrapper_main
$ Clone the circuit-training repo.
$  git clone https://github.com/google-research/circuit-training.git
```
### Place the placement_viewer in circuit_training/circuit_training/

## Testing: 
```
$ python3 -m circuit_training.placement_viewer.placement_viewer_test 
```

## Execution: 
```
$ python3 -m circuit_training.placement_viewer.placement_viewer  
  --init_file: Path to the init file. 
  --netlist_file: Path to the input netlist file. 
  --img_name: Path to/Prefix of the name of output image file.
```  
  
#### Example plot: Ariane RISC-V placement done by Circuit Training after training from scratch i.e.  [Full Scale Ariane example](https://github.com/google-research/circuit_training/blob/main/docs/ARIANE.md "Full Scale Ariane example") 

![picture alt](https://github.com/Maria-UET/MacroPlacement/blob/bc90e6deaceca15ff0ce846f7c441eddc3f44034/Utilities/placement_viewer/test_data/ariane/ariane_final_placement.png "Ariane RISC-V placement done by Circuit Training after training from scratch")
