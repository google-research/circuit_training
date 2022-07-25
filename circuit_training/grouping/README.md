# Grouping
Grouping (clustering) the standard cells in the circuit training netlist.

## Prerequisite

Before starting, download the following files. They will be referenced by
environment variables setup in the first step.

   * [hmetis binary](http://glaros.dtc.umn.edu/gkhome/metis/hmetis/download)
      (University of Minnesota)
   * [plc_wrapper_main binary](https://storage.googleapis.com/rl-infra-public/circuit-training/placement_cost/plc_wrapper_main)
   * Ariane [netlist](https://storage.googleapis.com/rl-infra-public/circuit-training/netlist/ariane.circuit_graph.pb.txt.gz)

Unzip the downloaded netlist file:

```shell
$ gzip -d ariane.circuit_graph.pb.txt.gz
```

### Note
If the netlist file is larger than 2 GB, split it into small files. This is
due to the text proto file size constraint. The smaller files can be 
passed as follows: 
`--netlist_file=/path/to/netlist_file_1.pb.txt,/path/to/netlist_file_2.pb.txt`

You can use the `split_proto_netlist_main` script from the
[grouping package](https://github.com/google-research/circuit_training/tree/main/circuit_training/grouping)
to split the netlist:

```shell
$ python split_proto_netlist_main \
      --file_name=netlist.pb.txt \
      --output_dir=/dir/to/output/
```

## 1. Setup the variables

```shell
$ export OUTPUT_DIR=/path/to/output
$ export BLOCK_NAME=ariane
$ export NETLIST_FILE=/path/to/the/netlist.pb.txt
$ export HMETIS_DIR=/path/to/the/hmetis/binary
$ export PLC_WRAPPER_MAIN=/path/to/the/plc_wrapper_main
```



## 2. Run the grouping (clustering) code

```shell
$ python circuittraining/grouping/groupermain \
--output_dir=$OUTPUT_DIR \
--netlist_file=$NETLIST_FILE \
--block_name=$BLOCK_NAME \
--hmetis_dir=$HMETIS_DIR \
--plc_wrapper_main=$PLC_WRAPPER_MAIN
```

#### Example output example

Metrics similar to the following will be output by the grouping code:

```shell
grouper.py:336] Wirelength: 4066003.636051
grouper.py:337] Wirelength cost: 0.248466
grouper.py:338] Congestion cost: 1.662051
grouper.py:339] Overlap cost: 0.000034
```

We expect the metrics of a clustered netlist to be within a small margin of the
original unclustered netlist. This gives us confidence that clustering is a
good approximation of the standard cells’ placements.

## F.A.Q.

**Q: Where are the nodes coordinates of the incoming netlist defined?**

**A**: We assume that a netlist that has all its nodes already placed. It does not have to be a good
placement. A low effort, fast version of placement results from any tool can be
used.

**Q: Could you give some hint about how "distance" and "threshold" parameters are determined, for example in the Ariane example?**

**A**  "distance" and "threshold" parameters are specific to each block and
technology. They need to be adjusted by trial and error. Current defaults are good
for Ariane.

**Q: What is the relationship between clustered netlist and adjacency matrix?**

**A**: A cluster (group) of standard cells is treated as a macro. We call them soft macros.
The original macro is called hard macro. The adjacency matrix is built using the
connections between the set of nodes including hard macros and soft macros.

**Q: Assuming the number of nets between cluster_a and cluster_b is
100, what is the value of adj[cluster_a][cluster_b] in the adjacency matrix?**

**A**: The adjacency value is computed as the number of connections between
clusters, in this case would be 100.

**Q: How does the distance of flip flops get involved in the adjacency matrix?**

**A**: It does not.

**Q: Are the values in the adjacency matrix normalized?**

**A**: No. We don't normalize them.
