# Blockage file format

A blockage is defined as a rectangular area which a hard macro or stdcells group
(soft macro) cannot be placed within. A blockage can be used to define a clock
strap blockage, or general macro blockage. Blockages are also used to support
rectilinear floorplans using a set of rectangular blockages.

The following is an example of the blockage file format.

```
# Blockage format follows <llx> <lly> <urx> <ury>
# All in micron units.
0.0 100.0 3000.0 300.0
3000.0 0.0 500.0 2000.0
```
