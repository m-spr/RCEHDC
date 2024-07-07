# Generic hardware accelerator design with VHDL

To develop a fully generic hardware accelerator that can address any problem without relying on High-Level Synthesis (HLS), we make extensive use of VHDL. In our approach, not only the bitwidths of the signals are generic, but also the implementation configuration is addressed by generic parameters in the top module, which makes the design very flexible and scalable. For this purpose, all submodules of the architecture are designed to be parameterizable by using one of the generic concepts, such as For generate and recursive hardware generate.

To achieve a generic or recursive coding structure, processing distribution must be symmetrical and well partitioned between the components. Detailed explanations of how these parameters are generated can be found [here] (./generate_config.md).

In the following, a few samples of the madules are discribed.

## Top Module: !!!!!name!!!!!

The top module, `hdctest.vhd`, serves as the primary entry point for our hardware accelerator. All the configurations related to the problem size are set in the generic parameters of this module. It controls the interaction between the various sub-modules and manages the entire control flow.

## Recursive Multiplexer:

The `recMux.vhd` module is designed to handle a dynamic number of inputs by using the recursive style. By utilizing the recursive support of VHDL, this multiplexer can adapt to different numbers of inputs, making it highly scalable and usable for different applications.

## Generic LFSR Design:

The file `BasedVectorLFSR.vhd` implements a linear feedback shift register (LFSR) with parallel outputs in a very generic way. The design is parameterizable so that different LFSR lengths and tap positions can be easily configured. This level of genericity ensures that the LFSR can be tailored to the specific requirements of different applications and provides an adaptable solution for base hypervectors.

## Encoding

In the encoding module, the number of dimensions (_d_) of the XOR and Popcount modules is instantiated by the generic for loop.