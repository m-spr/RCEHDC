VHDL modeling
======

To create a fully generic hardware accelerator capable of addressing any problem without relying on High-Level Synthesis (HLS), we extensively use VHDL. 
Our approach involves implementing designs in a highly generic and recursive manner. This allows for flexibility and scalability in hardware coding, 
and every significant or parameterizable component of the architecture is constructed using one of these two foundational concepts.
To have the generic or recursive coding structure, the design must be symmetric and well-divided between its components. 
Detailed explanations of how these parameters are generated can be found [here](./.md).??? or this file?

For example, in the encoding section, we can use a dimension size (_d_) number of XOR and Popcount operations. 
These can be easily generated in hardware using a "for generate" structure.
However, to make the multiplexer adaptable to different numbers of inputs, we employ VHDL's recursive support. 
This ensures the multiplexer is scalable and versatile in various applications.
