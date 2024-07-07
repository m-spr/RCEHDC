![logo](../figures/logo.jpg)

Welcome to the RE3HDC documentation!
=====================================

The reconfigurable Energy Efficient Encoding HDC (RE3HDC) is a framework dedicated to mapping *Hyperdimensional Computing* (HDC) also known as *binary Vector Symbolic Architectures* (VSA) to FPGA.
The ROC-E3HDC project is an experimental framework for the implementation of HDC on Xilinx FPGA boards. The main components of RE3HDC are shown in the figure below and can be described as follows:

![overview](../figures/overview.png){:width="300px"}


- **End-to-End Framework**
  - Uses an open source HDC training library ([Torchhd[^1]](https://github.com/torchhd))
  - Automatically generates a bitstream and hardware files for inferencing

- **Adjustable Pipeline and Fully Reconfigurable Hardware Architecture**
  - Parameterized hardware in VHDL
  - Scales based on problem size
  - Suitable for various problems without changing hardware discription by setting generic parameters and initial values
  
- **Automatic Optimization**
  - Optimizes HDC model for efficient hardware mapping by generating memory parameters on-the-fly 
  - Eliminates ineffective elements without sacrificing accuracy

[^1]: [Torchhd](https://github.com/torchhd)

RE3HDC tutorials Resources
===================
- [getting_started with RE3HDC](?)
- The RCD_E3HDC [examples repository](https://github.com/RE3HDC/examples)  
-  [E3HDC encoding](./_encoding.md)
-  [E3HDC paramiter generating?]
-  [RE3HDC architetcure and pipelining](./hardware_over.md)
-  [hardware discription]
-  [end_to_end_flow]
-  [source_code/RE3HDC]





* `List of [publications](https://xilinx.github.io/RCD_E3HDC/publications)
 
