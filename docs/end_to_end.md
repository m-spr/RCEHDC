END_to_END Flow
======


The focus of ER3HDC is to develop a lightweight FPGA-based accelerator for _HDC_ inference in IoT and low-power devices. Therefore, the training of the network is performed offline using the open-source Python library [TorchHD](https://github.com/torchhd). 

Besides the Base_level encoding in TorchHD, our lightweight approach for weight generators is defined in two classes to generate the BHV matrix and the ID-level matrix based on our system. These classes can be used beside the actual defined classes in TorchHD for the training phase. Within TorchHD, we reuse the existing base-level implementation, filling the BHV and ID-level matrices with the HVs generated by our hardware-friendly structures, as explained [here](./_encoding.md). This function provides the entire BHV matrix for the learning phase of HDC, as well as the configuration and initialization seed for the BHV-Gen module in hardware.

After training, the configuration of the hardware design top module is generated based on the feature size (_f_), dimension size (_d_). Then, the generated Class HyperVectors (CHV) are extracted from the trained TorchHD model and, based on the generated configuration for that dataset, initialized in different ROM files, and loaded into the architecture repository directory in our RE3HDC framework.

Lastly, the design can be synthesized using the .tcl file suitable for Vivado. Since all of the modules are designed to be configurable with the support of VHDL's recursive and for-generate style, there is no need for further modification to hardware codes. They produce correct hardware for any number of features and dimension sizes automatically, as discussed [here](./hardware_desc.md).