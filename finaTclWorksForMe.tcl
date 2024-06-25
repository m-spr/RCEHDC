start_gui
create_project DMA1 /localdata/sadmah00/github/RCD_E3HDC/DMA1 -part xc7z020clg400-1
set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]
set_property target_language VHDL [current_project]
--- open_project /localdata/sadmah00/github/RCD_E3HDC/DMA1/DMA1.xpr
import_files -norecurse {/localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full4_1.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full1_0.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full8_0.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full6_0.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full9_1.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full1_1.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full8_1.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full3_0.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full3_1.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full5_0.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full6_1.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full0_0.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full7_0.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full2_0.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full0_1.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full5_1.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full4_0.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full9_0.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full7_1.mif /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/full2_1.mif}
import_files -norecurse {/localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/fullconfComp.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/encoder.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/fulltop.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/id_level.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/classifier.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/popCount.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/comparator.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/XoringPopCtrl.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/confCompCtrl.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/hvTOcompIn.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/hdcTest.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/recMux.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/SeqAdderCtrl.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/XoringInputPop.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/regOne.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/SeqAdder.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/countingSimTop.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/BasedVectorLFSR.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/countingSim.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/reg.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/RSA.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/countingSimCtrl.vhd /localdata/sadmah00/OTFGEN_Hardware/RCD_HDC/source/hardware_source/NormalHDC/comparatorTop.vhd}
-Auto-update_compile_order -fileset sources_1
-Auto-update_compile_order -fileset sources_1

---IP part 

ipx::package_project -root_dir /localdata/sadmah00/github/RCD_E3HDC/DMA1.srcs/sources_1/imports -vendor user.org -library user -taxonomy /UserIP
set_property core_revision 2 [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity [ipx::current_core]
set_property  ip_repo_paths  /localdata/sadmah00/github/RCD_E3HDC/DMA1/DMA1.srcs/sources_1/imports [current_project]
update_ip_catalog

---grouping ports

ipx::add_bus_interface M_AXI [ipx::current_core]
set_property abstraction_type_vlnv xilinx.com:interface:axis_rtl:1.0 [ipx::get_bus_interfaces M_AXI -of_objects [ipx::current_core]]
set_property bus_type_vlnv xilinx.com:interface:axis:1.0 [ipx::get_bus_interfaces M_AXI -of_objects [ipx::current_core]]

set_property interface_mode master [ipx::get_bus_interfaces M_AXI -of_objects [ipx::current_core]]
ipx::add_port_map TVALID [ipx::get_bus_interfaces M_AXI -of_objects [ipx::current_core]]
set_property physical_name TVALID_S [ipx::get_port_maps TVALID -of_objects [ipx::get_bus_interfaces M_AXI -of_objects [ipx::current_core]]]
ipx::add_port_map TDATA [ipx::get_bus_interfaces M_AXI -of_objects [ipx::current_core]]
set_property physical_name TDATA_S [ipx::get_port_maps TDATA -of_objects [ipx::get_bus_interfaces M_AXI -of_objects [ipx::current_core]]]
ipx::add_port_map TLAST [ipx::get_bus_interfaces M_AXI -of_objects [ipx::current_core]]
set_property physical_name TLAST_S [ipx::get_port_maps TLAST -of_objects [ipx::get_bus_interfaces M_AXI -of_objects [ipx::current_core]]]
ipx::add_port_map TKEEP [ipx::get_bus_interfaces M_AXI -of_objects [ipx::current_core]]
set_property physical_name TKEEP_S [ipx::get_port_maps TKEEP -of_objects [ipx::get_bus_interfaces M_AXI -of_objects [ipx::current_core]]]
ipx::add_port_map TREADY [ipx::get_bus_interfaces M_AXI -of_objects [ipx::current_core]]
set_property physical_name TREADY_S [ipx::get_port_maps TREADY -of_objects [ipx::get_bus_interfaces M_AXI -of_objects [ipx::current_core]]]


ipx::add_bus_interface S_AXI [ipx::current_core]
set_property abstraction_type_vlnv xilinx.com:interface:axis_rtl:1.0 [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]
set_property bus_type_vlnv xilinx.com:interface:axis:1.0 [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]
set_property interface_mode slave [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]
ipx::add_port_map TVALID [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]
set_property physical_name TVALID_M [ipx::get_port_maps TVALID -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
ipx::add_port_map TLAST [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]
set_property physical_name TLAST_M [ipx::get_port_maps TLAST -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
ipx::add_port_map TDATA [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]
set_property physical_name TDATA_M [ipx::get_port_maps TDATA -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
ipx::add_port_map TKEEP [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]
set_property physical_name TKEEP_M [ipx::get_port_maps TKEEP -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]
ipx::add_port_map TREADY [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]
set_property physical_name TREADY_M [ipx::get_port_maps TREADY -of_objects [ipx::get_bus_interfaces S_AXI -of_objects [ipx::current_core]]]

ipx::associate_bus_interfaces -busif M_AXI -clock clk [ipx::current_core]
ipx::associate_bus_interfaces -busif S_AXI -clock clk [ipx::current_core]



---repackage IP-- 

--- might not needde ??? ---- ipx::merge_project_changes files [ipx::current_core]
set_property core_revision 3 [ipx::current_core]        ---- watch version
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity [ipx::current_core]
ipx::save_core [ipx::current_core]
update_ip_catalog -rebuild -repo_path /localdata/sadmah00/github/RCD_E3HDC/DMA1/DMA1.srcs/sources_1/imports
save_bd_design
ipx::merge_project_changes files [ipx::current_core]


---make diagram 
create_bd_design "design_1" -------change design_1 with E3HDC




update_compile_order -fileset sources_1
--------
startgroup
create_bd_cell -type ip -vlnv user.org:user:fulltopHDC:1.0 fulltopHDC_0         ---- add to digram
endgroup
delete_bd_objs [get_bd_cells fulltopHDC_0]  ---- remove form digram
---------
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
endgroup
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]

startgroup
create_bd_cell -type ip -vlnv user.org:user:fulltopHDC:1.0 fulltopHDC_0
endgroup
----------- apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (100 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins fulltopHDC_0/clk]




startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
endgroup
set_property -dict [list \
  CONFIG.c_include_s2mm {0} \
  CONFIG.c_include_sg {0} \
  CONFIG.c_sg_include_stscntrl_strm {0} \
] [get_bd_cells axi_dma_0]


startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_1
endgroup
set_property -dict [list \
  CONFIG.c_include_mm2s {0} \
  CONFIG.c_include_sg {0} \
  CONFIG.c_sg_include_stscntrl_strm {0} \
] [get_bd_cells axi_dma_1]



connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins fulltopHDC_0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins fulltopHDC_0/M_AXI] [get_bd_intf_pins axi_dma_1/S_AXIS_S2MM]


----ignore-----
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
endgroup
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
endgroup
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_cdma:4.1 axi_cdma_0
endgroup
delete_bd_objs [get_bd_cells axi_cdma_0]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_dma_0/S_AXI_LITE} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
----till here ------


startgroup
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_dma_0/S_AXI_LITE} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_dma_1/S_AXI_LITE} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins axi_dma_1/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (100 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins fulltopHDC_0/clk]
endgroup

startgroup
set_property -dict [list \
  CONFIG.PCW_USE_S_AXI_HP0 {1} \
  CONFIG.PCW_USE_S_AXI_HP1 {1} \
] [get_bd_cells processing_system7_0]
endgroup

startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0
endgroup
set_property CONFIG.NUM_SI {1} [get_bd_cells smartconnect_0]
connect_bd_intf_net [get_bd_intf_pins axi_dma_1/M_AXI_S2MM] [get_bd_intf_pins smartconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP1]

startgroup
copy_bd_objs /  [get_bd_cells {smartconnect_0}]
set_property location {6 1885 72} [get_bd_cells smartconnect_1]
endgroup
connect_bd_intf_net [get_bd_intf_pins smartconnect_1/S00_AXI] [get_bd_intf_pins axi_dma_0/M_AXI_MM2S]
connect_bd_intf_net [get_bd_intf_pins smartconnect_1/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

startgroup
--------   apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (100 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (100 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (100 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins processing_system7_0/S_AXI_HP1_ACLK]
endgroup




startgroup
set_property -dict [list \
  CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {50} \
  CONFIG.PCW_QSPI_GRP_SINGLE_SS_ENABLE {1} \
] [get_bd_cells processing_system7_0]
endgroup


regenerate_bd_layout
validate_bd_design
save_bd_design

----end bd design_1

make_wrapper -files [get_files /localdata/sadmah00/github/RCD_E3HDC/DMA1/DMA1.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse /localdata/sadmah00/github/RCD_E3HDC/DMA1/DMA1.gen/sources_1/bd/design_1/hdl/design_1_wrapper.vhd
update_compile_order -fileset sources_1
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1
launch_runs synth_1 -jobs 8
launch_runs impl_1 -jobs 8
launch_runs impl_1 -to_step write_bitstream -jobs 8

write_hw_platform -fixed -include_bit -force -file /localdata/sadmah00/github/RCD_E3HDC/DMA1/design_1_wrapper.xsa
write_bd_tcl -force /localdata/sadmah00/github/RCD_E3HDC/DMA1/design_1.tcl

file copy -force /localdata/sadmah00/github/RCD_E3HDC/DMA1/DMA1.runs/impl_1/design_1_wrapper.bit /localdata/sadmah00/github/RCD_E3HDC/DMA1/design_1.bit




open_bd_design {/localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/bd/design_1/design_1.bd}
write_hw_platform -fixed -include_bit -force -file /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/design_1_wrapper.xsa
write_bd_tcl -force /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/design_1.tcl
file copy -force /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.runs/impl_1/design_1_wrapper.bit /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/design_1.bit


export_simulation -of_objects [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/bd/design_1/design_1.bd] -directory /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.ip_user_files/sim_scripts -ip_user_files_dir /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.ip_user_files -ipstatic_source_dir /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.ip_user_files/ipstatic -lib_map_path [list {modelsim=/localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.cache/compile_simlib/modelsim} {questa=/localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.cache/compile_simlib/questa} {xcelium=/localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.cache/compile_simlib/xcelium} {vcs=/localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.cache/compile_simlib/vcs} {riviera=/localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet
report_ip_status -name ip_status 
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/fulltop.vhd] -no_script -reset -force -quiet
remove_files  /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/fulltop.vhd
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/hdcTest.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/classifier.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/id_level.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/popCount.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/BasedVectorLFSR.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/encoder.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/hvTOcompIn.vhd] -no_script -reset -force -quiet
remove_files  {/localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/hdcTest.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/classifier.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/id_level.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/popCount.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/BasedVectorLFSR.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/encoder.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/hvTOcompIn.vhd}
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/countingSimTop.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/comparatorTop.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/XoringInputPop.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/XoringPopCtrl.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/regOne.vhd] -no_script -reset -force -quiet
remove_files  {/localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/countingSimTop.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/comparatorTop.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/XoringInputPop.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/XoringPopCtrl.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/regOne.vhd}
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/comparator.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/reg.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/countingSim.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/SeqAdderCtrl.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/confCompCtrl.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/countingSimCtrl.vhd] -no_script -reset -force -quiet
remove_files  {/localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/comparator.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/reg.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/countingSim.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/SeqAdderCtrl.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/confCompCtrl.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/countingSimCtrl.vhd}
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/fullconfComp.vhd] -no_script -reset -force -quiet
remove_files  /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/fullconfComp.vhd
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/RSA.vhd] -no_script -reset -force -quiet
remove_files  /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/RSA.vhd
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/SeqAdder.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/recMux.vhd] -no_script -reset -force -quiet
remove_files  {/localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/SeqAdder.vhd /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_1/DMA_1.srcs/sources_1/imports/normalHDC/recMux.vhd}























