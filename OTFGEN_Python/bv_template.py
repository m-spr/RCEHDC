create_project_tcl_template="""
set PROJECT_NAME %s
set PROJECT_DIR %s
set BOARD %s
set CHVS "%s"
set HDC_DIR %s
set SOURCEFILES %s
set ENCODING %s
set VIVADO_VERSION %s
set FREQ_MHZ %d

#xc7z020clg400-1
#tul.com.tw:pynq-z2:part0:1.0

if {$BOARD == "PYNQ-Z2"} {
  create_project -force $PROJECT_NAME $PROJECT_DIR/$PROJECT_NAME -part xc7z020clg400-1
  #set ZYNQ_TYPE "zynq_7000"
  set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]
}
set_property target_language VHDL [current_project]
#open_project $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.xpr
import_files -norecurse $CHVS
import_files -norecurse $SOURCEFILES
if {$VIVADO_VERSION == "2022.2"} {
  -Auto-update_compile_order -fileset sources_1
  -Auto-update_compile_order -fileset sources_1
} else {
  update_compile_order -fileset sources_1
  update_compile_order -fileset sources_1
}
puts DONE
"""

create_ip_tcl_template="""
ipx::package_project -root_dir $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports -vendor user.org -library user -taxonomy /UserIP
set_property core_revision 2 [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity [ipx::current_core]
set_property  ip_repo_paths  $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports [current_project]
update_ip_catalog


#grouping ports

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

set NEW_CONSTANT_VALUE_1 {"%s"}
set NEW_CONSTANT_VALUE_2 {"%s"}
# Set the reference directory for source file relative paths 
# set origin_dir [file dirname [file normalize [info script]]]
# Open the file to be modified (FILE1) and the temporary file (FILE2).
set FILE1 [ open $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/BasedVectorLFSR.vhd r]
set FILE2 [ open $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/BasedVectorLFSR_temp.vhd w]

while { [ gets $FILE1 LINE ] >= 0 } {
    # Search for the constant declarations
    set INDEX_1 [ string match -nocase *constant*congigSigniture* $LINE ]
    set INDEX_2 [ string match -nocase *constant*congigInitialvalues* $LINE ]
  
    # Replace Configuration parameters with the desired values
    if { $INDEX_1 == 1 } {
        puts $FILE2 "CONSTANT congigSigniture : STD_LOGIC_VECTOR (n-1 DOWNTO 0) := ${NEW_CONSTANT_VALUE_1}; "
    } elseif { $INDEX_2 == 1 } {
        puts $FILE2 "CONSTANT congigInitialvalues : STD_LOGIC_VECTOR (n-1 DOWNTO 0) := ${NEW_CONSTANT_VALUE_2}; "
    } else {
    # Write this line, unchanged, to the temporary file if not found.
        puts $FILE2 $LINE
    }
}
    
# Close both files
close $FILE1
close $FILE2
# Rename temporary file
file rename -force $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/BasedVectorLFSR_temp.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/BasedVectorLFSR.vhd


#repackage IP 

set_property core_revision 3 [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity [ipx::current_core]
ipx::save_core [ipx::current_core]
update_ip_catalog -rebuild -repo_path $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports
#save_bd_design
ipx::merge_project_changes files [ipx::current_core]

puts DONE
"""


create_block_design="""

#make diagram 
create_bd_design "design_1"

update_compile_order -fileset sources_1

startgroup
create_bd_cell -type ip -vlnv user.org:user:fulltopHDC:1.0 fulltopHDC_0       
endgroup
delete_bd_objs [get_bd_cells fulltopHDC_0]  

startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
endgroup
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells processing_system7_0]

startgroup
create_bd_cell -type ip -vlnv user.org:user:fulltopHDC:1.0 fulltopHDC_0
endgroup

# remember to make them 8 bits

startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
endgroup
set_property -dict [list \
  CONFIG.c_include_s2mm {0} \
  CONFIG.c_include_sg {0} \
  CONFIG.c_sg_include_stscntrl_strm {0} \
] [get_bd_cells axi_dma_0]

startgroup
set_property -dict [list CONFIG.c_m_axis_mm2s_tdata_width {8}] [get_bd_cells axi_dma_0]
endgroup

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


startgroup
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_dma_0/S_AXI_LITE} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} Slave {/axi_dma_1/S_AXI_LITE} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins axi_DMA_1/S_AXI_LITE]
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
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (100 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (100 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins processing_system7_0/S_AXI_HP1_ACLK]
endgroup

connect_bd_net [get_bd_pins smartconnect_0/aresetn] [get_bd_pins rst_ps7_0_50M/peripheral_aresetn]
connect_bd_net [get_bd_pins smartconnect_1/aresetn] [get_bd_pins rst_ps7_0_50M/peripheral_aresetn]

startgroup
set_property -dict [list \
  CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ [expr int($FREQ_MHZ)] \
  CONFIG.PCW_QSPI_GRP_SINGLE_SS_ENABLE {1} \
] [get_bd_cells processing_system7_0]
endgroup

assign_bd_address
startgroup
set_property -dict [list CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ [expr int($FREQ_MHZ)] CONFIG.PCW_FCLK0_PERIPHERAL_CLKSRC {IO PLL}] [get_bd_cells processing_system7_0]
endgroup

set_property -dict [list CONFIG.FREQ_HZ [expr int($FREQ_MHZ*1000000)]] [get_bd_pins fulltopHDC_0/clk]
set_property -dict [list CONFIG.FREQ_HZ [expr int($FREQ_MHZ*1000000)]] [get_bd_intf_pins fulltopHDC_0/S_AXI]


startgroup
set_property -dict [list CONFIG.pixbit {%d} CONFIG.d {%d} CONFIG.lgf {%d} CONFIG.c {%d} CONFIG.featureSize {%d} CONFIG.n {%d} CONFIG.adI {%d} CONFIG.adz {%d} CONFIG.zComp {%d} CONFIG.lgCn {%d} CONFIG.logn {%d} CONFIG.r {%d} CONFIG.x {%d}] [get_bd_cells fulltopHDC_0]
endgroup



regenerate_bd_layout
validate_bd_design
save_bd_design

make_wrapper -files [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.gen/sources_1/bd/design_1/hdl/design_1_wrapper.vhd
update_compile_order -fileset sources_1
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

puts DONE
"""

launch_synth="""
launch_runs synth_1 -jobs 8
"""

launch_impl="""
launch_runs impl_1 -jobs 8
#launch_runs impl_1 -to_step write_bitstream -jobs 8
"""

generate_bitstream="""
launch_runs impl_1 -to_step write_bitstream -jobs 8
#write_bitstream -force design_1_wrapper.bit
#write_hw_platform -fixed -include_bit -force -file $PROJECT_DIR/$PROJECT_NAME/design_1_wrapper.xsa
#write_bd_tcl -force $PROJECT_DIR/$PROJECT_NAME/design_1.tcl
#file copy -force $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.runs/impl_1/design_1_wrapper.bit $PROJECT_DIR/$PROJECT_NAME/design_1.bit
#open_bd_design {$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/bd/design_1/design_1.bd}

"""

prepare_driver="""
#write_hw_platform -fixed -include_bit -force -file $PROJECT_DIR/$PROJECT_NAME/design_1_wrapper.xsa
write_bd_tcl -force $PROJECT_DIR/release/design_1.tcl
file copy -force $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.runs/impl_1/design_1_wrapper.bit $PROJECT_DIR/release/design_1.bit
file copy -force $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh $PROJECT_DIR/release/design_1.hwh
puts DONE
"""

"""
# DONE
export_simulation -of_objects [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/bd/design_1/design_1.bd] -directory $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.ip_user_files/sim_scripts -ip_user_files_dir $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.ip_user_files -ipstatic_source_dir $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.ip_user_files/ipstatic -lib_map_path [list {modelsim=$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.cache/compile_simlib/modelsim} {questa=$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.cache/compile_simlib/questa} {xcelium=$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.cache/compile_simlib/xcelium} {vcs=$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.cache/compile_simlib/vcs} {riviera=$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet
report_ip_status -name ip_status 
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/fulltop.vhd] -no_script -reset -force -quiet
remove_files  $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/fulltop.vhd
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/hdcTest.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/classifier.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/id_level.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/popCount.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/BasedVectorLFSR.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/encoder.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/hvTOcompIn.vhd] -no_script -reset -force -quiet
remove_files  {$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/hdcTest.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/classifier.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/id_level.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/popCount.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/BasedVectorLFSR.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/encoder.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/hvTOcompIn.vhd}
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/countingSimTop.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/comparatorTop.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/XoringInputPop.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/XoringPopCtrl.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/regOne.vhd] -no_script -reset -force -quiet
remove_files  {$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/countingSimTop.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/comparatorTop.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/XoringInputPop.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/XoringPopCtrl.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/regOne.vhd}
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/comparator.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/reg.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/countingSim.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/SeqAdderCtrl.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/confCompCtrl.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/countingSimCtrl.vhd] -no_script -reset -force -quiet
remove_files  {$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/comparator.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/reg.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/countingSim.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/SeqAdderCtrl.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/confCompCtrl.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/countingSimCtrl.vhd}
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/fullconfComp.vhd] -no_script -reset -force -quiet
remove_files  $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/fullconfComp.vhd
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/RSA.vhd] -no_script -reset -force -quiet
remove_files  $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/RSA.vhd
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/SeqAdder.vhd] -no_script -reset -force -quiet
export_ip_user_files -of_objects  [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/recMux.vhd] -no_script -reset -force -quiet
remove_files  {$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/SeqAdder.vhd $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/$ENCODING/recMux.vhd}














open_bd_design {$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/bd/design_1/design_1.bd}
write_hw_platform -fixed -include_bit -force -file /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_0/design_1.xsa
write_bd_tcl -force /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_0/design_1.tcl
file copy -force $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.runs/impl_1/design_1_wrapper.bit /localdata/sadmah00/github/RCD_E3HDC/vivado/DMA_0/design_1.bit










"""