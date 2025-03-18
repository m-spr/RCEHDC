create_project_tcl_template="""
set PROJECT_NAME %s
set PROJECT_DIR %s
set BOARD %s
set CHVS "%s"
set HDC_DIR %s
set SOURCEFILES "%s"
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
import_files -norecurse $PROJECT_DIR/mem/BV_img.coe
import_files -norecurse $PROJECT_DIR/mem/ID_img.coe
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
set FEATURES %d
set DIMENSIONS %d
set LEVELS %d

ipx::package_project -root_dir $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1 -vendor user.org -library user -taxonomy /UserIP
set_property core_revision 2 [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity [ipx::current_core]
set_property  ip_repo_paths  $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1 [current_project]

update_ip_catalog


set_property top fulltopHDC [current_fileset]
update_compile_order -fileset sources_1

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

puts DONE
"""

insert_block_mem="""
create_ip -name blk_mem_gen -vendor xilinx.com -library ip -version 8.4 -module_name blk_mem_gen_BV
set_property -dict [list \
  CONFIG.Coe_File ${PROJECT_DIR}/${PROJECT_NAME}/${PROJECT_NAME}.srcs/sources_1/imports/mem/BV_img.coe \
  CONFIG.Enable_A {Always_Enabled} \
  CONFIG.Load_Init_File {true} \
  CONFIG.Memory_Type {Single_Port_ROM} \
  CONFIG.Register_PortA_Output_of_Memory_Primitives {false} \
  CONFIG.Write_Depth_A $FEATURES \
  CONFIG.Write_Width_A $DIMENSIONS \
] [get_ips blk_mem_gen_BV]
set_property -dict [list CONFIG.Write_Width_A [expr int($DIMENSIONS)] CONFIG.Write_Depth_A [expr int($FEATURES)] CONFIG.Read_Width_A [expr int($DIMENSIONS)]] [get_ips blk_mem_gen_BV]

generate_target {instantiation_template} [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/ip/blk_mem_gen_BV/blk_mem_gen_BV.xci]
update_compile_order -fileset sources_1
generate_target all [get_files  $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/ip/blk_mem_gen_BV/blk_mem_gen_BV.xci]
catch { config_ip_cache -export [get_ips -all blk_mem_gen_BV] }
export_ip_user_files -of_objects [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/ip/blk_mem_gen_BV/blk_mem_gen_BV.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/ip/blk_mem_gen_BV/blk_mem_gen_BV.xci]
launch_runs blk_mem_gen_BV_synth_1 -jobs 8
export_simulation -of_objects [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/ip/blk_mem_gen_BV/blk_mem_gen_BV.xci] -directory $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.ip_user_files/sim_scripts -ip_user_files_dir $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.ip_user_files -ipstatic_source_dir $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.ip_user_files/ipstatic -lib_map_path [list {modelsim=$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.cache/compile_simlib/modelsim} {questa=$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.cache/compile_simlib/questa} {xcelium=$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.cache/compile_simlib/xcelium} {vcs=$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.cache/compile_simlib/vcs} {riviera=$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet


create_ip -name blk_mem_gen -vendor xilinx.com -library ip -version 8.4 -module_name blk_mem_gen_ID
set_property -dict [list \
  CONFIG.Coe_File ${PROJECT_DIR}/${PROJECT_NAME}/${PROJECT_NAME}.srcs/sources_1/imports/mem/ID_img.coe \
  CONFIG.Enable_A {Always_Enabled} \
  CONFIG.Load_Init_File {true} \
  CONFIG.Memory_Type {Single_Port_ROM} \
  CONFIG.Register_PortA_Output_of_Memory_Primitives {false} \
  CONFIG.Write_Depth_A $LEVELS \
  CONFIG.Write_Width_A $DIMENSIONS \
] [get_ips blk_mem_gen_ID]
set_property -dict [list CONFIG.Write_Width_A [expr int($DIMENSIONS)] CONFIG.Write_Depth_A [expr int($LEVELS)] CONFIG.Read_Width_A [expr int($DIMENSIONS)]] [get_ips blk_mem_gen_ID]

generate_target {instantiation_template} [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/ip/blk_mem_gen_ID/blk_mem_gen_ID.xci]
update_compile_order -fileset sources_1
generate_target all [get_files  $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/ip/blk_mem_gen_ID/blk_mem_gen_ID.xci]
catch { config_ip_cache -export [get_ips -all blk_mem_gen_ID] }
export_ip_user_files -of_objects [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/ip/blk_mem_gen_ID/blk_mem_gen_ID.xci] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/ip/blk_mem_gen_ID/blk_mem_gen_ID.xci]
launch_runs blk_mem_gen_ID_synth_1 -jobs 8
export_simulation -of_objects [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/ip/blk_mem_gen_ID/blk_mem_gen_ID.xci] -directory $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.ip_user_files/sim_scripts -ip_user_files_dir $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.ip_user_files -ipstatic_source_dir $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.ip_user_files/ipstatic -lib_map_path [list {modelsim=$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.cache/compile_simlib/modelsim} {questa=$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.cache/compile_simlib/questa} {xcelium=$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.cache/compile_simlib/xcelium} {vcs=$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.cache/compile_simlib/vcs} {riviera=$PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.cache/compile_simlib/riviera}] -use_ip_compiled_libs -force -quiet

"""

repackage_ip="""
#repackage IP 
set_property core_revision 3 [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity [ipx::current_core]
ipx::save_core [ipx::current_core]
update_ip_catalog -rebuild -repo_path $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1
ipx::merge_project_changes files [ipx::current_core]
ipx::merge_project_changes hdl_parameters [ipx::current_core]

update_compile_order -fileset sources_1
ipx::merge_project_changes files [ipx::current_core]

ipx::save_core [ipx::current_core]
#ipx::move_temp_component_back -component [ipx::current_core]
update_ip_catalog -rebuild -repo_path $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1



puts DONE
"""


create_block_design="""
set SPARSITY %d

#make diagram 
create_bd_design "design_1"

update_compile_order -fileset sources_1


startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
endgroup

startgroup
set_property -dict [list   CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ [expr int($FREQ_MHZ)]   CONFIG.PCW_QSPI_GRP_SINGLE_SS_ENABLE {1} ] [get_bd_cells processing_system7_0]
endgroup

startgroup
set_property -dict [list \
  CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ [expr int($FREQ_MHZ)] \
  CONFIG.PCW_QSPI_GRP_SINGLE_SS_ENABLE {1} \
] [get_bd_cells processing_system7_0]
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
#apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/processing_system7_0/FCLK_CLK0 (100 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins fulltopHDC_0/clk]
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

connect_bd_net [get_bd_pins fulltopHDC_0/clk] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins fulltopHDC_0/rst] [get_bd_pins rst_ps7_0_${FREQ_MHZ}M/peripheral_aresetn]
connect_bd_net [get_bd_pins smartconnect_0/aresetn] [get_bd_pins rst_ps7_0_${FREQ_MHZ}M/peripheral_aresetn]
connect_bd_net [get_bd_pins smartconnect_1/aresetn] [get_bd_pins rst_ps7_0_${FREQ_MHZ}M/peripheral_aresetn]
connect_bd_net [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins axi_dma_1/m_axi_s2mm_aclk] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins smartconnect_0/aclk] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins smartconnect_1/aclk] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net [get_bd_pins processing_system7_0/S_AXI_HP1_ACLK] [get_bd_pins processing_system7_0/FCLK_CLK0]
validate_bd_design

# connect_bd_net [get_bd_pins smartconnect_0/aresetn] [get_bd_pins rst_ps7_0_100M/peripheral_aresetn]
# connect_bd_net [get_bd_pins smartconnect_1/aresetn] [get_bd_pins rst_ps7_0_100M/peripheral_aresetn]

# connect_bd_net [get_bd_pins smartconnect_0/aresetn] [get_bd_pins rst_ps7_0_${FREQ_MHZ}M/peripheral_aresetn]
# connect_bd_net [get_bd_pins smartconnect_1/aresetn] [get_bd_pins rst_ps7_0_${FREQ_MHZ}M/peripheral_aresetn]


assign_bd_address

set_property -dict [list CONFIG.FREQ_HZ [expr int($FREQ_MHZ*1000000)]] [get_bd_pins fulltopHDC_0/clk]
set_property -dict [list CONFIG.FREQ_HZ [expr int($FREQ_MHZ*1000000)]] [get_bd_intf_pins fulltopHDC_0/S_AXI]


startgroup
set_property -dict [list CONFIG.pixbit {%d} CONFIG.d {%d} CONFIG.lgf {%d} CONFIG.c {%d} CONFIG.featureSize {%d} CONFIG.n {%d} CONFIG.adI {%d} CONFIG.adz {%d} CONFIG.zComp {%d} CONFIG.lgCn {%d} CONFIG.logn {%d} CONFIG.log2features {%d} CONFIG.log2id {%d} CONFIG.lenTKEEP_M {%d} CONFIG.lenTDATA_S {%d} CONFIG.lenTKEEP_S {%d} ] [get_bd_cells fulltopHDC_0]
endgroup

if { $SPARSITY > 0 } {
  set_property -dict [list CONFIG.sparse [expr int($SPARSITY)]] [get_bd_cells fulltopHDC_0]
}

regenerate_bd_layout
validate_bd_design
save_bd_design

make_wrapper -files [get_files $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.gen/sources_1/bd/design_1/hdl/design_1_wrapper.vhd
update_compile_order -fileset sources_1
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

#exec cp -r $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.srcs/sources_1/imports/mem $PROJECT_DIR/$PROJECT_NAME/$PROJECT_NAME.gen/sources_1/bd/design_1/ip/imports/mem


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
