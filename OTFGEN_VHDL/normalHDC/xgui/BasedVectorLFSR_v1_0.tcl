# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "n" -parent ${Page_0}
  ipgui::add_param $IPINST -name "pixbit" -parent ${Page_0}
  ipgui::add_param $IPINST -name "d" -parent ${Page_0}
  ipgui::add_param $IPINST -name "lgf" -parent ${Page_0}
  ipgui::add_param $IPINST -name "c" -parent ${Page_0}
  ipgui::add_param $IPINST -name "featureSize" -parent ${Page_0}
  ipgui::add_param $IPINST -name "adI" -parent ${Page_0}
  ipgui::add_param $IPINST -name "adz" -parent ${Page_0}
  ipgui::add_param $IPINST -name "zComp" -parent ${Page_0}
  ipgui::add_param $IPINST -name "lgCn" -parent ${Page_0}
  ipgui::add_param $IPINST -name "logn" -parent ${Page_0}
  ipgui::add_param $IPINST -name "r" -parent ${Page_0}
  ipgui::add_param $IPINST -name "x" -parent ${Page_0}


}

proc update_PARAM_VALUE.adI { PARAM_VALUE.adI } {
	# Procedure called to update adI when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.adI { PARAM_VALUE.adI } {
	# Procedure called to validate adI
	return true
}

proc update_PARAM_VALUE.adz { PARAM_VALUE.adz } {
	# Procedure called to update adz when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.adz { PARAM_VALUE.adz } {
	# Procedure called to validate adz
	return true
}

proc update_PARAM_VALUE.c { PARAM_VALUE.c } {
	# Procedure called to update c when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.c { PARAM_VALUE.c } {
	# Procedure called to validate c
	return true
}

proc update_PARAM_VALUE.d { PARAM_VALUE.d } {
	# Procedure called to update d when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.d { PARAM_VALUE.d } {
	# Procedure called to validate d
	return true
}

proc update_PARAM_VALUE.featureSize { PARAM_VALUE.featureSize } {
	# Procedure called to update featureSize when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.featureSize { PARAM_VALUE.featureSize } {
	# Procedure called to validate featureSize
	return true
}

proc update_PARAM_VALUE.lgCn { PARAM_VALUE.lgCn } {
	# Procedure called to update lgCn when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.lgCn { PARAM_VALUE.lgCn } {
	# Procedure called to validate lgCn
	return true
}

proc update_PARAM_VALUE.lgf { PARAM_VALUE.lgf } {
	# Procedure called to update lgf when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.lgf { PARAM_VALUE.lgf } {
	# Procedure called to validate lgf
	return true
}

proc update_PARAM_VALUE.logn { PARAM_VALUE.logn } {
	# Procedure called to update logn when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.logn { PARAM_VALUE.logn } {
	# Procedure called to validate logn
	return true
}

proc update_PARAM_VALUE.n { PARAM_VALUE.n } {
	# Procedure called to update n when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.n { PARAM_VALUE.n } {
	# Procedure called to validate n
	return true
}

proc update_PARAM_VALUE.pixbit { PARAM_VALUE.pixbit } {
	# Procedure called to update pixbit when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.pixbit { PARAM_VALUE.pixbit } {
	# Procedure called to validate pixbit
	return true
}

proc update_PARAM_VALUE.r { PARAM_VALUE.r } {
	# Procedure called to update r when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.r { PARAM_VALUE.r } {
	# Procedure called to validate r
	return true
}

proc update_PARAM_VALUE.x { PARAM_VALUE.x } {
	# Procedure called to update x when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.x { PARAM_VALUE.x } {
	# Procedure called to validate x
	return true
}

proc update_PARAM_VALUE.zComp { PARAM_VALUE.zComp } {
	# Procedure called to update zComp when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.zComp { PARAM_VALUE.zComp } {
	# Procedure called to validate zComp
	return true
}


proc update_MODELPARAM_VALUE.n { MODELPARAM_VALUE.n PARAM_VALUE.n } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.n}] ${MODELPARAM_VALUE.n}
}

proc update_MODELPARAM_VALUE.pixbit { MODELPARAM_VALUE.pixbit PARAM_VALUE.pixbit } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.pixbit}] ${MODELPARAM_VALUE.pixbit}
}

proc update_MODELPARAM_VALUE.d { MODELPARAM_VALUE.d PARAM_VALUE.d } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.d}] ${MODELPARAM_VALUE.d}
}

proc update_MODELPARAM_VALUE.lgf { MODELPARAM_VALUE.lgf PARAM_VALUE.lgf } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.lgf}] ${MODELPARAM_VALUE.lgf}
}

proc update_MODELPARAM_VALUE.c { MODELPARAM_VALUE.c PARAM_VALUE.c } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.c}] ${MODELPARAM_VALUE.c}
}

proc update_MODELPARAM_VALUE.featureSize { MODELPARAM_VALUE.featureSize PARAM_VALUE.featureSize } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.featureSize}] ${MODELPARAM_VALUE.featureSize}
}

proc update_MODELPARAM_VALUE.adI { MODELPARAM_VALUE.adI PARAM_VALUE.adI } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.adI}] ${MODELPARAM_VALUE.adI}
}

proc update_MODELPARAM_VALUE.adz { MODELPARAM_VALUE.adz PARAM_VALUE.adz } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.adz}] ${MODELPARAM_VALUE.adz}
}

proc update_MODELPARAM_VALUE.zComp { MODELPARAM_VALUE.zComp PARAM_VALUE.zComp } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.zComp}] ${MODELPARAM_VALUE.zComp}
}

proc update_MODELPARAM_VALUE.lgCn { MODELPARAM_VALUE.lgCn PARAM_VALUE.lgCn } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.lgCn}] ${MODELPARAM_VALUE.lgCn}
}

proc update_MODELPARAM_VALUE.logn { MODELPARAM_VALUE.logn PARAM_VALUE.logn } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.logn}] ${MODELPARAM_VALUE.logn}
}

proc update_MODELPARAM_VALUE.r { MODELPARAM_VALUE.r PARAM_VALUE.r } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.r}] ${MODELPARAM_VALUE.r}
}

proc update_MODELPARAM_VALUE.x { MODELPARAM_VALUE.x PARAM_VALUE.x } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.x}] ${MODELPARAM_VALUE.x}
}

