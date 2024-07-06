#This file gets called with a path to the project directory
#A config file should be in the directory, which specifies the training script and parameters of the HDC model
#make_proj.py will create the vivado project using templates from template.py

import os
import sys
import subprocess
import argparse
import json
import genConfig
import genMem
from glob import glob
import time

parser = argparse.ArgumentParser(description="Make Vivado project")
parser.add_argument("vivado_path", type=str, nargs='?', default="", help="Path to Vivado")
parser.add_argument("--version", type=str, nargs='?', default="2022.1", help="Vivado Version")
parser.add_argument("--pwd", type=str, nargs='?', default="", help="RCD_E3HDC directory")
parser.add_argument("--project_dir", type=str, nargs='?', default="", help="Path to project directory")

args = parser.parse_args()

def write_log(log, success, fail):
  while True:
    line = process.stdout.readline().decode("utf-8")
    log.writelines(line)
    if (success in line):
        break
    elif(fail in line): 
        print("Step failed. Please check the logs in the project directory")
        print("Exiting")
        sys.exit()

def read_log(file_path, success, fail, run):
    #wait until file exists
    print("-Waiting for pre-"+run+" steps to finish")
    while not os.path.exists(file_path):
        time.sleep(1)
    #read until success
    print("-Pre-"+run+" steps complete. Running "+ run)
    time.sleep(1)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            while True:
                line = f.readline()
                if line:
                    if (success in line):
                        break
                    elif (fail in line):
                        print("Step failed. Please check the logs in "+file_path)
                        print("Exiting")
                        sys.exit()
                else:
                    time.sleep(0.5)

#Read configuration file
PROJECT_DIR = args.project_dir
with open(PROJECT_DIR+"config.json") as f:
    config = json.load(f)
    #print(config)

PROJECT_NAME    = config["PROJECT_NAME"]
BOARD           = config["BOARD"]
TRAIN           = config["TRAIN"]
LFSR            = config["LFSR"]
DIMENSIONS      = config["DIMENSIONS"]
FEATURES        = config["FEATURES"]
NUM_LEVELS      = config["NUM_LEVELS"]
NUM_CLASSES     = config["NUM_CLASSES"]
SPARSE          = config["SPARSE"]
FREQ_MHZ        = config["FREQUENCY"]
HDC_DIR         = args.pwd+"/OTFGEN_VHDL"
VIVADO_VERSION  = args.version

sys.path.insert(1, args.project_dir)
os.makedirs(args.project_dir+"/mem", exist_ok=True)
os.makedirs(args.project_dir+"/model", exist_ok=True)
if TRAIN:
    #if TRAIN is set, train.py should be provided in the project directory
    #train.py should contain functions train and test and save model as well as encoder
    print("---Training HDC Model---")
    import hdc
    hdc.train()
    hdc.test()


print("---Generating HDC Accelerator---")
#Generate memory files from a given model
start = time.time()
print("1. Generating memory files and HDC Configuration")

if SPARSE:
    #get the list of elements to prune
    sparse_list = genConfig.class_sparsity(args.project_dir)
    #generate the resulting hardware configuration
    hdc_config = genConfig.sparseconfig(DIMENSIONS, FEATURES, len(sparse_list), NUM_LEVELS, NUM_CLASSES)
    os.makedirs(args.project_dir+"mem/"+"sparse", exist_ok=True)
    #generate memory files
    genMem.class_normalize_memory_sparse(sparse_list,
                                         2**hdc_config["n"],
                                         hdc_config["adI"],
                                         (2**hdc_config["n"]) * hdc_config["adI"] - hdc_config["sparsity"],
                                         args.project_dir)
    #get all paths to the CHV memory files seperated by one empty space
    CHVS = ' '.join(glob(args.project_dir+"mem/sparse/*.mif"))
    genMem.gen_sparsemodule(args.project_dir, sparse_list, DIMENSIONS)
else:
    #generate the resulting hardware configuration
    hdc_config = genConfig.config(DIMENSIONS, FEATURES, NUM_LEVELS, NUM_CLASSES)
    os.makedirs(args.project_dir+"mem/"+"normal", exist_ok=True)
    #generate memory files
    genMem.class_normalize_memory(2**hdc_config["n"],
                                  hdc_config["adI"],
                                  (2**hdc_config["n"])*hdc_config["adI"] - DIMENSIONS,
                                  args.project_dir)
    #get all paths to the CHV memory files seperated by one empty space
    CHVS = ' '.join(glob(args.project_dir+"mem/normal/*.mif"))

genMem.write_memory(args.project_dir, DIMENSIONS)

with open(PROJECT_DIR+"hdc_config.json", "w") as f:
    config = json.dump(hdc_config, f)


if LFSR:
    import lfsr_template as template
    if SPARSE:
        ENCODING = "SparseHDC"
        SOURCEFILES= (HDC_DIR
        +"/SparseHDC/BasedVectorLFSR.vhd "
        +HDC_DIR+"/SparseHDC/classifier.vhd "
        +HDC_DIR+"/SparseHDC/comparator.vhd "
        +args.project_dir+"connector.vhd "
        +HDC_DIR+"/SparseHDC/comparatorTop.vhd "
        +HDC_DIR+"/SparseHDC/confCompCtrl.vhd "
        +HDC_DIR+"/SparseHDC/countingSim.vhd "
        +HDC_DIR+"/SparseHDC/countingSimCtrl.vhd "
        +HDC_DIR+"/SparseHDC/countingSimTop.vhd "
        +HDC_DIR+"/SparseHDC/encoder.vhd "
        +HDC_DIR+"/SparseHDC/confComp.vhd "
        +HDC_DIR+"/SparseHDC/fulltop.vhd "
        +HDC_DIR+"/SparseHDC/hdcTest.vhd "
        +HDC_DIR+"/SparseHDC/hvTOcompIn.vhd "
        +HDC_DIR+"/SparseHDC/id_level.vhd "
        +HDC_DIR+"/SparseHDC/popCount.vhd "
        +HDC_DIR+"/SparseHDC/recMux.vhd "
        +HDC_DIR+"/SparseHDC/reg.vhd "
        +HDC_DIR+"/SparseHDC/regOne.vhd "
        +HDC_DIR+"/SparseHDC/RSA.vhd "
        +HDC_DIR+"/SparseHDC/SeqAdder.vhd "
        +HDC_DIR+"/SparseHDC/SeqAdderCtrl.vhd "
        +HDC_DIR+"/SparseHDC/XoringInputPop.vhd "
        +HDC_DIR+"/SparseHDC/XoringPopCtrl.vhd")
    else:
        ENCODING = "LFSRHDC"
        SOURCEFILES= (HDC_DIR
        +"/LFSRHDC/BasedVectorLFSR.vhd "
        +HDC_DIR+"/LFSRHDC/classifier.vhd "
        +HDC_DIR+"/LFSRHDC/comparator.vhd "
        +HDC_DIR+"/LFSRHDC/comparatorTop.vhd "
        +HDC_DIR+"/LFSRHDC/confCompCtrl.vhd "
        +HDC_DIR+"/LFSRHDC/countingSim.vhd "
        +HDC_DIR+"/LFSRHDC/countingSimCtrl.vhd "
        +HDC_DIR+"/LFSRHDC/countingSimTop.vhd "
        +HDC_DIR+"/LFSRHDC/encoder.vhd "
        +HDC_DIR+"/LFSRHDC/fullconfComp.vhd "
        +HDC_DIR+"/LFSRHDC/fulltop.vhd "
        +HDC_DIR+"/LFSRHDC/hdcTest.vhd "
        +HDC_DIR+"/LFSRHDC/hvTOcompIn.vhd "
        +HDC_DIR+"/LFSRHDC/id_level.vhd "
        +HDC_DIR+"/LFSRHDC/popCount.vhd "
        +HDC_DIR+"/LFSRHDC/recMux.vhd "
        +HDC_DIR+"/LFSRHDC/reg.vhd "
        +HDC_DIR+"/LFSRHDC/regOne.vhd "
        +HDC_DIR+"/LFSRHDC/RSA.vhd "
        +HDC_DIR+"/LFSRHDC/SeqAdder.vhd "
        +HDC_DIR+"/LFSRHDC/SeqAdderCtrl.vhd "
        +HDC_DIR+"/LFSRHDC/XoringInputPop.vhd "
        +HDC_DIR+"/LFSRHDC/XoringPopCtrl.vhd")
else:
    import bv_template as template
    ENCODING = "normalHDC"
    SOURCEFILES= (HDC_DIR
    +"/normalHDC/BasedVectorLFSR.vhd "
    +HDC_DIR+"/normalHDC/BV_mem.vhd "
    +HDC_DIR+"/normalHDC/classifier.vhd "
    +HDC_DIR+"/normalHDC/comparator.vhd "
    +HDC_DIR+"/normalHDC/comparatorTop.vhd "
    +HDC_DIR+"/normalHDC/confCompCtrl.vhd "
    +HDC_DIR+"/normalHDC/countingSim.vhd "
    +HDC_DIR+"/normalHDC/countingSimCtrl.vhd "
    +HDC_DIR+"/normalHDC/countingSimTop.vhd "
    +HDC_DIR+"/normalHDC/encoder.vhd "
    +HDC_DIR+"/normalHDC/fullconfComp.vhd "
    +HDC_DIR+"/normalHDC/fulltop.vhd "
    +HDC_DIR+"/normalHDC/hdcTest.vhd "
    +HDC_DIR+"/normalHDC/hvTOcompIn.vhd "
    +HDC_DIR+"/normalHDC/id_level.vhd "
    +HDC_DIR+"/normalHDC/popCount.vhd "
    +HDC_DIR+"/normalHDC/recMux.vhd "
    +HDC_DIR+"/normalHDC/reg.vhd "
    +HDC_DIR+"/normalHDC/regOne.vhd "
    +HDC_DIR+"/normalHDC/RSA.vhd "
    +HDC_DIR+"/normalHDC/SeqAdder.vhd "
    +HDC_DIR+"/normalHDC/SeqAdderCtrl.vhd "
    +HDC_DIR+"/normalHDC/XoringInputPop.vhd "
    +HDC_DIR+"/normalHDC/XoringPopCtrl.vhd")


print("2. Starting Vivado in TCL Mode")
source = 'source %s/settings64.sh \n'%(args.vivado_path)
cmds = [
    b'#!/bin/bash \n',
    source.encode('utf-8'),
    b'vivado -mode tcl \n'
    ]


bash_command = ["bash"]
process = subprocess.Popen(bash_command, stdin=subprocess.PIPE ,stdout=subprocess.PIPE, stderr=subprocess.PIPE)

for cmd in cmds:
    process.stdin.write(cmd)
    process.stdin.flush()

print("3. Create Vivado project")
#Prepare log file
log = open(args.project_dir+"create_project.log", "w")
create_project = (template.create_project_tcl_template % (PROJECT_NAME, PROJECT_DIR, BOARD, CHVS, HDC_DIR, SOURCEFILES, ENCODING, VIVADO_VERSION, FREQ_MHZ)).encode('utf-8')
process.stdin.write(create_project)
process.stdin.flush()

write_log(log, "DONE", "failed")
log.close()


print("4. Generate IP for HDC")
#Prepare log file
log = open(args.project_dir+"create_ip.log", "w")
#load initial values for LSFR
with open(args.project_dir+'mem/configSignature.txt', 'r') as f:
    signature = f.readline()
with open(args.project_dir+'mem/configInitialvalues.txt', 'r') as f:
    init = f.readline()
create_ip = (template.create_ip_tcl_template % (signature, init)).encode('utf-8')
process.stdin.write(create_ip)
process.stdin.flush()

write_log(log, "DONE", "failed")
log.close()


#We need to manually adjust the constant initialization values for LFSR

print("5. Create Block Design")
#Prepare log file
log = open(args.project_dir+"create_bd.log", "w")
create_bd = (template.create_block_design % (hdc_config["sparsity"],
                                             hdc_config["in_width"],
                                             hdc_config["dim_size"],
                                             hdc_config["lgf"],
                                             hdc_config["num_classes"],
                                             hdc_config["feature_size"],
                                             hdc_config["n"],
                                             hdc_config["adI"],
                                             hdc_config["adz"],
                                             hdc_config["zComp"],
                                             hdc_config["lgCn"],
                                             hdc_config["logn"],
                                             hdc_config["remainder"],
                                             hdc_config["x"]
                                             )).encode('utf-8')
process.stdin.write(create_bd)
process.stdin.flush()

write_log(log, "DONE", "failed")
log.close()

print("6. Run Synthesis")
# log = open(args.project_dir+"run_synthesis.log", "w")
launch_synth = template.launch_synth.encode('utf-8')
process.stdin.write(launch_synth)
process.stdin.flush()
read_log(args.project_dir+PROJECT_NAME+"/"+PROJECT_NAME+".runs/synth_1/runme.log", "synth_design completed successfully", "synth_design failed", "synthesis")

# write_log(log, "synth_design completed successfully", "synth_design failed")
# log.close()

print("7. Run Implementation")
# log = open(args.project_dir+"launch_implementation.log", "w")
launch_impl = template.launch_impl.encode('utf-8')
process.stdin.write(launch_impl)
process.stdin.flush()
read_log(args.project_dir+PROJECT_NAME+"/"+PROJECT_NAME+".runs/impl_1/runme.log", "report_power completed successfully", "phys_opt_design failed", "implementation")

# write_log(log, "launch_impl completed successfully", "launch_impl failed")
# log.close()

time.sleep(10)
print("8. Generate Bitstream")
os.makedirs(args.project_dir+"release", exist_ok=True)
# log = open(args.project_dir+"generate_bitstream.log", "w")
generate_bitstream = template.generate_bitstream.encode('utf-8')
process.stdin.write(generate_bitstream)
process.stdin.flush()

# write_log(log, "write_bitstream completed successfully", "write_bitstream failed")
# log.close()
read_log(args.project_dir+PROJECT_NAME+"/"+PROJECT_NAME+".runs/impl_1/runme.log", "write_bitstream completed successfully", "write_bitstream failed", "write bitstream")

#Need to wait a few seconds for vivado to properly finish writing all files
time.sleep(10)
print("9. Preparing Driver")
log = open(args.project_dir+"prepare_driver.log", "w")
prepare_driver = template.prepare_driver.encode('utf-8')
process.stdin.write(prepare_driver)
process.stdin.flush()
write_log(log, "DONE", "failed")
log.close()


# write_log(log, "write_bitstream completed successfully", "write_bitstream failed")
# log.close()

process.stdout.close()
process.stdin.close()

end = time.time()
print("Finished at " + time.strftime("%H:%M:%S", time.localtime()))
print("Hardware generation took " + str(end - start) + " seconds")