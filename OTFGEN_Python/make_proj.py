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
        break

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
                        break
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

if LFSR:
    import template
else:
    pass

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
    hdc_config = genConfig.sparseconfig(DIMENSIONS, SPARSE, FEATURES, NUM_LEVELS, NUM_CLASSES)
    os.makedirs(args.project_dir+"mem/"+"sparse", exist_ok=True)
    genMem.class_normalize_memory_sparse(2**hdc_config["n"], hdc_config["adI"], (2**hdc_config["n"])*hdc_config["adI"] - DIMENSIONS, args.project_dir)
    CHVS = ' '.join(glob(args.project_dir+"mem/sparse/*.mif"))
else:
    hdc_config = genConfig.config(DIMENSIONS, FEATURES, NUM_LEVELS, NUM_CLASSES)
    os.makedirs(args.project_dir+"mem/"+"normal", exist_ok=True)
    genMem.class_normalize_memory(2**hdc_config["n"], hdc_config["adI"], (2**hdc_config["n"])*hdc_config["adI"] - DIMENSIONS, args.project_dir)
    CHVS = ' '.join(glob(args.project_dir+"mem/normal/*.mif"))

genMem.write_memory(args.project_dir, DIMENSIONS)

with open(PROJECT_DIR+"hdc_config.json", "w") as f:
    config = json.dump(hdc_config, f)


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
create_project = (template.create_project_tcl_template % (PROJECT_NAME, PROJECT_DIR, BOARD, CHVS, HDC_DIR, VIVADO_VERSION, FREQ_MHZ)).encode('utf-8')
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
    #print(signature)
with open(args.project_dir+'mem/configInitialvalues.txt', 'r') as f:
    init = f.readline()
    #print(init)
create_ip = (template.create_ip_tcl_template % (signature, init)).encode('utf-8')
process.stdin.write(create_ip)
process.stdin.flush()

write_log(log, "DONE", "failed")
log.close()


#We need to manually adjust the constant initialization values for LFSR

print("5. Create Block Design")
#Prepare log file
log = open(args.project_dir+"create_bd.log", "w")
create_bd = (template.create_block_design % (hdc_config["in_width"],
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

time.sleep(5)
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