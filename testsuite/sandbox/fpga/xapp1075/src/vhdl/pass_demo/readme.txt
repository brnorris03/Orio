*******************************************************************************
** © Copyright 2010 Xilinx, Inc. All rights reserved.
** This file contains confidential and proprietary information of Xilinx, Inc. and 
** is protected under U.S. and international copyright and other intellectual property laws.
*******************************************************************************
**   ____  ____ 
**  /   /\/   / 
** /___/  \  /   Vendor: Xilinx 
** \   \   \/    
**  \   \        pass_demo\readme.txt
**  /   /        Date Last Modified: October 20, 2010
** /___/   /\    Date Created:       March 9, 2010
** \   \  /  \   
**  \___\/\___\ 
** 
**  Device: Virtex-6
**  Purpose: Virtex-6 Triple-Rate SDI Reference Design
**  Reference: 
**  Revision History: 
**      October 20, 2010: The SystemACE XO on the ML605 boards was recently changed
**      from 32 MHz to 33 MHz. This demo used this clock as a frequency reference,
**      so the demo did not work with newer ML605 boards. The demo was updated
**      to use a 27 MHz reference clock from the FMC card, making it compatible with
**      all versions of the ML605. This demo also incorporates updates to the
**      v6gtx_sdi_control module for reliable GTX TX initialization and improved
**      3G-SDI RX jitter tolerance.
**
**      April 12, 2010: Readme update.
**
**      March 9, 2010: First release.
**   
*******************************************************************************
**
**  Disclaimer: 
**
**      This disclaimer is not a license and does not grant any rights to the materials 
**              distributed herewith. Except as otherwise provided in a valid license issued to you 
**              by Xilinx, and to the maximum extent permitted by applicable law: 
**              (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, 
**              AND XILINX HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, 
**              INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR 
**              FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether in contract 
**              or tort, including negligence, or under any other theory of liability) for any loss or damage 
**              of any kind or nature related to, arising under or in connection with these materials, 
**              including for any direct, or any indirect, special, incidental, or consequential loss 
**              or damage (including loss of data, profits, goodwill, or any type of loss or damage suffered 
**              as a result of any action brought by a third party) even if such damage or loss was 
**              reasonably foreseeable or Xilinx had been advised of the possibility of the same.


**  Critical Applications:
**
**      Xilinx products are not designed or intended to be fail-safe, or for use in any application 
**      requiring fail-safe performance, such as life-support or safety devices or systems, 
**      Class III medical devices, nuclear facilities, applications related to the deployment of airbags,
**      or any other applications that could lead to death, personal injury, or severe property or 
**      environmental damage (individually and collectively, "Critical Applications"). Customer assumes 
**      the sole risk and liability of any use of Xilinx products in Critical Applications, subject only 
**      to applicable laws and regulations governing limitations on product liability.

**  THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.

*******************************************************************************

This folder contains the source code files specific to the v6sdi_pass_demo
project.

*******************************************************************************

** IMPORTANT NOTES **

1) The files in this reference design are intended to be used with XST.
Some of them contain XST-specific constraints that would need to be translated
if a different synthesizer is used.

2) The GTX wrapper files included with this demo are only provided to
allow the demo to be built for the ML605 board. 

3) The GTX wrapper files and .bit file have been optimized for the general ES
version of the Virtex-6 LX240T devices. They should also work on production 
versions of the the LX240T, but may not be completely optimized for these
production devices. These files may not work correctly on the limited early
ES (silicon version 1.0) LX240T devices. To create a version of this demo 
optimized for the production LX240T devices, new GTX wrappers should be 
generated using the RocketIO Wizard and the demo should be regenerated using 
the new GTX wrappers. Please read the instructions near the end of the 
Docs\V6 GTX Triple-Rate SDI.pdf file for details on mandatory editing that must 
be done to the GTX wrappers after they are created using the RocketIO Wizard.

*******************************************************************************

Building the demo

A prebuilt .bit file for this demo is included in the \Demo_Bit_Files folder.

To build this demo, create a new project choosing the Virtex-6 XCLX240T-FF1156-1
as the device. Add the v6sdi_pass_demo.v or .vhd file and the
v6sdi_pass_demo.ucf file to the project as source files.  Add all of the files
in the \V6_SDI folder to the project except the SMPTE352_vpid_insert.v/vhd,
triple_sdi_vpid_insert.v/.vhd, and wide_SRLC16E.v/.vhd files. Add all of the
files in the \misc_source folder to the project.  Also add all of the .v or .vhd
files in the \dru folder to the project. Copy the dru.ngc file in the \dru
folder into the ISE project directory. This file contains the pre-synthesized
DRU design. It should not be added to the project as a source file, but must be
present in the ISE project directory in order  for the tools to find it and and
incorporate it into the design. If the tools  can't find the dru.ngc file, an
error will occur when the project is implemented.

The .xco files found in \pass_demo directory must also be added to the project. 
It is generally best to copy them into the ISE project directory or into a 
subdirectory of the ISE project before adding them to the project. The files are
used to generate Coregen files required by the project. The files generated by 
Coregen will be located in the same directory where these files are located, and
it is generally better to have them stored with the project itself, rather than 
in the source code archive.

The v6sdi_pass_demo_name.txt file must be copied from the \pass_demo directory
into the ISE project directory. This file contains some data that is used by
the module that generates the messages displayed on the LCD on the ML605 board.
This file should not be added as a source file to the project, but must instead
be copied to the ISE project directory. If this file is not present in the
ISE project directory, XST will generate an error when it attempts to read this
file.

The only non-default setting used to build the .bit file for this demo was
the XST HDL option Safe Implementation which was set to Yes.

