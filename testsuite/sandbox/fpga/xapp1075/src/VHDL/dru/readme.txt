*******************************************************************************
** © Copyright 2010 Xilinx, Inc. All rights reserved.
** This file contains confidential and proprietary information of Xilinx, Inc. and 
** is protected under U.S. and international copyright and other intellectual property laws.
*******************************************************************************
**   ____  ____ 
**  /   /\/   / 
** /___/  \  /   Vendor: Xilinx 
** \   \   \/    
**  \   \        dru\readme.txt
**  /   /        Date Last Modified: April 12, 2010 
** /___/   /\    Date Created:       March 9, 2010
** \   \  /  \   
**  \___\/\___\ 
** 
**  Device: Virtex-6
**  Purpose: Virtex-6 Triple-Rate SDI Reference Design
**  Reference: 
**  Revision History: 
**      April 12, 2010: Readme update.
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

This folder contains the files needed to implement the data recovery unit
used to receive SD-SDI using oversampling. This is an optimized version of the
NI-DRU found in Xilinx application note XAPP875. The difference is that this
version supports only 11X oversampling of 270 Mb/s bit streams. By optimizing
the NI-DRU in this way, the design is much smaller than the fully programmable
NI-DRU found in XAPP875.

The .v or .vhd files found in this folder must be added to any project that
implements a triple-rate SDI receiver. The DRU is instantiated in the
triple_sdi_rx_20b_v6gtx.v and .vhd modules. Most of the DRU has been pre-
synthesized into the dru.ngc file also found in this folder. Do not add the 
dru.ngc file to the project as a source file. Instead, copy the dru.ngc file 
into the ISE project directory. If the dru.ngc file is not located in the ISE 
project directory, an error will be generated when the project is implemented.

The dru.ngc file included here is specifically designed for Virtex-6 FPGA
devices and can not be used with other devices.

Because the DRU has been optimized to only support 270 Mb/s, this is the only
SD-SDI bit rate that it supports. If other SD-SDI bit rates are required,
the fully programmable NI-DRU from XAPP875 can be used instead. This will require
other changes to the triple-rate SDI reference design to properly control the
NI-DRU and to implement a search algorithm to search through the various SD-SDI
bit rates when the SDI receiver is not locked. Xilinx does not currently have
a triple-rate SDI reference design that supports SD-SDI bit rates other than 
270 Mb/s.

