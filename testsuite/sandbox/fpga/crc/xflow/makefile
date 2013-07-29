# Pre-req: source <ISE_DS_install_dir>/settings64.sh

DEVICE=xc6vlx240t-ff1156-1

synth:
	xflow -p $(DEVICE) -synth synth.opt ../src/crc.v

impl:
	xflow -p $(DEVICE) -implement impl.opt crc.ngc

bitgen:
	xflow -p $(DEVICE) -implement impl.opt -config bitgen.opt crc.ngc

crc.bin: crc.bit
	promgen -w -p bin -o crc.bin -u 0 crc.bit

install: crc.bin
	stty --file=/dev/ttyUSB0 -opost #no post-processing, raw output
	cat counter.bin > /dev/ttyUSB0

clean:
	rm *.flw *.lst *.bld *.xrpt *.ngm *.xml *.mrp *.map *.html *.ptwx *.csv *.pad *.txt *.unroutes *.xpi *.tw* *.scr *.log
	rm *.ngd *.pcf *.ncd *.par *.his
	rm -rf xlnx_* _xmsgs
	rm *.drc *.bit *.xwbt *.bgn

cleansyn:
	rm *.scr *.log *.his *.xrpt
	rm -rf xst _xmsgs
