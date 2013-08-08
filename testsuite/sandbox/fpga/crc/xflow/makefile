# Pre-req: source <ISE_DS_install_dir>/settings64.sh

DEVICE=xc6vlx240t-ff1156-1

synth:
	xflow -p $(DEVICE) -synth synth.opt ../src/crc.v

synth2:
	xst -ifn crc_xst.scr -ofn crc_xst.log

impl:
	xflow -p $(DEVICE) -implement impl.opt crc.ngc

bitgen:
	xflow -p $(DEVICE) -implement impl.opt -config bitgen.opt crc.ngc

crc.bin: crc.bit
	promgen -w -p bin -o crc.bin -u 0 crc.bit

install: crc.bin
	stty --file=/dev/ttyUSB0 -opost #no post-processing, raw output
	cat counter.bin > /dev/ttyUSB0

power:
	xpwr crc.ncd crc.pcf

clean:
	rm -rf *.lst *.bld *.xrpt *.ngm *.xml *.mrp *.map *.html *.ptwx *.csv *.pad *.txt *.unroutes *.xpi *.tw* *.log
	rm -rf *.ngd *.ngc *.pcf *.ncd *.par *.his
	rm -rf xlnx_* _xmsgs xst
	rm -rf *.drc *.bit *.xwbt *.bgn

cleansyn:
	rm -rf *.log *.xrpt *.ngc
	rm -rf xst _xmsgs
