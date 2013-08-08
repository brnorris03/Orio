# Pre-req: source <ISE_DS_install_dir>/settings64.sh

DEVICE=xc6vlx240t-ff1156-1

synth:
	xflow -p $(DEVICE) -synth synth.opt ../src/crc.v

synth2: crc_xst.scr
	xst -ifn crc_xst.scr -ofn crc_xst.log

ngd: crc.ngc
	ngdbuild -p xc6vlx240t-ff1156-1 -nt timestamp -uc ../ucf/crc.ucf crc.ngc crc.ngd

map: crc.ngd crc.pcf
	map -o crc_map.ncd -w -pr b -timing -detail -global_opt power -logic_opt on -power xe -xe c crc.ngd crc.pcf
	#-ol std|high(default) #overall effort level
	#-cm area(default)|speed|balanced #cover mode
	#-global_opt off(default)|speed|area|power
	#-logic_opt on|off
	#-mt off(default)|2 #multi-threading
	#-power on|off(default)|high|xe #power optimization
	#-pr off(default)|i|o|b #pack registers in i/o
	#-xe n|c #extra effort level
	#-w #overwrite existing files
	#-timing #timing-driven par

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
