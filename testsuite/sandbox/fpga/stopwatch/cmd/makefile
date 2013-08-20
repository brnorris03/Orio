# Pre-req: source <ISE_DS_install_dir>/settings64.sh

DEVICE=xc6vlx240t-ff1156-1
EX=stopwatch

### Atomic workflow ###
syn:
	xst -ifn $(EX).xst

ngd: $(EX).ngc
	ngdbuild -p $(DEVICE) -nt timestamp -uc ../src/$(EX).ucf $(EX).ngc $(EX).ngd

map: crc.ngd 
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

par: crc_map.ncd
	par -w -ol high -xe c -power on crc_map.ncd crc.ncd crc.pcf
	#-mt off(default)|2|3|4
	#-ol high(default)|std
	#-power on|off(default)
	#-xe n|c

trace: crc.ncd
	trce -v 200 -u 100 -fastpaths -o crc.twr -xml crc.twx crc.ncd crc.pcf

genbit: crc.ncd crc.pcf
	bitgen -f crc.ut crc.ncd crc.bit crc.pcf


### Device programming ###
genprom: crc.bit
	promgen -w -p bin -o crc.bin -u 0 crc.bit

install: crc.bin
	stty --file=/dev/ttyUSB0 -opost #no post-processing, raw output
	cat crc.bin > /dev/ttyUSB0


### Analysis ###
power: crc.ncd
	xpwr crc.ncd crc.pcf


### Misc ###
clean:
	rm -rf *.lst *.bld *.xrpt *.ngm *.xml *.mrp *.map *.html *.ptwx *.csv *.pad *.txt *.unroutes *.xpi *.tw* *.log
	rm -rf *.ngd *.ngc *.pcf *.ncd *.par *.his
	rm -rf xlnx_* _xmsgs xst *.srp *.ngr
	rm -rf *.drc *.bit *.xwbt *.bgn
	rm -rf *.prm *.psr *.cfi *.en *.bin *.pwr

cleansyn:
	rm -rf *.log *.xrpt *.ngc
	rm -rf xst _xmsgs

