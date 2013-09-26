# Pre-req: source <ISE_DS_install_dir>/settings64.sh

DEVICE=xc6vlx240t-ff1156-1
NAME=v6sdi_4rx4tx_demo

### Atomic workflow ###
syn:
	xst -intstyle ise -ifn $(NAME).xst -ofn $(NAME).syr

ngd:
	ngdbuild -intstyle ise -dd _ngo -sd ../src/ipcore_dir -sd ../src/vhdl/dru -nt timestamp -i -p $(DEVICE) $(NAME).ngc $(NAME).ngd

map:
	map -intstyle ise -p $(DEVICE) -w -ol high -t 1 -xt 0 -register_duplication off -global_opt off -mt off -ir off -pr off -lc off -power off -o $(NAME)_map.ncd $(NAME).ngd $(NAME).pcf

par:
	par -w -intstyle ise -ol high -mt off $(NAME)_map.ncd $(NAME).ncd $(NAME).pcf

trace:
	trce -intstyle ise -v 3 -s 1 -n 3 -fastpaths -xml $(NAME).twx $(NAME).ncd -o $(NAME).twr $(NAME).pcf

### Exploration ###
xplor_bi:
	smartxplorer -p xc6vlx240t-ff1156-1 -ra -pwo on -area_report "slice_reg;lut;bram;slice;dsp48" -sd "../src/vhdl/;../src/ipcore_dir/" -wd tmp v6sdi_4rx4tx_demo.xst

### Misc ###
clean:
	rm -rf *.lst *.bld *.xrpt *.ngm *.xml *.mrp *.map *.html *.ptwx *.csv *.pad *.txt *.unroutes *.xpi *.tw* *.log
	rm -rf *.ngd *.ngc *.pcf *.ncd *.par *.his
	rm -rf xlnx_* _xmsgs xst _ngo *.ngr
	rm -rf *.drc *.bit *.xwbt *.bgn
	rm -rf *.prm *.psr *.cfi *.en *.bin *.pwr *.lso *.syr

