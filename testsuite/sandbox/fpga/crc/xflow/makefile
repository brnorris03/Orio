# Pre-req: source <ISE_DS_install_dir>/settings64.sh

impl:
	xflow -p xc6slx9-csg225-3 -implement impl.opt crc.ngc

bitgen:
	xflow -p xc6slx9-csg225-3 -implement impl.opt -config bitgen.opt crc.ngc

clean:
	rm *.flw *.lst *.bld *.xrpt *.ngm *.xml *.mrp *.map *.html *.ptwx *.csv *.pad *.txt *.unroutes *.xpi *.tw* *.scr *.log
	rm *.ngd *.pcf *.ncd *.par *.his
	rm -rf xlnx_* _xmsgs
	rm *.drc *.bit *.xwbt *.bgn

