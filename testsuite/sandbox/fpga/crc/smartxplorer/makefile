# Pre-req: source <ISE_DS_install_dir>/settings64.sh

DEVICE=xc6vlx240t-ff1156-1
EX=crc

### Smart explorer workflow ###
xplor:
	smartxplorer -p $(DEVICE) -ra -pwo on -area_report "slice_reg;lut;bram;slice;dsp48" -uc ../ucf/$(EX).ucf -sd "../src/" -wd tmp $(EX).xst

### Misc ###
clean:
	rm -rf tmp/*

