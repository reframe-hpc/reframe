gribdir = $(shell pwd)/cgribex-1.6.4

all:
	make -C cgribex-1.6.4 > cgribex_make.out
	make -C cgribex-1.6.4 install > cgribex_make_install.out
	cd 3dvar; cat config/mh-linux-x64 | sed "s?/scratch/owm/forCSCS/LETKF.r11529/cgribex-1.6.4?$(gribdir)?" > tmptmp.tmp
	cd 3dvar; cp tmptmp.tmp config/mh-linux-x64
	make -C 3dvar -f Makefile_cscs configure_cn > make_cscs_install.out
	make -C 3dvar -f Makefile_cscs cn >> make_cscs_install.out
	echo "SUCCESS"
