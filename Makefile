# Master Makefile for BrahmAstra

.PHONY: all clean nbody fof reionyuga dvisukta

all: nbody fof reionyuga dvisukta

nbody:
	$(MAKE) -C external/nbody nbody_comp

fof:
	$(MAKE) -C external/fof fof_main

reionyuga:
	$(MAKE) -C external/reionyuga ionz_main

dvisukta:
	$(MAKE) -C external/dvisukta bispec

clean:
	$(MAKE) -C external/nbody clean
	$(MAKE) -C external/fof clean
	$(MAKE) -C external/reionyuga clean
	$(MAKE) -C external/dvisukta clean
