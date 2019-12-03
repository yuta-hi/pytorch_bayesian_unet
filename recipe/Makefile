.PYONY: all

all: AIO.simg

AIO.def: AIO.def.in python_packages.txt
	sed "s/python_packages/`sort python_packages.txt | sed 's/\r//g' | perl -pe 's/\n/ /g'`/g" AIO.def.in > AIO.def

AIO.simg: AIO.def
	sudo singularity build AIO.simg AIO.def
