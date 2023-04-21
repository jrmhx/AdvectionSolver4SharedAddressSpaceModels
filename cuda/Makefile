# Makefile for COMP4300/8300 Assignment 2
# Peter Strazdins, SOCO ANU, 14 Apr 21 
.SUFFIXES:
.PRECIOUS: %.o

HDRS=serAdvect.h parAdvect.h
OBJS=serAdvect.o parAdvect.o
PROG=testAdvect
CCFLAGS=-O3 # -Xptxas -v

all: $(PROG) 

%: %.o $(OBJS)
	nvcc -o $* $*.o $(OBJS) -lm
%.o: %.cu $(HDRS)
	nvcc $(CCFLAGS) -c $*.cu
clean:
	rm -f *.o $(PROG)
