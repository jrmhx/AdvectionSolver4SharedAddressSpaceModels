# No Opt 2D
./testAdvect -g 32,32 -b 32,32 -d 3 4096 4096 100

# Shared Mem Opt 2D, No Pointer Swapping
./testAdvect -g 32,32 -b 32,32 -d 3 -o 4096 4096 100

# Shared Mem Opt 2D, With Pointer Swapping
./testAdvect -g 32,32 -b 32,32 -d 3 -o -w 1 4096 4096 100

# ./testAdvect -g 32,32 -b 32,32 -o -d 3 4096 4096 10

./testAdvect -h  -d 3 4096 4096 1

./testAdvect -s  -d 3 4096 4096 1

# ./testAdvect -g 2,2 -b 4,4 10 20

# ./testAdvect -g 2,2 -b 2,2 9 9

# ./testAdvect -g 2,2 -b 2,2 -o 9 9

# ./testAdvect -g 4,6 -b 7,3 -d 3 1000 1000

# ./testAdvect -g 4,6 -b 7,3 -o -d 3 1000 1000

# ./testAdvect -g 1,1 -b 3,3 -d 3 10 10 1

# ./testAdvect -g 1,1 -b 3,3 -d 3 -o 10 10 1


# q2_5 
#1
# ./testAdvect -g 8,8 -b 1,64 -d 3 4096 4096 10

# ./testAdvect -g 8,8 -b 2,32 -d 3 4096 4096 10

# ./testAdvect -g 8,8 -b 4,16 -d 3 4096 4096 10

# ./testAdvect -g 8,8 -b 8,8 -d 3 4096 4096 10

# ./testAdvect -g 8,8 -b 16,4 -d 3 4096 4096 10

# ./testAdvect -g 8,8 -b 32,2 -d 3 4096 4096 10

# ./testAdvect -g 8,8 -b 64,1 -d 3 4096 4096 10

# 2 

# ./testAdvect -g 256,256 -b 1,256 -d 3 4096 4096 10

# ./testAdvect -g 256,256 -b 2,128 -d 3 4096 4096 10

# ./testAdvect -g 256,256 -b 4,64 -d 3 4096 4096 10

# ./testAdvect -g 256,256 -b 8,32 -d 3 4096 4096 10

# ./testAdvect -g 256,256 -b 16,16 -d 3 4096 4096 10

# ./testAdvect -g 256,256 -b 32,8 -d 3 4096 4096 10

# ./testAdvect -g 256,256 -b 64,4 -d 3 4096 4096 10

# ./testAdvect -g 256,256 -b 128,2 -d 3 4096 4096 10

# ./testAdvect -g 256,256 -b 256,1 -d 3 4096 4096 10

# 3
# ./testAdvect -g 1,16384 -b 32,32 -d 3 4096 4096 10

# ./testAdvect -g 2,8192 -b 32,32 -d 3 4096 4096 10

# ./testAdvect -g 4,4096 -b 32,32 -d 3 4096 4096 10

# ./testAdvect -g 8,2048 -b 32,32 -d 3 4096 4096 10

# ./testAdvect -g 16,1024 -b 32,32 -d 3 4096 4096 10

# ./testAdvect -g 32,512 -b 32,32 -d 3 4096 4096 10

# ./testAdvect -g 64,256 -b 32,32 -d 3 4096 4096 10

# ./testAdvect -g 128,128 -b 32,32 -d 3 4096 4096 10

# ./testAdvect -g 256,64 -b 32,32 -d 3 4096 4096 10

# ./testAdvect -g 512,32 -b 32,32 -d 3 4096 4096 10

# ./testAdvect -g 1024,16 -b 32,32 -d 3 4096 4096 10

# ./testAdvect -g 2048,8 -b 32,32 -d 3 4096 4096 10

# ./testAdvect -g 4096,4 -b 32,32 -d 3 4096 4096 10

# ./testAdvect -g 8192,2 -b 32,32 -d 3 4096 4096 10

# ./testAdvect -g 16384,1 -b 32,32 -d 3 4096 4096 10