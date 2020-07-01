%mex -setup cpp

mex -g -O  CFLAGS='-Wall -Wextra -W -std=c99 -fPIC -lpthread -lrt ' opal_list.c

%mex snipe_opal_ssd.c
