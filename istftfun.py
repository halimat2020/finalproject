def stftfun(x,R,Nfft):
	import math
	import numpy as np
	[Nfft, Nc] = x.shape;
	n= range(1,R)-0.5
	wind= math.cos(math.pi*(n/R)-(math.pi/2))
	y= numpy.ifft(x)
	y=y[1:R,:]
	why= np.zeros(1,R/2*(Nc+1))
	end=len(why)

	j=0
	for i in range(1,Nc):
		for i in range(1,i2):
			why[j+i2]=why[j+i2]+(wind*y[:,i])
		j= j+R/2
	why[1:R/2] = why[1:R/2] / (wind[1:R/2]**2)
	#take care of last half-block
	why[end-R/2+(1:R/2)] = why[end-R/2+(1:R/2)] / (wind[R/2+1:R]**2)

	why = why[R+(1:N)]
	return(why)

