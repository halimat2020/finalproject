 # Tk label demo: Minimal demo
import Tkinter as Tk   	# for Python 3
import pyaudio
from myfunctions import clip16

import struct
import math
import numpy as np
def funN1(input_tuple):
	import stftfun
	import istftfun
	
	R = 512
	Nfft = 512
	N=len(input_tuple)


	x =stftfun(input_tuple[1:N/2], N/2, Nfft);

	#Set phase to zero in STFT-domain
	y1 = abs(x)
	#Synthesize 'robotic' speech signal
	y1 = istftfun(y1, N/2, N);
	
	outputpast= y1/max(abs(y1));

	#SECOND PART OF SIGNAL
	x2 =stftfun(input_tuple[N/2:N], N, Nfft);

	#Set phase to zero in STFT-domain
	y2 = abs(x2)
	#Synthesize 'robotic' speech signal
	y2 = istftfun(y2, R, N);
	outputnow= y2/max(abs(y2));
	#OVERLAP
	outputblock[N/2]=outputpast+outputnow

	#HOW TO SAVE PREVIOUS OUTPUT VALUE?
	return(outputblock)
def funN2(input_tuple):

	output_block = len(input_tuple) * [0]

	for n in range(0, BLOCKLEN):
	   
	    output_block[n] = abs((input_tuple[n]))
	return(output_block)


def funN6(input_tuple):
	#AM Modulation
	BLOCKLEN= 64
	output_block = BLOCKLEN * [0]

	# f0 = 0      # Normal audio
	f0 = 100   # Modulation frequency (Hz)
	om = 2*math.pi*f0/RATE
	theta = 0
	for n in range(0, BLOCKLEN):
	    # No processing:
        # output_block[n] = input_tuple[n]  
        # OR
        # Amplitude modulation:
		theta = theta + om
		output_block[n] =  input_tuple[n] * math.cos(theta)

    # keep theta betwen -pi and pi
	while theta > math.pi:
		theta = theta - 2*math.pi
	return(output_block)
def funN3(input_block):
	#block filtering
	import numpy as np
	import scipy.signal
	b0 =  0.008442692929081
	b2 = -0.016885385858161
	b4 =  0.008442692929081
	b = [b0, 0.0, b2, 0.0, b4]

	# a0 =  1.000000000000000
	a1 = -3.580673542760982
	a2 =  4.942669993770672
	a3 = -3.114402101627517
	a4 =  0.757546944478829
	a = [1.0, a1, a2, a3, a4]
	MAXVALUE = (2**15)-1  # Maximum allowed output signal value (because WIDTH = 2)
	ORDER = 4   # filter is fourth order
	states = np.zeros(ORDER)
	[output_block, states] = scipy.signal.lfilter(b, a, input_block, zi = states)

    # clipping
	output_block = np.clip(output_block, -MAXVALUE, MAXVALUE)

    # convert to integer
	output_block = output_block.astype(int)
	return(output_block)
def funN5(input_tuple):
	
	#vibrato
	from myfunctions import clip16
	f0 = 2
	W = 0.2   # W = 0 for no effect
	BUFFER_LEN =  64         # Set buffer length.
	buffer = np.zeros(BUFFER_LEN)   # list of zeros

	# Buffer (delay line) indices
	kr = 0  # read index
	kw = int(0.5 * BUFFER_LEN)  # write index (initialize to middle of buffer)
	
	# Get previous and next buffer values (since kr is fractional)
	for n in range(0, 64):


		kr_prev = int(math.floor(kr))
		frac = kr - kr_prev    # 0 <= frac < 1
		kr_next = kr_prev + 1
		if kr_next == BUFFER_LEN:
			kr_next = 0

		    # Compute output value using interpolation
		print(frac)
		print(buffer[kr_prev])
		print(buffer[kr_next])
		print(type(buffer))
		y0 = (1-frac) * buffer[kr_prev] + frac * buffer[kr_next]

		    # Update buffer
		buffer[kw] = input_tuple

		    # Increment read index
		kr = kr + 1 + W * math.sin( 2 * math.pi * f0 * n / RATE )
		        # Note: kr is fractional (not integer!)

		    # Ensure that 0 <= kr < BUFFER_LEN
		if kr >= BUFFER_LEN:
		        # End of buffer. Circle back to front.
			kr = kr - BUFFER_LEN

		    # Increment write index    
		kw = kw + 1
		if kw == BUFFER_LEN:
		        # End of buffer. Circle back to front.
			kw = 0
		output_bytes = struct.pack('h', int(clip16(y0)))
	return(output_bytes)
def funN7(input_tuple):
	#slow voice
	N=len(input_tuple)
	outputblock=[0]*N*2
	r=0
	for n in range(0, (N*2)):
		if (n%2)==0:
			outputblock[n]=input_tuple[r]
			r+=1
		else:
			outputblock[n]=0
	return(outputblock)



root = Tk.Tk()
gain = Tk.DoubleVar()
R = 64
gain.set(0.2 * 2**15)

BLOCKLEN = R      # Number of frames per block
WIDTH = 2           # Number of bytes per signal value
CHANNELS = 1        # mono
RATE = 8000        # Frame rate (frames/second)
RECORD_SECONDS = 5
def funback1():
	global effect_num
	effect_num=2
def funback2():
	global effect_num
	effect_num =2	
def funback3():
	global effect_num
	effect_num =3	
def funback5():
	global effect_num
	effect_num =5	
def funback6(): 
	global effect_num
	effect_num =6
def funback7():
	global effect_num
	effect_num =7


def fun_noeffect():
	global effect_num
	effect_num = 0

def fun_quit():
	global CONTINUE
	CONTINUE = False


p = pyaudio.PyAudio()

stream = p.open(
	format      = p.get_format_from_width(WIDTH),
	channels    = CHANNELS,
	rate        = RATE,
	input       = True,
	output      = True)

	
#output_block = np.zeros(BLOCKLEN)
output_block = [0]*BLOCKLEN

num_blocks = int(RATE / BLOCKLEN * RECORD_SECONDS)
global effect_num

print('* Recording for %.3f seconds' % RECORD_SECONDS)

effect_num = 0
CONTINUE = True


# Define Tk variable
x = Tk.DoubleVar() 
s = Tk.StringVar()

# Define widgets
S_gain = Tk.Scale(root, label = 'Gain', variable = gain, from_ = 0, to = 2**15-1)
L1 = Tk.Label(root, text = 'Choose an effect')
N1 = Tk.Button(root, text = 'Robot Voice', command = funback1)
N2 = Tk.Button(root, text = 'absolute value Voice', command = funback2)
N3 = Tk.Button(root, text = 'Echo Voice', command = funback3)
N5 = Tk.Button(root, text = 'weird geek', command = funback5)

N6 = Tk.Button(root, text = 'Helium Voice', command = funback6)
# using higher frequency
N7 = Tk.Button(root, text = 'Lazy Voice', command = funback7)
# using time delay


B_noeffect = Tk.Button(root, text = 'No Effect', command = fun_noeffect)

BQ = Tk.Button(root, text = 'Quit', command = fun_quit)

# Number of blocks to run for
num_blocks = int(RATE / BLOCKLEN * RECORD_SECONDS)

print('* Recording for %.3f seconds' % RECORD_SECONDS)


# Place widgets
L1.pack()
N1.pack()
N2.pack()
N3.pack()
N5.pack()
N6.pack()
N7.pack()
B_noeffect.pack()
S_gain.pack(side = Tk.LEFT)
BQ.pack()

# Start loop
# for i in range(0, num_blocks):
#MULTIPLY THE OUTPUT BY THE GAIN
while CONTINUE:
	root.update()

	# Get frames from audio input stream
	#input_bytes = stream.read(BLOCKLEN)       # BLOCKLEN = number of frames read
	input_bytes = stream.read(BLOCKLEN, exception_on_overflow = False)   # BLOCKLEN = number of frames read

	# Convert binary data to tuple of numbers
	input_tuple = struct.unpack('h' * BLOCKLEN, input_bytes)
	if effect_num == 1:
		output_block=funN1(input_tuple)
	#if effect_num == 5:
	#	output_block=funN5(input_tuple)
	if effect_num == 2:
		output_block=funN2(input_tuple)
	if effect_num == 6:
		output_block=funN6(input_tuple)
	if effect_num == 3:
		output_block=funN3(input_tuple)
	if effect_num == 8:
		output_block=funN8(input_tuple)
	if effect_num == 7:
		output_block = funN7(input_tuple)
	elif effect_num == 0:
		output_block = input_tuple
	if effect_num == 7:
		output_bytes = struct.pack('h' * BLOCKLEN*2, *output_block)
	else:
		output_bytes = struct.pack('h' * BLOCKLEN, *output_block)

	stream.write(output_bytes)

print('* Finished')

stream.stop_stream()
stream.close()
p.terminate()

