﻿
- Hello everyone, welcome to CS231.
I'm Song Han. Today I'm
going to give a guest lecture
on the efficient methods and
hardware for deep learning.
So I'm a fifth year PhD
candidate here at Stanford,
advised by Professor Bill Dally.
So, in this course we have seen
a lot of convolution neural
networks, recurrent
neural networks, or even
since last time, the
reinforcement learning.
They are spanning a lot of applications.
For example, the self-=driving
car, machine translation,
AlphaGo and Smart Robots.
And it's changing our
lives, but there is a recent
trend that in order to
achieve such high accuracy,
the models are getting larger and larger.
For example for ImageNet
recognition, the winner from
2012 to 2015, the model
size increased by 16X.
And just in one year,
for Baidu's deep speech
just in one year, the training
operations, the number
of training operations increased by 10X.
So such large model
creates lots of problems,
for example the model size
becomes larger and larger
so it's difficult for
them to be deployed either
on those for example,
on the mobile phones.
If the item is larger
than 100 megabytes, you
cannot download until
you connect to Wi-Fi.
So those product managers
and for example Baidu,
Facebook, they are very sensitive
to the size of the binary
size of their model.
And also for example, the
self-driving car, you can only
do those on over-the-air
update for the model
if the model is too large,
it's also difficult.
And the second challenge
for those large models is
that the training speed is extremely slow.
For example, the ResNet152,
which is only a few, less
than 1% actually, more
accurate than ResNet101.
Takes 1.5 weeks to train on four Maxwell
M40 GPUs for example.
Which greatly limits either
we are doing homework
or if the researcher's
designing new models is
getting pretty slow.
And the third challenge
for those bulky model is
the energy efficiency.
For example, the AlphaGo
beating Lee Sedol last year,
took 2000 CPUs and 300
GPUs, which cost $3,000
just to pay for the electric
bill, which is insane.
So either on those embedded
devices, those models
are draining your battery
power for on data-center
increases the total cost
of ownership of maintaining
a large data-center.
For example, Google in
their blog, they mentioned
if all the users using the
Google Voice Search for
just three minutes, they have
to double their data-center.
So that's a large cost.
So reducing such cost is very important.
And let's see where is
actually the energy consumed.
The large model means
lots of memory access.
You have to access, load
those models from the memory
means more energy.
If you look at how much
energy is consumed by loading
the memory versus how much is
consumed by multiplications
and add those arithmetic
operations, the memory access
is more than two or three
orders of magnitude,
more energy consuming than
those arithmetic operations.
So how to make deep
learning more efficient.
So we have to improve
energy efficiency by this
Algorithm and Hardware Co-Design.
So this is the previous
way, which is our hardware.
For example, we have some
benchmarks say Spec 2006
and then run those
benchmarks and tune your CPU
architectures for those benchmarks.
Now what we should do is
to open up the box to see
what can we do from algorithm
side first and see what
is the optimum question
mark processing unit.
That breaks the
boundary between the algorithm
hardware to improve
the overall efficiency.
So today's talk, I'm going
to have the following agenda.
We are going to cover four
aspects: The algorithm hardware
and inference and training.
So they form a small two by
two matrix, so includes the
algorithm for efficient inference,
hardware for efficient inference
and the algorithm for efficient training,
and lastly, the hardware
for efficient training.
For example, I'm going
to cover the TPU, I'm
going to cover the Volta.
But before I cover those
things, let's have three
slides for Hardware 101.
A brief introduction of
the families of hardware
in such a tree.
So in general, we can
have roughly two branches.
One is general purpose hardware.
It can do any applications
versus the specialized
hardware, which is tuned
for a specific kind of
applications, a domain of applications.
So the general purpose
hardware includes, the CPU
or the GPU, and their
difference is that CPU is
latency oriented, single threaded.
It's like a big elephant.
While the GPU is throughput oriented.
It has many small though
weak threads, but there
are thousands of such small weak cores.
Like a group of small ants,
where there are so many ants.
And specialized hardware,
roughly there are FPGAs and ASICs.
So FPGA stand for Field
Programmable Gate Array.
So it is programmable, hardware
programmable so its
logic can be changed.
So it's cheaper for you to try
new ideas and do prototype,
but it's less efficient.
It's in the middle between
the general purpose and
pure ASIC.
So ASIC stands for Application
Specific Integrated Circuit.
It has a fixed logic, just designed
for a certain application.
For example deep learning.
And Google's TPU is a kind of
ASIC and the neural networks
we train on, the earlier GPUs is here.
And another slide for
Hardware 101 is the number
representations.
So in this slide, I'm going
to convey you the idea that
all the numbers in computer
are not represented
by a real number.
It's not a real number, but
they are actually discrete.
Even for those floating
point with your 32 Bit.
Floating point numbers, their
resolution is not perfect.
It's not continuous, but it's discrete.
So for example FP32, meaning
using a 32 bit to represent
a floating point number.
So there are three components
in the representation.
The sign bit, the
exponent bit, the mantissa,
and the number it represents
is shown by minus 1 to the S
times 1.M times 2 to the exponent.
So similar there is FP16,
using a 16 bit to represent
a floating point number.
In particular, I'm going
to introduce Int8, where
the core TPU use, using an
integer to represent a fixed
point number.
So we have a certain number
of bits for the integer.
Followed by a radix point,
if we put different layers.
And lastly, the fractional bits.
So why do we prefer those
eight bit, or 16 bit
rather than those traditional like the
32 bit floating point.
That's the cost.
So, I generated the figure
from 45 nanometer technology
about the energy cost versus
the area cost for different
operations.
In particular, let's see
here, go you from 32 bit to
16 bit, we have about four
times reduction in energy
and also about four times
reduction in the area.
Area means money.
Every millimeter square takes
money to take out a chip
So it's very beneficial for
hardware design to go from
32 bit to 16 bit.
That's why you hear NVIDIA
from Pascal Architecture,
they said they're
starting to support FP16.
That's the reason why it's so beneficial.
For example, previous battery
level could last four hours,
now it becomes 16 hours.
That's what it means to reduce
the energy cost by four times.
But here still, there's a
problem of large energy costs
for reading the memory.
And let's see how can we deal
with this memory reference
so expensive, how do we deal
with this problem better?
So let's switch gear and
come to our topic directly.
So let's first introduce
algorithm for efficient inference.
So I'm going to cover six topics,
this is a really long slide.
So I'm going to relatively fast.
So the first idea I'm going
to talk about is pruning.
Pruning the neural networks.
For example, this is
original neural network.
So what I'm trying to do is,
can we remove some of the
weight and still have the same accuracy?
It's like pruning a tree, get rid
of those redundant connections.
This is first proposed by
Professor Yann LeCun back in 1989,
and I revisited this problem,
26 years later, on those
modern deep neural nets
to see how it works.
So not all parameters are useful actually.
For example, in this case, if
you want to fit a single line,
but you're using a quadratic
term, apparently the
0.01 is a redundant parameter.
So I'm going to train the
connectivity first and then
prune some of the connections.
And then train the remaining weights,
and through this process, it regulates.
And as a result, I can reduce
the number of connections,
and annex that from 16
million parameters to only
six million parameters,
which is 10 times less
the computation.
So this is the accuracy.
So the x-axis is how much
parameters to prune away
and the y-axis is the accuracy you have.
So we want to have less
parameters, but we also
want to have the same accuracy as before.
We don't want to sacrifice accuracy,
For example at 80%, we
locked zero away left 80%
of the parameters, but
accuracy jumped by 4%.
That's intolerable.
But the good thing is that
if we retrain the remaining
weights, the accuracy
can fully recover here.
And if we do this process iteratively
by pruning and retraining,
pruning and retraining,
we can fully recover the
accuracy not until we are
prune away 90% of the parameters.
So if you go back to home
and try it on your Ipad
or notebook, just zero away
50% of the parameters say
you went on your homework,
you will astonishingly find
that accuracy actually doesn't hurt.
So we just mentioned
convolution neural nets,
how about RNNs and LSTMs, so I
tried with this neural talk.
Again, pruning away 90% of
the rates doesn't hurt the
blue score.
And here are some visualizations.
For example, the original
picture, the neural talk says
a basketball player in a
white uniform is playing
with a ball.
Versus pruning away 90% it
says, a basketball player
in a white uniform is
playing with a basketball.
And on and so on.
But if you're too aggressive,
say you prune away
95% of the weights, the
network is going to get drunk.
It says, a man in a red shirt
and white and black shirt
is running through a field.
So there's really a limit,
a threshold, you have to
take care of during the pruning.
So interestingly, after
I did the work, did some
resource and research and
find actually the same
pruning procedure actually
happens to human brain
as well.
So when we were born, there
are about 50 trillion synapses
in the brain.
And at one year old, this number
surged into 1,000 trillion.
And as we become adolescent,
it becomes smaller actually,
500 trillion in the end,
according to the study by Nature.
So this is very interesting.
And also, the pruning changed
the weight distribution
because we are removing
those small connections
and after we retrain them,
that's why it becomes soft
in the end.
Yeah, question.
- [Student] Are you trying
to mean that it terms
of your mixed weights
during the training will be
just set at zero and
just start from scratch?
And these start from the
things that are at zero.
- Yeah. So the question is,
how do we deal with those
zero connections?
So we force them to be zero
in all the other iterations.
Question?
- [Student] How do you
pick which rates to drop?
- Yeah so very simple. Small
weights, drop it, sort it.
If it's small, just--
- [Student] Any threshold that I decide?
- Exactly, yeah.
So the next idea, weight sharing.
So now we have, remember
our end goal is to remove
connections so that we can
have less memory footprint
so that we can have more
energy efficient deployment.
Now we have less number
of parameters by pruning.
We want to have less number
of bits per parameter
so they're multiplied together
they get a small model.
So the idea is like this.
Not all numbers, not all the weights
has to be the exact number.
For example, 2.09, 2.12 or
all these four weights, you
just put them using 2.0 to represent them.
That's enough.
Otherwise too accurate number
is just leads to overfitting.
So the idea is I can
cluster the weights if they
are similar, just using
a centroid to represent
the number instead of using
the full precision weight.
So that every time I do the
inference, I just do inference
on this single number.
For example, this is a
four by four weight matrix
in a certain layer.
And what I'm going to do is do
k-means clustering by having
the similar weight
sharing the same centroid.
For example, 2.09, 2.12, I store index of
three pointing to here.
So that, the good thing is
we need to only store the
two bit index rather than the
32 bit, floating point number.
That's 16 times saving.
And how do we train such neural network?
They are binded together, so
after we get the gradient,
we color them in the same
pattern as the weight
and then we do a group by
operation by having all
the in that weights with the
same index grouped together.
And then we do a reduction
by summing them up.
And then multiplied by the learning rate
subtracted from the original centroid.
That's one iteration of
the SGD for such weight
shared neural network.
So remember previously,
after pruning this is
what the weight
distribution like and after
weight sharing, they become discrete.
There are only 16 different
values here, meaning
we can use four bits to
represent each number.
And by training on such
weight shared neural network,
training on such extremely
shared neural network,
these weights can adjust.
It is the subtle changes
that compensated for the
loss of accuracy.
So let's see, this is the
number of bits we give it,
this is the accuracy
for convolution layers.
Not until four bits, does
the accuracy begin to drop
and for those fully connected
layers, very astonishingly,
it's not until two bits,
only four number, does the
accuracy begins to drop.
And this result is per layer.
So we have covered two methods,
pruning and weight sharing.
What if we combine these
two methods together.
Do they work well?
So by combining those methods,
this is the compression
ratio with the smaller on the left.
And this is the accuracy.
We can combine it together
and make the model
about 3% of its original
size without hurting the
accuracy at all.
Compared with the each
working individual data by
10%, accuracy begins to drop.
And compared with the
cheap SVD method,
this has a better compression ratio.
And final idea is we can
apply the Huffman Coding
to use more number of bits
for those infrequent numbers,
infrequently appearing weights
and less number of bits
for those more frequently
appearing weights.
So by combining these three
methods, pruning, weight
sharing, and also Huffman
Coding, we can compress the
neural networks, state-of-the-art 
neural networks,
ranging from 10x to
49x without hurting the
prediction accuracy.
Sometimes a little bit better.
But maybe that is noise.
So the next question is, these
models are just pre-trained
models by say Google, Microsoft.
Can we make a compact
model, a pump compact model
to begin with?
Even before such compression?
So SqueezeNet, you may have
already worked with this
neural network model in a homework.
So the idea is we are having
a squeeze layer here to shield
at the three by three
convolution with fewer number of
channels.
So that's where squeeze comes from.
And here we have two branches,
rather than four branches
as in the inception model.
So as a result, the model
is extremely compact.
It doesn't have any 
fully connected layers.
Everything is fully convolutional.
The last layer is a global pooling.
So what if we apply deep
compression algorithm
on such already compact
model will it be getting even
smaller?
So this is AlexNet after
compression, this is SqueezeNet.
Even before compression, it's
50x smaller than AlexNet,
but has the same accuracy.
After compression 510x
smaller, but the same accuracy
only less than half a megabyte.
This means it's very easy
to fit such a small model
on the cache, which is literally
tens of megabyte SRAM.
So what does it mean?
It's possible to achieve speed up.
So this is the speedup, I
measured if all these fully
connected layers only for
now, on the CPU, GPU, and
the mobile GPU, before pruning
and after pruning the weights,
and on average, I observed
a 3x speedup in a CPU,
about 3X speedup on the GPU,
and roughly 5x speedup on
the mobile GPU, which is a
TK1.
And so is the energy efficiency.
In an average improvement
from 3x to 6x on a CPU, GPU,
and mobile GPU.
And these ideas are
used in these companies.
Having talked about when
pruning and when sharing,
which is a non-linear quantization method
and we're going to talk about
quantization, which is, why
do they use in the TPU design?
All the TPU designs use at
only eight bit for inference.
And the way, how they can
use that is because of the
quantization.
And let's see how does it work.
So quantization has this
complicated figure, but
the intuition is very simple.
You run the neural network
and train it with the normal
floating point numbers.
And quantize the weight
and activations by gather
the statistics for each layer.
For example, what is the maximum number,
minimum number,
and how many bits are enough
to represent this dynamic range.
Then you use that number of
bits for the integer part
and the rest of the eight bit or seven bit
for the other part of
the 8 bit representation.
And also we can fine tune in
the floating point format.
Or we can also use feed
forward with fixed point
and back propagation with
update with the floating
point number.
There are lots of different
ideas to have better accuracy.
And this is the result,
for how many number of bits
versus what is the accuracy.
For example, using a fixed,
8 bit, the accuracy for
GoogleNet doesn't drop significantly.
And for VGG-16, it also
remains pretty well for
the accuracy.
While circling down to
a six bit, the accuracy
begins to drop pretty dramatically.
Next idea, low rank approximation.
It turned out that for
a convolution layer,
you can break it into
two convolution layers.
One convolution here, followed
by a one by one convolution.
So that it's like you
break a complicated problem
into two separate small problems.
This is for convolution layer.
As we can see, achieving about
2x speedup, there's almost
no loss of accuracy.
And achieving a speedup
of 5x, roughly a 6%
loss of accuracy.
And this also works for
fully connected layers.
The simplest idea is using
the SVD to break it into
one matrix into two matrices.
And follow this idea, this
paper proposes to use the
Tensor Tree to break down one
fully connected layer into
a tree, lots of fully connected layers.
That's why it's called a tree.
So going even more crazy, can we use only
two weights or three weights
to represent a neural network?
A ternary weight or a binary weight.
We already seen this distribution
before, after pruning.
There's some positive
weights and negative weights.
Can we just use three numbers,
just use one, minus one, zero
to represent the neural network.
This is our recent paper
clear that we maintain
a full precision weight
during training time,
but at inference time, we
only keep the scaling factor
and the ternary weight.
So during inference, we
only need three weights.
That's very efficient and
making the model very small.
This is the proportion
of the positive zero
and negative weights, they can
change during the training.
So is their absolute value.
And this is the visualization of kernels
by this trained ternary quantization.
We can see some of them are
a corner detector like here.
And also here.
Some of them are maybe edge detector.
For example, this filter some of them
are corner detector like here this filter.
Actually we don't need
such fine grain resolution.
Just three weights are enough.
So this is the validation
accuracy on ImageNet with AlexNet.
So the threshline is the baseline accuracy
with floating point 32.
And the red line is our result.
Pretty much the same accuracy
converged compared with
the full precision weights.
Last idea, Winograd Transformation.
So this about how do we
implement deep neural nets,
how do we implement the convolutions.
So this is the conventional direct
convolution implementation method.
The slide credited to
Julien, a friend from Nvidia.
So originally, we just do the element wise
do a dot product for those
nine elements in the filter
and nine elements in the
image and then sum it up.
For example, for every
output we need nine times C
number of multiplication and adds.
Winograd Convolution is another
method, equivalent method.
It's not lost, it's an
equivalent method proposed at
first through this paper, Fast Algorithms
for Convolution Neural Networks.
That instead of directly
doing the convolution, move
it one by one, at first it
transforms the input feature
map to another feature map.
Which contains only the
weight, contains only 1, 0.5, 2
that can efficiently
implement it with shift.
And also transform the filter
into a four by four tensor.
So what we are going to do here
is sum over c and do an element-wise
element-wise product.
So there are only 16
multiplications happening here.
And then we do a inverse
transform to get four outputs.
So the transform and the
inverse transform can be
amortized and the multiplications,
whether it can ignored.
So in order to get four output,
we need nine times channel
times four, which is 36 times channel.
Multiplications originally
for the direct convolution
but now we need 16
times C of our output
So that is 2.25x less
number of multiplications to
perform the exact same multiplication.
And here is a speedup.
2.25x, so theoretically,
2.25x speedup and in real,
from cuDNN 5 they incorporated such
Winograd Convolution algorithm.
This is on the VGG net I
believe, the speedup is
roughly 1.7 to 2x speedup.
Pretty significant.
And after cuDNN 5, the
cuDNN begins to use the
Winograd Convolution algorithm.
Okay, so far we have covered
those efficient algorithms
for efficient inference.
We covered pruning, weight
sharing, quantization,
and also Winograd binary and ternary.
So now let's see what is the
optimal hardware for those
efficient inference?
And what is a Google TPU?
So there are a wide
range of domain specific
architectures or ASICS
for deep neural networks.
They have a common goal
is to minimize the memory
access to save power.
For example the Eyeriss from
MIT by using the RS Dataflow
to minimize the off chip direct access.
And DaDiannao from China
Academy of Science,
buffered all the weights on
chip DRAM instead of having
to go to off-chip DRAM.
So the TPU from Google is
using eight bit integer
to represent the numbers.
And at Stanford I proposed
the EIE architecture
that support those compressed and
sparse deep neural network inference.
So this is what the TPU looks like.
It's actually smartly, can
be put into the disk drive
up to four cards per server.
And this is the high-level architecture
for the Google TPU.
Don't be overwhelmed, it's
actually, the kernel part
here, is this giant matrix
multiplication unit.
So it's a 256 by 256
matrix multiplication unit.
So in one single cycle,
it can perform 64 kilo
those number of multiplication
and accumulate operations.
So running 700 Megahertz,
the throughput is 92
Teraops per second
because it's actually integer operation.
So we just about 25x as GPU
and more than 100x at the CPU.
And notice, TPU has a really
large software-managed
on-chip buffer.
It is 24 megabytes.
The cache for the CPU the
L3 cache is already
16 megabytes.
This is 24 megabytes
which is pretty large.
And it's powered by
two DDR3 DRAM channels.
So this is a little weak
because the bandwidth is
only 30 gigabytes per second
compared with the most
recent GPU that HBM, 900
Gigabytes per second.
The DDR4 is released in 2014,
so that makes sense because
the design is a little during
that day, used the DDR3.
But if you're using DDR4 or
even high-bandwidth memory,
the performance can be even boosted.
So this is a comparison
about Google's TPU compared
with the CPU, GPU of this K80
GPU by the way, and the TPU.
So the area is pretty much
smaller, like half the size of a
CPU and GPU and the power
consumption is roughly 75 watts.
And see this number, the
peak teraops per second
is much higher than the
CPU and GPU is, about 90
teraops per second, which is pretty high.
So here is a workload.
Thanks to David sharing the slide.
This is the workload at Google.
They did a benchmark on these TPUs.
So it's a little interesting
that convolution neural nets
only account for 5% of
data-center workload.
Most of them is multilayer perception,
those fully connected layers.
About 61% maybe for ads, I'm not sure.
And about 29% of the workload
in data-center is the
Long Short Term Memory.
For example, speech recognition,
or machine translation, I suspect.
Remember just now we have seen there are
90 teraops per second.
But what actually number
of teraops per second
can be achieved?
This is a basic tool to
measure the bottleneck
of a computer system.
Whether you are bottlenecked
by the arithmetic or
you are bottlenecked by
the memory bandwidth.
It's like if you have a bucket,
the lowest part of the
bucket determines how much
water we can hold in the bucket.
So in this region, you are bottlenecked
by the memory bandwidth.
So the x-axis is the arithmetic intensity.
Which is number of floating
point operations per byte
the ratio between the
computation and memory
of bandwidth overhead.
So the y-axis, is the actual
attainable performance.
Here is the peak performance for example.
When you do a lot of operation
after you fetch a single
piece of data, if you
can do a lot of operation
on top of it, then you are
bottlenecked by the arithmetic.
But after you fetch a lot
of data from the memory,
but you just do a tiny
little bit of arithmetic,
then you will be bottlenecked
by the memory bandwidth.
So how much you can fetch
from the memory determines
how much real performance you can get.
And remember there is a ratio.
When it is one here, this
region it happens to be the same
as the turning point is the actual
memory bandwidth of your system.
So let's see what is the life for the TPU.
The TPU's peak performance is really high,
about 90 Tops per second.
For those convolution nets,
they are pretty much saturating
the peak performance.
But there are lot of neural
networks that has a utlitization
less than 10%,
meaning that 90 T-ops
per second is actually
achieves about three to 12
T-ops per second in real case.
But why is it like that?
The reason is, in order to
have those real-time guarantee
that the user not wait for
too long, you cannot batch
a lot of user's images
or speech voice data
at the same time.
So as a result, for those
fully connect layers,
they have very little reuse,
so they are bottlenecked
by the memory bandwidth.
For those convolution neural
nets, for example this one,
this blue one, that
achieve 86, which is CNN0.
The ratio between the ops and
the number of memory is the highest.
It's pretty high, more than
2,000 compared with other
multilayer perceptron or
long short term memory
the ratio is pretty low.
So this figure compares, this
is the TPU and this one is
the CPU, this is the GPU.
Here is memory bandwidth,
the peak memory bandwidth
at a ratio of one here.
So TPU has the highest memory bandwidth.
And here is where are
these neural networks
lie on this curve.
So the asterisk is for the TPU.
It's still higher than other dots,
but if you're not comfortable
with this log scale figure,
this is what it's like
putting it in linear roofline.
So pretty much everything
disappeared except
for the TPU results.
So still, all these lines,
although they are higher
than the CPU and GPU,
it's still way below the
theoretical peak operations per second.
So as I mentioned before,
it is really bottlenecked
by the low latency requirement
so that it can have
a large batch size.
That's why you have low
operations per byte.
And how do you solve this problem?
You want to have less
number of memory footprint
so that it can reduce the
memory bandwidth requirement.
One solution is to compress
the model and the challenge
is how do we build a hardware
that can do inference
directly on the compressed model?
So I'm going to introduce my
design of EIE, the Efficient
Inference Engine, which
deals with those sparse
and the compressed model to
save the memory bandwidth.
And the rule of thumb, like
we mentioned before is taking
out one bit of sparsity first.
Anything times zero is zero.
So don't store it, don't compute on it.
And second idea is, you don't
need that much full precision,
but you can approximate it.
So by taking advantage
of the sparse weight, we
get about a 10x saving in
the computation, 5x less
memory footprint.
The 2x difference is
due to index overhead.
And by taking advantage
of the sparse activation,
meaning after bandwidth,
if activation is zero, then
ignore it.
You save another 3x of computation.
And then by such weight sharing mechanism,
you can use four bits to
represent each weight rather
than 32 bit.
That's another eight times
saving in the memory footprint.
So this is physically, logically
how the weights are stored.
A four by eight matrix,
and this is how physically
they are stored.
Only the non-zero weights are stored.
So you don't need to store those zeroes.
You'll save the bandwidth
fetching those zeroes.
And also I'm using the
relative index to further save
the number of memory overhead.
So in the computation
like this figure shows,
we are running the
multiplication only on non-zero.
If it's zero, then skip it.
Only broadcast it to the non-zero weights
and if it is zero, skip it.
If it's a non-zero, do the multiplication.
In another cycle, do the multiplication.
So the idea is anything
multiplied by zero is zero.
So this is a little complicated,
I'm going to go very quickly.
I'm going to have a lookup
table that decode the four bit
weight into the 16 bit
weight and using the four bit
relative index passed
through address accumulator
to get the 16 bit absolute index.
And this is what the hardware architecture
like in the high level.
You can feel free to refer
to my paper for detail.
Okay speedup.
So using such efficient
hardware architecture
and also model compression,
this is the original
result we have seen for
CPU, GPU, mobile GPU.
Now EIE is here.
189 times faster than the
CPU and about 13 times faster
than the GPU.
So this is the energy
efficiency on the log scale,
it's about 24,000x more
energy efficient than a CPU
and about 3000x more energy
efficient than a GPU.
It means for example,
previously if your battery can
last for one hour, now it can last for
3000 hours for example.
So if you say, ASIC is always
better than CPUs and GPUs
because it's customized hardware.
So this is comparing EIE with
the peer ASIC, for example
DaDianNao and the TrueNorth.
It has a better throughput,
better energy efficiency
by order of magnitude,
compared with other ASICs.
Not to mention that CPU, GPU and FPGAs.
So we have covered half of the journey.
We mentioned inference, we pretty much
covered everything for inference.
Now we are going to switch
gear and talk about training.
How do we train neural
networks efficiently,
how do we train it faster?
So again, we are starting
with algorithm first,
efficient algorithms
followed by the hardware
for efficient training.
So for efficient training
algorithms, I'm going to mention
four topics.
The first one is parallelization,
and then mixed precision
training, which was just
released about one month ago
and at NVIDIA GTC,
so it's fresh knowledge.
And then model distillation,
followed by my work on
Dense-Sparse-Dense training,
or better Regularization
technique.
So let's start with parallelization.
So this figure shows, anyone in the hardware community.
Most are very familiar with this figure.
So as time goes by, what is the trend?
For the number of transistors
is keeping increasing.
But the single threaded
performance is getting plateaued
in recent years.
And also the frequency is getting
plateaued in recent years.
Because of the power
constraint, to stop not scaling.
And interesting thing is the
number of cores is increasing.
So what we really need
to do is parallelization.
How do we parallelize the
problem to take advantage
of parallel processing?
Actually there are a lot of
opportunities for parallelism
in deep neural networks.
For example, we can do data parallel.
For example, feeding two
images into the same model
and run them at the same time.
This doesn't affect
latency for a single input.
It doesn't make it shorter,
but it makes batch size larger
basically if you have four
machines our effective batch
size becomes four times as before.
So it requires the
coordinated weight update.
For example, this is a paper from Google.
There is a parameter server
as a master and a couple of
slaves running their own piece
of training data and update
the gradient to the parameter
server and get the updated
weight for them individually,
that's how data parallelism is handled.
Another idea is there could
be a model parallelism.
You can sublet your model and handle it
to different processors
or different threads.
For example, there's this image,
you want to run convolution
on this image that is
six dimension for loop.
What you can do is you
can cut the input image by
two by two blocks so that
each thread, or each processor
handles one fourth of the image.
Although there's a small
halo here in between you
have to take care of.
And also, you can parallelize by the
output or input feature map.
And for those fully connect layers,
how do we parallelize the model?
It's even simpler.
You can cut the model into half
and hand it to different threads.
And the third idea, you can even do
hyper-parameter parallel.
For example, you can tune
your learning rate, your
weight decay for different machines
for those coarse-grained parallelism.
So there are so many
alternatives you have to tune.
Small summary of the parallelism.
There are lots of parallelisms
in deep neural networks.
For example, with data
parallelism, you can run multiple
training images, but you
cannot have unlimited number
of processors because you
are limited by batch size.
If it's too large, stochastic gradient descent
becomes gradient descent, that's not good.
You can also run the model parallelism.
Split the model, either
by cutting the image or
cutting the convolution weights.
Either cutting the image or cutting
the fully connected layers.
So it's very easy to get 16
to 64 GPUs training one model
in parallel, having very good speedup.
Almost linear speedup.
Okay, next interesting
thing, mixed precision with
FP16 or FP32.
So remember in the
beginning of this lecture,
I had a chart showing the
energy and area overhead for
a 16 bit versus a 32 bit.
Going from 32 bit to 16 bit,
you save about 4x the energy
and 4x the area.
So can we train a deep
neural network with such low
precision with floating point
16 bit rather than 32 bit?
It turns out we can do that partially.
By partially, I mean we
need FP32 in some places.
And where are those places?
So we can do the multiplication
in 16 bit as input.
And then we have to do the summation
in 32 bit accumulation.
And then convert the result
to 32 bit to store the weight.
So that's where the mixed
precision comes from.
So for example, we have
a master weight stored in
floating point 32, we down
converted it to floating
point 16 and then we do the
feed forward with 16 bit
weight, 16 bit activation,
we get a 16 bit activation
here in the end when we
are doing back propagation
of the computation is also done
with floating point 16 bit.
Very interesting here, for
the weights we get a floating
point 16 bit gradient here for the weight.
But when we are doing the
update, so W plus learning
rate times the gradient,
that operation has
to be done in 32 bit.
That's where the mixed
precision is coming from.
And see there are two
colors, which here is 16 bit,
here is the 32 bit.
That's where the mixed
precision comes from.
So does such low precision
sacrifice your prediction
accuracy for your model?
So this is the figure from
NVIDIA just released a couple
of weeks ago actually.
Thanks to Paulius giving me the slide.
The convergence between
floating point 32 versus
the multi tensor up, which
is basically the mixed
precision training, are
actually pretty much
the same for convergence.
If you zoom it in a little bit,
they are pretty much the same.
And for ResNet, the mixed
precision sometimes behaves
a little better than the
full precision weight.
Maybe because of noise.
But in the end, after you
train the model, this is
the result of AlexNet,
Inception V3, and ResNet-50
with FP32 versus FP16
mixed precision training.
The accuracy is pretty much the same
for these two methods.
A little bit worse, but not by too much.
So having talked about the
mixed precision training,
the next idea is to train
with model distillation.
For example, you can have
multiple neural networks,
Googlenet, Vggnet, Resnet for example.
And the question is, can
we take advantage of these
different models?
Of course we can do model
ensemble, can we utilitze them
as teacher, to teach a small
junior neural network to have
it perform as good as the
senior neural network.
So this is the idea.
You have multiple large
powerful senior neural networks
to teach this student model.
And hopefully it can get better results.
And the idea to do that
is, instead of using this
hard label, for example for
car, dog, cat, the probability
for dog is 100%, but the
output of the geometric
ensemble of those large
teacher neural networks
maybe the dog has 90%
and the cat is about 10%,
and the magic happens here.
You want to have a
softened result label here.
For example, the dog
is 30%, the cat is 20%.
Still the dog is higher than the cat.
So the prediction is
still correct, but it uses
this soft label to train
the student neural network
rather than use this hard label to train
the student neural network.
And mathematically, you
control how much do you make
it soft by this temperature
during the soft max
controlling by this temperature.
And the result is that,
starting with the trained model
that classifies 58.9% of
the test frames correctly,
the new model converges to 57%.
Only train on 3% of the data.
So that's the magic for model distillation
using this soft label.
And the last idea is my recent paper using
a better regularization
to train deep neural nets.
We have seen these two figures before.
We pruned the neural
network, having less number
of weights, but have the same accuracy.
Now what I did is to
recover and to retrain those
weights shown in red
and make everything train
out together to increase
the model capacity after
it is trained at a low dimensional space.
It's like you learn the trunk
first and then gradually
add those leaves and
learn everything together.
It turns out, on ImageNet it
performs relatively about 1% to
4% absolute improvement of accuracy.
And is also general purpose,
works on long-short term memory
and also recurrent neural
nets collaborated with Baidu.
So I also open sourced
this special training model
on the DSD Model Zoo, where
there are trained, all
these models, GoogleNet, VGG,
ResNet, and also SqueezeNet,
and also AlexNet.
So if you are interested,
feel free to check out this
Model Zoo and compare it
with the Caffe Model Zoo.
Here's some examples on
dense-spare-dense training helps
with image capture.
For example, this is a
very challenging figure.
The original baseline of
neural talk says a boy in
a red shirt is climbing a rock wall.
And the sparse model says
a young girl is jumping
off a tree, probably
mistaking the hair with either
the rock or the tree.
But then sparse-dense
training by using this kind of
regularization on a low
dimensional space, it says
a young girl in a pink shirt
is swinging on a swing.
And there are a lot of examples
due to the limit of time,
I will not go over them one by one.
For example, a group of
people are standing in front
of a building, there's no building.
A group of people are walking in the park.
Feel free to check out the
paper and see more interesting
results.
Okay finally, we come to
hardware for efficient training.
How to we take advantage of the algorithms
we just mentioned.
For example, parallelism,
mixed precision, how are
the hardware designed to actually
take advantage of such features.
First GPUs, this is the
Nvidia PASCAL GPU, GP100,
which was released last year.
So it supports up to 20 Teraflops on FP16.
It has 16 gigabytes of
high bandwidth memory.
750 gigabytes per second.
So remember, computation
and memory bandwidth are
the two factors determines
your overall performance.
Whichever is lower, it will suffer.
So this is a really high
bandwidth, 700 gigabytes
compared with DDR3 is just 10
or 30 gigabytes per second.
Consumes 300 Watts and
it's done in 16 nanometer process
and have a 160 gigabytes
per second NV Link.
So remember we have
computation, we have memory,
and the third thing is the communication.
All three factors has to
be balanced in order to
achieve a good performance.
So this is very powerful,
but even more exciting,
just about a month ago,
Jensen released the newest
architecture called the Volta GPUs.
And let's see what is
inside the Volta GPU.
Just released less than a
month ago, so it has 15 of
FP32 teraflops and what
is new here, there is 120
Tensor T-OPS, so specifically
designed for deep learning.
And we'll later cover
what is the tensor core.
And what is this 120 coming from.
And rather than 750
gigabytes per second, this
year, the HBM2, they are
using 900 gigabytes per second
memory bandwidth.
Very exciting.
And 12 nanometer process has
a die size of more than 800
millimeters square.
A really large chip and
supported by 300 gigabytes per
second NVLink.
So what's new in Volta, the
most interesting thing for us
for deep learning, is this
thing called Tensor Core.
So what is a Tensor Core?
Tensor Core is actually
an instruction that can
do the four by four matrix
times a four by four matrix.
The fused FMA stands Fused
Multiplication and Add
in this mixed precision operation.
Just in one single clock cycle.
So let's discern for a little
bit what does this mean.
So mixed precision is exactly
as we mentioned in the last
chapter, so we are having
FP16 for the multiplication,
but for accumulation, we
are doing it with FP32.
That's where the mixed
precision comes from.
So let's say how many
operations, if it's four
by four by four, it's 64
multiplications then just
in one single cycle.
That's 12x increase in
the speedup of the Volta
compared with the Pascal, which
is released just less year.
So this is the result for
matrix multiplication on
different sizes.
The speedup of Volta over
Pascal is roughly 3x faster
doing these matrix multiplications.
What we care more is not
only matrix multiplication
but actually running the deep neural nets.
So both for training and for inference.
And for training on
ResNet-50, by taking advantage
of this Tensor Core in this V100,
it is 2.4x faster than
the P100 using FP32.
So on the right hand side,
it compares the inference
speedup, given a 7 microsecond
latency requirement.
What is the number of images
per second it can process?
It has a measurement of throughput.
Again, the V100 over
P100, by taking advantage
of the Tensor Core, is
3.7 faster than the P100.
So this figure gives roughly
an idea, what is a Tensor Core,
what is an integer unit, what
is a floating point unit.
So this whole figure
is a single SM
stream multiprocessor.
So SM is partitioned into
four processing blocks.
One, two, three, four, right?
And in each block there
are eight FP64 cores here
and 16 FP32 and 16 INT32
cores here, units here.
And then there are two of
the new mixed precision
Tensor cores specifically
designed for deep learning.
And also there are the one
warp scheduler, dispatch unit
and Register File, as before.
So what is new here is
the Tensor core unit here.
So here is a figure comparing
the recent generations of
Nvidia GPUs from Kepler
to Maxwell to Pascal to Volta.
We can see everything
is keeping improving.
For example, the boost clock
has been increased from
about 800 MHz to 1.4 GHz.
And from the Volta generation
there begins to have
the Tensor core units here,
which has never existed before.
And before the Maxwell,
the GPUs are using the GDDR5,
and after the Pascal GPU,
the HBM begins to came into place,
the high-bandwidth memory.
750 gigabytes per second here.
900 gigabytes per second
compared with DDR3,
30 gigabytes per second.
And memory size actually
didn't increase by too much,
and the power consumption is actually
also remaining roughly the same.
But giving the increase of
computation, you can fit them
in the fixed power envelope
that's still an exciting thing.
And the manufacturing process
is actually improving from
28 nanometer, 16 nanometer,
all the way to 12 nanometer.
And the chip area are also increasing to
800 millimeter-squared,
that's really huge.
So, you may be interested
in the comparison of the GPU
with the TPU, right?
So how do they compare with each other?
So in the original TPU paper,
TPU actually designed
roughly in the year of 2015,
and this is comparison
of the Pascal P40 GPU
released in 2016.
So, TPU, the power consumption is lower,
is larger on chip memory of 24 megabytes,
really large on-chip SRAM
managed by the software.
And then both of them
support INT8 operations,
while the inferences per second
given a 10 nanometer latency
the comparison for TPU is 1X.
For the P40 it's about 2X.
So, just last week,
in the Google I/O,
a new nuclear bomb is landed on the Earth.
That is the Google Cloud TPU.
So now TPU not only support inference,
but also support training.
So there is a very limited
information we can get
beyond this Google Blog.
So their Cloud TPU delivers
up to 180 teraflops
to train and run machine learning models.
And this is multiple Cloud TPU,
making it into a TPU pod,
which is built with 16
the second generation TPUs
and delivers up to 11.5 teraflops
of machine learning acceleration.
So in the Google Blog, they mentioned that
one of the large scale translation models,
Google translation models, used
to take a full day to train
on 32 of best commercially-available
GPUs, probably P40
or P100, maybe.
And now it trains to the same accuracy,
just within one afternoon,
with just 1/8 of a TPU pod,
which is pretty exciting.
Okay, so as a little wrap-up.
We covered a lot of stuff, we've mentioned
the four dimension space
of algorithm and hardware,
inference and training, we
covered the algorithms for
inference, for example,
pruning and quantization,
Winograd Convolution, binary, ternary,
weight sharing, for example.
And then the hardware for
the efficient inference.
For example, the TPU,
that take advantage of INT8, integer 8.
And also my design of EIE
accelerator that take advantage
of the sparsity, anything
multiplied by zero is zero,
so don't store it, don't compute on it.
And also the efficient algorithm
for training, for example,
how do we do parallelization
and the most recent research on
how do we use mixed precision
training by taking advantage
of FP16 rather than FP32 to do training
which is four times saving the energy
and four times saving in the area,
which doesn't quite sacrifice
the accuracy you'll get from
the training.
And also Dense-Sparse-Dense
training using better regularization
sparse regularization, and also
the teacher-student model.
You have multiple teacher on
your network and have a small
student network that you
can distill the knowledge
from the teacher in your
network by a temperature.
And finally we covered the
hardware for efficient training
and introduced two nuclear bombs.
One is the Volta GPU, the
other is the TPU version two,
the Cloud TPU and also
the amazing Tensor cores
in the newest generation of Nvidia GPUs.
And we also revealed the
progression of a wide range,
the recent Nvidia GPUs
from the Kepler K40,
that's actually when
I started my research,
what we used in the beginning,
all the way to and then K40, M40,
and then Pascal and then
finally the exciting Volta GPU.
So every year there is a
nuclear bomb in the spring.
Okay, a little look ahead in the future.
So in the future of the city
we can imagine there are a lot
of AI applications using
smart society, smart care,
IOT devices, smart retail,
for example, the Amazon Go,
and also smart home, a lot of scenarios.
And it poses a lot of challenges
on the hardware design
that requires the low
latency, privacy, mobility
and energy efficiency.
You don't want your battery
to drain very quickly.
So it's both challenging
and very exciting era
for the code design for
both the machine learning
deep neural network model architectures
and also the hardware architecture.
So we have moved from
PC era to mobile era.
Now we are in the AI-First era,
and hope you are as excited
as I am for this kind of
brain-inspired cognitive
computing research.
Thank you for your attention,
I'm glad to take questions.
[applause]
We have five minutes.
Of course.
- [Student] Can you commercialize
the deep architecture?
- The architecture, yeah, some
of the ideas are pretty good.
I think there's opportunity.
Yeah.
Yeah.
The question is, what can we
do to make the hardware better?
Oh, right, the question is how do we,
the challenges and what
opportunity for those small
embedded devices around
deep neural network
or in general AI algorithms.
Yeah, so those are the
algorithm I discussed
in the beginning about inference.
Here.
These are the techniques
that can enable such
inference or AI running
on embedded devices,
by having less number of
weights, fewer bits per weight,
and also quantization,
low rank approximation.
The small matrix, same
accuracy, even going to binary,
or ternary weights having just two bits
to do the computation rather
than 16 or even 32 bit
and also the Winograd Transformation.
Those are also the enabling
algorithms for those
low-power embedded devices.
Okay, the question is, if it's
binary weight, the software
developers may be not able
to take advantage of it.
There is a way to take
advantage of binary weight.
So in one register there are 32 bit.
Now you can think of it
as a 32-way parallelism.
Each bit is a single operation.
So say previously we
have 10 ops per second.
Now you get 330 ops per second.
You can do this bitwise operations.
For example, XOR operations.
So one register file,
one operation becomes 32 operation.
So there is a paper called XORmad,
they very amazing implemented
on the Raspberry Pi using this feature
to do real-time detection,
very cool stuff.
Yeah.
Yeah, so the trade-off is
always so the power area
and performance in general,
all the hardware design
have to take into account
the performance, the power,
and also the area.
When machine learning
comes, there's a fourth
figure of merit which is the accuracy.
What is the accuracy?
And there is a fifth one
which is programmability.
So how general is your hardware?
For example, if Google just
want to use that for AI
and deep learning, it's totally fine
that we can have a fully
very specialized architecture
just for deep learning
to support convolution,
multi-layered perception,
long-short-term memory,
but GPUS, you also want
to have support for those
scientific computing
or graphics, AR and VR.
So that's a difference, first of all.
And TPU basically is a ASIC, right?
It's a very fixed function
but you can still program it
with those coarse instructions
so people from Google
roughly designed those coarse
granularity instruction.
For example, one instruction
just load the matrix,
store a matrix, do convolutions,
do matrix multiplications.
Those coarse-grain instructions
and they have a software-managed memory,
also called a scratchpad.
It's different from
cache where it determines
where to evict something
from the cache, but now,
since you know the computation pattern,
there's no need to do out-of-order execution,
to do branch prediction, no such things.
Everything is determined,
so you can take the multi of
it and maintain a fully
software-managed scratchpad
to reduce the data movement
and remember, data movement
is the key for reducing
the memory footprint
and energy consumption.
So, yeah.
Mobilia and Nobana architectures
actually I'm not quite
familiar, didn't prepare those slides, so,
comment it a little bit later, no.
Oh, yeah, of course.
Those are always and
can certainly be applied
to low-power embedded devices.
If you're interested, I can show you a...
Whoops.
Some examples of, oops.
Where is that?
Of my previous projects
running deep neural nets.
For example, on a drone,
this is using a Nvidia TK1
mobile GPU to do real-time
tracking and detection.
This is me playing my nunchaku.
Filmed by a drone to do the
detection and tracking.
And also, this FPGA doing
the deep neural network.
It's pretty small.
This large, doing the face-alignment and
detecting the eyes,
the nose and the mouth,
at a pretty high framerate.
Consuming only three watts.
This is a project I did
at Facebook doing the
deep neural nets on the mobile phone to do
image classification, for
example, it says it's a laptop,
or you can feed it with
an image and it says
it's a selfie, has person
and the face, et cetera.
So there's lots of opportunity for those
embedded or mobile-deployment
of deep neural nets.
No, there is a team doing that,
but I cannot comment too much, probably.
There is a team at Google
doing that sort of stuff, yeah.
Okay, thanks, everyone.
If you have any questions,
feel free to drop me a e-mail.
