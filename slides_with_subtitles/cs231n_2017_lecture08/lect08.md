﻿
- Hello?
Okay, it's after 12, so
I want to get started.
So today, lecture eight,
we're going to talk about
deep learning software.
This is a super exciting
topic because it changes
a lot every year.
But also means it's a lot
of work to give this lecture
'cause it changes a lot every year.
But as usual, a couple
administrative notes
before we dive into the material.
So as a reminder the
project proposals for your
course projects were due on Tuesday.
So hopefully you all turned that in,
and hopefully you all
have a somewhat good idea
of what kind of projects
you want to work on
for the class.
So we're in the process of
assigning TA's to projects
based on what the project area is
and the expertise of the TA's.
So we'll have some more
information about that
in the next couple days I think.
We're also in the process
of grading assignment one,
so stay tuned and we'll get
those grades back to you
as soon as we can.
Another reminder is that
assignment two has been out
for a while.
That's going to be due next week,
a week from today, Thursday.
And again, when working on assignment two,
remember to stop your
Google Cloud instances
when you're not working to
try to preserve your credits.
And another bit of
confusion, I just wanted to
re-emphasize is that for
assignment two you really
only need to use GPU instances
for the last notebook.
For all of the several
notebooks it's just in Python
and Numpy so you don't need
any GPUs for those questions.
So again, conserve your credits,
only use GPUs when you need them.
And the final reminder is
that the midterm is coming up.
It's kind of hard to
believe we're there already,
but the midterm will be in
class on Tuesday, five nine.
So the midterm will be more theoretical.
It'll be sort of pen and paper
working through different
kinds of, slightly more
theoretical questions
to check your understanding
of the material that we've
covered so far.
And I think we'll probably
post at least a short sort of
sample of the types of
questions to expect.
Question?
[student's words obscured
due to lack of microphone]
Oh yeah, question is
whether it's open-book,
so we're going to say
closed note, closed book.
So just,
Yeah, yeah, so that's what
we've done in the past
is just closed note,
closed book, relatively
just like want to check
that you understand
the intuition behind most of
the stuff we've presented.
So, a quick recap as a reminder
of what we were talking
about last time.
Last time we talked about
fancier optimization algorithms
for deep learning models
including SGD Momentum,
Nesterov, RMSProp and Adam.
And we saw that these
relatively small tweaks
on top of vanilla SGD, are
relatively easy to implement
but can make your networks
converge a bit faster.
We also talked about regularization,
especially dropout.
So remember dropout, you're
kind of randomly setting
parts of the network to zero
during the forward pass,
and then you kind of
marginalize out over that noise
in the back at test time.
And we saw that this was
kind of a general pattern
across many different
types of regularization
in deep learning, where
you might add some kind
of noise during training,
but then marginalize out
that noise at test time
so it's not stochastic
at test time.
We also talked about
transfer learning where you
can maybe download big
networks that were pre-trained
on some dataset and then
fine tune them for your
own problem.
And this is one way that you
can attack a lot of problems
in deep learning, even
if you don't have a huge
dataset of your own.
So today we're going to
shift gears a little bit
and talk about some of the nuts and bolts
about writing software and
how the hardware works.
And a little bit, diving
into a lot of details
about what the software
looks like that you actually
use to train these things in practice.
So we'll talk a little
bit about CPUs and GPUs
and then we'll talk about
several of the major
deep learning frameworks
that are out there in use
these days.
So first, we've sort of
mentioned this off hand
a bunch of different times,
that computers have CPUs,
computers have GPUs.
Deep learning uses GPUs,
but we weren't really
too explicit up to this
point about what exactly
these things are and
why one might be better
than another for different tasks.
So, who's built a computer before?
Just kind of show of hands.
So, maybe about a third
of you, half of you,
somewhere around that ballpark.
So this is a shot of my computer at home
that I built.
And you can see that there's
a lot of stuff going on
inside the computer,
maybe, hopefully you know
what most of these parts are.
And the CPU is the
Central Processing Unit.
That's this little chip
hidden under this cooling fan
right here near the top of the case.
And the CPU is actually
relatively small piece.
It's a relatively small
thing inside the case.
It's not taking up a lot of space.
And the GPUs are these
two big monster things
that are taking up a
gigantic amount of space
in the case.
They have their own cooling,
they're taking a lot of power.
They're quite large.
So, just in terms of how
much power they're using,
in terms of how big they
are, the GPUs are kind of
physically imposing and
taking up a lot of space
in the case.
So the question is what are these things
and why are they so
important for deep learning?
Well, the GPU is called a graphics card,
or Graphics Processing Unit.
And these were really developed,
originally for rendering
computer graphics, and
especially around games
and that sort of thing.
So another show of hands,
who plays video games at home
sometimes, from time to
time on their computer?
Yeah, so again, maybe
about half, good fraction.
So for those of you who've
played video games before
and who've built your own computers,
you probably have your own
opinions on this debate.
[laughs]
So this is one of those big
debates in computer science.
You know, there's like Intel versus AMD,
NVIDIA versus AMD for graphics cards.
It's up there with Vim
versus Emacs for text editor.
And pretty much any gamer
has their own opinions
on which of these two sides they prefer
for their own cards.
And in deep learning we
kind of have mostly picked
one side of this fight, and that's NVIDIA.
So if you guys have AMD cards,
you might be in a little
bit more trouble if you want
to use those for deep learning.
And really, NVIDIA's been
pushing a lot for deep learning
in the last several years.
It's been kind of a large focus
of some of their strategy.
And they put in a lot
effort into engineering
sort of good solutions
to make their hardware
better suited for deep learning.
So most people in deep learning
when we talk about GPUs,
we're pretty much exclusively
talking about NVIDIA GPUs.
Maybe in the future this'll
change a little bit,
and there might be new players coming up,
but at least for now
NVIDIA is pretty dominant.
So to give you an idea of
like what is the difference
between a CPU and a GPU,
I've kind of made a little
spread sheet here.
On the top we have two of
the kind of top end Intel
consumer CPUs, and on
the bottom we have two of
NVIDIA's sort of current
top end consumer GPUs.
And there's a couple general
trends to notice here.
Both GPUs and CPUs are
kind of a general purpose
computing machine where
they can execute programs
and do sort of arbitrary instructions,
but they're qualitatively
pretty different.
So CPUs tend to have just a few cores,
for consumer desktop CPUs these days,
they might have something like four or six
or maybe up to 10 cores.
With hyperthreading technology
that means they can run,
the hardware can physically
run, like maybe eight
or up to 20 threads concurrently.
So the CPU can maybe do 20
things in parallel at once.
So that's just not a gigantic number,
but those threads for a
CPU are pretty powerful.
They can actually do a lot of things,
they're very fast.
Every CPU instruction can
actually do quite a lot
of stuff.
And they can all work
pretty independently.
For GPUs it's a little bit different.
So for GPUs we see that
these sort of common top end
consumer GPUs have thousands of cores.
So the NVIDIA Titan XP
which is the current
top of the line consumer
GPU has 3840 cores.
So that's a crazy number.
That's like way more than
the 10 cores that you'll get
for a similarly priced CPU.
The downside of a GPU is
that each of those cores,
one, it runs at a much slower clock speed.
And two they really
can't do quite as much.
You can't really compare
CPU cores and GPU cores
apples to apples.
The GPU cores can't really
operate very independently.
They all kind of need to work together
and sort of paralyze one
task across many cores
rather than each core
totally doing its own thing.
So you can't really compare
these numbers directly.
But it should give you the sense that due
to the large number of
cores GPUs can sort of,
are really good for
parallel things where you
need to do a lot of things
all at the same time,
but those things are all
pretty much the same flavor.
Another thing to point
out between CPUs and GPUs
is this idea of memory.
Right, so CPUs have some cache on the CPU,
but that's relatively
small and the majority
of the memory for your
CPU is pulling from your
system memory, the RAM,
which will maybe be like
eight, 12, 16, 32 gigabytes
of RAM on a typical
consumer desktop these days.
Whereas GPUs actually
have their own RAM built
into the chip.
There's a pretty large
bottleneck communicating
between the RAM in your
system and the GPU,
so the GPUs typically have their own
relatively large block of
memory within the card itself.
And for the Titan XP, which
again is maybe the current
top of the line consumer card,
this thing has 12 gigabytes
of memory local to the GPU.
GPUs also have their own caching system
where there are sort of
multiple hierarchies of caching
between the 12 gigabytes of GPU memory
and the actual GPU cores.
And that's somewhat similar
to the caching hierarchy
that you might see in a CPU.
So, CPUs are kind of good for
general purpose processing.
They can do a lot of different things.
And GPUs are maybe more
specialized for these highly
paralyzable algorithms.
So the prototypical algorithm
of something that works
really really well and
is like perfectly suited
to a GPU is matrix multiplication.
So remember in matrix
multiplication on the left
we've got like a matrix
composed of a bunch of rows.
We multiply that on the right
by another matrix composed
of a bunch of columns
and then this produces
another, a final matrix
where each element in the
output matrix is a dot product
between one of the rows
and one of the columns of
the two input matrices.
And these dot products
are all independent.
Like you could imagine,
for this output matrix
you could split it up completely
and have each of those different elements
of the output matrix all
being computed in parallel
and they all sort of are
running the same computation
which is taking a dot
product of these two vectors.
But exactly where they're
reading that data from
is from different places
in the two input matrices.
So you could imagine that
for a GPU you can just
like blast this out and
have all of this elements
of the output matrix
all computed in parallel
and that could make this thing
computer super super fast
on GPU.
So that's kind of the
prototypical type of problem
that like where a GPU
is really well suited,
where a CPU might have
to go in and step through
sequentially and compute
each of these elements
one by one.
That picture is a little
bit of a caricature because
CPUs these days have multiple cores,
they can do vectorized
instructions as well,
but still, for these like
massively parallel problems
GPUs tend to have much better throughput.
Especially when these matrices
get really really big.
And by the way, convolution
is kind of the same
kind of story.
Where you know in convolution
we have this input tensor,
we have this weight tensor
and then every point in the
output tensor after a
convolution is again some inner
product between some part of the weights
and some part of the input.
And you can imagine that a
GPU could really paralyze
this computation, split it
all up across the many cores
and compute it very quickly.
So that's kind of the
general flavor of the types
of problems where GPUs give
you a huge speed advantage
over CPUs.
So you can actually write
programs that run directly
on GPUs.
So NVIDIA has this CUDA
abstraction that lets you write
code that kind of looks like C,
but executes directly on the GPUs.
But CUDA code is really really tricky.
It's actually really tough
to write CUDA code that's
performant and actually
squeezes all the juice out
of these GPUs.
You have to be very careful
managing the memory hierarchy
and making sure you
don't have cache misses
and branch mispredictions
and all that sort of stuff.
So it's actually really really
hard to write performant
CUDA code on your own.
So as a result NVIDIA has
released a lot of libraries
that implement common
computational primitives
that are very very highly
optimized for GPUs.
So for example NVIDIA has a
cuBLAS library that implements
different kinds of matrix multiplications
and different matrix operations
that are super optimized,
run really well on GPU,
get very close to sort of
theoretical peak hardware utilization.
Similarly they have a cuDNN
library which implements
things like convolution,
forward and backward passes,
batch normalization, recurrent networks,
all these kinds of
computational primitives
that we need in deep learning.
NVIDIA has gone in there and
released their own binaries
that compute these
primitives very efficiently
on NVIDIA hardware.
So in practice, you tend not
to end up writing your own
CUDA code for deep learning.
You typically are just
mostly calling into existing
code that other people have written.
Much of which is the stuff
which has been heavily
optimized by NVIDIA already.
There's another sort of
language called OpenCL
which is a bit more general.
Runs on more than just NVIDIA GPUs,
can run on AMD hardware, can run on CPUs,
but OpenCL, nobody's really
spent a really large amount
of effort and energy trying
to get optimized deep learning
primitives for OpenCL, so
it tends to be a lot less
performant the super
optimized versions in CUDA.
So maybe in the future we
might see a bit of a more open
standard and we might see
this across many different
more types of platforms,
but at least for now,
NVIDIA's kind of the main game
in town for deep learning.
So you can check, there's a
lot of different resources
for learning about how you can
do GPU programming yourself.
It's kind of fun.
It's sort of a different
paradigm of writing code
because it's this massively
parallel architecture,
but that's a bit beyond
the scope of this course.
And again, you don't really
need to write your own
CUDA code much in practice
for deep learning.
And in fact, I've never
written my own CUDA code
for any research project, so,
but it is kind of useful
to know like how it works
and what are the basic
ideas even if you're not
writing it yourself.
So if you want to look at
kind of CPU GPU performance
in practice, I did some
benchmarks last summer
comparing a decent Intel CPU
against a bunch of different
GPUs that were sort
of near top of the line at that time.
And these were my own
benchmarks that you can find
more details on GitHub,
but my findings were that
for things like VGG 16 and
19, ResNets, various ResNets,
then you typically see
something like a 65 to 75 times
speed up when running the
exact same computation
on a top of the line GPU, in
this case a Pascal Titan X,
versus a top of the line,
well, not quite top of the line
CPU, which in this case
was an Intel E5 processor.
Although, I'd like to make
one sort of caveat here
is that you always need
to be super careful
whenever you're reading
any kind of benchmarks
about deep learning, because
it's super easy to be
unfair between different things.
And you kind of need to know
a lot of the details about
what exactly is being
benchmarked in order to know
whether or not the comparison is fair.
So in this case I'll come
right out and tell you
that probably this comparison
is a little bit unfair
to CPU because I didn't
spend a lot of effort
trying to squeeze the maximal performance
out of CPUs.
I probably could have tuned
the blast libraries better
for the CPU performance.
And I probably could
have gotten these numbers
a bit better.
This was sort of out
of the box performance
between just installing
Torch, running it on a CPU,
just installing Torch running it on a GPU.
So this is kind of out
of the box performance,
but it's not really like
peak, possible, theoretical
throughput on the CPU.
But that being said, I
think there are still pretty
substantial speed ups to be had here.
Another kind of interesting
outcome from this benchmarking
was comparing these
optimized cuDNN libraries
from NVIDIA for convolution
and whatnot versus
sort of more naive CUDA
that had been hand written
out in the open source community.
And you can see that if you
compare the same networks
on the same hardware with
the same deep learning
framework and the only
difference is swapping out
these cuDNN versus sort of
hand written, less optimized
CUDA you can see something
like nearly a three X speed up
across the board when you
switch from the relatively
simple CUDA to these like
super optimized cuDNN
implementations.
So in general, whenever
you're writing code on GPU,
you should probably almost
always like just make sure
you're using cuDNN because
you're leaving probably
a three X performance boost
on the table if you're
not calling into cuDNN for your stuff.
So another problem that
comes up in practice,
when you're training these things is that
you know, your model is
maybe sitting on the GPU,
the weights of the model
are in that 12 gigabytes
of local storage on the
GPU, but your big dataset
is sitting over on the
right on a hard drive
or an SSD or something like that.
So if you're not careful
you can actually bottleneck
your training by just
trying to read the data
off the disk.
'Cause the GPU is super
fast, it can compute
forward and backward quite
fast, but if you're reading
sequentially off a spinning
disk, you can actually
bottleneck your training quite,
and that can be really
bad and slow you down.
So some solutions here
are that like you know
if your dataset's really
small, sometimes you might just
read the whole dataset into RAM.
Or even if your dataset isn't so small,
but you have a giant
server with a ton of RAM,
you might do that anyway.
You can also make sure
you're using an SSD instead
of a hard drive, that can help
a lot with read throughput.
Another common strategy
is to use multiple threads
on the CPU that are
pre-fetching data off RAM
or off disk, buffering it
in memory, in RAM so that
then you can continue
feeding that buffer data down
to the GPU with good performance.
This is a little bit painful to set up,
but again like, these
GPU's are so fast that
if you're not really
careful with trying to feed
them data as quickly as possible,
just reading the data
can sometimes bottleneck
the whole training process.
So that's something to be aware of.
So that's kind of the
brief introduction to like
sort of GPU CPU hardware
in practice when it comes
to deep learning.
And then I wanted to
switch gears a little bit
and talk about the
software side of things.
The various deep learning
frameworks that people are using
in practice.
But I guess before I move on,
is there any sort of
questions about CPU GPU?
Yeah, question?
[student's words obscured
due to lack of microphone]
Yeah, so the question
is what can you sort of,
what can you do mechanically
when you're coding
to avoid these problems?
Probably the biggest thing
you can do in software
is set up sort of pre-fetching on the CPU.
Like you couldn't like,
sort of a naive thing
would be you have this
sequential process where you
first read data off
disk, wait for the data,
wait for the minibatch to be read,
then feed the minibatch to the GPU,
then go forward and backward on the GPU,
then read another minibatch
and sort of do this all
in sequence.
And if you actually have multiple,
like instead you might have
CPU threads running in the
background that are
fetching data off the disk
such that while the,
you can sort of interleave
all of these things.
Like the GPU is computing,
the CPU background threads
are feeding data off disk
and your main thread is kind
of waiting for these things to,
just doing a bit of synchronization
between these things
so they're all happening in parallel.
And thankfully if you're using
some of these deep learning
frameworks that we're about to talk about,
then some of this work has
already been done for you
'cause it's a little bit painful.
So the landscape of
deep learning frameworks
is super fast moving.
So last year when I gave
this lecture I talked mostly
about Caffe, Torch, Theano and TensorFlow.
And when I last gave this talk,
again more than a year ago,
TensorFlow was relatively new.
It had not seen super widespread
adoption yet at that time.
But now I think in the
last year TensorFlow
has gotten much more popular.
It's probably the main framework
of choice for many people.
So that's a big change.
We've also seen a ton of new frameworks
sort of popping up like
mushrooms in the last year.
So in particular Caffe2 and
PyTorch are new frameworks
from Facebook that I think
are pretty interesting.
There's also a ton of other frameworks.
Paddle, Baidu has Paddle,
Microsoft has CNTK,
Amazon is mostly using
MXNet and there's a ton
of other frameworks as well,
but I'm less familiar with,
and really don't have time to get into.
But one interesting thing to
point out from this picture
is that kind of the first
generation of deep learning
frameworks that really saw wide adoption
were built in academia.
So Caffe was from Berkeley,
Torch was developed
originally NYU and also in
collaboration with Facebook.
And Theana was mostly build
at the University of Montreal.
But these kind of next
generation deep learning
frameworks all originated in industry.
So Caffe2 is from Facebook,
PyTorch is from Facebook.
TensorFlow is from Google.
So it's kind of an interesting
shift that we've seen
in the landscape over
the last couple of years
is that these ideas
have really moved a lot
from academia into industry.
And now industry is kind of
giving us these big powerful
nice frameworks to work with.
So today I wanted to
mostly talk about PyTorch
and TensorFlow 'cause I
personally think that those
are probably the ones you
should be focusing on for
a lot of research type
problems these days.
I'll also talk a bit
about Caffe and Caffe2.
But probably a little bit
less emphasis on those.
And before we move any farther,
I thought I should make
my own biases a little bit more explicit.
So I have mostly, I've
worked with Torch mostly
for the last several years.
And I've used it quite
a lot, I like it a lot.
And then in the last year I've
mostly switched to PyTorch
as my main research framework.
So I have a little bit
less experience with some
of these others, especially TensorFlow,
but I'll still try to do
my best to give you a fair
picture and a decent
overview of these things.
So, remember that in the
last several lectures
we've hammered this idea
of computational graphs in
sort of over and over.
That whenever you're doing deep learning,
you want to think about building
some computational graph
that computes whatever function
that you want to compute.
So in the case of a linear
classifier you'll combine
your data X and your weights
W with a matrix multiply.
You'll do some kind of
hinge loss to maybe have,
compute your loss.
You'll have some regularization term
and you imagine stitching
together all these different
operations into some graph structure.
Remember that these graph
structures can get pretty
complex in the case of a big neural net,
now there's many different layers,
many different activations.
Many different weights
spread all around in a pretty
complex graph.
And as you move to things
like neural turing machines
then you can get these really
crazy computational graphs
that you can't even really
draw because they're
so big and messy.
So the point of deep learning
frameworks is really,
there's really kind of three
main reasons why you might
want to use one of these
deep learning frameworks
rather than just writing your own code.
So the first would be that
these frameworks enable
you to easily build and
work with these big hairy
computational graphs
without kind of worrying
about a lot of those
bookkeeping details yourself.
Another major idea is that,
whenever we're working in deep learning
we always need to compute gradients.
We're always computing some loss,
we're always computer
gradient of our weight
with respect to the loss.
And we'd like to make this
automatically computing gradient,
you don't want to have to
write that code yourself.
You want that framework to
handle all these back propagation
details for you so you
can just think about
writing down the forward
pass of your network
and have the backward pass
sort of come out for free
without any additional work.
And finally you want all
this stuff to run efficiently
on GPUs so you don't have to
worry too much about these
low level hardware details
about cuBLAS and cuDNN
and CUDA and moving data
between the CPU and GPU memory.
You kind of want all those messy
details to be taken care of
for you.
So those are kind of
some of the major reasons
why you might choose to
use frameworks rather than
writing your own stuff from scratch.
So as kind of a concrete
example of a computational graph
we can maybe write down
this super simple thing.
Where we have three inputs, X, Y, and Z.
We're going to combine
X and Y to produce A.
Then we're going to combine
A and Z to produce B
and then finally we're going
to do some maybe summing out
operation on B to give
some scaler final result C.
So you've probably written
enough Numpy code at this point
to realize that it's
super easy to write down,
to implement this computational graph,
or rather to implement this
bit of computation in Numpy,
right?
You can just kind of write
down in Numpy that you want to
generate some random data, you
want to multiply two things,
you want to add two things, you
want to sum out a couple things.
And it's really easy to do this in Numpy.
But then the question is
like suppose that we want
to compute the gradient of C
with respect to X, Y, and Z.
So, if you're working in Numpy,
you kind of need to write out
this backward pass yourself.
And you've gotten a lot of
practice with this on the
homeworks, but it can be kind of a pain
and a little bit annoying
and messy once you get to
really big complicated things.
The other problem with
Numpy is that it doesn't run
on the GPU.
So Numpy is definitely CPU only.
And you're never going
to be able to experience
or take advantage of these
GPU accelerated speedups
if you're stuck working in Numpy.
And it's, again, it's a
pain to have to compute
your own gradients in
all these situations.
So, kind of the goal of most
deep learning frameworks
these days is to let you
write code in the forward pass
that looks very similar to Numpy,
but lets you run it on the GPU
and lets you automatically
compute gradients.
And that's kind of the big
picture goal of most of these
frameworks.
So if you imagine looking
at, if we look at an example
in TensorFlow of the exact
same computational graph,
we now see that in this forward pass,
you write this code that ends
up looking very very similar
to the Numpy forward pass
where you're kind of doing
these multiplication and
these addition operations.
But now TensorFlow has
this magic line that just
computes all the gradients for you.
So now you don't have go in and
write your own backward pass
and that's much more convenient.
The other nice thing about
TensorFlow is you can really
just, like with one line you
can switch all this computation
between CPU and GPU.
So here, if you just
add this with statement
before you're doing this forward pass,
you just can explicitly
tell the framework,
hey I want to run this code on the CPU.
But now if we just change that
with statement a little bit
with just with a one
character change in this case,
changing that C to a G,
now the code runs on GPU.
And now in this little code snippet,
we've solved these two problems.
We're running our code on the GPU
and we're having the framework
compute all the gradients
for us, so that's really nice.
And PyTorch kind looks
almost exactly the same.
So again, in PyTorch
you kind of write down,
you define some variables,
you have some forward pass
and the forward pass again
looks very similar to like,
in this case identical
to the Numpy code.
And then again, you can
just use PyTorch to compute
gradients, all your
gradients with just one line.
And now in PyTorch again,
it's really easy to switch
to GPU, you just need to
cast all your stuff to the
CUDA data type before
you rung your computation
and now everything runs
transparently on the GPU for you.
So if you kind of just look
at these three examples,
these three snippets of code side by side,
the Numpy, the TensorFlow and the PyTorch
you see that the TensorFlow
and the PyTorch code
in the forward pass looks
almost exactly like Numpy
which is great 'cause
Numpy has a beautiful API,
it's really easy to work with.
But we can compute gradients automatically
and we can run the GPU automatically.
So after that kind of introduction,
I wanted to dive in and
talk in a little bit more
detail about kind of
what's going on inside this
TensorFlow example.
So as a running example throughout
the rest of the lecture,
I'm going to use the training
a two-layer fully connected
ReLU network on random data
as kind of a running example
throughout the rest of the examples here.
And we're going to train this
thing with an L2 Euclidean
loss on random data.
So this is kind of a silly
network, it's not really doing
anything useful, but it does give you the,
it's relatively small, self contained,
the code fits on the slide
without being too small,
and it lets you demonstrate
kind of a lot of the useful
ideas inside these frameworks.
So here on the right, oh,
and then another note,
I'm kind of assuming
that Numpy and TensorFlow
have already been imported
in all these code snippets.
So in TensorFlow you would
typically divide your computation
into two major stages.
First, we're going to write
some code that defines
our computational graph,
and that's this red code
up in the top half.
And then after you define your graph,
you're going to run the
graph over and over again
and actually feed data into the graph
to perform whatever computation
you want it to perform.
So this is the really,
this is kind of the big
common pattern in TensorFlow.
You'll first have a bunch of
code that builds the graph
and then you'll go and
run the graph and reuse it
many many times.
So if you kind of dive
into the code of building
the graph in this case.
Up at the top you see that
we're defining this X, Y,
w1 and w2, and we're creating
these tf.placeholder objects.
So these are going to be
input nodes to the graph.
These are going to be sort
of entry points to the graph
where when we run the graph,
we're going to feed in data
and put them in through
these input slots in our
computational graph.
So this is not actually
like allocating any memory
right now.
We're just sort of setting
up these input slots
to the graph.
Then we're going to use those
input slots which are now
kind of like these symbolic variables
and we're going to perform
different TensorFlow operations
on these symbolic variables
in order to set up
what computation we want
to run on those variables.
So in this case we're doing
a matrix multiplication
between X and w1, we're
doing some tf.maximum to do a
ReLU nonlinearity and
then we're doing another
matrix multiplication to
compute our output predictions.
And then we're again using
a sort of basic Tensor
operations to compute
our Euclidean distance,
our L2 loss between our
prediction and the target Y.
Another thing to point out here is that
these lines of code are not
actually computing anything.
There's no data in the system right now.
We're just building up this
computational graph data
structure telling
TensorFlow which operations
we want to eventually run
once we put in real data.
So this is just building the graph,
this is not actually doing anything.
Then we have this magical line
where after we've computed
our loss with these symbolic operations,
then we can just ask TensorFlow to compute
the gradient of the loss
with respect to w1 and w2
in this one magical, beautiful line.
And this avoids you writing
all your own backprop code
that you had to do in the assignments.
But again there's no actual
computation happening here.
This is just sort of
adding extra operations
to the computational graph
where now the computational
graph has these additional
operations which will end up
computing these gradients for you.
So now at this point we've
computed our computational
graph, we have this big graph
in this graph data structure
in memory that knows what
operations we want to perform
to compute the loss in gradients.
And now we enter a TensorFlow
session to actually run
this graph and feed it with data.
So then, once we've entered the session,
then we actually need to
construct some concrete values
that will be fed to the graph.
So TensorFlow just expects
to receive data from
Numpy arrays in most cases.
So here we're just creating
concrete actual values
for X, Y, w1 and w2 using
Numpy and then storing these
in some dictionary.
And now here is where we're
actually running the graph.
So you can see that we're
calling a session.run
to actually execute
some part of the graph.
The first argument loss, tells
us which part of the graph
do we actually want as output.
And that, so we actually want the graph,
in this case we need to
tell it that we actually
want to compute loss and grad1 and grad w2
and we need to pass in with
this feed dict parameter
the actual concrete values
that will be fed to the graph.
And then after, in this one line,
it's going and running the
graph and then computing
those values for loss grad1 to grad w2
and then returning the
actual concrete values
for those in Numpy arrays again.
So now after you unpack this
output in the second line,
you get Numpy arrays, or you
get Numpy arrays with the loss
and the gradients.
So then you can go and
do whatever you want
with these values.
So then, this has only run sort
of one forward and backward
pass through our graph,
and it only takes a couple
extra lines if we actually
want to train the network.
So here we're, now we're
running the graph many times
in a loop so we're doing a four loop
and in each iteration of the loop,
we're calling session.run
asking it to compute
the loss and the gradients.
And now we're doing a
manual gradient discent step
using those computed gradients
to now update our current
values of the weights.
So if you actually run this
code and plot the losses,
then you'll see that the loss goes down
and the network is training and
this is working pretty well.
So this is kind of like a
super bare bones example
of training a fully connected
network in TensorFlow.
But there's a problem here.
So here, remember that
on the forward pass,
every time we execute this graph,
we're actually feeding in the weights.
We have the weights as Numpy arrays
and we're explicitly
feeding them into the graph.
And now when the graph finishes executing
it's going to give us these gradients.
And remember the gradients
are the same size
as the weights.
So this means that every time
we're running the graph here,
we're copying the weights
from Numpy arrays into
TensorFlow then getting the gradients
and then copying the
gradients from TensorFlow
back out to Numpy arrays.
So if you're just running on CPU,
this is maybe not a huge deal,
but remember we talked
about CPU GPU bottleneck
and how it's very expensive
actually to copy data
between CPU memory and GPU memory.
So if your network is very
large and your weights
and gradients were very big,
then doing something like
this would be super expensive
and super slow because we'd
be copying all kinds of data
back and forth between the
CPU and the GPU at every
time step.
So that's bad, we don't want to do that.
We need to fix that.
So, obviously TensorFlow
has some solution to this.
And the idea is that
now we want our weights,
w1 and w2, rather than being
placeholders where we're
going to, where we expect to
feed them in to the network
on every forward pass, instead
we define them as variables.
So a variable is something
is a value that lives inside
the computational graph
and it's going to persist
inside the computational
graph across different times
when you run the same graph.
So now instead of declaring
these w1 and w2 as placeholders,
instead we just construct
them as variables.
But now since they live inside the graph,
we also need to tell
TensorFlow how they should be
initialized, right?
Because in the previous
case we were feeding in
their values from outside the graph,
so we initialized them in Numpy,
but now because these things
live inside the graph,
TensorFlow is responsible
for initializing them.
So we need to pass in a
tf.randomnormal operation,
which again is not
actually initializing them
when we run this line, this
is just telling TensorFlow
how we want them to be initialized.
So it's a little bit of
confusing misdirection
going on here.
And now, remember in the previous example
we were actually updating
the weights outside
of the computational graph.
We, in the previous example,
we were computing the gradients
and then using them to update
the weights as Numpy arrays
and then feeding in the
updated weights at the next
time step.
But now because we want
these weights to live inside
the graph, this operation
of updating the weights
needs to also be an operation inside
the computational graph.
So now we used this assign
function which mutates
these variables inside
the computational graph
and now the mutated value will
persist across multiple runs
of the same graph.
So now when we run this graph
and when we train the network,
now we need to run the graph
once with a little bit of
special incantation to tell
TensorFlow to set up these
variables that are going
to live inside the graph.
And then once we've done
that initialization,
now we can run the graph
over and over again.
And here, we're now only
feeding in the data and labels
X and Y and the weights are
living inside the graph.
And here we've asked the network to,
we've asked TensorFlow to
compute the loss for us.
And then you might think that
this would train the network,
but there's actually a bug here.
So, if you actually run this code,
and you plot the loss, it doesn't train.
So that's bad, it's confusing,
like what's going on?
We wrote this assign
code, we ran the thing,
like we computed the
loss and the gradients
and our loss is flat, what's going on?
Any ideas?
[student's words obscured
due to lack of microphone]
Yeah so one hypothesis is
that maybe we're accidentally
re-initializing the w's
every time we call the graph.
That's a good hypothesis,
that's actually not the problem
in this case.
[student's words obscured
due to lack of microphone]
Yeah, so the answer is that
we actually need to explicitly
tell TensorFlow that we
want to run these new w1
and new w2 operations.
So we've built up this big
computational graph data
structure in memory and
now when we call run,
we only told TensorFlow that
we wanted to compute loss.
And if you look at the
dependencies among these different
operations inside the graph,
you see that in order to compute loss
we don't actually need to
perform this update operation.
So TensorFlow is smart and
it only computes the parts
of the graph that are necessary
for computing the output
that you asked it to compute.
So that's kind of a nice thing
because it means it's only
doing as much work as it needs to,
but in situations like this it
can be a little bit confusing
and lead to behavior
that you didn't expect.
So the solution in this case
is that we actually need to
explicitly tell TensorFlow
to perform those
update operations.
So one thing we could do,
which is what was suggested
is we could add new w1
and new w2 as outputs
and just tell TensorFlow
that we want to produce
these values as outputs.
But that's a problem
too because the values,
those new w1, new w2 values
are again these big tensors.
So now if we tell TensorFlow
we want those as output,
we're going to again get
this copying behavior
between CPU and GPU at ever iteration.
So that's bad, we don't want that.
So there's a little
trick you can do instead.
Which is that we add kind of
a dummy node to the graph.
With these fake data dependencies
and we just say that
this dummy node updates,
has these data dependencies
of new w1 and new w2.
And now when we actually run the graph,
we tell it to compute both
the loss and this dummy node.
And this dummy node
doesn't actually return
any value it just returns
none, but because of this
dependency that we've put
into the node it ensures
that when we run the updates value,
we actually also run
these update operations.
So, question?
[student's words obscured
due to lack of microphone]
Is there a reason why we didn't
put X and Y into the graph?
And that it stayed as Numpy.
So in this example we're
reusing X and Y on every,
we're reusing the same X
and Y on every iteration.
So you're right, we could
have just also stuck those
in the graph, but in a
more realistic scenario,
X and Y will be minibatches
of data so those will actually
change at every iteration
and we will want to feed
different values for
those at every iteration.
So in this case, they could
have stayed in the graph,
but in most cases they will change,
so we don't want them
to live in the graph.
Oh, another question?
[student's words obscured
due to lack of microphone]
Yeah, so we've told it,
we had put into TensorFlow
that the outputs we want
are loss and updates.
Updates is not actually a real value.
So when updates evaluates
it just returns none.
But because of this dependency
we've told it that updates
depends on these assign operations.
But these assign operations live inside
the computational graph and
all live inside GPU memory.
So then we're doing
these update operations
entirely on the GPU and
we're no longer copying the
updated values back out of the graph.
[student's words obscured
due to lack of microphone]
So the question is does
tf.group return none?
So this gets into the
trickiness of TensorFlow.
So tf.group returns some
crazy TensorFlow value.
It sort of returns some like
internal TensorFlow node
operation that we need to
continue building the graph.
But when you execute the graph,
and when you tell, inside the session.run,
when we told it we want it
to compute the concrete value
from updates, then that returns none.
So whenever you're working with TensorFlow
you have this funny indirection
between building the graph
and the actual output values
during building the graph
is some funny weird object,
and then you actually get
a concrete value when you run the graph.
So here after you run updates,
then the output is none.
Does that clear it up a little bit?
[student's words obscured
due to lack of microphone]
So the question is why is loss a value
and why is updates none?
That's just the way that updates works.
So loss is a value when we compute,
when we tell TensorFlow
we want to run a tensor,
then we get the concrete value.
Updates is this kind of
special other data type
that does not return a value,
it instead returns none.
So it's kind of some TensorFlow
magic that's going on there.
Maybe we can talk offline
if you're still confused.
[student's words obscured
due to lack of microphone]
Yeah, yeah, that behavior is
coming from the group method.
So now, we kind of have
this weird pattern where we
wanted to do these
different assign operations,
we have to use this funny tf.group thing.
That's kind of a pain, so
thankfully TensorFlow gives
you some convenience
operations that kind of do that
kind of stuff for you.
And that's called an optimizer.
So here we're using a
tf.train.GradientDescentOptimizer
and we're telling it what
learning rate we want to use.
And you can imagine that
there's, there's RMSprop,
there's all kinds of different
optimization algorithms here.
And now we call optimizer.minimize of loss
and now this is a pretty magical,
this is a pretty magical thing,
because now this call is
aware that these variables
w1 and w2 are marked as
trainable by default,
so then internally, inside
this optimizer.minimize
it's going in and adding
nodes to the graph
which will compute gradient
of loss with respect
to w1 and w2 and then it's
also performing that update
operation for you and it's
doing the grouping operation
for you and it's doing the assigns.
It's like doing a lot of
magical stuff inside there.
But then it ends up giving
you this magical updates value
which, if you dig through the
code they're actually using
tf.group so it looks very
similar internally to what
we saw before.
And now when we run the
graph inside our loop
we do the same pattern of
telling it to compute loss
and updates.
And every time we tell the
graph to compute updates,
then it'll actually go
and update the graph.
Question?
[student's words obscured
due to lack of microphone]
Yeah, so what is the
tf.GlobalVariablesInitializer?
So that's initializing w1
and w2 because these are
variables which live inside the graph.
So we need to, when we
saw this, when we create
the tf.variable we have
this tf.randomnormal
which is this initialization so the
tf.GlobalVariablesInitializer
is causing the
tf.randomnormal to actually run
and generate concrete values
to initialize those variables.
[student's words obscured
due to lack of microphone]
Sorry, what was the question?
[student's words obscured
due to lack of microphone]
So it knows that a
placeholder is going to be fed
outside of the graph and a
variable is something that
lives inside the graph.
So I don't know all the
details about how it decides,
what exactly it decides
to run with that call.
I think you'd need to dig
through the code to figure
that out, or maybe it's
documented somewhere.
So but now we've kind of got this,
again we've got this full
example of training a
network in TensorFlow
and we're kind of adding
bells and whistles to make it
a little bit more convenient.
So we can also here,
in the previous example
we were computing the loss
explicitly using our own
tensor operations, TensorFlow
you can always do that,
you can use basic tensor
operations to compute
just about anything you want.
But TensorFlow also gives
you a bunch of convenience
functions that compute these
common neural network things
for you.
So in this case we can use
tf.losses.mean_squared_error
and it just does the L2
loss for us so we don't have
to compute it ourself in terms
of basic tensor operations.
So another kind of weirdness
here is that it was kind of
annoying that we had to
explicitly define our inputs
and define our weights and
then like chain them together
in the forward pass
using a matrix multiply.
And in this example we've
actually not put biases
in the layer because that
would be kind of an extra,
then we'd have to initialize biases,
we'd have to get them in the right shape,
we'd have to broadcast the
biases against the output
of the matrix multiply
and you can see that that
would kind of be a lot of code.
It would be kind of annoying write.
And once you get to like convolutions
and batch normalizations
and other types of layers
this kind of basic way of working,
of having these variables,
having these inputs and outputs
and combining them all together with basic
computational graph operations
could be a little bit
unwieldy and it could
be really annoying to
make sure you initialize
the weights with the right
shapes and all that sort of stuff.
So as a result, there's a
bunch of sort of higher level
libraries that wrap around TensorFlow
and handle some of these details for you.
So one example that ships with TensorFlow,
is this tf.layers inside.
So now in this code example
you can see that our code
is only explicitly
declaring the X and the Y
which are the placeholders
for the data and the labels.
And now we say that H=tf.layers.dense,
we give it the input X
and we tell it units=H.
This is again kind of a magical line
because inside this line,
it's kind of setting up
w1 and b1, the bias, it's
setting up variables for those
with the right shapes that
are kind of inside the graph
but a little bit hidden from us.
And it's using this
xavier initializer object
to set up an initialization
strategy for those.
So before we were doing
that explicitly ourselves
with the tf.randomnormal business,
but now here it's kind of
handling some of those details
for us and it's just spitting out an H,
which is again the same
sort of H that we saw
in the previous layer, it's
just doing some of those
details for us.
And you can see here,
we're also passing an
activation=tf.nn.relu so it's
even doing the activation,
the relu activation function
inside this layer for us.
So it's taking care of a
lot of these architectural
details for us.
Question?
[student's words obscured
due to lack of microphone]
Question is does the
xavier initializer default
to particular distribution?
I'm sure it has some default,
I'm not sure what it is.
I think you'll have to
look at the documentation.
But it seems to be a
reasonable strategy, I guess.
And in fact if you run this code,
it converges much faster
than the previous one
because the initialization is better.
And you can see that
we're using two calls to
tf.layers and this lets us build our model
without doing all these
explicit bookkeeping details
ourself.
So this is maybe a little
bit more convenient.
But tf.contrib.layer is really
not the only game in town.
There's like a lot of different
higher level libraries
that people build on top of TensorFlow.
And it's kind of due to this
basic impotence mis-match
where the computational graph
is relatively low level thing,
but when we're working
with neural networks
we have this concept of layers and weights
and some layers have weights
associated with them,
and we typically think at
a slightly higher level
of abstraction than this
raw computational graph.
So that's what these various
packages are trying to
help you out and let you
work at this higher layer
of abstraction.
So another very popular
package that you may have
seen before is Keras.
Keras is a very beautiful,
nice API that sits on top of
TensorFlow and handles
sort of building up these
computational graph for
you up in the back end.
By the way, Keras also
supports Theano as a back end,
so that's also kind of nice.
And in this example you
can see we build the model
as a sequence of layers.
We build some optimizer object
and we call model.compile
and this does a lot of magic
in the back end to build the graph.
And now we can call model.fit
and that does the whole
training procedure for us magically.
So I don't know all the
details of how this works,
but I know Keras is very popular,
so you might consider using
it if you're talking about
TensorFlow.
Question?
[student's words obscured
due to lack of microphone]
Yeah, so the question is
like why there's no explicit
CPU, GPU going on here.
So I've kind of left that
out to keep the code clean.
But you saw at the beginning examples
it was pretty easy to
flop all these things
between CPU and GPU and there
was either some global flag
or some different data type
or some with statement and
it's usually relatively simple
and just about one line
to swap in each case.
But exactly what that line looks like
differs a bit depending on the situation.
So there's actually like
this whole large set
of higher level TensorFlow
wrappers that you might see
out there in the wild.
And it seems that like
even people within Google
can't really agree on which
one is the right one to use.
So Keras and TFLearn are
third party libraries
that are out there on the
internet by other people.
But there's these three different ones,
tf.layers, TF-Slim and tf.contrib.learn
that all ship with TensorFlow,
that are all kind of
doing a slightly different version of this
higher level wrapper thing.
There's another framework
also from Google,
but not shipping with
TensorFlow called Pretty Tensor
that does the same sort of thing.
And I guess none of these
were good enough for DeepMind,
because they went ahead a couple weeks ago
and wrote and released
their very own high level
TensorFlow wrapper called Sonnet.
So I wouldn't begrudge you
if you were kind of confused
by all these things.
There's a lot of different choices.
They don't always play
nicely with each other.
But you have a lot of
options, so that's good.
TensorFlow has pretrained models.
There's some examples in
TF-Slim, and in Keras.
'Cause remember retrained
models are super important
when you're training your own things.
There's also this idea of Tensorboard
where you can load up your,
I don't want to get into details,
but Tensorboard you can
add sort of instrumentation
to your code and then
plot losses and things
as you go through the training process.
TensorFlow also let's you run distributed
where you can break up
a computational graph
run on different machines.
That's super cool but I
think probably not anyone
outside of Google is really
using that to great success
these days, but if you do
want to run distributed stuff
probably TensorFlow is the
main game in town for that.
A side note is that a lot
of the design of TensorFlow
is kind of spiritually inspired
by this earlier framework
called Theano from Montreal.
I don't want to go
through the details here,
just if you go through
these slides on your own,
you can see that the code
for Theano ends up looking
very similar to TensorFlow.
Where we define some variables,
we do some forward pass,
we compute some gradients,
and we compile some function,
then we run the function
over and over to train the network.
So it kind of looks a lot like TensorFlow.
So we still have a lot to get through,
so I'm going to move on to PyTorch
and maybe take questions at the end.
So, PyTorch from Facebook
is kind of different from
TensorFlow in that we have
sort of three explicit
different layers of
abstraction inside PyTorch.
So PyTorch has this tensor
object which is just like a
Numpy array.
It's just an imperative array,
it doesn't know anything
about deep learning,
but it can run with GPU.
We have this variable
object which is a node in a
computational graph which
builds up computational graphs,
lets you compute gradients,
that sort of thing.
And we have a module object
which is a neural network
layer that you can compose
together these modules
to build big networks.
So if you kind of want to
think about rough equivalents
between PyTorch and TensorFlow
you can think of the
PyTorch tensor as fulfilling the same role
as the Numpy array in TensorFlow.
The PyTorch variable is similar
to the TensorFlow tensor
or variable or placeholder,
which are all sort of nodes
in a computational graph.
And now the PyTorch module
is kind of equivalent
to these higher level things
from tf.slim or tf.layers
or sonnet or these other
higher level frameworks.
So right away one thing
to notice about PyTorch
is that because it ships with
this high level abstraction
and like one really nice
higher level abstraction
called modules on its own,
there's sort of less choice
involved.
Just stick with nnmodules
and you'll be good to go.
You don't need to worry about
which higher level wrapper
to use.
So PyTorch tensors, as I said,
are just like Numpy arrays
so here on the right we've done
an entire two layer network
using entirely PyTorch tensors.
One thing to note is that
we're not importing Numpy here
at all anymore.
We're just doing all these
operations using PyTorch tensors.
And this code looks exactly
like the two layer net code
that you wrote in Numpy
on the first homework.
So you set up some random
data, you use some operations
to compute the forward pass.
And then we're explicitly
viewing the backward pass
ourself.
Just sort of backhopping
through the network,
through the operations, just
as you did on homework one.
And now we're doing a
manual update of the weights
using a learning rate and
using our computed gradients.
But the major difference
between the PyTorch tensor
and Numpy arrays is that they run on GPU
so all you have to do
to make this code run on
GPU is use a different data type.
Rather than using torch.FloatTensor,
you do torch.cuda.FloatTensor,
cast all of your tensors
to this new datatype and
everything runs magically
on the GPU.
You should think of PyTorch
tensors as just Numpy plus GPU.
That's exactly what it
is, nothing specific
to deep learning.
So the next layer of abstraction
in PyTorch is the variable.
So this is, once we moved
from tensors to variables
now we're building computational graphs
and we're able to take
gradients automatically
and everything like that.
So here, if X is a variable,
then x.data is a tensor
and x.grad is another variable
containing the gradients
of the loss with respect to that tensor.
So x.grad.data is an
actual tensor containing
those gradients.
And PyTorch tensors and variables
have the exact same API.
So any code that worked on
PyTorch tensors you can just
make them variables instead
and run the same code,
except now you're building
up a computational graph
rather than just doing
these imperative operations.
So here when we create these variables
each call to the variable
constructor wraps a PyTorch
tensor and then also gives
a flag whether or not
we want to compute gradients
with respect to this variable.
And now in the forward
pass it looks exactly like
it did before in the variable
in the case with tensors
because they have the same API.
So now we're computing our predictions,
we're computing our loss
in kind of this imperative
kind of way.
And then we call loss.backwards
and now all these gradients
come out for us.
And then we can make
a gradient update step
on our weights using the
gradients that are now present
in the w1.grad.data.
So this ends up looking
quite like the Numpy case,
except all the gradients come for free.
One thing to note that's
kind of different between
PyTorch and TensorFlow is
that in a TensorFlow case
we were building up this explicit graph,
then running the graph many times.
Here in PyTorch, instead
we're building up a new graph
every time we do a forward pass.
And this makes the code
look a bit cleaner.
And it has some other
implications that we'll
get to in a bit.
So in PyTorch you can define
your own new autograd functions
by defining the forward and
backward in terms of tensors.
This ends up looking kind
of like the module layers
code that you write for homework two.
Where you can implement
forward and backward using
tensor operations and then
stick these things inside
computational graph.
So here we're defining our own relu
and then we can actually
go in and use our own relu
operation and now stick it
inside our computational graph
and define our own operations this way.
But most of the time you
will probably not need
to define your own autograd operations.
Most of the times the
operations you need will
mostly be already implemented for you.
So in TensorFlow we saw,
if we can move to something
like Keras or TF.Learn
and this gives us a higher
level API to work with,
rather than this raw computational graphs.
The equivalent in PyTorch
is the nn package.
Where it provides these high
level wrappers for working
with these things.
But unlike TensorFlow
there's only one of them.
And it works pretty well,
so just use that if you're
using PyTorch.
So here, this ends up
kind of looking like Keras
where we define our model
as some sequence of layers.
Our linear and relu operations.
And we use some loss function
defined in the nn package
that's our mean squared error loss.
And now inside each iteration of our loop
we can run data forward
through the model to get
our predictions.
We can run the predictions
forward through the loss function
to get our scale or loss,
then we can call loss.backward,
get all our gradients
for free and then loop over
the parameters of the models
and do our explicit gradient
descent step to update
the models.
And again we see that we're
sort of building up this
new computational graph every
time we do a forward pass.
And just like we saw in TensorFlow,
PyTorch provides these
optimizer operations
that kind of abstract
away this updating logic
and implement fancier
update rules like Adam
and whatnot.
So here we're constructing
an optimizer object
telling it that we want
it to optimize over the
parameters of the model.
Giving it some learning rate
under the hyper parameters.
And now after we compute our gradients
we can just call
optimizer.step and it updates
all the parameters of the
model for us right here.
So another common thing
you'll do in PyTorch
a lot is define your own nn modules.
So typically you'll write your own class
which defines you entire model as a single
new nn module class.
And a module is just kind
of a neural network layer
that can contain either
other other modules
or trainable weights or
other other kinds of state.
So in this case we can redo
the two layer net example
by defining our own nn module class.
So now here in the
initializer of the class
we're assigning this linear1 and linear2.
We're constructing
these new module objects
and then store them
inside of our own class.
And now in the forward pass
we can use both our own
internal modules as well as
arbitrary autograd operations
on variables to compute
the output of our network.
So here we receive the, inside
this forward method here,
the input acts as a variable,
then we pass the variable
to our self.linear1
for the first layer.
We use an autograd op
clamp to complete the relu,
we pass the output of
that to the second linear
and then that gives us our output.
And now the rest of this
code for training this thing
looks pretty much the same.
Where we build an optimizer and loop over
and on ever iteration
feed data to the model,
compute the gradients with loss.backwards,
call optimizer.step.
So this is like relatively characteristic
of what you might see
in a lot of PyTorch type
training scenarios.
Where you define your own class,
defining your own model
that contains other modules
and whatnot and then you
have some explicit training
loop like this that
runs it and updates it.
One kind of nice quality
of life thing that you have
in PyTorch is a dataloader.
So a dataloader can handle
building minibatches for you.
It can handle some of the
multi-threading that we talked
about for you, where it can
actually use multiple threads
in the background to
build many batches for you
and stream off disk.
So here a dataloader wraps
a dataset and provides
some of these abstractions for you.
And in practice when you
want to run your own data,
you typically will write
your own dataset class
which knows how to read
your particular type of data
off whatever source you
want and then wrap it in
a data loader and train with that.
So, here we can see that
now we're iterating over
the dataloader object
and at every iteration
this is yielding minibatches of data.
And it's internally handling
the shuffling of the data
and multithreaded dataloading
and all this sort of stuff
for you.
So this is kind of a
completely PyTorch example
and a lot of PyTorch
training code ends up looking
something like this.
PyTorch provides pretrained models.
And this is probably the
slickest pretrained model
experience I've ever seen.
You just say torchvision.models.alexnet
pretained=true.
That'll go down in the background,
download the pretrained
weights for you if you
don't already have them,
and then it's right
there, you're good to go.
So this is super easy to use.
PyTorch also has, there's
also a package called Visdom
that lets you visualize some
of these loss statistics
somewhat similar to Tensorboard.
So that's kind of nice,
I haven't actually gotten
a chance to play around with
this myself so I can't really
speak to how useful it is,
but one of the major
differences between Tensorboard
and Visdom is that Tensorboard
actually lets you visualize
the structure of the computational graph.
Which is really cool, a really
useful debugging strategy.
And Visdom does not have
that functionality yet.
But I've never really used
this myself so I can't really
speak to its utility.
As a bit of an aside, PyTorch
is kind of an evolution of,
kind of a newer updated
version of an older framework
called Torch which I worked
with a lot in the last
couple of years.
And I don't want to go
through the details here,
but PyTorch is pretty much
better in a lot of ways
than the old Lua Torch, but
they actually share a lot
of the same back end C code
for computing with tensors
and GPU operations on tensors and whatnot.
So if you look through this Torch example,
some of it ends up looking
kind of similar to PyTorch,
some of it's a bit different.
Maybe you can step through this offline.
But kind of the high
level differences between
Torch and PyTorch are that
Torch is actually in Lua,
not Python, unlike these other things.
So learning Lua is a bit of
a turn off for some people.
Torch doesn't have autograd.
Torch is also older, so it's more stable,
less susceptible to bugs,
there's maybe more example code
for Torch.
They're about the same speeds,
that's not really a concern.
But in PyTorch it's in
Python which is great,
you've got autograd which
makes it a lot simpler
to write complex models.
In Lua Torch you end up
writing a lot of your own
back prop code sometimes, so
that's a little bit annoying.
But PyTorch is newer,
there's less existing code,
it's still subject to change.
So it's a little bit more of an adventure.
But at least for me, I kind of prefer,
I don't really see much reason for myself
to use Torch over PyTorch
anymore at this time.
So I'm pretty much using
PyTorch exclusively for
all my work these days.
We talked about this a
little bit about this idea
of static versus dynamic graphs.
And this is one of the main
distinguishing features
between PyTorch and TensorFlow.
So we saw in TensorFlow
you have these two stages
of operation where first you build up this
computational graph, then you
run the computational graph
over and over again many
many times reusing that same
graph.
That's called a static
computational graph 'cause there's
only one of them.
And we saw PyTorch is quite
different where we're actually
building up this new computational graph,
this new fresh thing
on every forward pass.
That's called a dynamic
computational graph.
For kind of simple cases,
with kind of feed forward
neural networks, it doesn't
really make a huge difference,
the code ends up kind of similarly
and they work kind of similarly,
but I do want to talk a bit
about some of the implications
of static versus dynamic.
And what are the tradeoffs of those two.
So one kind of nice
idea with static graphs
is that because we're
kind of building up one
computational graph once, and
then reusing it many times,
the framework might have
the opportunity to go in
and do optimizations on that graph.
And kind of fuse some operations,
reorder some operations,
figure out the most
efficient way to operate
that graph so it can be really efficient.
And because we're going
to reuse that graph
many times, maybe that
optimization process
is expensive up front,
but we can amortize that
cost with the speedups
that we've gotten when we run
the graph many many times.
So as kind of a concrete example,
maybe if you write some
graph which has convolution
and relu operations kind
of one after another,
you might imagine that
some fancy graph optimizer
could go in and actually
output, like emit custom code
which has fused operations,
fusing the convolution
and the relu so now it's
computing the same thing
as the code you wrote, but
now might be able to be
executed more efficiently.
So I'm not too sure on exactly
what the state in practice
of TensorFlow graph
optimization is right now,
but at least in principle,
this is one place where
static graph really, you
can have the potential for
doing this optimization in static graphs
where maybe it would be not so
tractable for dynamic graphs.
Another kind of subtle point
about static versus dynamic
is this idea of serialization.
So with a static graph you
can imagine that you write
this code that builds up the graph
and then once you've built the graph,
you have this data structure
in memory that represents
the entire structure of your network.
And now you could take that data structure
and just serialize it to disk.
And now you've got the whole
structure of your network
saved in some file.
And then you could later
rear load that thing
and then run that computational
graph without access
to the original code that built it.
So this would be kind of nice
in a deployment scenario.
You might imagine that you
might want to train your
network in Python because it's
maybe easier to work with,
but then after you serialize that network
and then you could deploy
it now in maybe a C++
environment where you don't
need to use the original
code that built the graph.
So that's kind of a nice
advantage of static graphs.
Whereas with a dynamic graph,
because we're interleaving
these processes of graph
building and graph execution,
you kind of need the
original code at all times
if you want to reuse
that model in the future.
On the other hand, some
advantages for dynamic graphs
are that it kind of makes,
it just makes your code
a lot cleaner and a lot
easier in a lot of scenarios.
So for example, suppose
that we want to do some
conditional operation where
depending on the value
of some variable Z, we want
to do different operations
to compute Y.
Where if Z is positive, we
want to use one weight matrix,
if Z is negative we want to
use a different weight matrix.
And we just want to switch off
between these two alternatives.
In PyTorch because we're
using dynamic graphs,
it's super simple.
Your code kind of looks
exactly like you would expect,
exactly what you would do in Numpy.
You can just use normal
Python control flow
to handle this thing.
And now because we're building
up the graph each time,
each time we perform this
operation will take one
of the two paths and build
up maybe a different graph
on each forward pass, but
for any graph that we do
end up building up, we can
back propagate through it
just fine.
And the code is very
clean, easy to work with.
Now in TensorFlow the
situations is a little bit more
complicated because we
build the graph once,
this control flow operator
kind of needs to be
an explicit operator in
the TensorFlow graph.
And now, so them you can
see that we have this
tf.cond call which is kind
of like a TensorFlow version
of an if statement,
but now it's baked into
the computational graph
rather than using sort of
Python control flow.
And the problem is that
because we only build the graph
once, all the potential
paths of control flow that
our program might flow
through need to be baked
into the graph at the time we
construct it before we ever
run it.
So that means that any kind
of control flow operators
that you want to have need
to be not Python control flow
operators, you need to
use some kind of magic,
special tensor flow
operations to do control flow.
In this case this tf.cond.
Another kind of similar
situation happens if you want to
have loops.
So suppose that we want to
compute some kind of recurrent
relationships where maybe Y
T is equal to Y T minus one
plus X T times some weight
matrix W and depending on
each time we do this,
every time we compute this,
we might have a different
sized sequence of data.
And no matter the length
of our sequence of data,
we just want to compute this
same recurrence relation
no matter the size of the input sequence.
So in PyTorch this is super easy.
We can just kind of use a
normal for loop in Python
to just loop over the number
of times that we want to
unroll and now depending on
the size of the input data,
our computational graph will
end up as different sizes,
but that's fine, we can
just back propagate through
each one, one at a time.
Now in TensorFlow this
becomes a little bit uglier.
And again, because we need
to construct the graph
all at once up front, this
control flow looping construct
again needs to be an explicit
node in the TensorFlow graph.
So I hope you remember
your functional programming
because you'll have to use
those kinds of operators
to implement looping
constructs in TensorFlow.
So in this case, for this
particular recurrence relationship
you can use a foldl operation and pass in,
sort of implement this particular
loop in terms of a foldl.
But what this basically means
is that you have this sense
that TensorFlow is almost
building its own entire
programming language,
using the language of
computational graphs.
And any kind of control flow operator,
or any kind of data
structure needs to be rolled
into the computational graph
so you can't really utilize
all your favorite paradigms
for working imperatively
in Python.
You kind of need to relearn
a whole separate set
of control flow operators.
And if you want to do
any kinds of control flow
inside your computational
graph using TensorFlow.
So at least for me, I find
that kind of confusing,
a little bit hard to wrap
my head around sometimes,
and I kind of like that
using PyTorch dynamic graphs,
you can just use your favorite
imperative programming
constructs and it all works just fine.
By the way, there actually
is some very new library
called TensorFlow Fold which
is another one of these
layers on top of TensorFlow
that lets you implement
dynamic graphs, you kind
of write your own code
using TensorFlow Fold that
looks kind of like a dynamic
graph operation and then
TensorFlow Fold does some magic
for you and somehow implements
that in terms of the
static TensorFlow graphs.
This is a super new paper
that's being presented
at ICLR this week in France.
So I haven't had the chance
to like dive in and play
with this yet.
But my initial impression
was that it does add some
amount of dynamic graphs to
TensorFlow but it is still
a bit more awkward to work
with than the sort of native
dynamic graphs you have in PyTorch.
So then, I thought it
might be nice to motivate
like why would we care about
dynamic graphs in general?
So one option is recurrent networks.
So you can see that for
something like image captioning
we use a recurrent network
which operates over
sequences of different lengths.
In this case, the sentence
that we want to generate
as a caption is a sequence
and that sequence can vary
depending on our input data.
So now you can see that we
have this dynamism in the thing
where depending on the
size of the sentence,
our computational graph
might need to have more
or fewer elements.
So that's one kind of common
application of dynamic graphs.
For those of you who
took CS224N last quarter,
you saw this idea of recursive networks
where sometimes in natural
language processing
you might, for example,
compute a parsed tree
of a sentence and then
you want to have a neural
network kind of operate
recursively up this parse tree.
So having a neural network
that kind of works,
it's not just a sequential
sequence of layers,
but instead it's kind of
working over some graph
or tree structure instead
where now each data point
might have a different
graph or tree structure
so the structure of
the computational graph
then kind of mirrors the
structure of the input data.
And it could vary from
data point to data point.
So this type of thing seems
kind of complicated and
hairy to implement using TensorFlow,
but in PyTorch you can just kind of use
like normal Python control
flow and it'll work out
just fine.
Another bit of more researchy
application is this really
cool idea that I like
called neuromodule networks
for visual question answering.
So here the idea is that we
want to ask some questions
about images where we
maybe input this image
of cats and dogs, there's some question,
what color is the cat, and
then internally the system
can read the question and
that has these different
specialized neural network
modules for performing
operations like asking for
colors and finding cats.
And then depending on
the text of the question,
it can compile this custom
architecture for answering
the question.
And now if we asked a different question,
like are there more cats than dogs?
Now we have maybe the
same basic set of modules
for doing things like finding
cats and dogs and counting,
but they're arranged in a different order.
So we get this dynamism again
where different data points
might give rise to different
computational graphs.
But this is a bit more
of a researchy thing
and maybe not so main stream right now.
But as kind of a bigger
point, I think that there's
a lot of cool, creative
applications that people
could do with dynamic computational graphs
and maybe there aren't so many right now,
just because it's been so
painful to work with them.
So I think that there's
a lot of opportunity
for doing cool, creative things with
dynamic computational graphs.
And maybe if you come up with cool ideas,
we'll feature it in lecture next year.
So I wanted to talk
very briefly about Caffe
which is this framework from Berkeley.
Which Caffe is somewhat
different from the other
deep learning frameworks
where you in many cases
you can actually train
networks without writing
any code yourself.
You kind of just call into
these pre-existing binaries,
set up some configuration
files and in many cases
you can train on data without
writing any of your own code.
So, you may be first,
you convert your data
into some format like HDF5
or LMDB and there exists
some scripts inside Caffe
that can just convert like
folders of images and text files
into these formats for you.
You need to define, now
instead of writing code
to define the structure of
your computational graph,
instead you edit some text
file called a prototxt
which sets up the structure
of the computational graph.
Here the structure is that
we read from some input
HDF5 file, we perform some inner product,
we compute some loss
and the whole structure
of the graph is set up in this text file.
One kind of downside
here is that these files
can get really ugly for
very large networks.
So for something like the
152 layer ResNet model,
which by the way was
trained in Caffe originally,
then this prototxt file ends
up almost 7000 lines long.
So people are not writing these by hand.
People will sometimes will
like write python scripts
to generate these prototxt files.
[laughter]
Then you're kind in the
realm of rolling your own
computational graph abstraction.
That's probably not a good
idea, but I've seen that before.
Then, rather than having
some optimizer object,
instead there's some solver,
you define some solver things
inside another prototxt.
This defines your learning rate,
your optimization algorithm and whatnot.
And then once you do all these things,
you can just run the Caffe
binary with the train command
and it all happens magically.
Cafee has a model zoo with a
bunch of pretrained models,
that's pretty useful.
Caffe has a Python
interface but it's not super
well documented.
You kind of need to read the
source code of the python
interface to see what it can do,
so that's kind of annoying.
But it does work.
So, kind of my general thing
about Caffe is that it's
maybe good for feed forward models,
it's maybe good for production scenarios,
because it doesn't depend on Python.
But probably for research
these days, I've seen Caffe
being used maybe a little bit less.
Although I think it is
still pretty commonly used
in industry again for production.
I promise one slide, one
or two slides on Caffe 2.
So Caffe 2 is the successor to
Caffe which is from Facebook.
It's super new, it was
only released a week ago.
[laughter]
So I really haven't had
the time to form a super
educated opinion about Caffe 2 yet,
but it uses static graphs
kind of similar to TensorFlow.
Kind of like Caffe one
the core is written in C++
and they have some Python interface.
The difference is that
now you no longer need to
write your own Python scripts
to generate prototxt files.
You can kind of define your
computational graph structure
all in Python, kind of
looking with an API that looks
kind of like TensorFlow.
But then you can spit out,
you can serialize this
computational graph
structure to a prototxt file.
And then once your model
is trained and whatnot,
then we get this benefit that
we talked about of static
graphs where you can, you
don't need the original
training code now in order
to deploy a trained model.
So one interesting thing
is that you've seen Google
maybe has one major
deep running framework,
which is TensorFlow, where
Facebook has these two,
PyTorch and Caffe 2.
So these are kind of
different philosophies.
Google's kind of trying to
build one framework to rule
them all that maybe works
for every possible scenario
for deep learning.
This is kind of nice because
it consolidates all efforts
onto one framework.
It means you only need to learn one thing
and it'll work across
many different scenarios
including like distributed
systems, production,
deployment, mobile, research, everything.
Only need to learn one framework
to do all these things.
Whereas Facebook is taking a
bit of a different approach.
Where PyTorch is really more specialized,
more geared towards research
so in terms of writing
research code and quickly
iterating on your ideas,
that's super easy in
PyTorch, but for things like
running in production,
running on mobile devices,
PyTorch doesn't have a
lot of great support.
Instead, Caffe 2 is kind
of geared toward those more
production oriented use cases.
So my kind of general study,
my general, overall advice
about like which framework
to use for which problems
is kind of that both,
I think TensorFlow is a
pretty safe bet for just about
any project that you
want to start new, right?
Because it is sort of one
framework to rule them all,
it can be used for just
about any circumstance.
However, you probably
need to pair it with a
higher level wrapper and
if you want dynamic graphs,
you're maybe out of luck.
Some of the code ends up
looking a little bit uglier
in my opinion, but maybe that's
kind of a cosmetic detail
and it doesn't really matter that much.
I personally think PyTorch
is really great for research.
If you're focused on just
writing research code,
I think PyTorch is a great choice.
But it's a bit newer, has
less community support,
less code out there, so it
could be a bit of an adventure.
If you want more of a well
trodden path, TensorFlow
might be a better choice.
If you're interested in
production deployment,
you should probably look at
Caffe, Caffe 2 or TensorFlow.
And if you're really focused
on mobile deployment,
I think TensorFlow and Caffe
2 both have some built in
support for that.
So it's kind of unfortunately,
there's not just like
one global best framework,
it kind of depends
on what you're actually trying to do,
what applications you anticipate
but theses are kind of
my general advice on those things.
So next time we'll talk
about some case studies
about various CNN architectures.
