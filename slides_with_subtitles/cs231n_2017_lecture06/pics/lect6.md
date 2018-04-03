﻿
<img src="./pics/cs231n_2017_lecture6-01.jpg" width="700">
Okay, let's get started.
Alright, so welcome to lecture five.
Today we're going to be getting
to the title of the class,
Convolutional Neural Networks.
Okay, so a couple of
administrative details
before we get started.
Assignment one is due Thursday,
April 20, 11:59 p.m. on Canvas.
We're also going to be releasing
assignment two on Thursday.
Okay, so a quick review of last time.
We talked about neural
networks, and how we had
the running example of
the linear score function
that we talked about through
the first few lectures.
And then we turned this
into a neural network
by stacking these linear
layers on top of each other
with non-linearities in between.
And we also saw that
this could help address
the mode problem where
we are able to learn
intermediate templates
that are looking for,
for example, different
types of cars, right.
A red car versus a yellow car and so on.
And to combine these
together to come up with
the final score function for a class.
Okay, so today we're going to talk about
convolutional neural networks,
which is basically the same sort of idea,
but now we're going to
learn convolutional layers
that reason on top of basically explicitly
trying to maintain spatial structure.
So, let's first talk a little bit about
the history of neural
networks, and then also
how convolutional neural
networks were developed.
So we can go all the way back
to 1957 with Frank Rosenblatt,
who developed the Mark
I Perceptron machine,
which was the first
implementation of an algorithm
called the perceptron, which
had sort of the similar idea
of getting score functions,
right, using some,
you know, W times X plus a bias.
But here the outputs are going
to be either one or a zero.
And then in this case
we have an update rule,
so an update rule for our weights, W,
which also look kind of similar
to the type of update rule
that we're also seeing in
backprop, but in this case
there was no principled
backpropagation technique yet,
we just sort of took the
weights and adjusted them
in the direction towards
the target that we wanted.
So in 1960, we had Widrow and Hoff,
who developed Adaline and
Madaline, which was the first time
that we were able to
get, to start to stack
these linear layers into
multilayer perceptron networks.
And so this is starting to now
look kind of like this idea
of neural network layers, but
we still didn't have backprop
or any sort of principled
way to train this.
And so the first time
backprop was really introduced
was in 1986 with Rumelhart.
And so here we can start
seeing, you know, these kinds of
equations with the chain
rule and the update rules
that we're starting to
get familiar with, right,
and so this is the first time we started
to have a principled way to train
these kinds of network architectures.
And so after that, you know,
it still wasn't able to scale
to very large neural networks,
and so there was sort of
a period in which there wasn't a whole lot
of new things happening
here, or a lot of popular use
of these kinds of networks.
And so this really started
being reinvigorated
around the 2000s, so in
2006, there was this paper
by Geoff Hinton and Ruslan Salakhutdinov,
which basically showed that we could train
a deep neural network,
and show that we could
do this effectively.
But it was still not quite
the sort of modern iteration
of neural networks.
It required really careful initialization
in order to be able to do backprop,
and so what they had
here was they would have
this first pre-training
stage, where you model
each hidden layer through this kind of,
through a restricted Boltzmann machine,
and so you're going to get
some initialized weights
by training each of
these layers iteratively.
And so once you get all
of these hidden layers
you then use that to
initialize your, you know,
your full neural network,
and then from there
you do backprop and fine tuning of that.
And so when we really started
to get the first really strong
results using neural networks,
and what sort of really
sparked the whole craze
of starting to use these
kinds of networks really
widely was at around 2012,
where we had first the strongest results
using for speech recognition,
and so this is work out
of Geoff Hinton's lab
for acoustic modeling
and speech recognition.
And then for image recognition,
2012 was the landmark paper
from Alex Krizhevsky
in Geoff Hinton's lab,
which introduced the first
convolutional neural network
architecture that was able to do,
get really strong results
on ImageNet classification.
And so it took the ImageNet,
image classification benchmark,
and was able to dramatically reduce
the error on that benchmark.
And so since then, you
know, ConvNets have gotten
really widely used in all
kinds of applications.
So now let's step back and
take a look at what gave rise
to convolutional neural
networks specifically.
And so we can go back to the 1950s,
where Hubel and Wiesel did
a series of experiments
trying to understand how neurons
in the visual cortex worked,
and they studied this
specifically for cats.
And so we talked a little bit
about this in lecture one,
but basically in these
experiments they put electrodes
in the cat, into the cat brain,
and they gave the cat
different visual stimulus.
Right, and so, things like, you know,
different kinds of edges, oriented edges,
different sorts of
shapes, and they measured
the response of the
neurons to these stimuli.
And so there were a couple
of important conclusions
that they were able to
make, and observations.
And so the first thing
found that, you know,
there's sort of this topographical
mapping in the cortex.
So nearby cells in the
cortex also represent
nearby regions in the visual field.
And so you can see for
example, on the right here
where if you take kind
of the spatial mapping
and map this onto a visual cortex
there's more peripheral
regions are these blue areas,
you know, farther away from the center.
And so they also discovered
that these neurons
had a hierarchical organization.
And so if you look at different
types of visual stimuli
they were able to find
that at the earliest layers
retinal ganglion cells
were responsive to things
that looked kind of like
circular regions of spots.
And then on top of that
there are simple cells,
and these simple cells are
responsive to oriented edges,
so different orientation
of the light stimulus.
And then going further,
they discover that these
were then connected to more complex cells,
which were responsive to
both light orientation
as well as movement, and so on.
And you get, you know,
increasing complexity,
for example, hypercomplex
cells are now responsive
to movement with kind
of an endpoint, right,
and so now you're starting
to get the idea of corners
and then blobs and so on.
And so
then in 1980, the neocognitron
was the first example
of a network architecture, a model,
that had this idea of
simple and complex cells
that Hubel and Wiesel had discovered.
And in this case Fukushima put these into
these alternating layers of
simple and complex cells,
where you had these simple cells
that had modifiable parameters,
and then complex cells
on top of these that
performed a sort of pooling
so that it was invariant to, you know,
different minor modifications
from the simple cells.
And so this is work that
was in the 1980s, right,
and so by 1998 Yann LeCun
basically showed the first example
of applying backpropagation
and gradient-based learning
to train convolutional neural networks
that did really well on
document recognition.
And specifically they
were able to do a good job
of recognizing digits of zip codes.
And so these were then used pretty widely
for zip code recognition
in the postal service.
But beyond that it
wasn't able to scale yet
to more challenging and
complex data, right,
digits are still fairly simple
and a limited set to recognize.
And so this is where
Alex Krizhevsky, in 2012,
gave the modern incarnation of
convolutional neural networks
and his network we sort of
colloquially call AlexNet.
But this network really
didn't look so much different
than the convolutional neural networks
that Yann LeCun was dealing with.
They're now, you know,
they were scaled now
to be larger and deeper and able to,
the most important parts
were that they were now able
to take advantage of
the large amount of data
that's now available, in web
images, in ImageNet data set.
As well as take advantage
of the parallel computing power in GPUs.
And so we'll talk more about that later.
But fast forwarding
today, so now, you know,
ConvNets are used everywhere.
And so we have the initial
classification results
on ImageNet from Alex Krizhevsky.
This is able to do a really
good job of image retrieval.
You can see that when we're
trying to retrieve a flower
for example, the features that are learned
are really powerful for
doing similarity matching.
We also have ConvNets that
are used for detection.
So we're able to do a really
good job of localizing
where in an image is, for
example, a bus, or a boat,
and so on, and draw precise
bounding boxes around that.
We're able to go even deeper
beyond that to do segmentation,
right, and so these are now richer tasks
where we're not looking
for just the bounding box
but we're actually going
to label every pixel
in the outline of, you know,
trees, and people, and so on.
And these kind of algorithms are used in,
for example, self-driving cars,
and a lot of this is powered
by GPUs as I mentioned earlier,
that's able to do parallel processing
and able to efficiently
train and run these ConvNets.
And so we have modern
powerful GPUs as well as ones
that work in embedded
systems, for example,
that you would use in a self-driving car.
So we can also look at some
of the other applications
that ConvNets are used for.
So, face-recognition, right,
we can put an input image
of a face and get out a
likelihood of who this person is.
ConvNets are applied to video,
and so this is an example
of a video network that
looks at both images
as well as temporal information,
and from there is able to classify videos.
We're also able to do pose recognition.
Being able to recognize, you know,
shoulders, elbows, and different joints.
And so here are some images
of our fabulous TA, Lane,
in various kinds of pretty
non-standard human poses.
But ConvNets are able
to do a pretty good job
of pose recognition these days.
They're also used in game playing.
So some of the work in
reinforcement learning,
deeper enforcement learning
that you may have seen,
playing Atari games, and Go, and so on,
and ConvNets are an important
part of all of these.
Some other applications,
so they're being used for
interpretation and
diagnosis of medical images,
for classification of galaxies,
for street sign recognition.
There's also whale recognition,
this is from a recent Kaggle Challenge.
We also have examples of
looking at aerial maps
and being able to draw
out where are the streets
on these maps, where are buildings,
and being able to segment all of these.
And then beyond recognition
of classification detection,
these types of tasks, we also have tasks
like image captioning,
where given an image,
we want to write a sentence description
about what's in the image.
And so this is something
that we'll go into
a little bit later in the class.
And we also have, you know,
really, really fancy and cool
kind of artwork that we can
do using neural networks.
And so on the left is an
example of a deep dream,
where we're able to take
images and kind of hallucinate
different kinds of objects
and concepts in the image.
There's also neural style type
work, where we take an image
and we're able to re-render this image
using a style of a particular
artist and artwork, right.
And so here we can take, for
example, Van Gogh on the right,
Starry Night, and use that to redraw
our original image using that style.
And Justin has done a lot of work in this
and so if you guys are interested,
these are images produced
by some of his code
and you guys should talk
to him more about it.
Okay, so basically, you know,
this is just a small sample
of where ConvNets are being used today.
But there's really a huge amount
that can be done with this,
right, and so, you know,
for you guys' projects,
sort of, you know, let
your imagination go wild
and we're excited to see
what sorts of applications
you can come up with.
So today we're going to talk about
how convolutional neural networks work.
And again, same as with neural
networks, we're going to first
talk about how they work
from a functional perspective
without any of the brain analogies.
And then we'll talk briefly
about some of these connections.
Okay, so, last lecture, we talked about
this idea of a fully connected layer.
And how, you know, for
a fully connected layer
what we're doing is we operate
on top of these vectors,
right, and so let's say we
have, you know, an image,
a 3D image, 32 by 32 by three,
so some of the images that we
were looking at previously.
We'll take that, we'll stretch
all of the pixels out, right,
and then we have this
3072 dimensional vector,
for example in this case.
And then we have these weights, right,
so we're going to multiply
this by a weight matrix.
And so here for example our W
we're going to say is 10 by 3072.
And then we're going
to get the activations,
the output of this layer,
right, and so in this case,
we take each of our 10 rows
and we do this dot product
with 3072 dimensional input.
And from there we get this one number
that's kind of the value of that neuron.
And so in this case we're going to have
10 of these neuron outputs.
And so a convolutional
layer, so the main difference
between this and the fully connected layer
that we've been talking about
is that here we want to
preserve spatial structure.
And so taking this 32 by 32 by three image
that we had earlier, instead
of stretching this all out
into one long vector, we're
now going to keep the structure
of this image, right, this
three dimensional input.
And then what we're going to do is
our weights are going to
be these small filters,
so in this case for example, a
five by five by three filter,
and we're going to take this filter
and we're going to slide
it over the image spatially
and compute dot products
at every spatial location.
And so we're going to go into
detail of exactly how this works.
So, our filters, first of all,
always extend the full
depth of the input volume.
And so they're going to be
just a smaller spatial area,
so in this case five by five, right,
instead of our full 32
by 32 spatial input,
but they're always going to go
through the full depth, right,
so here we're going to
take five by five by three.
And then we're going to take this filter
and at a given spatial location
we're going to do a dot product
between this filter and
then a chunk of a image.
So we're just going to overlay this filter
on top of a spatial location in the image,
right, and then do the dot product,
the multiplication of each
element of that filter
with each corresponding element
in that spatial location
that we've just plopped it on top of.
And then this is going
to give us a dot product.
So in this case, we have
five times five times three,
this is the number of multiplications
that we're going to do,
right, plus the bias term.
And so this is basically
taking our filter W
and basically doing W transpose
times X and plus bias.
So is that clear how this works?
Yeah, question.
[faint speaking]
Yeah, so the question is,
when we do the dot product
do we turn the five by five
by three into one vector?
Yeah, in essence that's what you're doing.
You can, I mean, you
can think of it as just
plopping it on and doing the
element-wise multiplication
at each location, but this is
going to give you the same result
as if you stretched out
the filter at that point,
stretched out the input
volume that it's laid over,
and then took the dot product,
and that's what's written
here, yeah, question.
[faint speaking]
Oh, this is, so the question is,
any intuition for why
this is a W transpose?
And this was just, not really,
this is just the notation
that we have here
to make the math work
out as a dot product.
So it just depends on whether,
how you're representing W
and whether in this case
if we look at the W matrix
this happens to be each column
and so we're just taking
the transpose to get a row out of it.
But there's no intuition here,
we're just taking the filters of W
and we're stretching it
out into a one D vector,
and in order for it to be a dot product
it has to be like a one
by, one by N vector.
[faint speaking]
Okay, so the question is,
is W here not five by five
by three, it's one by 75.
So that's the case, right, if we're going
to do this dot product
of W transpose times X,
we have to stretch it out first
before we do the dot product.
So we take the five by five by three,
and we just take all these values
and stretch it out into a long vector.
And so again, similar
to the other question,
the actual operation that we're doing here
is plopping our filter on top of
a spatial location in the image
and multiplying all of the
corresponding values together,
but in order just to make it
kind of an easy expression
similar to what we've seen before
we can also just stretch
each of these out,
make sure that dimensions
are transposed correctly
so that it works out as a dot product.
Yeah, question.
[faint speaking]
Okay, the question is,
how do we slide the filter over the image.
We'll go into that next, yes.
[faint speaking]
Okay, so the question is,
should we rotate the kernel
by 180 degrees to better
match the convolution,
the definition of a convolution.
And so the answer is that
we'll also show the equation
for this later, but
we're using convolution
as kind of a looser definition
of what's happening.
So for people from signal processing,
what we are actually technically doing,
if you want to call this a convolution,
is we're convolving with the
flipped version of the filter.
But for the most part, we
just don't worry about this
and we just, yeah, do this operation
and it's like a convolution in spirit.
Okay, so...
Okay, so we had a question
earlier, how do we, you know,
slide this over all the spatial locations.
Right, so what we're going to do is
we're going to take this
filter, we're going to start
at the upper left-hand
corner and basically center
our filter on top of every
pixel in this input volume.
And at every position, we're
going to do this dot product
and this will produce one value
in our output activation map.
And so then we're going
to just slide this around.
The simplest version
is just at every pixel
we're going to do this
operation and fill in
the corresponding point
in our output activation.
You can see here that the
dimensions are not exactly
what would happen, right,
if you're going to do this.
I had 32 by 32 in the input
and I'm having 28 by 28 in the output,
and so we'll go into
examples later of the math
of exactly how this is going
to work out dimension-wise,
but basically you have a choice
of how you're going to slide this,
whether you go at every
pixel or whether you slide,
let's say, you know, two
input values over at a time,
two pixels over at a time,
and so you can get different size outputs
depending on how you choose to slide.
But you're basically doing this
operation in a grid fashion.
Okay, so what we just saw earlier,
this is taking one filter, sliding it over
all of the spatial locations in the image
and then we're going to get
this activation map out, right,
which is the value of that
filter at every spatial location.
And so when we're dealing
with a convolutional layer,
we want to work with
multiple filters, right,
because each filter is kind
of looking for a specific
type of template or concept
in the input volume.
And so we're going to have
a set of multiple filters,
and so here I'm going
to take a second filter,
this green filter, which is
again five by five by three,
I'm going to slide this over
all of the spatial locations
in my input volume, and
then I'm going to get out
this second green activation
map also of the same size.
And we can do this for as many filters
as we want to have in this layer.
So for example, if we have six filters,
six of these five by five filters,
then we're going to get in
total six activation maps out.
All of, so we're going
to get this output volume
that's going to be
basically six by 28 by 28.
Right, and so a preview
of how we're going to use
these convolutional layers
in our convolutional network
is that our ConvNet is
basically going to be
a sequence of these convolutional layers
stacked on top of each other,
same way as what we had
with the simple linear layers
in their neural network.
And then we're going to intersperse these
with activation functions,
so for example, a ReLU
activation function.
Right, and so you're going to
get something like Conv, ReLU,
and usually also some pooling layers,
and then you're just going
to get a sequence of these
each creating an output
that's now going to be
the input to the next convolutional layer.
Okay, and so each of these
layers, as I said earlier,
has multiple filters, right, many filters.
And each of the filter is
producing an activation map.
And so when you look at
multiple of these layers
stacked together in a ConvNet,
what ends up happening
is you end up learning this
hierarching of filters,
where the filters at the
earlier layers usually represent
low-level features that
you're looking for.
So things kind of like edges, right.
And then at the mid-level,
you're going to get more
complex kinds of features,
so maybe it's looking more for things
like corners and blobs and so on.
And then at higher-level features,
you're going to get
things that are starting
to more resemble concepts than blobs.
And we'll go into more
detail later in the class
in how you can actually
visualize all these features
and try and interpret what your network,
what kinds of features
your network is learning.
But the important thing for
now is just to understand
that what these features end up being
when you have a whole stack of these,
is these types of simple
to more complex features.
[faint speaking]
Yeah.
Oh, okay.
Oh, okay, so the question
is, what's the intuition
for increasing the depth each time.
So here I had three filters
in the original layer
and then six filters in the next layer.
Right, and so this is
mostly a design choice.
You know, people in practice have found
certain types of these
configurations to work better.
And so later on we'll go into
case studies of different
kinds of convolutional
neural network architectures
and design choices for these
and why certain ones
work better than others.
But yeah, basically the choice of,
you're going to have many design choices
in a convolutional neural network,
the size of your filter, the stride,
how many filters you have,
and so we'll talk about
this all more later.
Question.
[faint speaking]
Yeah, so the question is,
as we're sliding this filter
over the image spatially it
looks like we're sampling
the edges and corners less
than the other locations.
Yeah, that's a really good point,
and we'll talk I think in a few slides
about how we try and compensate for that.
Okay, so each of these
convolutional layers
that we have stacked together,
we saw how we're starting
with more simpler features
and then aggregating these
into more complex features later on.
And so in practice this is compatible
with what Hubel and Wiesel
noticed in their experiments,
right, that we had these simple cells
at the earlier stages of processing,
followed by more complex cells later on.
And so even though we didn't explicitly
force our ConvNet to learn
these kinds of features,
in practice when you give it this type of
hierarchical structure and
train it using backpropagation,
these are the kinds of filters
that end up being learned.
[faint speaking]
Okay, so yeah, so the question is,
what are we seeing in
these visualizations.
And so, alright so, in
these visualizations, like,
if we look at this Conv1, the
first convolutional layer,
each of these grid, each part
of this grid is a one neuron.
And so what we've visualized here
is what the input looks
like that maximizes
the activation of that particular neuron.
So what sort of image you would get
that would give you the largest value,
make that neuron fire and
have the largest value.
And so the way we do this is basically
by doing backpropagation from
a particular neuron activation
and seeing what in the input will trigger,
will give you the highest
values of this neuron.
And this is something
that we'll talk about
in much more depth in a later lecture
about how we create all
of these visualizations.
But basically each element of these grids
is showing what in the
input would look like
that basically maximizes the
activation of the neuron.
So in a sense, what is
the neuron looking for?
Okay, so here is an example
of some of the activation maps
produced by each filter, right.
So we can visualize up here on the top
we have this whole row of
example five by five filters,
and so this is basically a real
case from a trained ConvNet
where each of these is
what a five by five filter
looks like, and then as we
convolve this over an image,
so in this case this I think
it's like a corner of a car,
the car light, what the
activation looks like.
Right, and so here for example,
if we look at this first
one, this red filter,
filter like with a red box around it,
we'll see that it's looking for,
the template looks like an
edge, right, an oriented edge.
And so if you slide it over the image,
it'll have a high value,
a more white value
where there are edges in
this type of orientation.
And so each of these activation
maps is kind of the output
of sliding one of these filters over
and where these filters
are causing, you know,
where this sort of template
is more present in the image.
And so the reason we call
these convolutional is because
this is related to the
convolution of two signals,
and so someone pointed out earlier
that this is basically this
convolution equation over here,
for people who have
seen convolutions before
in signal processing, and in practice
it's actually more like a correlation
where we're convolving
with the flipped version
of the filter, but this
is kind of a subtlety,
it's not really important for
the purposes of this class.
But basically if you're
writing out what you're doing,
it has an expression that
looks something like this,
which is the standard
definition of a convolution.
But this is basically
just taking a filter,
sliding it spatially over the image
and computing the dot
product at every location.
Okay, so you know, as I
had mentioned earlier,
like what our total
convolutional neural network
is going to look like is we're
going to have an input image,
and then we're going to pass it through
this sequence of layers, right,
where we're going to have a
convolutional layer first.
We usually have our
non-linear layer after that.
So ReLU is something
that's very commonly used
that we're going to talk about more later.
And then we have these Conv,
ReLU, Conv, ReLU layers,
and then once in a while
we'll use a pooling layer
that we'll talk about later as well
that basically downsamples the
size of our activation maps.
And then finally at the end
of this we'll take our last
convolutional layer output
and then we're going to use
a fully connected layer
that we've seen before,
connected to all of these
convolutional outputs,
and use that to get a final score function
basically like what we've
already been working with.
Okay, so now let's work out some examples
of how the spatial dimensions work out.
So let's take our 32 by 32
by three image as before,
right, and we have our five
by five by three filter
that we're going to slide over this image.
And we're going to see how
we're going to use that
to produce exactly this
28 by 28 activation map.
So let's assume that we actually
have a seven by seven input
just to be simpler, and let's assume
we have a three by three filter.
So what we're going to do is
we're going to take this filter,
plop it down in our
upper left-hand corner,
right, and we're going to
multiply, do the dot product,
multiply all these values
together to get our first value,
and this is going to go into
the upper left-hand value
of our activation map.
Right, and then what
we're going to do next
is we're just going to take this filter,
slide it one position to the right,
and then we're going to get
another value out from here.
And so we can continue with
this to have another value,
another, and in the end
what we're going to get
is a five by five output, right,
because what fit was
basically sliding this filter
a total of five spatial
locations horizontally
and five spatial locations vertically.
Okay, so as I said before
there's different kinds of
design choices that we can make.
Right, so previously I
slid it at every single
spatial location and the
interval at which I slide
I'm going to call the stride.
And so previously we
used the stride of one.
And so now let's see what happens
if we have a stride of two.
Right, so now we're going
to take our first location
the same as before, and
then we're going to skip
this time two pixels over
and we're going to get
our next value centered at this location.
Right, and so now if
we use a stride of two,
we have in total three
of these that can fit,
and so we're going to get
a three by three output.
Okay, and so what happens when
we have a stride of three,
what's the output size of this?
And so in this case, right, we have three,
we slide it over by three again,
and the problem is that here
it actually doesn't fit.
Right, so we slide it over by three
and now it doesn't fit
nicely within the image.
And so what we in practice we
just, it just doesn't work.
We don't do convolutions like this
because it's going to lead to
asymmetric outputs happening.
Right, and so just kind
of looking at the way
that we computed how many, what
the output size is going to be,
this actually can work into a nice formula
where we take our
dimension of our input N,
we have our filter size
F, we have our stride
at which we're sliding along,
and our final output size,
the spatial dimension of each output size
is going to be N minus F
divided by the stride plus one,
right, and you can kind of
see this as a, you know,
if I'm going to take my
filter, let's say I fill it in
at the very last possible
position that it can be in
and then take all the pixels before that,
how many instances of moving
by this stride can I fit in.
Right, and so that's how this
equation kind of works out.
And so as we saw before,
right, if we have N equal seven
and F equals three, if
we want a stride of one
we plug it into this
formula, we get five by five
as we had before, and the
same thing we had for two.
And with a stride of three,
this doesn't really work out.
And so in practice it's actually common
to zero pad the borders in order to make
the size work out to what we want it to.
And so this is kind of
related to a question earlier,
which is what do we do,
right, at the corners.
And so what in practice happens is
we're going to actually pad
our input image with zeros
and so now you're going to
be able to place a filter
centered at the upper
right-hand pixel location
of your actual input image.
Okay, so here's a question,
so who can tell me
if I have my same input, seven by seven,
three by three filter, stride one,
but now I pad with a one pixel border,
what's the size of my output going to be?
[faint speaking]
So, I heard some sixes, heard some sev,
so remember we have this
formula that we had before.
So if we plug in N is equal
to seven, F is equal to three,
right, and then our
stride is equal to one.
So what we actually get, so
actually this is giving us
seven, four, so seven
minus three is four,
divided by one plus one is five.
And so this is what we had before.
So we actually need to adjust
this formula a little bit,
right, so this was actually,
this formula is the case
where we don't have zero padded pixels.
But if we do pad it, then if
you now take your new output
and you slide it along,
you'll see that actually
seven of the filters fit,
so you get a seven by seven output.
And plugging in our
original formula, right,
so our N now is not seven, it's nine,
so if we go back here
we have N equals nine
minus a filter size of
three, which gives six.
Right, divided by our
stride, which is one,
and so still six, and then
plus one we get seven.
Right, and so once you've padded it
you want to incorporate this
padding into your formula.
Yes, question.
[faint speaking]
Seven, okay, so the question is,
what's the actual output of the size,
is it seven by seven or
seven by seven by three?
The output is going to be seven by seven
by the number of filters that you have.
So remember each filter is
going to do a dot product
through the entire depth
of your input volume.
But then that's going to
produce one number, right,
so each filter is, let's
see if we can go back here.
Each filter is producing
a one by seven by seven
in this case activation map
output, and so the depth
is going to be the number
of filters that we have.
[faint speaking]
Sorry, let me just, one second go back.
Okay, can you repeat your question again?
[muffled speaking]
Okay, so the question is, how
does this connect to before
when we had a 32 by 32
by three input, right.
So our input had depth
and here in this example
I'm showing a 2D example with no depth.
And so yeah, I'm showing
this for simplicity
but in practice you're going to take your,
you're going to multiply
throughout the entire depth
as we had before, so you're going to,
your filter is going to be
in this case a three be three
spatial filter by whatever
input depth that you had.
So three by three by three in this case.
Yeah, everything else stays the same.
Yes, question.
[muffled speaking]
Yeah, so the question
is, does the zero padding
add some sort of extraneous
features at the corners?
And yeah, so I mean, we're
doing our best to still,
get some value and do, like,
process that region of the image,
and so zero padding is
kind of one way to do this,
where I guess we can, we are detecting
part of this template in this region.
There's also other ways
to do this that, you know,
you can try and like,
mirror the values here
or extend them, and so it
doesn't have to be zero padding,
but in practice this is one
thing that works reasonably.
And so, yeah, so there is a
little bit of kind of artifacts
at the edge and we sort of just,
you do your best to deal with it.
And in practice this works reasonably.
I think there was another question.
Yeah, question.
[faint speaking]
So if we have non-square
images, do we ever use a stride
that's different
horizontally and vertically?
So, I mean, there's nothing
stopping you from doing that,
you could, but in practice we just usually
take the same stride, we
usually operate square regions
and we just, yeah we usually just
take the same stride everywhere
and it's sort of like,
in a sense it's a little bit like,
it's a little bit like the
resolution at which you're,
you know, looking at this image,
and so usually there's kind
of, you might want to match
sort of your horizontal
and vertical resolutions.
But, yeah, so in practice you could
but really people don't do that.
Okay, another question.
[faint speaking]
So the question is, why
do we do zero padding?
So the way we do zero padding
is to maintain the same
input size as we had before.
Right, so we started with seven by seven,
and if we looked at just
starting your filter
from the upper left-hand
corner, filling everything in,
right, then we get a smaller size output,
but we would like to maintain
our full size output.
Okay, so,
yeah, so we saw how padding
can basically help you
maintain the size of the
output that you want,
as well as apply your filter at these,
like, corner regions and edge regions.
And so in general in terms of choosing,
you know, your stride, your
filter, your filter size,
your stride size, zero
padding, what's common to see
is filters of size three
by three, five by five,
seven by seven, these are
pretty common filter sizes.
And so each of these, for three by three
you will want to zero pad with one
in order to maintain
the same spatial size.
If you're going to do five by five,
you can work out the math,
but it's going to come out
to you want to zero pad by two.
And then for seven you
want to zero pad by three.
Okay, and so again you
know, the motivation
for doing this type of zero padding
and trying to maintain
the input size, right,
so we kind of alluded to this before,
but if you have multiple of
these layers stacked together...
So if you have multiple of
these layers stacked together
you'll see that, you know,
if we don't do this kind of
zero padding, or any kind of padding,
we're going to really
quickly shrink the size
of the outputs that we have.
Right, and so this is not
something that we want.
Like, you can imagine if you
have a pretty deep network
then very quickly your, the
size of your activation maps
is going to shrink to
something very small.
And this is bad both because
we're kind of losing out
on some of this information, right,
now you're using a much
smaller number of values
in order to represent your original image,
so you don't want that.
And then at the same time also as
we talked about this earlier, your also kind of
losing sort of some of
this edge information,
corner information that each time
we're losing out and
shrinking that further.
Okay, so let's go through
a couple more examples
of computing some of these sizes.
So let's say that we have an input volume
which is 32 by 32 by three.
And here we have 10 five by five filters.
Let's use stride one and pad two.
And so who can tell me
what's the output volume size of this?
So you can think about
the formula earlier.
Sorry, what was it?
[faint speaking]
32 by 32 by 10, yes that's correct.
And so the way we can see this, right,
is so we have our input size, F is 32.
Then in this case we want to augment it
by the padding that we added onto this.
So we padded it two in
each dimension, right,
so we're actually going to get,
total width and total height's
going to be 32 plus four on each side.
And then minus our filter size five,
divided by one plus one and we get 32.
So our output is going to
be 32 by 32 for each filter.
And then we have 10 filters total,
so we have 10 of these activation maps,
and our total output volume
is going to be 32 by 32 by 10.
Okay, next question,
so what's the number of
parameters in this layer?
So remember we have 10
five by five filters.
[faint speaking]
I kind of heard something,
but it was quiet.
Can you guys speak up?
250, okay so I heard 250, which is close,
but remember that we're
also, our input volume,
each of these filters
goes through by depth.
So maybe this wasn't clearly written here
because each of the filters
is five by five spatially,
but implicitly we also have
the depth in here, right.
It's going to go through the whole volume.
So I heard, yeah, 750 I heard.
Almost there, this is
kind of a trick question
'cause also remember
we usually always have
a bias term, right, so
in practice each filter
has five by five by three
weights, plus our one bias term,
we have 76 parameters per filter,
and then we have 10 of these total,
and so there's 760 total parameters.
Okay, and so here's just a summary
of the convolutional layer
that you guys can read
a little bit more carefully later on.
But we have our input volume
of a certain dimension,
we have all of these choice,
we have our filters, right,
where we have number of
filters, the filter size,
the stride of the size,
the amount of zero padding,
and you basically can use all of these,
go through the computations
that we talked about earlier
in order to find out what
your output volume is actually
going to be and how many total
parameters that you have.
And so some common settings of this.
You know, we talked earlier
about common filter sizes
of three by three, five by five.
Stride is usually one
and two is pretty common.
And then your padding P is
going to be whatever fits,
like, whatever will
preserve your spatial extent
is what's common.
And then the total number of filters K,
usually we use powers of two
just to be nice, so, you know,
32, 64, 128 and so on, 512,
these are pretty common
numbers that you'll see.
And just as an aside,
we can also do a one by one convolution,
this still makes perfect sense where
given a one by one convolution
we still slide it over
each spatial extent,
but now, you know, the spatial region
is not really five by five
it's just kind of the
trivial case of one by one,
but we are still having this filter
go through the entire depth.
Right, so this is going
to be a dot product
through the entire depth
of your input volume.
And so the output here, right,
if we have an input volume
of 56 by 56 by 64 depth and
we're going to do one by one
convolution with 32 filters,
then our output is going to be
56 by 56 by our number of filters, 32.
Okay, and so here's an example
of a convolutional layer
in TORCH, a deep learning framework.
And so you'll see that,
you know, last lecture
we talked about how you can go into these
deep learning frameworks,
you can see these definitions
of each layer, right,
where they have kind of
the forward pass and the backward pass
implemented for each layer.
And so you'll see convolutions,
spatial convolution is going
to be just one of these,
and then the arguments
that it's going to take
are going to be all of these
design choices of, you know,
I mean, I guess your
input and output sizes,
but also your choices of
like your kernel width,
your kernel size, padding,
and these kinds of things.
Right, and so if we look at
another framework, Caffe,
you'll see something very similar,
where again now when you're
defining your network
you define networks in Caffe
using this kind of, you know,
proto text file where you're specifying
each of your design choices for your layer
and you can see for a convolutional layer
will say things like, you
know, the number of outputs
that we have, this is going
to be the number of filters
for Caffe, as well as the kernel
size and stride and so on.
Okay, and so I guess before I go on,
any questions about convolution,
how the convolution operation works?
Yes, question.
[faint speaking]
Yeah, so the question is,
what's the intuition behind
how you choose your stride.
And so at one sense it's
kind of the resolution
at which you slide it on, and
usually the reason behind this
is because when we have a larger stride
what we end up getting as the output
is a down sampled image, right,
and so what this downsampled
image lets us have is both,
it's a way, it's kind of
like pooling in a sense
but it's just a different
and sometimes works better
way of doing pooling is one
of the intuitions behind this,
'cause you get the same effect
of downsampling your image,
and then also as you're doing
this you're reducing the size
of the activation maps
that you're dealing with
at each layer, right, and so
this also affects later on
the total number of
parameters that you have
because for example at the
end of all your Conv layers,
now you might put on fully
connected layers on top,
for example, and now the
fully connected layer's
going to be connected to every value
of your convolutional output, right,
and so a smaller one will
give you smaller number
of parameters, and so now
you can get into, like,
basically thinking about
trade offs of, you know,
number of parameters you
have, the size of your model,
overfitting, things
like that, and so yeah,
these are kind of some of the things
that you want to think about
with choosing your stride.
Okay, so now if we look a
little bit at kind of the,
you know, brain neuron view
of a convolutional layer,
similar to what we
looked at for the neurons
in the last lecture.
So what we have is that
at every spatial location,
we take a dot product between a filter
and a specific part of the image, right,
and we get one number out from here.
And so this is the same idea
of doing these types
of dot products, right,
taking your input, weighting
it by these Ws, right,
values of your filter, these
weights that are the synapses,
and getting a value out.
But the main difference
here is just that now
your neuron has local connectivity.
So instead of being connected
to the entire input,
it's just looking at a local
region spatially of your image.
And so this looks at a local region
and then now you're going
to get kind of, you know,
this, how much this
neuron is being triggered
at every spatial location in your image.
Right, so now you preserve
the spatial structure
and you can say, you
know, be able to reason
on top of these kinds of
activation maps in later layers.
And just a little bit of terminology,
again for, you know, we have
this five by five filter,
we can also call this a
five by five receptive field
for the neuron, because this is,
the receptive field is
basically the, you know,
input field that this field of vision
that this neuron is receiving, right,
and so that's just another common term
that you'll hear for this.
And then again remember each
of these five by five filters
we're sliding them over
the spatial locations
but they're the same set of weights,
they share the same parameters.
Okay, and so, you know, as we talked about
what we're going to get at this output
is going to be this volume, right,
where spatially we have,
you know, let's say 28 by 28
and then our number of
filters is the depth.
And so for example with five filters,
what we're going to
get out is this 3D grid
that's 28 by 28 by five.
And so if you look at the filters across
in one spatial location
of the activation volume
and going through depth
these five neurons,
all of these neurons,
basically the way you can interpret this
is they're all looking at the same region
in the input volume,
but they're just looking
for different things, right.
So they're different filters
applied to the same spatial
location in the image.
And so just a reminder
again kind of comparing
with the fully connected layer
that we talked about earlier.
In that case, right, if we
look at each of the neurons
in our activation or
output, each of the neurons
was connected to the
entire stretched out input,
so it looked at the
entire full input volume,
compared to now where each one
just looks at this local spatial region.
Question.
[muffled talking]
Okay, so the question
is, within a given layer,
are the filters completely symmetric?
So what do you mean by
symmetric exactly, I guess?
Right, so okay, so the
filters, are the filters doing,
they're doing the same dimension,
the same calculation, yes.
Okay, so is there anything different
other than they have the
same parameter values?
No, so you're exactly right,
we're just taking a filter
with a given set of, you know,
five by five by three parameter values,
and we just slide this
in exactly the same way
over the entire input volume
to get an activation map.
Okay, so you know, we've
gone into a lot of detail
in what these convolutional
layers look like,
and so now I'm just going to go briefly
through the other layers that we have
that form this entire
convolutional network.
Right, so remember again,
we have convolutional layers
interspersed with pooling
layers once in a while
as well as these non-linearities.
Okay, so what the pooling layers do
is that they make the representations
smaller and more manageable, right,
so we talked about this earlier with
someone asked a question of
why we would want to make
the representation smaller.
And so this is again for it to have fewer,
it effects the number of
parameters that you have at the end
as well as basically does some, you know,
invariance over a given region.
And so what the pooling layer does
is it does exactly just downsamples,
and it takes your input
volume, so for example,
224 by 224 by 64, and
spatially downsamples this.
So in the end you'll get out 112 by 112.
And it's important to note
this doesn't do anything
in the depth, right, we're
only pooling spatially.
So the number of, your input depth
is going to be the same
as your output depth.
And so, for example, a common
way to do this is max pooling.
So in this case our pooling
layer also has a filter size
and this filter size is
going to be the region
at which we pool over,
right, so in this case
if we have two by two filters,
we're going to slide this,
and so, here, we also have
stride two in this case,
so we're going to take this filter
and we're going to slide
it along our input volume
in exactly the same way
as we did for convolution.
But here instead of
doing these dot products,
we just take the maximum value
of the input volume in that region.
Right, so here if we
look at the red values,
the value of that will
be six is the largest.
If we look at the greens
it's going to give an eight,
and then we have a three and a four.
Yes, question.
[muffled speaking]
Yeah, so the question is, is
it typical to set up the stride
so that there isn't an overlap?
And yeah, so for the pooling layers it is,
I think the more common thing to do
is to have them not have any overlap,
and I guess the way you
can think about this
is basically we just want to downsample
and so it makes sense to
kind of look at this region
and just get one value
to represent this region
and then just look at the
next region and so on.
Yeah, question.
[faint speaking]
Okay, so the question
is, why is max pooling
better than just taking the,
doing something like average pooling?
Yes, that's a good point,
like, average pooling
is also something that you can do,
and intuition behind why
max pooling is commonly used
is that it can have
this interpretation of,
you know, if this is, these
are activations of my neurons,
right, and so each value is kind of
how much this neuron
fired in this location,
how much this filter
fired in this location.
And so you can think of
max pooling as saying,
you know, giving a signal of
how much did this filter fire
at any location in this image.
Right, and if we're
thinking about detecting,
you know, doing recognition,
this might make some intuitive
sense where you're saying,
well, you know, whether a
light or whether some aspect
of your image that you're looking for,
whether it happens anywhere in this region
we want to fire at with a high value.
Question.
[muffled speaking]
Yeah, so the question is,
since pooling and stride
both have the same effect of downsampling,
can you just use stride
instead of pooling and so on?
Yeah, and so in practice I think
looking at more recent
neural network architectures
people have begun to use stride more
in order to do the downsampling
instead of just pooling.
And I think this gets into
things like, you know,
also like fractional strides
and things that you can do.
But in practice this in a
sense maybe has a little bit
better way to get better
results using that, so.
Yeah, so I think using
stride is definitely,
you can do it and people are doing it.
Okay, so let's see, where were we.
Okay, so yeah, so with
these pooling layers,
so again, there's right, some
design choices that you make,
you take this input volume of W by H by D,
and then you're going to
set your hyperparameters
for design choices of your filter size
or the spatial extent over
which you are pooling,
as well as your stride, and
then you can again compute
your output volume using the
same equation that you used
earlier for convolution, it
still applies here, right,
so we still have our W total extent
minus filter size divided
by stride plus one.
Okay, and so just one other thing to note,
it's also, typically people
don't really use zero padding
for the pooling layers
because you're just trying
to do a direct downsampling, right,
so there isn't this problem of like,
applying a filter at the corner
and having some part of the
filter go off your input volume.
And so for pooling we don't
usually have to worry about this
and we just directly downsample.
And so some common settings
for the pooling layer
is a filter size of two by
two or three by three strides.
Two by two, you know, you can have,
also you can still have
pooling of two by two
even with a filter size of three by three,
I think someone asked that earlier,
but in practice it's pretty
common just to have two by two.
Okay, so now we've talked about
these convolutional layers,
the ReLU layers were the
same as what we had before
with the, you know, just
the base neural network
that we talked about last lecture.
So we intersperse these and
then we have a pooling layer
every once in a while when we
feel like downsampling, right.
And then the last thing is that at the end
we want to have a fully connected layer.
And so this will be just exactly the same
as the fully connected layers
that you've seen before.
So in this case now what we do
is we take the convolutional
network output,
at the last layer we have some volume,
so we're going to have width
by height by some depth,
and we just take all of these
and we essentially just
stretch these out, right.
And so now we're going
to get the same kind of,
you know, basically 1D
input that we're used to
for a vanilla neural network,
and then we're going to apply
this fully connected layer on top,
so now we're going to have connections
to every one of these
convolutional map outputs.
And so what you can think
of this is basically,
now instead of preserving, you know,
before we were preserving
spatial structure,
right, and so but at the
last layer at the end,
we want to aggregate all of this together
and we want to reason basically on top of
all of this as we had before.
And so what you get from that is just our
score outputs as we had earlier.
Okay, so--
- [Student] This is
sort of a silly question
about this visual.
Like what are the 16 pixels
that are on the far right,
like what should be interpreting those as?
- Okay, so the question
is, what are the 16 pixels
that are on the far
right, do you mean the--
- [Student] Like that column of--
- [Instructor] Oh, each column.
- [Student] The column
on the far right, yeah.
- [Instructor] The green
ones or the black ones?
- [Student] The ones labeled pool.
- The one with hold on, pool.
Oh, okay, yeah, so the question is
how do we interpret this column,
right, for example at pool.
And so what we're showing
here is each of these columns
is the output activation maps, right,
the output from one of these layers.
And so starting from the
beginning, we have our car,
after the convolutional layer
we now have these activation
maps of each of the filters
slid spatially over the input image.
Then we pass that through a ReLU,
so you can see the values
coming out from there.
And then going all the way over,
and so what you get for the pooling layer
is that it's really just taking
the output of the ReLU layer
that came just before it
and then it's pooling it.
So it's going to downsample it,
right, and then it's going to take
the max value in each filter location.
And so now if you look at
this pool layer output,
like, for example, the last
one that you were mentioning,
it looks the same as this ReLU output
except that it's downsampled
and that it has this kind of
max value at every spatial location
and so that's the minor difference
that you'll see between those two.
[distant speaking]
So the question is, now this looks like
just a very small amount
of information, right,
so how can it know to
classify it from here?
And so the way that you
should think about this
is that each of these values
inside one of these pool
outputs is actually,
it's the accumulation of all
the processing that you've done
throughout this entire network, right.
So it's at the very top of your hierarchy,
and so each actually represents
kind of a higher level concept.
So we saw before, you know,
for example, Hubel and Wiesel
and building up these
hierarchical filters,
where at the bottom level
we're looking for edges, right,
or things like very simple
structures, like edges.
And so after your convolutional layer
the outputs that you see
here in this first column
is basically how much do
specific, for example, edges,
fire at different locations in the image.
But then as you go through
you're going to get more complex,
it's looking for more
complex things, right,
and so the next convolutional layer
is going to fire at how much, you know,
let's say certain kinds of
corners show up in the image,
right, because it's reasoning.
Its input is not the original image,
its input is the output, it's
already the edge maps, right,
so it's reasoning on top of edge maps,
and so that allows it to get more complex,
detect more complex things.
And so by the time you get all the way up
to this last pooling layer,
each value is representing
how much a relatively complex
sort of template is firing.
Right, and so because of
that now you can just have
a fully connected layer,
you're just aggregating
all of this information together to get,
you know, a score for your class.
So each of these values is how much
a pretty complicated
complex concept is firing.
Question.
[faint speaking]
So the question is, when
do you know you've done
enough pooling to do the classification?
And the answer is you just try and see.
So in practice, you know,
these are all design choices
and you can think about this
a little bit intuitively,
right, like you want to pool
but if you pool too much
you're going to have very few values
representing your entire image and so on,
so it's just kind of a trade off.
Something reasonable
versus people have tried
a lot of different configurations
so you'll probably cross validate, right,
and try over different pooling sizes,
different filter sizes,
different number of layers,
and see what works best for
your problem because yeah,
like every problem with
different data is going to,
you know, different set of these sorts
of hyperparameters might work best.
Okay, so last thing, just
wanted to point you guys
to this demo of training a ConvNet,
which was created by Andre Karpathy,
the originator of this class.
And so he wrote up this demo
where you can basically
train a ConvNet on CIFAR-10,
the dataset that we've seen
before, right, with 10 classes.
And what's nice about
this demo is you can,
it basically plots for you
what each of these filters
look like, what the
activation maps look like.
So some of the images I showed earlier
were taken from this demo.
And so you can go try it
out, play around with it,
and you know, just go through
and try and get a sense
for what these activation maps look like.
And just one thing to note,
usually the first layer
activation maps are,
you can interpret them, right,
because they're operating
directly on the input image
so you can see what these templates mean.
As you get to higher level layers
it starts getting really hard,
like how do you actually
interpret what do these mean.
So for the most part it's
just hard to interpret
so you shouldn't, you know, don't worry
if you can't really make
sense of what's going on.
But it's still nice just
to see the entire flow
and what outputs are coming out.
Okay, so in summary, so
today we talked about
how convolutional neural networks work,
how they're basically stacks
of these convolutional and pooling layers
followed by fully connected
layers at the end.
There's been a trend towards
having smaller filters
and deeper architectures,
so we'll talk more
about case studies for
some of these later on.
There's also been a trend
towards getting rid of these
pooling and fully
connected layers entirely.
So just keeping these, just
having, you know, Conv layers,
very deep networks of Conv layers,
so again we'll discuss
all of this later on.
And then typical architectures
again look like this,
you know, as we had earlier.
Conv, ReLU for some N number of steps
followed by a pool every once in a while,
this whole thing repeated
some number of times,
and then followed by fully
connected ReLU layers
that we saw earlier, you know, one or two
or just a few of these,
and then a softmax at the
end for your class scores.
And so, you know, some typical values
you might have N up to five of these.
You're going to have pretty deep layers
of Conv, ReLU, pool
sequences, and then usually
just a couple of these fully
connected layers at the end.
But we'll also go into
some newer architectures
like ResNet and GoogLeNet,
which challenge this
and will give pretty different
types of architectures.
Okay, thank you and
see you guys next time.

