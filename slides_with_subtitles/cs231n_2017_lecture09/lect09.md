﻿
- All right welcome to lecture nine.
So today we will be talking
about CNN Architectures.
And just a few administrative points
before we get started,
assignment two is due Thursday.
The mid term will be in
class on Tuesday May ninth,
so next week and it will
cover material through Tuesday
through this coming Thursday May fourth.
So everything up to
recurrent neural networks
are going to be fair game.
The poster session
we've decided on a time,
it's going to be Tuesday June sixth
from twelve to three p.m.
So this is the last week of classes.
So we have our our poster
session a little bit early
during the last week so that after that,
once you guys get feedback
you still have some time
to work for your final report
which will be due finals week.
Okay, so just a quick review of last time.
Last time we talked
about different kinds of
deep learning frameworks.
We talked about you know
PyTorch, TensorFlow,
Caffe2
and we saw that using
these kinds of frameworks
we were able to easily build
big computational graphs,
for example very large neural
networks and comm nets,
and be able to really
easily compute gradients
in these graphs.
So to compute all of the
gradients for all the intermediate
variables weights inputs and
use that to train our models
and to run all this efficiently on GPUs
And we saw that for a
lot of these frameworks
the way this works is by
working with these modularized
layers that you guys have
been working writing with,
in your home works as well
where we have a forward pass,
we have a backward pass,
and then in our final model architecture,
all we need to do then is to just define
all of these sequence of layers together.
So using that we're able
to very easily be able to
build up very complex
network architectures.
So today we're going to talk
about some specific kinds
of CNN Architectures that are
used today in cutting edge
applications and research.
And so we'll go into depth
in some of the most commonly
used architectures for
these that are winners
of ImageNet classification benchmarks.
So in chronological
order AlexNet, VGG net,
GoogLeNet, and ResNet.
And so these will go into a lot of depth.
And then I'll also after
that, briefly go through
some other architectures that are not
as prominently used these
days, but are interesting
either from a historical perspective,
or as recent areas of research.
Okay, so just a quick review.
We talked a long time ago about LeNet,
which was one of the first
instantiations of a comNet
that was successfully used in practice.
And so this was the comNet
that took an input image,
used com filters five by five filters
applied at stride one and
had a couple of conv layers,
a few pooling layers and then
some fully connected layers
at the end.
And this fairly simple comNet
was very successfully applied
to digit recognition.
So AlexNet from 2012 which
you guys have also heard
already before in previous classes,
was the first large scale
convolutional neural network
that was able to do well on
the ImageNet classification
task so in 2012 AlexNet was
entered in the competition,
and was able to outperform
all previous non deep
learning based models
by a significant margin,
and so this was the comNet
that started the spree
of comNet research and usage afterwards.
And so the basic comNet
AlexNet architecture
is a conv layer followed by pooling layer,
normalization, com pool norm,
and then a few more conv
layers, a pooling layer,
and then several fully
connected layers afterwards.
So this actually looks very
similar to the LeNet network
that we just saw.
There's just more layers in total.
There is five of these conv layers,
and two fully connected layers before
the final fully connected
layer going to the output
classes.
So let's first get a sense
of the sizes involved
in the AlexNet.
So if we look at the input to the AlexNet
this was trained on ImageNet, with inputs
at a size 227 by 227 by 3 images.
And if we look at this first
layer which is a conv layer
for the AlexNet, it's 11 by 11 filters,
96 of these applied at stride 4.
So let's just think
about this for a moment.
What's the output volume
size of this first layer?
And there's a hint.
So remember we have our input size,
we have our convolutional filters, ray.
And we have this formula,
which is the hint over here
that gives you the size
of the output dimensions
after applying com right?
So remember it was the full
image, minus the filter size,
divided by the stride, plus one.
So given that that's
written up here for you 55,
does anyone have a guess at
what's the final output size
after this conv layer?
[student speaks off mic]
- So I had 55 by 55 by 96, yep.
That's correct.
Right so our spatial
dimensions at the output
are going to be 55 in each
dimension and then we have
96 total filters so the
depth after our conv layer
is going to be 96.
So that's the output volume.
And what's the total number
of parameters in this layer?
So remember we have 96 11 by 11 filters.
[student speaks off mic]
- [Lecturer] 96 by 11 by 11, almost.
So yes, so I had another by three,
yes that's correct.
So each of the filters is going to
see through a local region
of 11 by 11 by three,
right because the input depth was three.
And so, that's each filter
size, times we have 96
of these total.
And so there's 35K parameters
in this first layer.
Okay, so now if we look
at the second layer
this is a pooling layer
right and in this case
we have three three by three
filters applied at stride two.
So what's the output volume
of this layer after pooling?
And again we have a hint, very
similar to the last question.
Okay, 27 by 27 by 96.
Yes that's correct.
Right so the pooling layer
is basically going to use
this formula that we had here.
Again because these are pooling
applied at a stride of two
so we're going to use the
same formula to determine
the spatial dimensions and
so the spatial dimensions
are going to be 27 by
27, and pooling preserves
the depth.
So we had 96 as depth as input,
and it's still going to be 96 depth
at output.
And next question.
What's the number of
parameters in this layer?
I hear some muttering.
[student answers off mic]
- Nothing.
Okay.
Yes, so pooling layer
has no parameters, so,
kind of a trick question.
Okay, so we can basically, yes, question?
[student speaks off mic]
- The question is, why are
there no parameters in the
pooling layer?
The parameters are the weights right,
that we're trying to learn.
And so convolutional layers
have weights that we learn
but pooling all we do is have a rule,
we look at the pooling region,
and we take the max.
So there's no parameters that are learned.
So we can keep on doing
this and you can just repeat
the process and it's kind of
a good exercise to go through
this and figure out the
sizes, the parameters,
at every layer.
And so if you do this all the way,
you can look at this is
the final architecture
that you can work with.
There's 11 by 11 filters at the beginning,
then five by five and some
three by three filters.
And so these are generally
pretty familiar looking sizes
that you've seen before
and then at the end
we have a couple of fully connected layers
of size 4096 and finally the last layer,
is FC8 going to the soft max,
which is going to the
1000 ImageNet classes.
And just a couple of details about this,
it was the first use of
the ReLu non-linearity
that we've talked about
that's the most commonly used
non-linearity.
They used local response
normalization layers
basically trying to
normalize the response across
neighboring channels but this
is something that's not really
used anymore.
It turned out not to, other people showed
that it didn't have so much of an effect.
There's a lot of heavy data augmentation,
and so you can look in the
paper for more details,
but things like flipping,
jittering, cropping,
color normalization all of these things
which you'll probably
find useful for you when
you're working on your
projects for example,
so a lot of data augmentation here.
They also use dropout batch size of 128,
and learned with SGD with
momentum which we talked about
in an earlier lecture,
and basically just started
with a base learning
rate of 1e negative 2.
Every time it plateaus,
reduce by a factor of 10
and then just keep going.
Until they finish training
and a little bit of weight
decay and in the end,
in order to get the best
numbers they also did
an ensembling of models and
so training multiple of these,
averaging them together and
this also gives an improvement
in performance.
And so one other thing I want to point out
is that if you look at this
AlexNet diagram up here,
it looks kind of like the
normal comNet diagrams
that we've been seeing,
except for one difference,
which is that it's, you
can see it's kind of split
in these two different rows
or columns going across.
And so the reason for this
is mostly historical note,
so AlexNet was trained
on GTX580 GPUs older GPUs
that only had three gigs of memory.
So it couldn't actually fit
this entire network on here,
and so what they ended up doing,
was they spread the
network across two GPUs.
So on each GPU you would
have half of the neurons,
or half of the feature maps.
And so for example if you
look at this first conv layer,
we have 55 by 55 by 96 output,
but if you look at this diagram carefully,
you can zoom in later in the actual paper,
you can see that, it's actually only 48
depth-wise, on each GPU,
and so they just spread
it, the feature maps,
directly in half.
And so what happens is that
for most of these layers,
for example com one, two, four and five,
the connections are only with feature maps
on the same GPU, so you
would take as input,
half of the feature maps
that were on the the same GPU
as before and you don't
look at the full 96
feature maps for example.
You just take as input the
48 in that first layer.
And then there's a few
layers so com three,
as well as FC six, seven and eight,
where here are the GPUs
do talk to each other
and so there's connections
with all feature maps
in the preceding layer.
so there's communication across the GPUs,
and each of these neurons
are then connected
to the full depth of the
previous input layer.
Question.
- [Student] It says the
full simplified AlexNetwork
architecture.
[mumbles]
- Oh okay, so the question
is why does it say
full simplified AlexNet architecture here?
It just says that because I
didn't put all the details
on here, so for example this
is the full set of layers
in the architecture, and
the strides and so on,
but for example the normalization
layer, there's other,
these details are not written on here.
And then just one little note,
if you look at the paper
and try and write out
the math and architectures and so on,
there's a little bit of
an issue on the very first
layer they'll say if
you'll look in the figure
they'll say 224 by 224 ,
but there's actually some
kind of funny pattern
going on and so the
numbers actually work out
if you look at it as 227.
AlexNet was the winner of
the ImageNet classification
benchmark in 2012, you can see that
it cut the error rate
by quite a large margin.
It was the first CNN base
winner, and it was widely used
as a base to our architecture
almost ubiquitously from then
until a couple years ago.
It's still used quite a bit.
It's used in transfer learning
for lots of different tasks
and so it was used for
basically a long time,
and it was very famous and
now though there's been
some more recent architectures
that have generally
just had better performance
and so we'll talk about these
next and these are going to be
the more common architectures
that you'll be wanting to use in practice.
So just quickly first in
2013 the ImageNet challenge
was won by something called a ZFNet.
Yes, question.
[student speaks off mic]
- So the question is intuition why AlexNet
was so much better than
the ones that came before,
DefLearning comNets [mumbles] this is just
a very different kind of
approach in architecture.
So this was the first deep
learning based approach
first comNet that was used.
So in 2013 the challenge
was won by something called
a ZFNet [Zeller Fergus Net]
named after the creators.
And so this mostly was
improving hyper parameters
over the AlexNet.
It had the same number of layers,
the same general structure
and they made a few
changes things like
changing the stride size,
different numbers of filters
and after playing around
with these hyper parameters more,
they were able to improve the error rate.
But it's still basically the same idea.
So in 2014 there are a
couple of architectures
that were now more significantly different
and made another jump in performance,
and the main difference with
these networks first of all
was much deeper networks.
So from the eight layer
network that was in 2012
and 2013, now in 2014 we
had two very close winners
that were around 19 layers and 22 layers.
So significantly deeper.
And the winner of this
was GoogleNet, from Google
but very close behind was
something called VGGNet
from Oxford, and on actually
the localization challenge
VGG got first place in
some of the other tracks.
So these were both very,
very strong networks.
So let's first look at VGG
in a little bit more detail.
And so the VGG network is the
idea of much deeper networks
and with much smaller filters.
So they increased the number of layers
from eight layers in AlexNet
right to now they had
models with 16 to 19 layers in VGGNet.
And one key thing that they
did was they kept very small
filter so only three by
three conv all the way,
which is basically the
smallest com filter size
that is looking at a little
bit of the neighboring pixels.
And they just kept this
very simple structure
of three by three convs
with the periodic pooling
all the way through the network.
And it's very simple elegant
network architecture,
was able to get 7.3% top five error
on the ImageNet challenge.
So first the question of
why use smaller filters.
So when we take these
small filters now we have
fewer parameters and we
try and stack more of them
instead of having larger filters,
have smaller filters
with more depth instead,
have more of these filters instead,
what happens is that you end
up having the same effective
receptive field as if you
only have one seven by seven
convolutional layer.
So here's a question, what is
the effective receptive field
of three of these three
by three conv layers
with stride one?
So if you were to stack three
three by three conv layers
with Stride one what's the
effective receptive field,
the total area of the input,
spatial area of the input
that enure at the top
layer of the three layers
is looking at.
So I heard fifteen pixels,
why fifteen pixels?
- [Student] Okay, so the
reason given was because
they overlap--
- Okay, so the reason given
was because they overlap.
So it's on the right track.
What actually is happening
though is you have to see,
at the first layer, the
receptive field is going to be
three by three right?
And then at the second layer,
each of these neurons in the second layer
is going to look at three
by three other first layer
filters, but the corners
of these three by three
have an additional pixel on each side,
that is looking at in
the original input layer.
So the second layer is actually
looking at five by five
receptive field and then
if you do this again,
the third layer is
looking at three by three
in the second layer but this is going to,
if you just draw out this
pyramid is looking at
seven by seven in the input layer.
So the effective receptive field here
is going to be seven by seven.
Which is the same as one
seven by seven conv layer.
So what happens is that
this has the same effective
receptive field as a
seven by seven conv layer
but it's deeper.
It's able to have more
non-linearities in there,
and it's also fewer parameters.
So if you look at the
total number of parameters,
each of these conv filters
for the three by threes
is going to have nine parameters
in each conv [mumbles]
three times three, and
then times the input depth,
so three times three times
C, times this total number
of output feature maps, which is again C
is we're going to preserve the total
number of channels.
So you get three times three,
times C times C for each of these layers,
and we have three layers
so it's going to be
three times this number,
compared to if you had a
single seven by seven layer
then you get, by the same reasoning,
seven squared times C squared.
So you're going to have
fewer parameters total,
which is nice.
So now if we look at
this full network here
there's a lot of numbers up
here that you can go back
and look at more carefully
but if we look at all
of the sizes and number
of parameters the same way
that we calculated the
example for AlexNet,
this is a good exercise to go through,
we can see that you
know going the same way
we have a couple of these conv
layers and a pooling layer
a couple more conv layers,
pooling layer, several more
conv layers and so on.
And so this just keeps going up.
And if you counted the total
number of convolutional
and fully connected layers,
we're going to have 16
in this case for VGG 16,
and then VGG 19, it's just a very similar
architecture, but with a few
more conv layers in there.
And so the total memory
usage of this network,
so just making a forward
pass through counting up
all of these numbers so
in the memory numbers here
written in terms of the total numbers,
like we calculated earlier,
and if you look at four bytes per number,
this is going to be
about 100 megs per image,
and so this is the scale
of the memory usage
that's happening and this is
only for a forward pass right,
when you do a backward pass
you're going to have to store
more and so this is
pretty heavy memory wise.
100 megs per image, if
you have on five gigs
of total memory, then
you're only going to be able
to store about 50 of these.
And so also the total number
of parameters here we have
is 138 million parameters in this network,
and this compares with
60 million for AlexNet.
Question?
[student speaks off mic]
- So the question is what
do we mean by deeper,
is it the number of
filters, number of layers?
So deeper in this case is
always referring to layers.
So there are two usages of the word depth
which is confusing one is
the depth rate per channel,
width by height by depth, you can use
the word depth here,
but in general we talk about
the depth of a network,
this is going to be the
total number of layers
in the network, and usually in particular
we're counting the total
number of weight layers.
So the total number of
layers with trainable weight,
so convolutional layers
and fully connected layers.
[student mumbles off mic]
- Okay, so the question
is, within each layer
what do different filters need?
And so we talked about this
back in the comNet lecture,
so you can also go back and refer to that,
but each filter is a set of
let's say three by three convs,
so each filter is looking at a,
is a set of weight looking at
a three by three value input
input depth, and this
produces one feature map,
one activation map of
all the responses of the
different spatial locations.
And then we have we can have
as many filters as we want
right so for example 96 and each of these
is going to produce a feature map.
And so it's just like
each filter corresponds
to a different pattern
that we're looking for
in the input that we
convolve around and we see
the responses everywhere in the input,
we create a map of these
and then another filter
will we convolve over the
image and create another map.
Question.
[student speaks off mic]
- So question is, is
there intuition behind,
as you go deeper into the network
we have more channel depth
so more number of filters
right and so you can have
any design that you want so
you don't have to do this.
In practice you will see this
happen a lot of the times
and one of the reasons is
people try and maintain
kind of a relatively
constant level of compute,
so as you go higher up or
deeper into your network,
you're usually also using
basically down sampling
and having smaller total
spatial area and then so then
they also increase now you
increase by depth a little bit,
it's not as expensive
now to increase by depth
because it's spatially smaller and so,
yeah that's just a reason.
Question.
[student speaks off mic]
- So performance-wise is
there any reason to use
SBN [mumbles] instead
of SouthMax [mumbles],
so no, for a classifier
you can use either one,
and you did that earlier
in the class as well,
but in general SouthMax losses,
have generally worked
well and been standard use
for classification here.
Okay yeah one more question.
[student mumbles off mic]
- Yes, so the question
is, we don't have to store
all of the memory like we
can throw away the parts
that we don't need and so on?
And yes this is true.
Some of this you don't need to keep,
but you're also going to
be doing a backwards pass
through ware for the most part,
when you were doing the chain
rule and so on you needed
a lot of these activations
as part of it and so in
large part a lot of this
does need to be kept.
So if we look at the distribution
of where memory is used
and where parameters are,
you can see that a lot
of memories in these early
layers right where you still have
spatial dimensions you're
going to have more memory usage
and then a lot of the
parameters are actually in
the last layers, the
fully connected layers
have a huge number of parameters right,
because we have all of
these dense connections.
And so that's something
just to know and then
keep in mind so later on we'll
see some networks actually
get rid of these fully
connected layers and be able
to save a lot on the number of parameters.
And then just one last thing to point out,
you'll also see different ways of calling
all of these layers right.
So here I've written out
exactly what the layers are.
conv3-64 means three by three convs
with 64 total filters.
But for VGGNet on this
diagram on the right here
there's also common ways
that people will look
at each group of filters,
so each orange block here, as in conv1
part one, so conv1-1, conv1-2,
and so on.
So just something to keep in mind.
So VGGNet ended up getting
second place in the
ImageNet 2014 classification challenge,
first in localization.
They followed a very
similar training procedure
as Alex Krizhevsky for the AlexNet.
They didn't use local
response normalization,
so as I mentioned earlier,
they found out this
didn't really help them,
and so they took it out.
You'll see VGG 16 and VGG
19 are common variants
of the cycle here, and this is just
the number of layers, 19
is slightly deeper than 16.
In practice VGG 19 works
very little bit better,
and there's a little
bit more memory usage,
so you can use either but
16 is very commonly used.
For best results, like
AlexNet, they did ensembling
in order to average several models,
and you get better results.
And they also showed in their work that
the FC7 features of the last
fully connected layer before
going to the 1000 ImageNet classes.
The 4096 size layer just before that,
is a good feature representation,
that can even just be used as is,
to extract these features from other data,
and generalized these other tasks as well.
And so FC7 is a good
feature representation.
Yeah question.
[student speaks off mic]
- Sorry what was the question?
Okay, so the question is
what is localization here?
And so this is a task,
and we'll talk about it
a little bit more in a later lecture
on detection and localization
so I don't want to
go into detail here but
it's basically an image,
not just classifying What's
the class of the image,
but also drawing a bounding
box around where that
object is in the image.
And the difference with detection,
which is a very related
task is that detection
there can be multiple instances
of this object in the image
localization we're
assuming there's just one,
this classification but we just how this
additional bounding box.
So we looked at VGG which
was one of the deep networks
from 2014 and then now
we'll talk about GoogleNet
which was the other one that won
the classification challenge.
So GoogleNet again was
a much deeper network
with 22 layers but one
of the main insights
and special things about
GoogleNet is that it really
looked at this problem of
computational efficiency
and it tried to design a
network architecture that was
very efficient in the amount of compute.
And so they did this using
this inception module
which we'll go into more
detail and basically stacking
a lot of these inception
modules on top of each other.
There's also no fully connected
layers in this network,
so they got rid of that
were able to save a lot
of parameters and so in total
there's only five million
parameters which is twelve
times less than AlexNet,
which had 60 million even
though it's much deeper now.
It got 6.7% top five error.
So what's the inception module?
So the idea behind the inception module
is that they wanted to design
a good local network typology
and it has this idea
of this local topology
that's you know you can
think of it as a network
within a network and
then stack a lot of these
local typologies one on top of each other.
And so in this local
network that they're calling
an inception module what they're
doing is they're basically
applying several different
kinds of filter operations
in parallel on top of the
same input coming into
this same layer.
So we have our input coming
in from the previous layer
and then we're going to do
different kinds of convolutions.
So a one by one conv, right
a three by three conv,
five by five conv, and then they also
have a pooling operation
in this case three by three
pooling, and so you get
all of these different
outputs from these different layers,
and then what they do is
they concatenate all these
filter outputs together depth wise, and so
then this creates one
tenser output at the end
that is going tom pass
on to the next layer.
So if we look at just a
naive way of doing this
we just do exactly that we
have all of these different
operations we get the outputs
we concatenate them together.
So what's the problem with this?
And it turns out that
computational complexity
is going to be a problem here.
So if we look more
carefully at an example,
so here just for as an example
I've put one by one conv,
128 filter so three by
three conv 192 filters,
five by five convs and 96 filters.
Assume everything has basically the stride
that's going to maintain
the spatial dimensions,
and that we have this input coming in.
So what is the output size
of the one by one filter
with 128 , one by one
conv with 128 filters?
Who has a guess?
OK so I heard 28 by 28,
by 128 which is correct.
So right by one by one conv
we're going to maintain
spatial dimensions and
then on top of that,
each conv filter is going to look through
the entire 256 depth of the input,
but then the output is going to be,
we have a 28 by 28 feature map
for each of the 128 filters that we have
in this conv layer.
So we get 28 by 28 by 128.
OK and then now if we do the same thing
and we look at the filter
sizes of the output sizes sorry
of all of the different
filters here, after the
three by three conv we're
going to have this volume
of 28 by 28 by 192 right
after five by five conv
we have 96 filters here.
So 28 by 28 by 96,
and then out pooling layer is just going
to keep the same spatial
dimension here, so pooling layer
will preserve it in depth,
and here because of our stride,
we're also going to preserve
our spatial dimensions.
And so now if we look at
the output size after filter
concatenation what we're
going to get is 28 by 28,
these are all 28 by 28, and
we concatenating depth wise.
So we get 28 by 28 times
all of these added together,
and the total output size is going to be
28 by 28 by 672.
So the input to our
inception module was 28 by 28
by 256, then the output
from this module is 28 by 28
by 672.
So we kept the same spatial dimensions,
and we blew up the depth.
Question.
[student speaks off mic]
OK So in this case, yeah, the question is,
how are we getting 28
by 28 for everything?
So here we're doing all the zero padding
in order to maintain
the spatial dimensions,
and that way we can do this filter
concatenation depth-wise.
Question in the back.
[student speaks off mic]
- OK The question is what's
the 256 deep at the input,
and so this is not the
input to the network,
this is the input just
to this local module
that I'm looking at.
So in this case 256 is
the depth of the previous
inception module that
came just before this.
And so now coming out
we have 28 by 28 by 672,
and that's going to be
the input to the next
inception module.
Question.
[student speaks off mic]
- Okay the question is, how
did we get 28 by 28 by 128
for the first one, the first conv,
and this is basically it's a
one by one convolution right,
so we're going to take
this one by one convolution
slide it across our 28 by
28 by 256 input spatially
where it's at each location,
it's going to multiply,
it's going to do a [mumbles]
through the entire 256
depth, and so we do this
one by one conv slide it over spatially
and we get a feature map
out that's 28 by 28 by one.
There's one number at each
spatial location coming out,
and each filter produces
one of these 28 by 28
by one maps, and we have
here a total 128 filters,
and that's going to
produce 28 by 28, by 128.
OK so if you look at
the number of operations
that are happening in
the convolutional layer,
let's look at the first one for
example this one by one conv
as I was just saying at each
each location we're doing
a one by one by 256 dot product.
So there's 256 multiply
operations happening here
and then for each filter
map we have 28 by 28
spatial locations, so
that's the first 28 times 28
first two numbers that
are multiplied here.
These are the spatial
locations for each filter map,
and so we have to do this
to 25 60 multiplication
each one of these then
we have 128 total filters
at this layer, or we're
producing 128 total
feature maps.
And so the total number
of these operations here
is going to be 28 times 28
times 128 times 256.
And so this is going to be the same for,
you can think about this
for the three by three conv,
and the five by five conv,
that's exactly the same
principle.
And in total we're going to
get 854 million operations
that are happening here.
- [Student] And the 128,
192, and 96 are just values
[mumbles]
- Question the 128, 192 and
256 are values that I picked.
Yes, these are not values
that I just came up with.
They are similar to the
ones that you will see
in like a particular
layer of inception net,
so in GoogleNet basically,
each module has a different
set of these kinds of
parameters, and I picked one
that was similar to one of these.
And so this is very expensive
computationally right,
these these operations.
And then the other thing
that I also want to note
is that the pooling layer also
adds to this problem because
it preserves the whole feature depth.
So at every layer your total
depth can only grow right,
you're going to take
the full featured depth
from your pooling layer, as
well as all the additional
feature maps from the conv
layers and add these up together.
So here our input was 256
depth and our output is
672 depth and you're just
going to keep increasing this
as you go up.
So how do we deal with this
and how do we keep this
more manageable?
And so one of the key
insights that GoogleNet used
was that well we can we
can address this by using
bottleneck layers and try and
project these feature maps
to lower dimension before our
our convolutional operations,
so before our expensive layers.
And so what exactly does that mean?
So reminder one by one
convolution, I guess
we were just going through
this but it's taking your input
volume, it's performing a
dot product at each spatial
location and what it does is
it preserves spatial dimension
but it reduces the depth and
it reduces that by projecting
your input depth to a lower dimension.
It just takes it's basically
like a linear combination
of your input feature maps.
And so this main idea is
that it's projecting your
depth down and so the inception module
takes these one by one convs
and adds these at a bunch
of places in these modules
where there's going to be,
in order to alleviate
this expensive compute.
So before the three by three
and five by five conv layers,
it puts in one of these
one by one convolutions.
And then after the
pooling layer it also puts
an additional one by one convolution.
Right so these are the one
by one bottleneck layers
that are added in.
And so how does this change the math
that we were looking at earlier?
So now basically what's
happening is that we still
have the same input here 28 by 28 by 256,
but these one by one convs
are going to reduce the depth
dimension and so you can see
before the three by three
convs, if I put a one by
one conv with 64 filters,
my output from that is going to be,
28 by 28 by 64.
So instead of now going into
the three by three convs
afterwards instead of 28
by 28 by 256 coming in,
we only have a 28 by 28,
by 64 block coming in.
And so this is now
reducing the smaller input
going into these conv
layers, the same thing for
the five by five conv, and
then for the pooling layer,
after the pooling comes
out, we're going to
reduce the depth after this.
And so, if you work out
the math the same way
for all of the convolutional ops here,
adding in now all these one by one convs
on top of the three by
threes and five by fives,
the total number of operations
is 358 million operations,
so it's much less than the
854 million that we had
in the naive version, and
so you can see how you
can use this one by one
conv, and the filter size
for that to control your computation.
Yes, question in the back.
[student speaks off mic]
- Yes, so the question
is, have you looked into
what information might be
lost by doing this one by one
conv at the beginning.
And so there might be
some information loss,
but at the same time if
you're doing these projections
you're taking a linear
combination of these input
feature maps which has redundancy in them,
you're taking combinations of them,
and you're also introducing
an additional non-linearity
after the one by one
conv, so it also actually
helps in that way with
adding a little bit more
depth and so, I don't think
there's a rigorous analysis
of this, but basically in
general this works better
and there's reasons why it helps as well.
OK so here we have, we're
basically using these one by one
convs to help manage our
computational complexity,
and then what GooleNet
does is it takes these
inception modules and it's going to stack
all these together.
So this is a full inception architecture.
And if we look at this a
little bit more detail,
so here I've flipped it,
because it's so big, it's not going to fit
vertically any more on the slide.
So what we start with is
we first have this stem
network, so this is more
the kind of vanilla plain
conv net that we've seen earlier [mumbles]
six sequence of layers.
So conv pool a couple
of convs in another pool
just to get started and then after that
we have all of our different
our multiple inception
modules all stacked on top of each other,
and then on top we have
our classifier output.
And notice here that
they've really removed
the expensive fully connected layers
it turns out that the model
works great without them,
even and you reduce a lot of parameters.
And then what they also have here is,
you can see these couple
of extra stems coming out
and these are auxiliary
classification outputs
and so these are also you know
just a little mini networks
with an average pooling,
a one by one conv,
a couple of fully connected
layers here going to
the soft Max and also a 1000 way SoftMax
with the ImageNet classes.
And so you're actually
using your ImageNet training
classification loss in
three separate places here.
The standard end of the
network, as well as in these
two places earlier on in
the network, and the reason
they do that is just
this is a deep network
and they found that having
these additional auxiliary
classification outputs,
you get more gradient
training injected at the earlier layers,
and so more just helpful signal flowing in
because these intermediate
layers should also be
helpful.
You should be able to do classification
based off some of these as well.
And so this is the full architecture,
there's 22 total layers
with weights and so
within each of these modules
each of those one by one,
three by three, five by
five is a weight layer,
just including all of
these parallel layers,
and in general it's a relatively
more carefully designed
architecture and part of this
is based on some of these
intuitions that we're talking
about and part of them
also is just you know
Google the authors they had
huge clusters and they're
cross validating across
all kinds of design
choices and this is what
ended up working well.
Question?
[student speaks off mic]
- Yeah so the question is,
are the auxiliary outputs
actually useful for the
final classification,
to use these as well?
I think when they're training them
they do average all these
for the losses coming out.
I think they are helpful.
I can't remember if in
the final architecture,
whether they average all
of these or just take one,
it seems very possible that
they would use all of them,
but you'll need to check on that.
[student speaks off mic]
- So the question is for
the bottleneck layers,
is it possible to use some
other types of dimensionality
reduction and yes you can use
other kinds of dimensionality
reduction.
The benefits here of
this one by one conv is,
you're getting this effect,
but it's all, you know
it's a conv layer just like any other.
You have the soul network of these,
you just train it this full network
back [mumbles] through everything,
and it's learning how to combine the
previous feature maps.
Okay yeah, question in the back.
[student speaks off mic]
- Yes so, question is
are any weights shared
or all they all separate and yeah,
all of these layers have separate weights.
Question.
[student speaks off mic]
- Yes so the question is why do we have
to inject gradients at earlier layers?
So our classification
output at the very end,
where we get a gradient on this, it's
passed all the way back
through the chain roll
but the problem is when
you have very deep networks
and you're going all the
way back through these,
some of this gradient
signal can become minimized
and lost closer to the beginning,
and so that's why having
these additional ones in earlier parts
can help provide some additional signal.
[student mumbles off mic]
- So the question is are you
doing back prop all the times
for each output.
No it's just one back
prop all the way through,
and you can think of these three,
you can think of there being kind of like
an addition at the end
of these if you were to
draw up your computational
graph, and so you get your
final signal and you can
just take all of these
gradients and just back plot
them all the way through.
So it's as if they were
added together at the end
in a computational graph.
OK so in the interest of
time because we still have
a lot to get through, can
take other questions offline.
Okay so GoogleNet basically 22 layers.
It has an efficient inception module,
there's no fully connected layers.
12 times fewer parameters than AlexNet,
and it's the ILSVRC 2014
classification winner.
And so now let's look at the 2015 winner,
which is the ResNet network and so here
this idea is really, this
revolution of depth net right.
We were starting to increase
depth in 2014, and here we've
just had this hugely
deeper model at 152 layers
was the ResNet architecture.
And so now let's look at that
in a little bit more detail.
So the ResNet architecture,
is getting extremely
deep networks, much deeper
than any other networks
before and it's doing this using this idea
of residual connections
which we'll talk about.
And so, they had 152
layer model for ImageNet.
They were able to get 3.5
of 7% top 5 error with this
and the really special
thing is that they swept
all classification and
detection contests in the
ImageNet mart benchmark
and this other benchmark
called COCO.
It just basically won everything.
So it was just clearly
better than everything else.
And so now let's go into a
little bit of the motivation
behind ResNet and residual connections
that we'll talk about.
And the question that they
started off by trying to answer
is what happens when we try
and stack deeper and deeper
layers on a plain
convolutional neural network?
So if we take something like VGG
or some normal network that's
just stacks of conv and
pool layers on top of each
other can we just continuously
extend these, get deeper
layers and just do better?
And and the answer is no.
So if you so if you look at what happens
when you get deeper, so here
I'm comparing a 20 layer
network and a 56 layer network
and so this is just a plain
kind of network you'll see
that in the test error here
on the right the 56 layer
network is doing worse
than the 28 layer network.
So the deeper network was
not able to do better.
But then the really weird thing is now
if you look at the training error right
we here have again the 20 layer network
and a 56 layer network.
The 56 layer network, one of
the obvious problems you think,
I have a really deep network,
I have tons of parameters
maybe it's probably starting
to over fit at some point.
But what actually happens is
that when you're over fitting
you would expect to have very good,
very low training error rate,
and just bad test error,
but what's happening here is
that in the training error
the 56 layer network is
also doing worse than
the 20 layer network.
And so even though the
deeper model performs worse,
this is not caused by over-fitting.
And so the hypothesis
of the ResNet creators
is that the problem is actually
an optimization problem.
Deeper models are just harder to optimize,
than more shallow networks.
And the reasoning was that well,
a deeper model should be
able to perform at least
as well as a shallower model.
You can have actually a
solution by construction
where you just take the learned layers
from your shallower model, you just
copy these over and then
for the remaining additional
deeper layers you just
add identity mappings.
So by construction this
should be working just as well
as the shallower layer.
And your model that weren't
able to learn properly,
it should be able to learn at least this.
And so motivated by
this their solution was
well how can we make it
easier for our architecture,
our model to learn these
kinds of solutions,
or at least something like this?
And so their idea is well
instead of just stacking
all these layers on top
of each other and having
every layer try and learn
some underlying mapping
of a desired function, lets
instead have these blocks,
where we try and fit a residual mapping,
instead of a direct mapping.
And so what this looks
like is here on this right
where the input to these block
is just the input coming in
and here we are going to
use our, here on the side,
we're going to use our
layers to try and fit
some residual of our desire to H of X,
minus X instead of the desired
function H of X directly.
And so basically at the
end of this block we take
the step connection on
this right here, this loop,
where we just take our input,
we just use pass it through
as an identity, and so if
we had no weight layers
in between it was just
going to be the identity
it would be the same thing
as the output, but now we use
our additional weight
layers to learn some delta,
for some residual from our X.
And so now the output
of this is going to be
just our original R X plus some residual
that we're going to call it.
It's basically a delta
and so the idea is that
now the output it should
be easy for example,
in the case where identity is ideal,
to just squash all of
these weights of F of X
from our weight layers
just set it to all zero
for example, then we're
just going to get identity
as the output, and we can get something,
for example, close to this
solution by construction
that we had earlier.
Right, so this is just
a network architecture
that says okay, let's try and fit this,
learn how our weight layers
residual, and be something
close, that way it'll more
likely be something close to X,
it's just modifying X,
than to learn exactly
this full mapping of what it should be.
Okay, any questions about this?
[student speaks off mic]
- Question is is there the same dimension?
So yes these two paths
are the same dimension.
In general either it's the same dimension,
or what they actually
do is they have these
projections and shortcuts
and they have different ways
of padding to make things work
out to be the same dimension.
Depth wise.
Yes
- [Student] When you use the word residual
you were talking about [mumbles off mic]
- So the question is what
exactly do we mean by
residual this output
of this transformation
is a residual?
So we can think of our output
here right as this F of X
plus X, where F of X is the
output of our transformation
and then X is our input,
just passed through
by the identity.
So we'd like to using a plain layer,
what we're trying to do is learn something
like H of X, but what we saw
earlier is that it's hard
to learn H of X.
It's a good H of X as we
get very deep networks.
And so here the idea is
let's try and break it down
instead of as H of X is
equal to F of X plus,
and let's just try and learn F of X.
And so instead of learning
directly this H of X
we just want to learn what
is it that we need to add
or subtract to our input as
we move on to the next layer.
So you can think of it as
kind of modifying this input,
in place in a sense.
We have--
[interrupted by student mumbling off mic]
- The question is, when we're
saying the word residual
are we talking about F of X?
Yeah.
So F of X is what we're
calling the residual.
And it just has that meaning.
Yes another question.
[student mumbles off mic]
- So the question is in
practice do we just sum
F of X and X together, or
do we learn some weighted
combination and you just do a direct sum.
Because when you do a direct sum,
this is the idea of let
me just learn what is it
I have to add or subtract onto X.
Is this clear to everybody,
the main intuition?
Question.
[student speaks off mic]
- Yeah, so the question
is not clear why is it
that learning the
residual should be easier
than learning the direct mapping?
And so this is just their hypotheses,
and a hypotheses is that if
we're learning the residual
you just have to learn
what's the delta to X right?
And if our hypotheses is that generally
even something like our
solution by construction,
where we had some number
of these shallow layers
that were learned and we had
all these identity mappings
at the top this was a
solution that should have been
good, and so that implies that
maybe a lot of these layers,
actually something just close to identity,
would be a good layer
And so because of that,
now we formulate this
as being able to learn the identity
plus just a little delta.
And if really the identity
is best we just make
F of X squashes transformation
to just be zero,
which is something that's relatively,
might seem easier to learn,
also we're able to get
things that are close
to identity mappings.
And so again this is not
something that's necessarily
proven or anything it's just
the intuition and hypothesis,
and then we'll also see
later some works where people
are actually trying to
challenge this and say oh maybe
it's not actually the residuals
that are so necessary,
but at least this is the
hypothesis for this paper,
and in practice using this model,
it was able to do very well.
Question.
[student speaks off mic]
- Yes so the question is
have people tried other ways
of combining the inputs
from previous layers and yes
so this is basically a very
active area of research
on and how we formulate
all these connections,
and what's connected to what
in all of these structures.
So we'll see a few more
examples of different network
architectures briefly later
but this is an active area
of research.
OK so we basically have all
of these residual blocks
that are stacked on top of each other.
We can see the full resident architecture.
Each of these residual blocks
has two three by three conv
layers as part of this block
and there's also been work
just saying that this happens
to be a good configuration
that works well.
We stack all these blocks
together very deeply.
Another thing like with
this very deep architecture
it's basically also
enabling up to 150 layers
deep of this, and then
what we do is we stack
all these and periodically we also double
the number of filters
and down sample spatially
using stride two when we do that.
And then we have this additional [mumbles]
at the very beginning of our network
and at the end we also hear,
don't have any fully connected layers
and we just have a global
average pooling layer
that's going to average
over everything spatially,
and then be input into the
last 1000 way classification.
So this is the full ResNet architecture
and it's very simple and
elegant just stacking up
all of these ResNet blocks
on top of each other,
and they have total depths
of up to 34, 50, 100,
and they tried up to 152 for ImageNet.
OK so one additional
thing just to know is that
for a very deep network,
so the ones that are more
than 50 layers deep, they
also use bottleneck layers
similar to what GoogleNet did
in order to improve efficiency
and so within each block
now you're going to,
what they did is, have this
one by one conv filter,
that first projects it
down to a smaller depth.
So again if we are looking
at let's say 28 by 28
by 256 implant, we do
this one by one conv,
it's taking it's
projecting the depth down.
We get 28 by 28 by 64.
Now your convolution
your three by three conv,
in here they only have
one, is operating over this
reduced step so it's going
to be less expensive,
and then afterwards they have another
one by one conv that
projects the depth back up
to 256, and so, this is
the actual block that
you'll see in deeper networks.
So in practice the ResNet
also uses batch normalization
after every conv layer, they
use Xavier initialization
with an extra scaling factor
that they helped introduce
to improve the initialization
trained with SGD + momentum.
Their learning rate they
use a similar learning rate
type of schedule where you
decay your learning rate
when your validation error plateaus.
Mini batch size 256, a
little bit of weight decay
and no drop out.
And so experimentally they
were able to show that they
were able to train these
very deep networks,
without degrading.
They were able to have
basically good gradient flow
coming all the way back
down through the network.
They tried up to 152 layers on ImageNet,
1200 on Cifar, which is a,
you have played with it,
but a smaller data set
and they also saw that now
you're deeper networks are
able to achieve lower training
errors as expected.
So you don't have the same strange plots
that we saw earlier where the behavior
was in the wrong direction.
And so from here they were
able to sweep first place
at all of the ILSVRC competitions,
and all of the COCO competitions in 2015
by a significant margins.
Their total top five error
was 3.6 % for a classification
and this is actually better
than human performance
in the ImageNet paper.
There was also a human
metric that came from
actually [mumbles] our
lab Andre Kapathy spent
like a week training
himself and then basically
did all of, did this task himself
and was I think somewhere around 5-ish %,
and so I was basically able to do
better than the then that human at least.
Okay, so these are kind
of the main networks
that have been used recently.
We had AlexNet starting off with first,
VGG and GoogleNet are still very popular,
but ResNet is the most
recent best performing model
that if you're looking for
something training a new network
ResNet is available, you should try
working with it.
So just quickly looking at
some of this getting a better
sense of the complexity involved.
So here we have some
plots that are sorted by
performance so this is
top one accuracy here,
and higher is better.
And so you'll see a lot
of these models that we
talked about, as well as
some different versions
of them so, this
GoogleNet inception thing,
I think there's like V2,
V3 and the best one here
is V4, which is actually
a ResNet plus inception
combination, so these are just kind of
more incremental, smaller
changes that they've
built on top of them,
and so that's the best
performing model here.
And if we look on the
right, these plots of their
computational complexity here it's sorted.
The Y axis is your top one accuracy
so higher is better.
The X axis is your operations
and so the more to the right,
the more ops you're doing,
the more computationally
expensive and then the bigger the circle,
your circle is your memory usage,
so the gray circles are referenced here,
but the bigger the circle
the more memory usage
and so here we can see
that VGG these green ones
are kind of the least efficient.
They have the biggest memory,
the most operations,
but they they do pretty well.
GoogleNet is the most efficient here.
It's way down on the operation side,
as well as a small little
circle for memory usage.
AlexNet, our earlier
model, has lowest accuracy.
It's relatively smaller compute, because
it's a smaller network, but
it's also not particularly
memory efficient.
And then ResNet here, we
have moderate efficiency.
It's kind of in the middle,
both in terms of memory
and operations, and it
has the highest accuracy.
And so here also are
some additional plots.
You can look at these
more on your own time,
but this plot on the left is
showing the forward pass time
and so this is in milliseconds
and you can up at the top
VGG forward passes about 200
milliseconds you can get about
five frames per second with this,
and this is sorted in order.
There's also this plot on
the right looking at power
consumption and if you look
more at this paper here,
there's further analysis of
these kinds of computational
comparisons.
So these were the main
architectures that you should
really know in-depth and be familiar with,
and be thinking about actively using.
But now I'm going just
to go briefly through
some other architectures
that are just good
to know either historical inspirations
or more recent areas of research.
So the first one Network in Network,
this is from 2014, and
the idea behind this
is that we have these
vanilla convolutional layers
but we also have these,
this introduces the idea of
MLP conv layers they call
it, which are micro networks
or basically network within networth, the
name of the paper.
Where within each conv
layer trying to stack an MLP
with a couple of fully
connected layers on top of
just the standard conv
and be able to compute
more abstract features for these local
patches right.
So instead of sliding
just a conv filter around,
it's sliding a slightly
more complex hierarchical
set of filters around
and using that to get the
activation maps.
And so, it uses these fully connected,
or basically one by one
conv kind of layers.
It's going to stack them all up like the
bottom diagram here where
we just have these networks
within networks stacked
in each of the layers.
And the main reason to know this is just
it was kind of a precursor
to GoogleNet and ResNet
in 2014 with this idea
of bottleneck layers
that you saw used very heavily in there.
And it also had a little bit
of philosophical inspiration
for GoogleNet for this idea
of a local network typology
network in network that they also used,
with a different kind of structure.
Now I'm going to talk
about a series of works,
on, or works since ResNet
that are mostly geared
towards improving resNet
and so this is more recent
research has been done since then.
I'm going to go over these pretty fast,
and so just at a very high level.
If you're interested in
any of these you should
look at the papers, to have more details.
So the authors of ResNet
a little bit later on
in 2016 also had this paper
where they improved the
ResNet block design.
And so they basically
adjusted what were the layers
that were in the ResNet block path,
and showed this new
structure was able to have
a more direct path in order
for propagating information
throughout the network,
and you want to have a good
path to propagate
information all the way up,
and then back up all the way down again.
And so they showed that this
new block was better for that
and was able to give better performance.
There's also a Wide Residual
networks which this paper
argued that while ResNets
made networks much deeper
as well as added these
residual connections
and their argument was
that residuals are really
the important factor.
Having this residual construction,
and not necessarily having
extremely deep networks.
And so what they did was they
used wider residual blocks,
and so what this means is
just more filters in every
conv layer.
So before we might have
F filters per layer
and they use these factors
of K and said well,
every layer it's going to be
F times K filters instead.
And so, using these
wider layers they showed
that their 50 layer wide
ResNet was able to out-perform
the 152 layer original ResNet,
and it also had the
additional advantages of
increasing with this,
even with the same amount
of parameters, tit's more
computationally efficient
because you can parallelize
these with operations
more easily.
Right just convolutions with more neurons
just spread across more kernels
as opposed to depth
that's more sequential,
so it's more computationally
efficient to increase
your width.
So here you can see
this work is starting to
trying to understand the
contributions of width
and depth and residual connections,
and making some arguments
for one way versus the other.
And this other paper around the same time,
I think maybe a little
bit later, is ResNeXt,
and so this is again,
the creators of ResNet
continuing to work on
pushing the architecture.
And here they also had
this idea of okay, let's
indeed tackle this width
thing more but instead of just
increasing the width
of this residual block
through more filters they have structure.
And so within each residual
block, multiple parallel
pathways and they're going to call
the total number of these
pathways the cardinality.
And so it's basically
taking the one ResNet block
with the bottlenecks and having
it be relatively thinner,
but having multiple of
these done in parallel.
And so here you can also
see that this both have some
relation to this idea of wide networks,
as well as to has some connection
to the inception module
as well right where we
have these parallel,
these layers operating in parallel.
And so now this ResNeXt has
some flavor of that as well.
So another approach
towards improving ResNets
was this idea called Stochastic
Depth and in this work
the motivation is well let's look more
at this depth problem.
Once you get deeper and
deeper the typical problems
that you're going to have
vanishing gradients right.
You're not able to, your
gradients will get smaller
and eventually vanish as
you're trying to back propagate
them over very long layers,
or a large number of layers.
And so what their motivation
is well let's try to have
short networks during training
and they use this idea
of dropping out a subset of
the layers during training.
And so for a subset of the
layers they just drop out
the weights and they just set
it to identity connection,
and now what you get is you
have these shorter networks
during training, you can pass back your
gradients better.
It's also a little more
efficient, and then it's
kind of like the drop out right.
It has this sort of flavor
that you've seen before.
And then at test time you want
to use the full deep network
that you've trained.
So these are some of the
works that looking at the
resident architecture, trying
to understand different
aspects of it and trying
to improve ResNet training.
And so there's also some
works now that are going
beyond ResNet that are
saying well what are some non
ResNet architectures that
maybe can also work better,
or comparable or better to ResNets.
And so one idea is
FractalNet, which came out
pretty recently, and the
argument in FractalNet
is that while residual
representations maybe
are not actually necessary,
so this goes back
to what we were talking about earlier.
What's the motivation of
residual networks and it seems
to make sense and there's, you know,
good reasons for why this
should help but in this paper
they're saying that well here
is a different architecture
that we're introducing, there's
no residual representations.
We think that the key is
more about transitioning
effectively from shallow to deep networks,
and so they have this fractal architecture
which has if you look on the right here,
these layers where they compose
it in this fractal fashion.
And so there's both
shallow and deep pathways
to your output.
And so they have these
different length pathways,
they train them with
dropping out sub paths,
and so again it has this
dropout kind of flavor,
and then at test time they'll
use the entire fractal network
and they show that this was able to
get very good performance.
There's another idea
called Densely Connected
convolutional Networks,
DenseNet, and this idea
is now we have these
blocks that are called
dense blocks.
And within each block
each layer is going to be
connected to every other layer after it,
in this feed forward fashion.
So within this block,
your input to the block
is also the input to
every other conv layer,
and as you compute each conv output,
those outputs are now connected to every
layer after and then,
these are all concatenated
as input to the conv
layer, and they do some
they have some other
processes for reducing
the dimensions and keeping efficient.
And so their main takeaway from this,
is that they argue that
this is alleviating
a vanishing gradient problem
because you have all of these
very dense connections.
It strengthens feature propagation
and then also encourages
future use right because
there are so many of these
connections each feature
map that you're learning
is input in multiple
later layers and being
used multiple times.
So these are just a
couple of ideas that are
you know alternatives or
what can we do that's not
ResNets and yet is still performing either
comparably or better to
ResNets and so this is
another very active area
of current research.
You can see that a lot of this is looking
at the way how different layers
are connected to each other
and how depth is managed
in these networks.
And so one last thing
that I wanted to mention
quickly, is just efficient networks.
So this idea of efficiency
and you saw that GoogleNet
was a work that was
looking into this direction
of how can we have efficient
networks which are important
for you know a lot of
practical usage both training
as well as especially
deployment and so this is
another recent network
that's called SqueezeNet
which is looking at
very efficient networks.
They have these things
called fire modules,
which consists of a
squeeze layer with a lot of
one by one filters and
then this feeds then into
an expand layer with one by
one and three by three filters,
and they're showing that with
this kind of architecture
they're able to get AlexNet
level accuracy on ImageNet,
but with 50 times fewer parameters,
and then you can further do
network compression on this
to get up to 500 times
smaller than AlexNet
and just have the whole
network just be 0.5 megs.
And so this is a direction
of how do we have
efficient networks model compression
that we'll cover more in a lecture later,
but just giving you a hint of that.
OK so today in summary we've
talked about different kinds
of CNN Architectures.
We looked in-depth at four
of the main architectures
that you'll see in wide usage.
AlexNet, one of the early,
very popular networks.
VGG and GoogleNet which
are still widely used.
But ResNet is kind of
taking over as the thing
that you should be
looking most when you can.
We also looked at these other networks
in a little bit more depth at a brief
level overview.
And so the takeaway that these
models that are available
they're in a lot of
[mumbles] so you can use them
when you need them.
There's a trend toward
extremely deep networks,
but there's also significant
research now around
the design of how do we connect layers,
skip connections, what
is connected to what,
and also using these to
design your architecture
to improve gradient flow.
There's an even more recent
trend towards examining
what's the necessity
of depth versus width,
residual connections.
Trade offs, what's
actually helping matters,
and so there's a lot of these recent works
in this direction that you can look into
some of the ones I pointed
out if you are interested.
And next time we'll talk about
Recurrent neural networks.
Thanks.

