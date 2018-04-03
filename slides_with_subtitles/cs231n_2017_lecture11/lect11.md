﻿
- Hello, hi.
So I want to get started.
Welcome to CS 231N Lecture 11.
We're going to talk about
today detection segmentation
and a whole bunch of other
really exciting topics
around core computer vision tasks.
But as usual, a couple
administrative notes.
So last time you obviously
took the midterm,
we didn't have lecture,
hopefully that went okay
for all of you but so we're
going to work on grading
the midterm this week, but as a reminder
please don't make any public discussions
about the midterm questions
or answers or whatever
until at least tomorrow
because there are still
some people taking makeup midterms today
and throughout the rest of the week
so we just ask you that
you refrain from talking
publicly about midterm questions.
Why don't you wait until Monday?
[laughing]
Okay, great.
So we're also starting to
work on midterm grading.
We'll get those back to
you as soon as you can,
as soon as we can.
We're also starting to work
on grading assignment two
so there's a lot of grading
being done this week.
The TA's are pretty busy.
Also a reminder for you guys,
hopefully you've been working
hard on your projects now that most of you
are done with the midterm
so your project milestones
will be due on Tuesday so
any sort of last minute
changes that you had in your projects,
I know some people
decided to switch projects
after the proposal, some
teams reshuffled a little bit,
that's fine but your
milestone should reflect
the project that you're actually doing
for the rest of the quarter.
So hopefully that's going out well.
I know there's been a
lot of worry and stress
on Piazza, wondering
about assignment three.
So we're working on that as hard as we can
but that's actually a
bit of a new assignment,
it's changing a bit from last year
so it will be out as soon as possible,
hopefully today or tomorrow.
Although we promise that
whenever it comes out
you'll have two weeks to finish it
so try not to stress
out about that too much.
But I'm pretty excited,
I think assignment three
will be really cool, has a lot of cool,
it'll cover a lot of really cool material.
So another thing, last time in lecture
we mentioned this thing
called the Train Game
which is this really cool
thing we've been working on
sort of as a side project a little bit.
So this is an interactive
tool that you guys can go on
and use to explore a
little bit the process
of tuning hyperparameters
in practice so we hope that,
so this is again totally
not required for the course.
Totally optional, but
if you do we will offer
a small amount of extra
credit for those of you
who want to do well and
participate on this.
And we'll send out
exactly some more details
later this afternoon on Piazza.
But just a bit of a demo for
what exactly is this thing.
So you'll get to go in
and we've changed the name
from Train Game to HyperQuest
because you're questing
to solve, to find the best
hyperparameters for your model
so this is really cool,
it'll be an interactive tool
that you can use to explore
the training of hyperparameters
interactively in your browser.
So you'll login with
your student ID and name.
You'll fill out a little survey with some
of your experience on deep learning
then you'll read some instructions.
So in this game you'll be
shown some random data set
on every trial.
This data set might be
images or it might be vectors
and your goal is to
train a model by picking
the right hyperparameters
interactively to perform
as well as you can on the validation set
of this random data set.
And it'll sort of keep
track of your performance
over time and there'll be a leaderboard,
it'll be really cool.
So every time you play the game,
you'll get some statistics
about your data set.
In this case we're doing
a classification problem
with 10 classes.
You can see down at the bottom
you have these statistics
about random data set, we have 10 classes.
The input data size is three by 32 by 32
so this is some image
data set and we can see
that in this case we have 8500 examples
in the training set and 1500
examples in the validation set.
These are all random, they'll change
a little bit every time.
Based on these data set statistics
you'll make some choices
on your initial learning rate,
your initial network size,
and your initial dropout rate.
Then you'll see a screen
like this where it'll run
one epoch with those
chosen hyperparameters,
show you on the right
here you'll see two plots.
One is your training and validation loss
for that first epoch.
Then you'll see your training
and validation accuracy
for that first epoch and
based on the gaps that you see
in these two graphs you can
make choices interactively
to change the learning
rates and hyperparameters
for the next epoch.
So then you can either
choose to continue training
with the current or
changed hyperparameters,
you can also stop training,
or you can revert to
go back to the previous checkpoint
in case things got really messed up.
So then you'll get to make some choice,
so here we'll decide to continue training
and in this case you could
go and set new learning rates
and new hyperparameters for
the next epoch of training.
You can also, kind of interesting here,
you can actually grow
the network interactively
during training in this demo.
There's this cool trick
from a couple recent papers
where you can either take existing layers
and make them wider or add
new layers to the network
in the middle of training
while still maintaining
the same function in the
network so you can do that
to increase the size of
your network in the middle
of training here which is kind of cool.
So then you'll make
choices over several epochs
and eventually your
final validation accuracy
will be recorded and we'll
have some leaderboard
that compares your score on that data set
to some simple baseline models.
And depending on how well
you do on this leaderboard
we'll again offer some small
amounts of extra credit
for those of you who
choose to participate.
So this is again, totally
optional, but I think
it can be a really cool
learning experience for you guys
to play around with and
explore how hyperparameters
affect the learning process.
Also, it's really useful for us.
You'll help science out by
participating in this experiment.
We're pretty interested in
seeing how people behave
when they train neural networks
so you'll be helping us out
as well if you decide to play this.
But again, totally optional, up to you.
Any questions on that?
Hopefully at some point but it's.
So the question was will this be a paper
or whatever eventually?
Hopefully but it's really
early stages of this project
so I can't make any
promises but I hope so.
But I think it'll be really cool.
[laughing]
Yeah, so the question is
how can you add layers
during training?
I don't really want to
get into that right now
but the paper to read is
Net2Net by Ian Goodfellow's
one of the authors and
there's another paper
from Microsoft called Network Morphism.
So if you read those two papers
you can see how this works.
Okay, so last time, a bit of a reminder
before we had the midterm
last time we talked
about recurrent neural networks.
We saw that recurrent
neural networks can be used
for different types of problems.
In addition to one to one
we can do one to many,
many to one, many to many.
We saw how this can apply
to language modeling
and we saw some cool examples
of applying neural networks
to model different sorts of
languages at the character level
and we sampled these
artificial math and Shakespeare
and C source code.
We also saw how similar
things could be applied
to image captioning by connecting
a CNN feature extractor
together with an RNN language model.
And we saw some really
cool examples of that.
We also talked about the
different types of RNN's.
We talked about this Vanilla RNN.
I also want to mention that
this is sometimes called
a Simple RNN or an Elman RNN so you'll see
all of these different
terms in literature.
We also talked about the Long
Short Term Memory or LSTM.
And we talked about how the gradient,
the LSTM has this crazy set of equations
but it makes sense because it
helps improve gradient flow
during back propagation
and helps this thing model
more longer term dependencies
in our sequences.
So today we're going to
switch gears and talk about
a whole bunch of different exciting tasks.
We're going to talk about, so
so far we've been talking about
mostly the image classification problem.
Today we're going to talk
about various types of other
computer vision tasks where
you actually want to go in
and say things about the spatial
pixels inside your images
so we'll see segmentation,
localization, detection,
a couple other different
computer vision tasks
and how you can approach these
with convolutional neural networks.
So as a bit of refresher,
so far the main thing
we've been talking about in this class
is image classification so
here we're going to have
some input image come in.
That input image will go through
some deep convolutional network,
that network will give
us some feature vector
of maybe 4096 dimensions
in the case of AlexNet RGB
and then from that final feature vector
we'll have some fully-connected,
some final fully-connected layer
that gives us 1000 numbers
for the different class scores
that we care about where
1000 is maybe the number
of classes in ImageNet in this example.
And then at the end of the day
what the network does is we input an image
and then we output a single category label
saying what is the content of
this entire image as a whole.
But this is maybe the
most basic possible task
in computer vision and
there's a whole bunch
of other interesting types of tasks
that we might want to
solve using deep learning.
So today we're going to talk about several
of these different tasks and
step through each of these
and see how they all
work with deep learning.
So we'll talk about these more in detail
about what each problem is as we get to it
but this is kind of a summary slide
that we'll talk first about
semantic segmentation.
We'll talk about classification
and localization,
then we'll talk about object detection,
and finally a couple brief words
about instance segmentation.
So first is the problem
of semantic segmentation.
In the problem of semantic segmentation,
we want to input an image
and then output a decision
of a category for every
pixel in that image
so for every pixel in this, so
this input image for example
is this cat walking through
the field, he's very cute.
And in the output we want to say
for every pixel is that pixel
a cat or grass or sky or trees
or background or some
other set of categories.
So we're going to have
some set of categories
just like we did in the
image classification case
but now rather than
assigning a single category
labeled to the entire
image, we want to produce
a category label for each
pixel of the input image.
And this is called semantic segmentation.
So one interesting thing
about semantic segmentation
is that it does not
differentiate instances
so in this example on the
right we have this image
with two cows where
they're standing right next
to each other and when
we're talking about semantic
segmentation we're just
labeling all the pixels
independently for what is
the category of that pixel.
So in the case like this
where we have two cows
right next to each other
the output does not make
any distinguishing, does not distinguish
between these two cows.
Instead we just get a whole mass of pixels
that are all labeled as cow.
So this is a bit of a shortcoming
of semantic segmentation
and we'll see how we can fix this later
when we move to instance segmentation.
But at least for now we'll just talk about
semantic segmentation first.
So you can imagine maybe using a class,
so one potential approach for attacking
semantic segmentation might
be through classification.
So there's this, you could use this idea
of a sliding window approach
to semantic segmentation.
So you might imagine that
we take our input image
and we break it up into many
many small, tiny local crops
of the image so in this
example we've taken
maybe three crops from
around the head of this cow
and then you could imagine
taking each of those crops
and now treating this as
a classification problem.
Saying for this crop, what is the category
of the central pixel of the crop?
And then we could use
all the same machinery
that we've developed for
classifying entire images
but now just apply it on crops rather than
on the entire image.
And this would probably
work to some extent
but it's probably not a very good idea.
So this would end up being super super
computationally expensive
because we want to label
every pixel in the image,
we would need a separate
crop for every pixel in
that image and this would be
super super expensive to
run forward and backward
passes through.
And moreover, we're actually,
if you think about this
we can actually share
computation between different
patches so if you're trying
to classify two patches
that are right next to each
other and actually overlap
then the convolutional
features of those patches
will end up going through
the same convolutional layers
and we can actually share
a lot of the computation
when applying this to separate passes
or when applying this type of approach
to separate patches in the image.
So this is actually a terrible
idea and nobody does this
and you should probably not do this
but it's at least the first
thing you might think of
if you were trying to think
about semantic segmentation.
Then the next idea that works a bit better
is this idea of a fully
convolutional network right.
So rather than extracting
individual patches from the image
and classifying these
patches independently,
we can imagine just having
our network be a whole giant
stack of convolutional layers
with no fully connected
layers or anything so in this
case we just have a bunch
of convolutional layers that
are all maybe three by three
with zero padding or something like that
so that each convolutional
layer preserves the spatial size
of the input and now if we pass our image
through a whole stack of
these convolutional layers,
then the final convolutional
layer could just output
a tensor of something by C by H by W
where C is the number of
categories that we care about
and you could see this
tensor as just giving
our classification scores for every pixel
in the input image at every
location in the input image.
And we could compute this all at once
with just some giant stack
of convolutional layers.
And then you could imagine
training this thing
by putting a classification
loss at every pixel
of this output, taking an
average over those pixels
in space, and just training
this kind of network
through normal, regular back propagation.
Question?
Oh, the question is how do you develop
training data for this?
It's very expensive right.
So the training data for this would be
we need to label every
pixel in those input images
so there's tools that
people sometimes have online
where you can go in and
sort of draw contours
around the objects and
then fill in regions
but in general getting
this kind of training data
is very expensive.
Yeah, the question is
what is the loss function?
So here since we're making
a classification decision
per pixel then we put a cross entropy loss
on every pixel of the output.
So we have the ground truth category label
for every pixel in the output,
then we compute across entropy loss
between every pixel in the output
and the ground truth pixels and then
take either a sum or an average over space
and then sum or average
over the mini-batch.
Question?
Yeah, yeah.
Yeah, the question is do we assume
that we know the categories?
So yes, we do assume that we
know the categories up front
so this is just like the
image classification case.
So an image classification we
know at the start of training
based on our data set that
maybe there's 10 or 20
or 100 or 1000 classes that we care about
for this data set and
then here we are fixed
to that set of classes that
are fixed for the data set.
So this model is relatively simple
and you can imagine this
working reasonably well
assuming that you tuned all
the hyperparameters right
but it's kind of a problem right.
So in this setup, since
we're applying a bunch
of convolutions that
are all keeping the same
spatial size of the input image,
this would be super super expensive right.
If you wanted to do
convolutions that maybe have
64 or 128 or 256 channels for
those convolutional filters
which is pretty common in
a lot of these networks,
then running those convolutions
on this high resolution
input image over a
sequence of layers would be
extremely computationally expensive
and would take a ton of memory.
So in practice, you don't
usually see networks
with this architecture.
Instead you tend to see
networks that look something
like this where we have some downsampling
and then some upsampling
of the feature map
inside the image.
So rather than doing all the convolutions
of the full spatial
resolution of the image,
we'll maybe go through a small number
of convolutional layers
at the original resolution
then downsample that
feature map using something
like max pooling or strided convolutions
and sort of downsample, downsample,
so we have convolutions in downsampling
and convolutions in downsampling
that look much like a lot of
the classification networks
that you see but now
the difference is that
rather than transitioning
to a fully connected layer
like you might do in an
image classification setup,
instead we want to increase
the spatial resolution
of our predictions in the
second half of the network
so that our output image
can now be the same size
as our input image and this ends up being
much more computationally efficient
because you can make the network very deep
and work at a lower spatial resolution
for many of the layers at
the inside of the network.
So we've already seen
examples of downsampling
when it comes to convolutional networks.
We've seen that you can
do strided convolutions
or various types of pooling
to reduce the spatial size
of the image inside a
network but we haven't
really talked about
upsampling and the question
you might be wondering is
what are these upsampling
layers actually look
like inside the network?
And what are our strategies
for increasing the size
of a feature map inside the network?
Sorry, was there a question in the back?
Yeah, so the question
is how do we upsample?
And the answer is that's the topic
of the next couple slides.
[laughing]
So one strategy for
upsampling is something like
unpooling so we have
this notion of pooling
to downsample so we talked
about average pooling
or max pooling so when we
talked about average pooling
we're kind of taking a spatial average
within a receptive field
of each pooling region.
One kind of analog for
upsampling is this idea
of nearest neighbor unpooling.
So here on the left we see this example
of nearest neighbor
unpooling where our input
is maybe some two by
two grid and our output
is a four by four grid
and now in our output
we've done a two by two
stride two nearest neighbor
unpooling or upsampling
where we've just duplicated
that element for every
point in our two by two
receptive field of the unpooling region.
Another thing you might see
is this bed of nails unpooling
or bed of nails upsampling
where you'll just take,
again we have a two by two receptive field
for our unpooling regions
and then you'll take the,
in this case you make it all
zeros except for one element
of the unpooling region so
in this case we've taken
all of our inputs and
always put them in the upper
left hand corner of this unpooling region
and everything else is zeros.
And this is kind of like a bed of nails
because the zeros are very flat,
then you've got these things poking up
for the values at these
various non-zero regions.
Another thing that you see
sometimes which was alluded to
by the question a minute ago
is this idea of max unpooling
so in a lot of these networks
they tend to be symmetrical
where we have a downsampling
portion of the network
and then an upsampling
portion of the network
with a symmetry between those
two portions of the network.
So sometimes what you'll see
is this idea of max unpooling
where for each unpooling,
for each upsampling layer,
it is associated with
one of the pooling layers
in the first half of the network
and now in the first half,
in the downsampling when we do max pooling
we'll actually remember which
element of the receptive field
during max pooling was
used to do the max pooling
and now when we go through
the rest of the network
then we'll do something that
looks like this bed of nails
upsampling except rather than
always putting the elements
in the same position,
instead we'll stick it
into the position that was
used in the corresponding
max pooling step earlier in the network.
I'm not sure if that explanation was clear
but hopefully the picture makes sense.
Yeah, so then you just end up
filling the rest with zeros.
So then you fill the rest with zeros
and then you stick the elements
from the low resolution
patch up into the high resolution patch
at the points where the
max pooling took place
at the corresponding max pooling there.
Okay, so that's kind
of an interesting idea.
Sorry, question?
Oh yeah, so the question
is why is this a good idea?
Why might this matter?
So the idea is that when we're
doing semantic segmentation
we want our predictions
to be pixel perfect right.
We kind of want to get
those sharp boundaries
and those tiny details in
our predictive segmentation
so now if you're doing this max pooling,
there's this sort of
heterogeneity that's happening
inside the feature map
due to the max pooling
where from the low resolution
image you don't know,
you're sort of losing spatial
information in some sense
by you don't know where that
feature vector came from
in the local receptive
field after max pooling.
So if you actually unpool
by putting the vector
in the same slot you might
think that that might help us
handle these fine details
a little bit better
and help us preserve some
of that spatial information
that was lost during max pooling.
Question?
The question is does this make
things easier for back prop?
Yeah, I guess, I don't think
it changes the back prop
dynamics too much because
storing these indices
is not a huge computational overhead.
They're pretty small in
comparison to everything else.
So another thing that you'll see sometimes
is this idea of transpose convolution.
So transpose convolution,
so for these various types
of unpooling that we just talked about,
these bed of nails, this nearest neighbor,
this max unpooling, all
of these are kind of
a fixed function, they're
not really learning exactly
how to do the upsampling so
if you think about something
like strided convolution,
strided convolution
is kind of like a learnable
layer that learns the way
that the network wants
to perform downsampling
at that layer.
And by analogy with that
there's this type of layer
called a transpose
convolution that lets us do
kind of learnable upsampling.
So it will both upsample the feature map
and learn some weights about how it wants
to do that upsampling.
And this is really just
another type of convolution
so to see how this works
remember how a normal
three by three stride one pad
one convolution would work.
That for this kind of normal convolution
that we've seen many
times now in this class,
our input might by four by four,
our output might be four by four,
and now we'll have this
three by three kernel
and we'll take an inner product between,
we'll plop down that kernel
at the corner of the image,
take an inner product,
and that inner product
will give us the value and the activation
in the upper left hand
corner of our output.
And we'll repeat this
for every receptive field
in the image.
Now if we talk about strided convolution
then strided convolution ends
up looking pretty similar.
However, our input is
maybe a four by four region
and our output is a two by two region.
But we still have this idea of taking,
of there being some three
by three filter or kernel
that we plop down in
the corner of the image,
take an inner product
and use that to compute
a value of the activation and the output.
But now with strided
convolution the idea is that
we're moving that, rather
than popping down that filter
at every possible point in the input,
instead we're going to move
the filter by two pixels
in the input every time we
move the filter by one pixel,
every time we move by
one pixel in the output.
Right so this stride
of two gives us a ratio
between how much do we move in the input
versus how much do we move in the output.
So when you do a strided
convolution with stride two
this ends up downsampling
the image or the feature map
by a factor of two in
kind of a learnable way.
And now a transpose convolution
is sort of the opposite
in a way so here our input
will be a two by two region
and our output will be
a four by four region.
But now the operation that we perform
with transpose convolution
is a little bit different.
Now so rather than taking an inner product
instead what we're going
to do is we're going to
take the value of our input feature map
at that upper left hand
corner and that'll be
some scalar value in the
upper left hand corner.
We're going to multiply the
filter by that scalar value
and then copy those values
over to this three by three
region in the output so rather
than taking an inner product
with our filter and the
input, instead our input
gives weights that we will
use to weight the filter
and then our output will be
weighted copies of the filter
that are weighted by
the values in the input.
And now we can do this
sort of same ratio trick
in order to upsample so
now when we move one pixel
in the input now we can
plop our filter down
two pixels away in the output
and it's the same trick
that now the blue pixel in
the input is some scalar value
and we'll take that scalar value,
multiply it by the values in the filter,
and copy those weighted filter values
into this new region in the output.
The tricky part is that
sometimes these receptive fields
in the output can overlap
now and now when these
receptive fields in the output overlap
we just sum the results in the output.
So then you can imagine
repeating this everywhere
and repeating this process everywhere
and this ends up doing sort
of a learnable upsampling
where we use these learned
convolutional filter weights
to upsample the image and
increase the spatial size.
By the way, you'll see this operation go
by a lot of different names in literature.
Sometimes this gets called
things like deconvolution
which I think is kind of a
bad name but you'll see it
out there in papers so from a
signal processing perspective
deconvolution means the inverse
operation to convolution
which this is not however
you'll frequently see this
type of layer called a deconvolution layer
in some deep learning
papers so be aware of that,
watch out for that terminology.
You'll also sometimes see
this called upconvolution
which is kind of a cute name.
Sometimes it gets called
fractionally strided convolution
because if we think of the
stride as the ratio in step
between the input and the output
then now this is something
like a stride one half
convolution because of this ratio
of one to two between steps in the input
and steps in the output.
This also sometimes gets
called a backwards strided
convolution because if you think about it
and work through the math
this ends up being the same,
the forward pass of a
transpose convolution
ends up being the same
mathematical operation
as the backwards pass
in a normal convolution
so you might have to take my word for it,
that might not be super obvious
when you first look at this
but that's kind of a neat
fact so you'll sometimes
see that name as well.
And as maybe a bit of
a more concrete example
of what this looks like I think
it's maybe a little easier
to see in one dimension so if we imagine,
so here we're doing a three
by three transpose convolution
in one dimension.
Sorry, not three by three, a three by one
transpose convolution in one dimension.
So our filter here is just three numbers.
Our input is two numbers
and now you can see
that in our output we've
taken the values in the input,
used them to weight the
values of the filter
and plopped down those
weighted filters in the output
with a stride of two and now
where these receptive fields
overlap in the output then we sum.
So you might be wondering,
this is kind of a funny name.
Where does the name transpose
convolution come from
and why is that actually my preferred name
for this operation?
So that comes from this kind of
neat interpretation of convolution.
So it turns out that any
time you do convolution
you can always write convolution
as a matrix multiplication.
So again, this is kind of easier to see
with a one-dimensional example
but here we've got some weight.
So we're doing a
one-dimensional convolution
of a weight vector x
which has three elements,
and an input vector, a vector,
which has four elements,
A, B, C, D.
So here we're doing a
three by one convolution
with stride one and you
can see that we can frame
this whole operation as
a matrix multiplication
where we take our convolutional kernel x
and turn it into some matrix capital X
which contains copies of
that convolutional kernel
that are offset by different regions.
And now we can take this
giant weight matrix X
and do a matrix vector
multiplication between x
and our input a and this
just produces the same result
as convolution.
And now with transpose convolution means
that we're going to take
this same weight matrix
but now we're going to
multiply by the transpose
of that same weight matrix.
So here you can see the same
example for this stride one
convolution on the left and
the corresponding stride one
transpose convolution on the right.
And if you work through
the details you'll see
that when it comes to stride one,
a stride one transpose
convolution also ends up being
a stride one normal convolution
so there's a little bit
of details in the way that
the border and the padding
are handled but it's
fundamentally the same operation.
But now things look different
when you talk about a stride of two.
So again, here on the left
we can take a stride two
convolution and write out
this stride two convolution
as a matrix multiplication.
And now the corresponding
transpose convolution
is no longer a convolution so if you look
through this weight matrix and think about
how convolutions end up
getting represented in this way
then now this transposed
matrix for the stride two
convolution is something
fundamentally different
from the original normal
convolution operation
so that's kind of the
reasoning behind the name
and that's why I think that's
kind of the nicest name
to call this operation by.
Sorry, was there a question?
Sorry?
It's very possible there's
a typo in the slide
so please point out on
Piazza and I'll fix it
but I hope the idea was clear.
Is there another question?
Okay, thank you [laughing].
Yeah, so, oh no lots of questions.
Yeah, so the issue is why
do we sum and not average?
So the reason we sum is due
to this transpose convolution
formula zone so that's
the reason why we sum
but you're right that you actually,
this is kind of a problem
that the magnitudes
will actually vary in the output depending
on how many receptive
fields were in the output.
So actually in practice this
is something that people
started to point out very
recently and somewhat
switched away from this
stride, so using three by three
stride two transpose
convolution upsampling
can sometimes produce some
checkerboard artifacts
in the output exactly due to that problem.
So what I've seen in a
couple more recent papers
is maybe to use four by four stride two
or two by two stride two
transpose convolution
for upsampling and that helps alleviate
that problem a little bit.
Yeah, so the question is what
is a stride half convolution
and where does that terminology come from?
I think that was from my paper.
So that was actually, yes
that was definitely this.
So at the time I was writing that paper
I was kind of into the name
fractionally strided convolution
but after thinking about
it a bit more I think
transpose convolution is
probably the right name.
So then this idea of semantic segmentation
actually ends up being pretty natural.
You just have this giant
convolutional network
with downsampling and
upsampling inside the network
and now our downsampling will
be by strided convolution
or pooling, our upsampling will
be by transpose convolution
or various types of
unpooling or upsampling
and we can train this
whole thing end to end
with back propagation using
this cross entropy loss
over every pixel.
So this is actually pretty
cool that we can take
a lot of the same machinery
that we already learned
for image classification
and now just apply it
very easily to extend
to new types of problems
so that's super cool.
So the next task that I want
to talk about is this idea
of classification plus localization.
So we've talked about
image classification a lot
where we want to just
assign a category label
to the input image but
sometimes you might want to know
a little bit more about the image.
In addition to predicting
what the category is,
in this case the cat, you
might also want to know
where is that object in the image?
So in addition to predicting
the category label cat,
you might also want to draw a bounding box
around the region of
the cat in that image.
And classification plus localization,
the distinction here between
this and object detection
is that in the localization
scenario you assume
ahead of time that you know
there's exactly one object
in the image that you're looking
for or maybe more than one
but you know ahead of time
that we're going to make
some classification
decision about this image
and we're going to produce
exactly one bounding box
that's going to tell us
where that object is located
in the image so we
sometimes call that task
classification plus localization.
And again, we can reuse a
lot of the same machinery
that we've already learned
from image classification
in order to tackle this problem.
So kind of a basic
architecture for this problem
looks something like this.
So again, we have our input image,
we feed our input image through some giant
convolutional network, this is Alex,
this is AlexNet for
example, which will give us
some final vector summarizing
the content of the image.
Then just like before we'll
have some fully connected layer
that goes from that final
vector to our class scores.
But now we'll also have
another fully connected layer
that goes from that
vector to four numbers.
Where the four numbers are something like
the height, the width,
and the x and y positions
of that bounding box.
And now our network will
produce these two different
outputs, one is this set of class scores,
and the other are these four
numbers giving the coordinates
of the bounding box in the input image.
And now during training time,
when we train this network
we'll actually have two
losses so in this scenario
we're sort of assuming a
fully supervised setting
so we assume that each
of our training images
is annotated with both a
category label and also
a ground truth bounding box
for that category in the image.
So now we have two loss functions.
We have our favorite
softmax loss that we compute
using the ground truth category label
and the predicted class scores,
and we also have some
kind of loss that gives us
some measure of dissimilarity
between our predicted
coordinates for the bounding box
and our actual coordinates
for the bounding box.
So one very simple thing
is to just take an L2 loss
between those two and that's
kind of the simplest thing
that you'll see in
practice although sometimes
people play around with
this and maybe use L1
or smooth L1 or they
parametrize the bounding box
a little bit differently but
the idea is always the same,
that you have some regression loss
between your predicted
bounding box coordinates
and the ground truth
bounding box coordinates.
Question?
Sorry, go ahead.
So the question is, is this a good idea
to do all at the same time?
Like what happens if you misclassify,
should you even look
at the box coordinates?
So sometimes people get fancy with it,
so in general it works okay.
It's not a big problem, you
can actually train a network
to do both of these
things at the same time
and it'll figure it out but
sometimes things can get tricky
in terms of misclassification
so sometimes what you'll see
for example is that rather
than predicting a single box
you might make predictions
like a separate prediction
of the box for each category
and then only apply loss
to the predicted box corresponding
to the ground truth category.
So people do get a little
bit fancy with these things
that sometimes helps a bit in practice.
But at least this basic
setup, it might not be perfect
or it might not be
optimal but it will work
and it will do something.
Was there a question in the back?
Yeah, so that's the
question is do these losses
have different units, do
they dominate the gradient?
So this is what we call a multi-task loss
so whenever we're taking
derivatives we always
want to take derivative
of a scalar with respect
to our network parameters
and use that derivative
to take gradient steps.
But now we've got two scalars
that we want to both minimize
so what you tend to do in
practice is have some additional
hyperparameter that
gives you some weighting
between these two losses so
you'll take a weighted sum
of these two different loss functions
to give our final scalar loss.
And then you'll take your
gradients with respect
to this weighted sum of the two losses.
And this ends up being
really really tricky
because this weighting
parameter is a hyperparameter
that you need to set but
it's kind of different
from some of the other hyperparameters
that we've seen so far in the past right
because this weighting hyperparameter
actually changes the
value of the loss function
so one thing that you might often look at
when you're trying to set hyperparameters
is you might make different
hyperparameter choices
and see what happens to the loss
under different choices
of hyperparameters.
But in this case because
the loss actually,
because the hyperparameter
affects the absolute value
of the loss making those
comparisons becomes kind of tricky.
So setting that hyperparameter
is somewhat difficult.
And in practice, you
kind of need to take it
on a case by case basis
for exactly the problem
you're solving but my
general strategy for this
is to have some other
metric of performance
that you care about other
than the actual loss value
which then you actually use
that final performance metric
to make your cross validation
choices rather than looking
at the value of the loss
to make those choices.
Question?
So the question is why do
we do this all at once?
Why not do this separately?
Yeah, so the question is why
don't we fix the big network
and then just only learn
separate fully connected layers
for these two tasks?
People do do that sometimes
and in fact that's probably
the first thing you
should try if you're faced
with a situation like this but in general
whenever you're doing transfer learning
you always get better
performance if you fine tune
the whole system jointly
because there's probably
some mismatch between the features,
if you train on ImageNet and
then you use that network
for your data set you're going
to get better performance
on your data set if you can
also change the network.
But one trick you might
see in practice sometimes
is that you might freeze that network
then train those two things
separately until convergence
and then after they
converge then you go back
and jointly fine tune the whole system.
So that's a trick that sometimes people do
in practice in that situation.
And as I've kind of
alluded to this big network
is often a pre-trained
network that is taken
from ImageNet for example.
So a bit of an aside,
this idea of predicting
some fixed number of
positions in the image
can be applied to a lot
of different problems
beyond just classification
plus localization.
One kind of cool example
is human pose estimation.
So here we want to take an input image
is a picture of a person.
We want to output the
positions of the joints
for that person and this
actually allows the network
to predict what is the pose of the human.
Where are his arms, where are
his legs, stuff like that,
and generally most people have
the same number of joints.
That's a bit of a simplifying assumption,
it might not always be true
but it works for the network.
So for example one
parameterization that you might see
in some data sets is
define a person's pose
by 14 joint positions.
Their feet and their knees and their hips
and something like that and
now when we train the network
then we're going to input
this image of a person
and now we're going to output
14 numbers in this case
giving the x and y coordinates
for each of those 14 joints.
And then you apply some
kind of regression loss
on each of those 14
different predicted points
and just train this network
with back propagation again.
Yeah, so you might see an L2
loss but people play around
with other regression losses here as well.
Question?
So the question is what do I mean
when I say regression loss?
So I mean something
other than cross entropy
or softmax right.
When I say regression loss I usually mean
like an L2 Euclidean loss or an L1 loss
or sometimes a smooth L1 loss.
But in general classification
versus regression
is whether your output is
categorical or continuous
so if you're expecting
a categorical output
like you ultimately want to
make a classification decision
over some fixed number of categories
then you'll think about
a cross entropy loss,
softmax loss or these
SVM margin type losses
that we talked about already in the class.
But if your expected output is
to be some continuous value,
in this case the position of these points,
then your output is
continuous so you tend to use
different types of losses
in those situations.
Typically an L2, L1, different
kinds of things there.
So sorry for not clarifying that earlier.
But the bigger point
here is that for any time
you know that you want
to make some fixed number
of outputs from your network,
if you know for example.
Maybe you knew that you wanted to,
you knew that you always
are going to have pictures
of a cat and a dog and
you want to predict both
the bounding box of the cat
and the bounding box of the dog
in that case you'd know
that you have a fixed number
of outputs for each input
so you might imagine
hooking up this type of regression
classification plus localization framework
for that problem as well.
So this idea of some fixed
number of regression outputs
can be applied to a lot
of different problems
including pose estimation.
So the next task that I want to
talk about is object detection
and this is a really meaty topic.
This is kind of a core
problem in computer vision
and you could probably
teach a whole seminar class
on just the history of object detection
and various techniques applied there.
So I'll be relatively
brief and try to go over
the main big ideas of object
detection plus deep learning
that have been used in
the last couple of years.
But the idea in object detection is that
we again start with some
fixed set of categories
that we care about, maybe cats
and dogs and fish or whatever
but some fixed set of categories
that we're interested in.
And now our task is that
given our input image,
every time one of those
categories appears in the image,
we want to draw a box around
it and we want to predict
the category of that
box so this is different
from classification plus localization
because there might be a
varying number of outputs
for every input image.
You don't know ahead of time
how many objects you expect
to find in each image so that's,
this ends up being a
pretty challenging problem.
So we've seen graphs, so
this is kind of interesting.
We've seen this graph
many times of the ImageNet
classification performance
as a function of years
and we saw that it just got
better and better every year
and there's been a similar
trend with object detection
because object detection
has again been one
of these core problems in computer vision
that people have cared
about for a very long time.
So this slide is due to Ross Girshick
who's worked on this
problem a lot and it shows
the progression of object
detection performance
on this one particular
data set called PASCAL VOC
which has been relatively
used for a long time
in the object detection community.
And you can see that up until about 2012
performance on object
detection started to stagnate
and slow down a little
bit and then in 2013
was when some of the first
deep learning approaches
to object detection came
around and you could see
that performance just shot up very quickly
getting better and better year over year.
One thing you might notice is
that this plot ends in 2015
and it's actually continued
to go up since then
so the current state of
the art in this data set
is well over 80 and in
fact a lot of recent papers
don't even report results
on this data set anymore
because it's considered too easy.
So it's a little bit hard to know,
I'm not actually sure what is
the state of the art number
on this data set but it's
off the top of this plot.
Sorry, did you have a question?
Nevermind.
Okay, so as I already
said this is different
from localization because
there might be differing
numbers of objects for each image.
So for example in this
cat on the upper left
there's only one object
so we only need to predict
four numbers but now for
this image in the middle
there's three animals there
so we need our network
to predict 12 numbers, four coordinates
for each bounding box.
Or in this example of many
many ducks then you want
your network to predict
a whole bunch of numbers.
Again, four numbers for each duck.
So this is quite different
from object detection.
Sorry object detection is quite
different from localization
because in object detection
you might have varying numbers
of objects in the image and
you don't know ahead of time
how many you expect to find.
So as a result, it's kind of
tricky if you want to think
of object detection as
a regression problem.
So instead, people tend to
work, use kind of a different
paradigm when thinking
about object detection.
So one approach that's very
common and has been used
for a long time in computer
vision is this idea
of sliding window approaches
to object detection.
So this is kind of similar to this idea
of taking small patches and applying that
for semantic segmentation and we can apply
a similar idea for object detection.
So the ideas is that
we'll take different crops
from the input image, in
this case we've got this crop
in the lower left hand corner of our image
and now we take that crop,
feed it through our convolutional network
and our convolutional network does
a classification decision
on that input crop.
It'll say that there's no dog
here, there's no cat here,
and then in addition to the
categories that we care about
we'll add an additional
category called background
and now our network can predict background
in case it doesn't see
any of the categories
that we care about, so
then when we take this crop
from the lower left hand corner here
then our network would
hopefully predict background
and say that no, there's no object here.
Now if we take a different
crop then our network
would predict dog yes,
cat no, background no.
We take a different crop we get dog yes,
cat no, background no.
Or a different crop, dog
no, cat yes, background no.
Does anyone see a problem here?
Yeah, the question is how
do you choose the crops?
So this is a huge problem right.
Because there could be any
number of objects in this image,
these objects could appear
at any location in the image,
these objects could appear
at any size in the image,
these objects could also
appear at any aspect ratio
in the image, so if you want
to do kind of a brute force
sliding window approach
you'd end up having to test
thousands, tens of thousands,
many many many many
different crops in order
to tackle this problem
with a brute force
sliding window approach.
And in the case where
every one of those crops
is going to be fed through a
giant convolutional network,
this would be completely
computationally intractable.
So in practice people don't
ever do this sort of brute force
sliding window approach
for object detection
using convolutional networks.
Instead there's this cool line of work
called region proposals that comes from,
this is not using deep learning typically.
These are slightly more
traditional computer vision
techniques but the idea is
that a region proposal network
kind of uses more traditional
signal processing,
image processing type
things to make some list
of proposals for where,
so given an input image,
a region proposal network
will then give you something
like a thousand boxes where
an object might be present.
So you can imagine that
maybe we do some local,
we look for edges in the
image and try to draw boxes
that contain closed edges
or something like that.
These various types of
image processing approaches,
but these region proposal
networks will basically look
for blobby regions in our
input image and then give us
some set of candidate proposal regions
where objects might be potentially found.
And these are relatively fast-ish to run
so one common example of
a region proposal method
that you might see is something
called Selective Search
which I think actually gives
you 2000 region proposals,
not the 1000 that it says on the slide.
So you kind of run this
thing and then after
about two seconds of turning on your CPU
it'll spit out 2000 region
proposals in the input image
where objects are likely to be found
so there'll be a lot of noise in those.
Most of them will not be true objects
but there's a pretty high recall.
If there is an object in
the image then it does tend
to get covered by these region proposals
from Selective Search.
So now rather than applying
our classification network
to every possible location
and scale in the image
instead what we can do is
first apply one of these
region proposal networks to get some set
of proposal regions where
objects are likely located
and now apply a convolutional
network for classification
to each of these proposal
regions and this will end up
being much more computationally tractable
than trying to do all
possible locations and scales.
And this idea all came
together in this paper
called R-CNN from a few years
ago that does exactly that.
So given our input image in this case
we'll run some region proposal network
to get our proposals, these
are also sometimes called
regions of interest or ROI's
so again Selective Search
gives you something like
2000 regions of interest.
Now one of the problems
here is that these input,
these regions in the input
image could have different sizes
but if we're going to run them all
through a convolutional
network our classification,
our convolutional networks
for classification
all want images of the
same input size typically
due to the fully connected
net layers and whatnot
so we need to take each
of these region proposals
and warp them to that fixed square size
that is expected as input
to our downstream network.
So we'll crop out those region proposal,
those regions corresponding
to the region proposals,
we'll warp them to that fixed size,
and then we'll run each of them
through a convolutional network
which will then use in this case an SVM
to make a classification
decision for each of those,
to predict categories
for each of those crops.
And then I lost a slide.
But it'll also, not shown
in the slide right now
but in addition R-CNN also
predicts a regression,
like a correction to the bounding box
in addition for each of
these input region proposals
because the problem is that
your input region proposals
are kind of generally in the
right position for an object
but they might not be perfect
so in addition R-CNN will,
in addition to category labels
for each of these proposals,
it'll also predict four
numbers that are kind of an
offset or a correction to
the box that was predicted
at the region proposal stage.
So then again, this is a multi-task loss
and you would train this whole thing.
Sorry was there a question?
The question is how much does the change
in aspect ratio impact accuracy?
It's a little bit hard to say.
I think there's some
controlled experiments
in some of these papers but I'm not sure
I can give a generic answer to that.
Question?
The question is is it necessary
for regions of interest to be rectangles?
So they typically are
because it's tough to warp
these non-region things but once you move
to something like instant segmentation
then you sometimes get proposals
that are not rectangles.
If you actually do care
about predicting things
that are not rectangles.
Is there another question?
Yeah, so the question is are
the region proposals learned
so in R-CNN it's a traditional thing.
These are not learned, this is
kind of some fixed algorithm
that someone wrote down but
we'll see in a couple minutes
that we can actually, we've
changed that a little bit
in the last couple of years.
Is there another question?
The question is is the
offset always inside
the region of interest?
The answer is no, it doesn't have to be.
You might imagine that
suppose the region of interest
put a box around a person
but missed the head
then you could imagine
the network inferring
that oh this is a person but
people usually have heads
so the network showed the box
should be a little bit higher.
So sometimes the final predicted boxes
will be outside the region of interest.
Question?
Yeah.
Yeah the question is
you have a lot of ROI's
that don't correspond to true objects?
And like we said, in
addition to the classes
that you actually care
about you add an additional
background class so your
class scores can also
predict background to say
that there was no object here.
Question?
Yeah, so the question is
what kind of data do we need
and yeah, this is fully
supervised in the sense that
our training data has each
image, consists of images.
Each image has all the
object categories marked
with bounding boxes for each
instance of that category.
There are definitely papers
that try to approach this
like oh what if you don't have the data.
What if you only have
that data for some images?
Or what if that data is noisy but at least
in the generic case you
assume full supervision
of all objects in the
images at training time.
Okay, so I think we've
kind of alluded to this
but there's kind of a lot of problems
with this R-CNN framework.
And actually if you look at
the figure here on the right
you can see that additional
bounding box head
so I'll put it back.
But this is kind of still
computationally pretty expensive
because if we've got
2000 region proposals,
we're running each of those
proposals independently,
that can be pretty expensive.
There's also this question
of relying on this
fixed region proposal network,
this fixed region proposals,
we're not learning them so
that's kind of a problem.
And just in practice it
ends up being pretty slow
so in the original implementation R-CNN
would actually dump all
the features to disk
so it'd take hundreds of
gigabytes of disk space
to store all these features.
Then training would be super
slow since you have to make
all these different
forward and backward passes
through the image and it
took something like 84 hours
is one number they've
recorded for training time
so this is super super slow.
And now at test time it's also super slow,
something like roughly 30
seconds minute per image
because you need to run
thousands of forward passes
through the convolutional network
for each of these region proposals
so this ends up being pretty slow.
Thankfully we have fast
R-CNN that fixed a lot
of these problems so when we do fast R-CNN
then it's going to look kind of the same.
We're going to start with our input image
but now rather than processing
each region of interest
separately instead we're
going to run the entire image
through some convolutional
layers all at once
to give this high resolution
convolutional feature map
corresponding to the entire image.
And now we still are using
some region proposals
from some fixed thing
like Selective Search
but rather than cropping
out the pixels of the image
corresponding to the region proposals,
instead we imagine projecting
those region proposals
onto this convolutional feature map
and then taking crops from
the convolutional feature map
corresponding to each proposal rather
than taking crops directly from the image.
And this allows us to reuse
a lot of this expensive
convolutional computation
across the entire image
when we have many many crops per image.
But again, if we have some
fully connected layers
downstream those fully connected layers
are expecting some fixed-size input
so now we need to do some
reshaping of those crops
from the convolutional feature map
and they do that in a differentiable way
using something they call
an ROI pooling layer.
Once you have these warped crops
from the convolutional feature map
then you can run these things through some
fully connected layers and
predict your classification
scores and your linear regression offsets
to the bounding boxes.
And now when we train
this thing then we again
have a multi-task loss that trades off
between these two constraints
and during back propagation
we can back prop through this entire thing
and learn it all jointly.
This ROI pooling, it looks
kind of like max pooling.
I don't really want to get into
the details of that right now.
And in terms of speed if we
look at R-CNN versus fast R-CNN
versus this other model called SPP net
which is kind of in between the two,
then you can see that at
training time fast R-CNN
is something like 10 times faster to train
because we're sharing all this computation
between different feature maps.
And now at test time
fast R-CNN is super fast
and in fact fast R-CNN
is so fast at test time
that its computation time
is actually dominated
by computing region proposals.
So we said that computing
these 2000 region proposals
using Selective Search takes
something like two seconds
and now once we've got
all these region proposals
then because we're processing
them all sort of in a shared
way by sharing these
expensive convolutions
across the entire image that
we can process all of these
region proposals in less
than a second altogether.
So fast R-CNN ends up being bottlenecked
by just the computing of
these region proposals.
Thankfully we've solved this
problem with faster R-CNN.
So the idea in faster
R-CNN is to just make,
so the problem was the
computing the region proposals
using this fixed function
was a bottleneck.
So instead we'll just
make the network itself
predict its own region proposals.
And so the way that this
sort of works is that again,
we take our input image,
run the entire input image
altogether through some
convolutional layers
to get some convolutional feature map
representing the entire
high resolution image
and now there's a separate
region proposal network
which works on top of those
convolutional features
and predicts its own region
proposals inside the network.
Now once we have those
predicted region proposals
then it looks just like fast R-CNN
where now we take crops
from those region proposals
from the convolutional features,
pass them up to the rest of the network.
And now we talked about multi-task losses
and multi-task training networks
to do multiple things at once.
Well now we're telling the
network to do four things
all at once so balancing out this four-way
multi-task loss is kind of tricky.
But because the region proposal network
needs to do two things: it needs to say
for each potential
proposal is it an object
or not an object, it
needs to actually regress
the bounding box coordinates
for each of those proposals,
and now the final network at the end
needs to do these two things again.
Make final classification decisions
for what are the class scores
for each of these proposals,
and also have a second round
of bounding box regression
to again correct any errors that may have
come from the region proposal stage.
Question?
So the question is that
sometimes multi-task learning
might be seen as regularization
and are we getting that affect here?
I'm not sure if there's been
super controlled studies
on that but actually
in the original version
of the faster R-CNN paper
they did a little bit
of experimentation like what if we share
the region proposal network,
what if we don't share?
What if we learn separate
convolutional networks
for the region proposal network
versus the classification network?
And I think there were minor differences
but it wasn't a dramatic
difference either way.
So in practice it's kind
of nicer to only learn one
because it's computationally cheaper.
Sorry, question?
Yeah the question is how do you train
this region proposal network
because you don't know,
you don't have ground
truth region proposals
for the region proposal network.
So that's a little bit hairy.
I don't want to get too
much into those details
but the idea is that at any
time you have a region proposal
which has more than some
threshold of overlap
with any of the ground truth objects
then you say that that is
the positive region proposal
and you should predict
that as the region proposal
and any potential proposal
which has very low overlap
with any ground truth objects
should be predicted as a negative.
But there's a lot of dark
magic hyperparameters
in that process and
that's a little bit hairy.
Question?
Yeah, so the question is what
is the classification loss
on the region proposal
network and the answer is
that it's making a binary,
so I didn't want to get
into too much of the
details of that architecture
'cause it's a little bit hairy
but it's making binary decisions.
So it has some set of potential regions
that it's considering and it's making
a binary decision for each one.
Is this an object or not an object?
So it's like a binary classification loss.
So once you train this
thing then faster R-CNN
ends up being pretty darn fast.
So now because we've
eliminated this overhead
from computing region
proposals outside the network,
now faster R-CNN ends
up being very very fast
compared to these other alternatives.
Also, one interesting thing
is that because we're learning
the region proposals
here you might imagine
maybe what if there was some mismatch
between this fixed region
proposal algorithm and my data?
So in this case once you're learning
your own region proposals
then you can overcome
that mismatch if your region proposals
are somewhat weird or
different than other data sets.
So this whole family of R-CNN methods,
R stands for region, so these
are all region-based methods
because there's some
kind of region proposal
and then we're doing some processing,
some independent processing for each
of those potential regions.
So this whole family of methods are called
these region-based methods
for object detection.
But there's another family of methods
that you sometimes see
for object detection
which is sort of all feed
forward in a single pass.
So one of these is YOLO
for You Only Look Once.
And another is SSD for
Single Shot Detection
and these two came out
somewhat around the same time.
But the idea is that rather
than doing independent
processing for each of
these potential regions
instead we want to try to treat this
like a regression problem and just make
all these predictions all at once
with some big convolutional network.
So now given our input image you imagine
dividing that input image
into some coarse grid,
in this case it's a seven by seven grid
and now within each of those grid cells
you imagine some set
of base bounding boxes.
Here I've drawn three base bounding boxes
like a tall one, a wide
one, and a square one
but in practice you would
use more than three.
So now for each of these grid cells
and for each of these base bounding boxes
you want to predict several things.
One, you want to predict an
offset off the base bounding box
to predict what is the true location
of the object off this base bounding box.
And you also want to predict
classification scores
so maybe a classification score for each
of these base bounding boxes.
How likely is it that an
object of this category
appears in this bounding box.
So then at the end we end up predicting
from our input image, we end up predicting
this giant tensor of seven
by seven grid by 5B + C.
So that's just where we
have B base bounding boxes,
we have five numbers for
each giving our offset
and our confidence for
that base bounding box
and C classification scores
for our C categories.
So then we kind of see object
detection as this input
of an image, output of this
three dimensional tensor
and you can imagine just
training this whole thing
with a giant convolutional network.
And that's kind of what
these single shot methods do
where they just, and again
matching the ground truth
objects into these potential base boxes
becomes a little bit hairy but
that's what these methods do.
And by the way, the
region proposal network
that gets used in faster
R-CNN ends up looking
quite similar to these
where they have some set
of base bounding boxes
over some gridded image,
another region proposal
network does some regression
plus some classification.
So there's kind of some
overlapping ideas here.
So in faster R-CNN we're
kind of treating the object,
the region proposal step
as kind of this fixed
end-to-end regression problem
and then we do the separate
per region processing but now
with these single shot methods
we only do that first step and just do all
of our object detection
with a single forward pass.
So object detection has a
ton of different variables.
There could be different
base networks like VGG,
ResNet, we've seen
different metastrategies
for object detection
including this faster R-CNN
type region based family of methods,
this single shot detection
family of methods.
There's kind of a hybrid
that I didn't talk about
called R-FCN which is somewhat in between.
There's a lot of different hyperparameters
like what is the image size,
how many region proposals do you use.
And there's actually
this really cool paper
that will appear at CVPR this
summer that does a really
controlled experimentation
around a lot of these
different variables and tries to tell you
how do these methods all perform
under these different variables.
So if you're interested I'd
encourage you to check it out
but kind of one of the
key takeaways is that
the faster R-CNN style
of region based methods
tends to give higher
accuracies but ends up being
much slower than the single shot methods
because the single shot
methods don't require
this per region processing.
But I encourage you to
check out this paper
if you want more details.
Also as a bit of aside,
I had this fun paper
with Andre a couple years ago that kind of
combined object detection
with image captioning
and did this problem
called dense captioning
so now the idea is that
rather than predicting
a fixed category label for each region,
instead we want to write
a caption for each region.
And again, we had some data
set that had this sort of data
where we had a data set of
regions together with captions
and then we sort of trained
this giant end-to-end model
that just predicted these
captions all jointly.
And this ends up looking
somewhat like faster R-CNN
where you have some region proposal stage
then a bounding box, then
some per region processing.
But rather than a SVM or a softmax loss
instead those per region
processing has a whole
RNN language model that predicts
a caption for each region.
So that ends up looking quite
a bit like faster R-CNN.
There's a video here but I think
we're running out of time so I'll skip it.
But the idea here is
that once you have this,
you can kind of tie together
a lot of these ideas
and if you have some new
problem that you're interested
in tackling like dense captioning,
you can recycle a lot of the components
that you've learned from other problems
like object detection and image captioning
and kind of stitch together
one end-to-end network
that produces the outputs
that you care about
for your problem.
So the last task that I want to talk about
is this idea of instance segmentation.
So here instance segmentation is
in some ways like the full problem
We're given an input image
and we want to predict one,
the locations and identities
of objects in that image
similar to object detection,
but rather than just
predicting a bounding box
for each of those objects,
instead we want to predict
a whole segmentation mask
for each of those objects
and predict which pixels
in the input image corresponds
to each object instance.
So this is kind of like a hybrid
between semantic segmentation
and object detection
because like object
detection we can handle
multiple objects and we
differentiate the identities
of different instances so in this example
since there are two dogs in the image
and instance segmentation method
actually distinguishes
between the two dog instances
and the output and kind of
like semantic segmentation
we have this pixel wise accuracy
where for each of these
objects we want to say
which pixels belong to that object.
So there's been a lot of different methods
that people have tackled, for
instance segmentation as well,
but the current state of
the art is this new paper
called Mask R-CNN that
actually just came out
on archive about a month ago
so this is not yet published,
this is like super fresh stuff.
And this ends up looking
a lot like faster R-CNN.
So it has this multi-stage
processing approach
where we take our whole input image,
that whole input image goes
into some convolutional
network and some learned
region proposal network
that's exactly the same as faster R-CNN
and now once we have our
learned region proposals
then we project those proposals
onto our convolutional feature map
just like we did in fast and faster R-CNN.
But now rather than just
making a classification
and a bounding box for regression decision
for each of those boxes we in addition
want to predict a segmentation mask
for each of those bounding box,
for each of those region proposals.
So now it kind of looks like a mini,
like a semantic segmentation problem
inside each of the region proposals
that we're getting from our
region proposal network.
So now after we do this
ROI aligning to warp
our features corresponding
to the region of proposal
into the right shape, then we
have two different branches.
One branch will come up that looks exact,
and this first branch at
the top looks just like
faster R-CNN and it will
predict classification scores
telling us what is the
category corresponding
to that region of
proposal or alternatively
whether or not it's background.
And we'll also predict some
bounding box coordinates
that regressed off the
region proposal coordinates.
And now in addition we'll
have this branch at the bottom
which looks basically like
a semantic segmentation
mini network which will
classify for each pixel
in that input region proposal
whether or not it's an object
so this mask R-CNN problem,
this mask R-CNN architecture
just kind of unifies all
of these different problems
that we've been talking
about today into one nice
jointly end-to-end trainable model.
And it's really cool and it actually works
really really well so when
you look at the examples
in the paper they're kind of amazing.
They look kind of indistinguishable
from ground truth.
So in this example on the left you can see
that there are these two people standing
in front of motorcycles,
it's drawn the boxes
around these people, it's
also gone in and labeled
all the pixels of those
people and it's really small
but actually in the
background on that image
on the left there's also
a whole crowd of people
standing very small in the background.
It's also drawn boxes around each of those
and grabbed the pixels
of each of those images.
And you can see that this is just,
it ends up working really really well
and it's a relatively simple addition
on top of the existing
faster R-CNN framework.
So I told you that mask
R-CNN unifies everything
we talked about today and it also does
pose estimation by the way.
So we talked about, you
can do pose estimation
by predicting these joint coordinates
for each of the joints of the person
so you can do mask R-CNN to
do joint object detection,
pose estimation, and
instance segmentation.
And the only addition we need to make
is that for each of these region proposals
we add an additional little branch
that predicts these
coordinates of the joints
for the instance of the
current region proposal.
So now this is just another loss,
like another layer that we add,
another head coming out of the network
and an additional term
in our multi-task loss.
But once we add this one little branch
then you can do all of these
different problems jointly
and you get results looking
something like this.
Where now this network, like
a single feed forward network
is deciding how many
people are in the image,
detecting where those people are,
figuring out the pixels
corresponding to each
of those people and also
drawing a skeleton estimating
the pose of those people
and this works really well
even in crowded scenes like this classroom
where there's a ton of people sitting
and they all overlap each other
and it just seems to work incredibly well.
And because it's built on
the faster R-CNN framework
it also runs relatively close to real time
so this is running something
like five frames per second
on a GPU because this is all sort of done
in the single forward pass of the network.
So this is again, a super new paper
but I think that this will probably get
a lot of attention in the coming months.
So just to recap, we've talked.
Sorry question?
The question is how much
training data do you need?
So all of these instant
segmentation results
were trained on the
Microsoft Coco data set
so Microsoft Coco is roughly
200,000 training images.
It has 80 categories that it cares about
so in each of those
200,000 training images
it has all the instances of
those 80 categories labeled.
So there's something like
200,000 images for training
and there's something
like I think an average
of fivee or six instances per image.
So it actually is quite a lot of data.
And for Microsoft Coco for all the people
in Microsoft Coco they
also have all the joints
annotated as well so this
actually does have quite a lot
of supervision at training
time you're right,
and actually is trained
with quite a lot of data.
So I think one really
interesting topic to study
moving forward is that we kind of know
that if you have a lot of
data to solve some problem,
at this point we're relatively
confident that you can
stitch up some convolutional network
that can probably do a
reasonable job at that problem
but figuring out ways to
get performance like this
with less training data
is a super interesting
and active area of research and I think
that's something people will be spending
a lot of their efforts working
on in the next few years.
So just to recap, today we
had kind of a whirlwind tour
of a whole bunch of different
computer vision topics
and we saw how a lot of the
machinery that we built up
from image classification can
be applied relatively easily
to tackle these different
computer vision topics.
And next time we'll talk about,
we'll have a really fun lecture
on visualizing CNN features.
Well also talk about DeepDream
and neural style transfer.
