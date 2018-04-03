﻿
- Okay we have a lot to cover
today so let's get started.
Today we'll be talking
about Generative Models.
And before we start, a few
administrative details.
So midterm grades will be
released on Gradescope this week
A reminder that A3 is
due next Friday May 26th.
The HyperQuest deadline for
extra credit you can do this
still until Sunday May 21st.
And our poster session is
June 6th from 12 to 3 P.M..
Okay so an overview of what
we're going to talk about today
we're going to switch gears a little bit
and take a look at
unsupervised learning today.
And in particular we're going
to talk about generative
models which is a type
of unsupervised learning.
And we'll look at three
types of generative models.
So pixelRNNs and pixelCNNs
variational autoencoders
and Generative Adversarial networks.
So so far in this class we've
talked a lot about supervised
learning and different kinds
of supervised learning problems.
So in the supervised learning
set up we have our data X and
then we have some labels Y.
And our goal is to learn
a function that's mapping
from our data X to our labels Y.
And these labels can take
many different types of forms.
So for example, we've
looked at classification
where our input is an image
and we want to output Y, a
class label for the category.
We've talked about object
detection where now our input
is still an image but
here we want to output the
bounding boxes of instances of
up to multiple dogs or cats.
We've talked about semantic
segmentation where here we have
a label for every pixel the
category that every pixel
belongs to.
And we've also talked
about image captioning
where here our label is now a sentence
and so it's now in the
form of natural language.
So unsupervised learning in this set up,
it's a type of learning where here we have
unlabeled training data and
our goal now is to learn some
underlying hidden structure of the data.
Right, so an example of
this can be something like
clustering which you guys
might have seen before
where here the goal is to
find groups within the data
that are similar through
some type of metric.
For example, K means clustering.
Another example of an
unsupervised learning task
is a dimensionality reduction.
So in this problem want
to find axes along which
our training data has the most variation,
and so these axes are part
of the underlying structure
of the data.
And then we can use this
to reduce of dimensionality
of the data such that the
data has significant variation
among each of the remaining dimensions.
Right, so this example
here we start off with data
in three dimensions and
we're going to find two
axes of variation in this case
and reduce our data projected down to 2D.
Another example of unsupervised learning
is learning feature
representations for data.
We've seen how to do this
in supervised ways before
where we used the supervised loss,
for example classification.
Where we have the classification label.
We have something like a Softmax loss
And we can train a neural network where
we can interpret activations for example
our FC7 layers as some kind of future
representation for the data.
And in an unsupervised setting,
for example here
autoencoders which we'll talk
more about later
In this case our loss is now trying to
reconstruct the input data to basically,
you have a good reconstruction
of our input data
and use this to learn features.
So we're learning a feature
representation without
using any additional external labels.
And finally another example
of unsupervised learning
is density estimation where
in this case we want to
estimate the underlying
distribution of our data.
So for example in this top case over here,
we have points in 1-d and we can try
and fit a Gaussian into this density
and in this bottom example
over here it's 2D data
and here again we're trying
to estimate the density and
we can model this density.
We want to fit a model such
that the density is higher
where there's more points concentrated.
And so to summarize the
differences in unsupervised
learning which we've looked a lot so far,
we want to use label data to learn
a function mapping from X to Y
and an unsupervised
learning we use no labels
and instead we try to learn
some underlying hidden
structure of the data,
whether this is grouping,
acts as a variation or
underlying density estimation.
And unsupervised learning is a huge
and really exciting area of research and
and some of the reasons are
that training data is really
cheap, it doesn't use labels
so we're able to learn
from a lot of data at one time
and basically utilize a lot
more data than if we required annotating
or finding labels for data.
And unsupervised learning
is still relatively
unsolved research area by comparison.
There's a lot of open problems in this,
but it also, it holds the potential of
if you're able to successfully learn
and represent a lot of
the underlying structure
in the data then this also takes you a
long way towards the Holy Grail
of trying to understand the
structure of the visual world.
So that's a little bit of kind
of a high-level big picture
view of unsupervised learning.
And today will focus more
specifically on generative models
which is a class of
models for unsupervised
learning where given training
data our goal is to try and
generate new samples from
the same distribution.
Right, so we have training
data over here generated
from some distribution P data
and we want to learn a model, P model
to generate samples from
the same distribution
and so we want to learn P
model to be similar to P data.
And generative models
address density estimations.
So this problem that we
saw earlier of trying
to estimate the underlying
distribution of your
training data which is a core problem
in unsupervised learning.
And we'll see that there's
several flavors of this.
We can use generative models
to do explicit density
estimation where we're
going to explicitly define
and solve for our P model
or we can also do implicit
density estimation
where in this case we'll
learn a model that can
produce samples from P model
without explicitly defining it.
So, why do we care
about generative models?
Why is this a really
interesting core problem
in unsupervised learning?
Well there's a lot of
things that we can do
with generative models.
If we're able to create
realistic samples from the data
distributions that we want
we can do really cool things
with this, right?
We can generate just
beautiful samples to start
with so on the left you can
see a completely new samples of
just generated by these generative models.
Also in the center here
generated samples of
images we can also do tasks
like super resolution,
colorization so hallucinating
or filling in these edges
with generated ideas of colors
and what the purse should look like.
We can also use generative
models of time series data
for simulation and planning
and so this will be useful in
for reinforcement learning applications
which we'll talk a bit more
about reinforcement learning
in a later lecture.
And training generative
models can also enable
inference of latent representations.
Learning latent features
that can be useful
as general features for downstream tasks.
So if we look at types
of generative models
these can be organized
into the taxonomy here
where we have these two major
branches that we talked about,
explicit density models and
implicit density models.
And then we can also get down into many
of these other sub categories.
And well we can refer to
this figure is adapted
from a tutorial on GANs
from Ian Goodfellow
and so if you're interested in some
of these different taxonomy
and categorizations of
generative models this is a
good resource that you can take
a look at.
But today we're going to
discuss three of the most
popular types of generative
models that are in use
and in research today.
And so we'll talk first briefly
about pixelRNNs and CNNs
And then we'll talk about
variational autoencoders.
These are both types of
explicit density models.
One that's using a tractable density
and another that's using
an approximate density
And then we'll talk about
generative adversarial networks,
GANs which are a type of
implicit density estimation.
So let's first talk
about pixelRNNs and CNNs.
So these are a type of fully
visible belief networks
which are modeling a density explicitly
so in this case what
they do is we have this
image data X that we have
and we want to model the
probability or likelihood
of this image P of X.
Right and so in this case,
for these kinds of models,
we use the chain rule to
decompose this likelihood
into a product of one
dimensional distribution.
So we have here the
probability of each pixel X I
conditioned on all previous
pixels X1 through XI - 1.
and your likelihood all
right, your joint likelihood
of all the pixels in your image
is going to be the product
of all of these pixels together,
all of these likelihoods together.
And then once we define this likelihood,
in order to train this
model we can just maximize
the likelihood of our training data
under this defined density.
So if we look at this this
distribution over pixel values
right, we have this P of
XI given all the previous
pixel values, well this is a
really complex distribution.
So how can we model this?
Well we've seen before that
if we want to have complex
transformations we can do
these using neural networks.
Neural networks are a good
way to express complex
transformations.
And so what we'll do is
we'll use a neural network
to express this complex
function that we have
of the distribution.
And one thing you'll see here is that,
okay even if we're going to
use a neural network for this
another thing we have to take
care of is how do we order
the pixels.
Right, I said here that
we have a distribution
for P of XI given all previous pixels
but what does all
previous the pixels mean?
So we'll take a look at that.
So PixelRNN was a model proposed in 2016
that basically defines a way
for setting up and optimizing
this problem and so
how this model works is
that we're going to
generate pixels starting
in a corner of the image.
So we can look at this grid
as basically the pixels
of your image and so what
we're going to do is start
from the pixel in the
upper left-hand corner
and then we're going to
sequentially generate pixels based
on these connections from the arrows
that you can see here.
And each of the dependencies
on the previous pixels
in this ordering is going
to be modeled using an RNN
or more specifically an
LSTM which we've seen before
in lecture.
Right so using this we can
basically continue to move
forward just moving
down a long is diagonal
and generating all of these
pixel values dependent
on the pixels that they're connected to.
And so this works really
well but the drawback here
is that this sequential generation, right,
so it's actually quite slow to do this.
You can imagine you know if
you're going to generate a new
image instead of all of these
feed forward networks that we
see, we've seen with CNNs.
Here we're going to have
to iteratively go through
and generate all these
images, all these pixels.
So a little bit later, after a pixelRNN,
another model called
pixelCNN was introduced.
And this has very
similar setup as pixelCNN
and we're still going to
do this image generation
starting from the corner of
the of the image and expanding
outwards but the difference now
is that now instead of using
 
255
00:12:43,074 --> 00:12:45,480
an RNN to model all these dependencies
we're going to use the CNN instead.
And we're now going to use a
CNN over a a context region
that you can see here around
in the particular pixel
that we're going to generate now.
Right so we take the pixels around it,
this gray area within the
region that's already been
generated and then we can
pass this through a CNN
and use that to generate
our next pixel value.
And so what this is going to
give is this is going to give
This is a CNN, a neural
network at each pixel location
right and so the output of
this is going to be a soft
max loss over the pixel values here.
In this case we have a 0 to
255 and then we can train this
by maximizing the likelihood
of the training images.
Right so we say that basically
we want to take a training
image we're going to do
this generation process
and at each pixel location
we have the ground truth
training data image
value that we have here
and this is a quick basically the label
or the the the classification
label that we want
our pixel to be which of these 255 values
and we can train this
using a Softmax loss.
Right and so basically
the effect of doing this
is that we're going to
maximize the likelihood
of our training data
pixels being generated.
Okay any questions about this?
Yes.
[student's words obscured
due to lack of microphone]
Yeah, so the question is,
I thought we were talking
about unsupervised learning,
why do we have basically
a classification label here?
The reason is that this loss,
this output that we have
is the value of the input training data.
So we have no external labels, right?
We didn't go and have to
manually collect any labels
for this, we're just taking our input data
and saying that this is what
we used for the last function.
[student's words obscured
due to lack of microphone]
The question is, is
this like bag of words?
I would say it's not really bag of words,
it's more saying that we
want where we're outputting
a distribution over pixel
values at each location
of our image right, and what we want to do
is we want to maximize the
likelihood of our input,
our training data being
produced, being generated.
Right so, in that sense, this
is why it's using our input
data to create our loss.
So using pixelCNN training
is faster than pixelRNN
because here now right
at every pixel location
we want to maximize the value of our,
we want to maximize the
likelihood of our training data
showing up and so we have all
of these values already right,
just from our training data
and so we can do this much
faster but a generation time
for a test time we want to
generate a completely new
image right, just starting from
the corner and we're not,
we're not trying to do any type
of learning so in that
generation time we still
have to generate each
of these pixel locations
before we can generate the next location.
And so generation time here
it still slow even though
training time is faster.
Question.
[student's words obscured
due to lack of microphone]
So the question is, is
this training a sensitive
distribution to what you
pick for the first pixel?
Yeah, so it is dependent on
what you have as the initial
pixel distribution and then
everything is conditioned
based on that.
So again, how do you
pick this distribution?
So at training time you
have these distributions
from your training data
and then at generation time
you can just initialize
this with either uniform
or from your training
data, however you want.
And then once you have that
everything else is conditioned
based on that.
Question.
[student's words obscured
due to lack of microphone]
Yeah so the question is is
there a way that we define
this in this chain rule
fashion instead of predicting
all the pixels at one time?
And so we'll see, we'll see
models later that do do this,
but what the chain rule allows
us to do is it allows us
to find this very tractable
density that we can then
basically optimize and do,
directly optimizes likelihood
Okay so these are some
examples of generations
from this model and so here
on the left you can see
generations where the
training data is CIFAR-10,
CIFAR-10 dataset.
And so you can see that in
general they are starting
to capture statistics of natural images.
You can see general types of blobs
and kind of things that look
like parts of natural images
coming out.
On the right here it's ImageNet,
we can again see samples
from here and
these are starting to
look like natural images
but they're still not, there's
still room for improvement.
You can still see that there
are differences obviously
with regional training images
and some of the semantics
are not clear in here.
So, to summarize this,
pixelRNNs and CNNs allow you
to explicitly compute likelihood P of X.
It's an explicit density
that we can optimize.
And being able to do this
also has another benefit
of giving a good evaluation metric.
You know you can kind of measure
how good your samples are
by this likelihood of the
data that you can compute.
And it's able to produce
pretty good samples
but it's still an active area of research
and the main disadvantage
of these methods is that
the generation is sequential
and so it can be pretty slow.
And these kinds of methods
have also been used
for generating audio for example.
And you can look online for
some pretty interesting examples
of this, but again the drawback
is that it takes a long time
to generate these samples.
And so there's a lot of work,
has been work since then
on still on improving pixelCNN performance
And so all kinds of different
you know architecture changes
add the loss function
formulating this differently
on different types of training tricks
And so if you're interested
in learning more about this
you can look at some of
these papers on PixelCNN
and then other pixelCNN plus
plus better improved version
that came out this year.
Okay so now we're going
to talk about another type
of generative models call
variational autoencoders.
And so far we saw that
pixelCNNs defined a tractable
density function, right,
using this this definition
and based on that we can
optimize directly optimize
the likelihood of the training data.
So with variational autoencoders
now we're going to define
an intractable density function.
We're now going to model this
with an additional latent
variable Z and we'll talk in more detail
about how this looks.
And so our data likelihood
P of X is now basically
has to be this integral right,
taking the expectation over
all possible values of Z.
And so this now is going to be a problem.
We'll see that we cannot
optimize this directly.
And so instead what we have
to do is we have to derive
and optimize a lower bound
on the likelihood instead.
Yeah, question.
So the question is is what is Z?
Z is a latent variable
and I'll go through this
in much more detail.
So let's talk about some background first.
Variational autoencoders
are related to a type of
unsupervised learning
model called autoencoders.
And so we'll talk little bit
more first about autoencoders
and what they are and then
I'll explain how variational
autoencoders are related
and build off of this
and allow you to generate data.
So with autoencoders we don't
use this to generate data,
but it's an unsupervised
approach for learning a lower
dimensional feature representation
from unlabeled training data.
All right so in this case
we have our input data X
and then we're going to
want to learn some features
that we call Z.
And then we'll have an encoder
that's going to be a mapping,
a function mapping
from this input data
to our feature Z.
And this encoder can take
many different forms right,
they would generally use
neural networks so originally
these models have been
around, autoencoders have been
around for a long time.
So in the 2000s we used linear
layers of non-linearities,
then later on we had fully
connected deeper networks
and then after that we moved
on to using CNNs for these
encoders.
So we take our input data
X and then we map this
to some feature Z.
And Z we usually have as,
we usually specify this
to be smaller than X and we
perform basically dimensionality
reduction because of that.
So the question who has an
idea of why do we want to do
dimensionality reduction here?
Why do we want Z to be smaller than X?
Yeah.
[student's words obscured
due to lack of microphone]
So the answer I heard is Z
should represent the most
important features in
X and that's correct.
So we want Z to be able to
learn features that can capture
meaningful factors of
variation in the data.
Right this makes them good features.
So how can we learn this
feature representation?
Well the way autoencoders
do this is that we train
the model such that the features
can be used to reconstruct
our original data.
So what we want is we want to
have input data that we use
an encoder to map it to some
lower dimensional features Z.
This is the output of the encoder network,
and we want to be able to
take these features that were
produced based on this input
data and then use a decoder
a second network and be
able to output now something
of the same size dimensionality
as X and have it be similar
to X right so we want to be
able to reconstruct the original
data.
And again for the decoder we
are basically using same types
of networks as encoders so
it's usually a little bit
symmetric and now we can use CNN networks
for most of these.
Okay so the process is going
to be we're going to take
our input data right we pass
it through our encoder first
which is going to be something
for example like a four layer
convolutional network and
then we're going to pass it,
get these features and then
we're going to pass it through
a decoder which is a four layer
for example upconvolutional
network and then get a
reconstructed data out at the end
of this.
Right in the reason why we
have a convolutional network
for the encoder and an
upconvolutional network
for the decoder is because at
the encoder we're basically
taking it from this high
dimensional input to these lower
dimensional features and now
we want to go the other way
go from our low dimensional
features back out to our
high dimensional reconstructed input.
And so in order to get this
effect that we said we wanted
before of being able to
reconstruct our input data
we'll use something like
an L2 loss function.
Right that basically just
says let me make my pixels
of my input data to be the same as my,
my pixels in my reconstructed
data to be the same
as the pixels of my input data.
An important thing to notice here,
this relates back to a
question that we had earlier,
is that even though we have
this loss function here,
there's no, there's no external
labels that are being used
in training this.
All we have is our training
data that we're going to use
both to pass through the
network as well as to compute
our loss function.
So once we have this
after training this model
what we can do is we can
throw away this decoder.
All this was used was too
to be able to produce our
reconstruction input and
be able to compute our loss
function.
And we can use the encoder
that we have which produces our
feature mapping and we
can use this to initialize
a supervised model.
Right and so for example we
can now go from this input
to our features and then
have an additional classifier
network on top of this that
now we can use to output
a class label for example for
classification problem
we can have external labels from here
and use our standard loss
functions like Softmax.
And so the value of this is
that we basically were able
to use a lot of unlabeled
training data to try and learn
good general feature representations.
Right, and now we can use this
to initialize a supervised
learning problem where sometimes
we don't have so much data
we only have small data.
And we've seen in previous
homeworks and classes
that with small data it's
hard to learn a model, right?
You can have over fitting
and all kinds of problems
and so this allows you to
initialize your model first
with better features.
Okay so we saw that autoencoders
are able to reconstruct
data and are able to, as
a result, learn features
to initialize, that we can
use to initialize a supervised
model.
And we saw that these
features that we learned
have this intuition of being
able to capture factors
of variation in the training data.
All right so based on this
intuition of okay these,
we can have this latent
this vector Z which has
factors of variation in our training data.
Now a natural question is
well can we use a similar type
of setup to generate new images?
And so now we will talk about
variational autoencoders
which is a probabillstic spin
on autoencoders that will let
us sample from the model in
order to generate new data.
Okay any questions on autoencoders first?
Okay, so variational autoencoders.
All right so here we assume
that our training data
that we have X I from one to N
is generated from some
underlying, unobserved
latent representation Z.
Right, so it's this intuition
that Z is some vector
right which element of Z
is capturing how little
or how much of some factor
of variation that we have
in our training data.
Right so the intuition is,
you know, maybe these could
be something like different
kinds of attributes.
Let's say we're trying to generate faces,
it could be how much of
a smile is on the face,
it could be position of the eyebrows hair
orientation of the head.
These are all possible
types of latent factors
that could be learned.
Right, and so our generation
process is that we're going to
sample from a prior over Z.
Right so for each of these
attributes for example,
you know, how much smile that there is,
we can have a prior over
what sort of distribution
we think that there should be for this so,
a gaussian is something
that's a natural prior
that we can use for each
of these factors of Z
and then we're going
to generate our data X
by sampling from a conditional,
conditional distribution
P of X given Z.
So we sample Z first, we sample
a value for each of these
latent factors and then we'll use that
and sample our image X from here.
And so the true parameters
of this generation process
are theta, theta star right?
So we have the parameters of our prior
and our conditional distributions
and what we want to do is in
order to have a generative
model be able to generate new data
we want to estimate these
parameters of our true parameters
Okay so let's first talk
about how should we represent
this model.
All right, so if we're going to
have a model for this generator
process, well we've already
said before that we can choose
our prior P of Z to be something simple.
Something like a Gaussian, right?
And this is the reasonable
thing to choose for
for latent attributes.
Now for our conditional
distribution P of X given Z
this is much more complex right,
because we need to use
this to generate an image
and so for P of X given
Z, well as we saw before,
when we have some type of
complex function that we want
to represent we can represent
this with a neural network.
And so that's a natural
choice for let's try and model
P of X given Z with a neural network.
And we're going to call
this the decoder network.
Right, so we're going to
think about taking some latent
representation and trying to
decode this into the image
that it's specifying.
So now how can we train this model?
Right, we want to be able to
train this model so that we can
learn an estimate of these parameters.
So if we remember our strategy
from training generative
models, back from are fully
visible belief networks,
our pixelRNNs and CNNs,
a straightforward natural
strategy is to try
and learn these model
parameters in order to maximize
the likelihood of the training data.
Right, so we saw earlier
that in this case,
with our latent variable
Z, we're going to have
to write out P of X taking
expectation over all possible
values of Z which is
continuous and so we get this
expression here.
Right so now we have it with this latent Z
and now if we're going to, if
you want to try and maximize
its likelihood, well what's the problem?
Can we just take this take
gradients and maximize
this likelihood?
[student's words obscured
due to lack of microphone]
Right, so this integral is
not going to be tractable,
that's correct.
So let's take a look at this
in a little bit more detail.
Right, so we have our
data likelihood term here.
And the first time is P of Z.
And here we already said
earlier, we can just choose this
to be a simple Gaussian
prior, so this is fine.
P of X given Z, well we
said we were going to
specify a decoder neural network.
So given any Z, we can get
P of X given Z from here.
It's the output of our neural network.
But then what's the problem here?
Okay this was supposed to
be a different unhappy face
but somehow I don't know what happened,
in the process of translation,
it turned into a crying black ghost
but what this is symbolizing
is that basically if we want
to compute P of X given Z
for every Z this is now intractable right,
we cannot compute this integral.
So data likelihood is intractable
and it turns out that if
we look at other terms
in this model if we look
at our posterior density,
So P of our posterior of Z given X,
then this is going to be P of X given Z
times P of Z over P of X by Bayes' rule
and this is also going
to be intractable, right.
We have P of X given Z
is okay, P of Z is okay,
but we have this P of X our likelihood
which has the integral
and it's intractable.
So we can't directly optimizes this.
but we'll see that a solution,
a solution that will enable
us to learn this model
is if in addition to
using a decoder network
defining this neural network
to model P of X given Z.
If we now define an
additional encoder network
Q of Z given X we're going
to call this an encoder
because we want to turn our input X into,
get the likelihood of Z given X,
we're going to encode this into Z.
And defined this network to approximate
the P of Z given X.
Right this was posterior
density term now is also
intractable.
If we use this additional
network to approximate this
then we'll see that this will
actually allow us to derive
a lower bound on the data
likelihood that is tractable
and which we can optimize.
Okay so first just to be a
little bit more concrete about
these encoder and decoder
networks that I specified,
in variational autoencoders we
want the model probabilistic
generation of data.
So in autoencoders we already talked
about this concept of having
an encoder going from input X
to some feature Z and a
decoder network going from Z
back out to some image X.
And so here we go to again
have an encoder network
and a decoder network but we're going
to make these probabilistic.
So now our encoder network
Q of Z given X with
parameters phi are going to output a mean
and a diagonal covariance and from here,
this will be the direct
outputs of our encoder
network and the same thing for our
decoder network which
is going to start from Z
and now it's going to output the mean
and the diagonal covariance of some X,
same dimension as the input given Z
And then this decoder network
has different parameters
theta.
And now in order to
actually get our Z and our,
This should be Z given X and X given Z.
We'll sample from these distributions.
So now our encoder and our decoder network
are producing distributions
over Z and X respectively
and will sample from this distribution
in order to get a value from here.
So you can see how this is
taking us on the direction
towards being able to sample
and generate new data.
And just one thing to note is that
these encoder and decoder networks,
you'll also hear different terms for them.
The encoder network can
also be kind of recognition
or inference network because
we're trying to form
inference of this latent
representation of Z given
X and then for the decoder
network, this is what we'll
use to perform generation.
Right so you also hear
generation network being used.
Okay so now equipped with our
encoder and decoder networks,
let's try and work out
the data likelihood again.
and we'll use the log of
the data likelihood here.
So we'll see that if we
want the log of P of X right
we can write this out as like a P of X but
take the expectation with respect to Z.
So Z samples from our
distribution of Q of Z given
X that we've now defined
using the encoder network.
And we can do this because
P of X doesn't depend on Z.
Right 'cause Z is not part of that.
And so we'll see that taking
the expectation with respect
to Z is going to come in handy later on.
Okay so now from this
original expression we can
now expand it out to be
log of P of X given Z,
P of Z over P of Z given
X using Bayes' rule.
And so this is just
directly writing this out.
And then taking this we
can also now multiply it
by a constant.
Right, so Q of Z given
X over Q of Z given X.
This is one we can do this.
It doesn't change it but it's
going to be helpful later on.
So given that what we'll
do is we'll write it
out into these three separate terms.
And you can work out this
math later on by yourself
but it's essentially just
using logarithm rules
taking all of these
terms that we had in the
line above and just separating it out into
these three different terms
that will have nice meanings.
Right so if we look at this,
the first term that we get
separated out is log of P
given X and then expectation
of log of P given X and
then we're going to have
two KL terms, right.
This is basically KL divergence term
to say how close these two distributions are.
So how close is a distribution
Q of Z given X to P of Z.
So it's just the, it's exactly
this expectation term above.
And it's just a distance
metric for distributions.
And so we'll see that,
right, we saw that these are
nice KL terms that we can write out.
And now if we look at these
three terms that we have here,
the first term is P of X
given Z, which is provided
by our decoder network.
And we're able to compute
an estimate of these term
through sampling and we'll see that we can
do a sampling that's
differentiable through something
called the re-parametrization
trick which is a
detail that you can look
at this paper if you're
interested.
But basically we can
now compute this term.
And then these KL terms,
the second KL term
is a KL between two Gaussians,
so our Q of Z given X,
remember our encoder produced
this distribution which had
a mean and a covariance,
it was a nice Gaussian.
And then also our prior P of
Z which is also a Gaussian.
And so this has a nice, when you have a KL
of two Gaussians you have
a nice closed form solution
that you can have.
And then this third KL term now,
this is a KL of Q given
X with a P of Z given X.
But we know that P of Z
given X was this intractable
posterior that we saw earlier, right?
That we didn't want to
compute that's why we had
this approximation using Q.
And so this term is still is a problem.
But one thing we do know
about this term is that KL
divergence, it's a distance
between two distributions
is always greater than or
equal to zero by definition.
And so what we can do with this is that,
well what we have here, the
two terms that we can work
nicely with, this is a,
this is a tractable lower
bound which we can actually
take gradient of and optimize.
P of X given Z is
differentiable and the KL terms
are also, the close form
solution is also differentiable.
And this is a lower bound
because we know that the KL
term on the right, the
ugly one is greater than
or equal it zero.
So we have a lower bound.
And so what we'll do to train
a variational autoencoder
is that we take this
lower bound and we instead
optimize and maximize
this lower bound instead.
So we're optimizing a lower
bound on the likelihood
of our data.
So that means that our data
is always going to have
a likelihood that's at
least as high as this lower
bound that we're maximizing.
And so we want to find
the parameters theta,
estimate parameters theta
and phi that allows us to
maximize this.
And then one last sort of
intuition about this lower bound
that we have is that this first term
is expectation over all samples of Z
sampled from passing our X
through the encoder network
sampling Z taking expectation
over all of these samples
of likelihood of X given Z
and so this is a reconstruction, right?
This is basically saying,
if I want this to be big
I want this likelihood P
of X given Z to be high,
so it's kind of like
trying to do a good job
reconstructing the data.
So similar to what we had
from our autoencoder before.
But the second term here is
saying make this KL small.
Make our approximate
posterior distribution close
to our prior distribution.
And this basically is
saying that well we want our
latent variable Z to be following this,
have this distribution
type, distribution shape
that we would like it to have.
Okay so any questions about this?
I think this is a lot
of math that if you guys
are interested you should go
back and kind of work through
all of the derivations yourself.
Yeah.
[student's words obscured
due to lack of microphone]
So the question is why
do we specify the prior
and the latent variables as Gaussian?
And the reason is that well we're defining
some sort of generative process right,
of sampling Z first and
then sampling X first.
And defining it as a
Gaussian is a reasonable type
of prior that we can say
makes sense for these types
of latent attributes to
be distributed according
to some sort of Gaussian, and
then this lets us now then
optimize our model.
Okay, so we talked about how
we can deride this lower bound
and now let's put this all
together and walk through
the process of the training of the AE.
Right so here's the bound
that we want to optimize,
to maximize.
And now for a forward pass.
We're going to proceed
in the following manner.
We have our input data
X, so we'll a mini batch
of input data.
And then we'll pass it
through our encoder network
so we'll get Q of Z given X.
And from this Q of Z given
X, this'll be the terms
that we use to compute the KL term.
And then from here we'll
sample Z from this distribution
of Z given X so we have a
sample of the latent factors
that we can infer from X.
And then from here we're
going to pass a Z through
another, our second decoder network.
And from the decoder network
we'll get this output
for the mean and variance
on our distribution for
X given Z and then
finally we can sample now
our X given Z from this distribution
and here this will produce
some sample output.
And when we're training
we're going to take this
distribution and say well
our loss term is going to be
log of our training image
pixel values given Z.
So our loss functions going
to say let's maximize the
likelihood of this original
input being reconstructed.
And so now for every mini batch of input
we're going to compute this forward pass.
Get all these terms that we need
and then this is all
differentiable so then we just
backprop though all of this
and then get our gradient,
we update our model and
we use this to continuously
update our parameters,
our generator and decoder
network parameters theta
and phi in order to maximize
the likelihood of the trained data.
Okay so once we've trained our VAE,
so now to generate data,
what we can do is we can
use just the decoder network.
All right, so from here
we can sample Z now,
instead of sampling Z from
this posterior that we had
during training, while
during generation we sample
from our true generative process.
So we sample from our
prior that we specify.
And then we're going to then
sample our data X from here.
And we'll see that this
can produce, in this case,
train on MNIST, these are
samples of digits generated
from a VAE trained on MNIST.
And you can see that, you
know, we talked about this idea
of Z representing these
latent factors where we can
bury Z right according to
our sample from different
parts of our prior and
then get different kind of
interpretable meanings from here.
So here we can see that this is
the data manifold for two dimensional Z.
So if we have a two dimensional
Z and we take Z and let's
say some range from you know,
from different percentiles
of the distribution, and
we vary Z1 and we vary Z2,
then you can see how the
image generated from every
combination of Z1 and
Z2 that we have here,
you can see it's transitioning
smoothly across all
of these different variations.
And you know our prior on
Z was, it was diagonal,
so we chose this in order
to encourage this to be
independent latent variables
that can then encode
interpretable factors of variation.
So because of this now we'll
have different dimensions
of Z, encoding different
interpretable factors
of variation.
So, in this example train now on Faces,
we'll see as we vary
Z1, going up and down,
you'll see the amount of smile changing.
So from a frown at the
top to like a big smile
at the bottom and then as we go vary Z2,
from left to right, you can
see the head pose changing.
From one direction all
the way to the other.
And so one additional
thing I want to point out
is that as a result of doing this,
these Z variables are also good feature
representations.
Because they encode how
much of these different
these different interpretable
semantics that we have.
And so we can use our Q of Z given X,
the encoder that we've
learned and give it an input
images X, we can map this
to Z and use the Z as
features that we can
use for downstream tasks
like supervision, or
like classification or
other tasks.
Okay so just another
couple of examples of data
generated from VAEs.
So on the left here we have
data generated on CIFAR-10,
trained on CIFAR-10, and
then on the right we have
data trained and generated on Faces.
And we'll see so we can
see that in general VAEs
are able to generate recognizable data.
One of the main drawbacks
of VAEs is that they tend
to still have a bit of
a blurry aspect to them.
You can see this in the
faces and so this is still
an active area of research.
Okay so to summarize VAEs,
they're a probabilistic spin
on traditional autoencoders.
So instead of deterministically
taking your input X
and going to Z, feature Z and
then back to reconstructing X,
now we have this idea of
distributions and sampling
involved which allows us to generate data.
And in order to train
this, VAEs are defining an
intractable density.
So we can derive and
optimize a lower bound,
a variational lower bound, so
variational means basically
using approximations to handle
these types of intractable
expressions.
And so this is why this is
called a variational autoencoder.
And so some of the
advantages of this approach
is that VAEs are, they're
a principled approach
to generative models and they
also allow this inference
query so being able to infer
things like Q of Z given X.
That we said could be useful
feature representations
for other tasks.
So disadvantages of VAEs are
that while we're maximizing
the lower bound of the
likelihood, which is okay
like you know in general this
is still pushing us in the
right direction and there's
more other theoretical analysis of this.
So you know, it's doing okay,
but it's maybe not still
as direct an optimization
and evaluation as the pixel
RNNs and CNNs that we saw earlier,
but which had, and then,
also the VAE samples are
tending to be a little bit
blurrier and of lower quality
compared to state of the art
samples that we can see
from other generative models
such as GANs that we'll talk about next.
And so VAEs now are still,
they're still an active
area of research.
People are working on more
flexible approximations,
so richer approximate posteriors,
so instead of just a
diagonal Gaussian some richer
functions for this.
And then also, another area
that people have been working
on is incorporating more
structure in these latent
variables.
So now we had all of these
independent latent variables
but people are working on
having modeling structure
in here, groupings,
other types of structure.
Okay, so yeah, question.
[student's words obscured
due to lack of microphone]
Yeah, so the question is we're
deciding the dimensionality
of the latent variable.
Yeah, that's something that you specify.
Okay, so we've talked so
far about pixelCNNs and VAEs
and now we'll take a look
at a third and very popular
type of generative model called GANs.
So the models that we've seen
so far, pixelCNNs and RNNs
define a tractable density function.
And they optimize the
likelihood of the trained data.
And then VAEs in contrast to
that now have this additional
latent variable Z that they
define in the generative
process.
And so having the Z has
a lot of nice properties
that we talked about, but
they are also cause us to have
this intractable density
function that we can't
optimize directly and so
we derive and optimize
a lower bound on the likelihood instead.
And so now what if we
just give up on explicitly
modeling this density at all?
And we say well what we
want is just the ability
to sample and to have nice
samples from our distribution.
So this is the approach that GANs take.
So in GANs we don't work with
an explicit density function,
but instead we're going to
take a game-theoretic approach
and we're going to learn to
generate from our training
distribution through a set
up of a two player game,
and we'll talk about this in more detail.
So, in the GAN set up we're
saying, okay well what we want,
what we care about is we
want to be able to sample
from a complex high dimensional
training distribution.
So if we think about well
we want to produce samples
from this distribution,
there's no direct way
that we can do this.
We have this very complex distribution,
we can't just take samples from here.
So the solution that we're
going to take is that we can,
however, sample from
simpler distributions.
For example random noise, right?
Gaussians are, these we can sample from.
And so what we're going to
do is we're going to learn
a transformation from
these simple distributions
directly to the training
distribution that we want.
So the question, what can we
used to represent this complex
distribution?
Neural network, I heard the answer.
So when we want to model
some kind of complex function
or transformation we use a neural network.
Okay so what we're going to
do is we're going to take
in the GAN set up, we're
going to take some input
which is a vector of some
dimension that we specify
of random noise and then we're
going to pass this through
a generator network, and then
we're going to get as output
directly a sample from
the training distribution.
So every input of random
noise we want to correspond to
a sample from the training distribution.
And so the way we're going to
train and learn this network
is that we're going to look
at this as a two player game.
So we have two players, a
generator network as well
as an additional discriminator
network that I'll show next.
And our generator network is
going to try to, as player one,
it's going to try to fool the
discriminator by generating
real looking images.
And then our second player,
our discriminator network
is then going to try to
distinguish between real and fake
images.
So it wants to do as good
a job as possible of trying
to determine which of these
images are counterfeit
or fake images generated
by this generator.
Okay so what this looks like is,
we have our random noise going
to our generator network,
generator network is generating
these images that we're
going to call, they're
fake from our generator.
And then we're going to also
have real images that we
take from our training
set and then we want the
discriminator to be able
to distinguish between
real and fake images.
Output real and fake for each images.
So the idea is if we're
able to have a very good
discriminator, we want to
train a good discriminator,
if it can do a good job of
discriminating real versus fake,
and then if our generator
network is able to generate,
if it's able to do well
and generate fake images
that can successfully
fool this discriminator,
then we have a good generative model.
We're generating images that
look like images from the
training set.
Okay, so we have these two
players and so we're going to
train this jointly in a
minimax game formulation.
So this minimax objective
function is what we have here.
We're going to take, it's going
to be minimum over theta G
our parameters of our generator network G,
and maximum over parameter Zeta
of our Discriminator network
D, of this objective, right, these terms.
And so if we look at these
terms, what this is saying
is well this first thing,
expectation over data
of log of D given X.
This log of D of X is
the discriminator output
for real data X.
This is going to be likelihood
of real data being real
from the data distribution P data.
And then the second term
here, expectation of Z drawn
from P of Z, Z drawn from
P of Z means samples from
our generator network and
this term D of G of Z that
we have here is the output
of our discriminator
for generated fake data for our,
what does the discriminator
output of G of Z which is
our fake data.
And so if we think about
this is trying to do,
our discriminator wants to
maximize this objective, right,
it's a max over theta D such
that D of X is close to one.
It's close to real, it's
high for the real data.
And then D of G of X, what
it thinks of the fake data
on the left here is small, we
want this to be close to zero.
So if we're able to maximize
this, this means discriminator
is doing a good job of
distinguishing between real and zero.
Basically classifying
between real and fake data.
And then our generator, here
we want the generator to
minimize this objective such
that D of G of Z is close
to one.
So if this D of G of Z is
close to one over here,
then the one minus side is
small and basically we want to,
if we minimize this term
then, then it's having
discriminator think that our
fake data's actually real.
So that means that our generator
is producing real samples.
Okay so this is the
important objective of GANs
to try and understand so are
there any questions about this?
[student's words obscured
due to lack of microphone]
I'm not sure I understand
your question, can you,
[student's words obscured
due to lack of microphone]
Yeah, so the question is
is this basically trying
to have the first network
produce real looking images
that our second network,
the discriminator cannot
distinguish between.
Okay, so the question is how
do we actually label the data
or do the training for these networks.
We'll see how to train the networks next.
But in terms of like what
is the data label basically,
this is unsupervised, so
there's no data labeling.
But data generated from
the generator network,
the fake images have a label
of basically zero or fake.
And we can take training
images that are real images
and this basically has
a label of one or real.
So when we have, the loss
function for our discriminator
is using this.
It's trying to output a zero
for the generator images
and a one for the real images.
So there's no external labels.
[student's words obscured
due to lack of microphone]
So the question is the label
for the generator network
will be the output for
the discriminator network.
The generator is not really doing,
it's not really doing
classifications necessarily.
What it's objective is
is here, D of G of Z,
it wants this to be high.
So given a fixed discriminator,
it wants to learn the
generator parameter
such that this is high.
So we'll take the fixed
discriminator output and use that
to do the backprop.
Okay so in order to train
this, what we're going to do
is we're going to alternate
between gradient ascent
on our discriminator, so we're
trying to learn theta beta
to maximizing this objective.
And then gradient
descent on the generator.
So taking gradient ascent
on these parameters theta G
such that we're minimizing
this and this objective.
And here we are only taking
this right part over here
because that's the only
part that's dependent on
theta G parameters.
Okay so this is how we can train this GAN.
We can alternate between
training our discriminator
and our generator in this
game, each trying to fool
the other or generator trying
to fool the discriminator.
But one thing that is important
to note is that in practice
this generator objective as
we've just defined actually
doesn't work that well.
And the reason for this is
we have to look at the loss
landscape.
So if we look at the loss
landscape over here for
D of G of X,
if we apply here one minus D of G of X
which is what we want to
minimize for the generator,
it has this shape here.
So we want to minimize this
and it turns out the slope
of this loss is actually going
to be higher towards the right.
High when D of G of Z is closer to one.
So that means that when our
generator is doing a good job
of fooling the discriminator,
we're going to have
a high gradient, more
higher gradient terms.
And on the other hand
when we have bad samples,
our generator has not
learned a good job yet,
it's not good at generating yet,
then this is when the
discriminator can easily tell
it's now closer to this
zero region on the X axis.
Then here the gradient's relatively flat.
And so what this actually
means is that our
our gradient signal is
dominated by region where the
sample is already pretty good.
Whereas we actually want it to
learn a lot when the samples
are bad, right?
These are training samples
that we want to learn from.
And so in order to, so this
basically makes it hard
to learn and so in order
to improve learning,
what we're going to do
is define a different,
slightly different objective
function for the gradient.
Where now we're going to
do gradient ascent instead.
And so instead of minimizing
the likelihood of our
discriminator being correct,
which is what we had earlier,
now we'll kind of flip
it and say let's maximize
the likelihood of our
discriminator being wrong.
And so this will produce this
objective here of maximizing,
maximizing log of D of G of X.
And so, now basically
we want to, there should be a negative
sign here.
But basically we want to now
maximize this flip objective
instead and what this now does
is if we plot this function
on the right here, then we
have a high gradient signal
in this region on the left
where we have bad samples,
and now the flatter region
is to the right where we
would have good samples.
So now we're going to
learn more from regions
of bad samples.
And so this has the same
objective of fooling
the discriminator but it
actually works much better
in practice and for a lot
of work on GANs that are
using these kind of
vanilla GAN formulation
is actually using this objective.
Okay so just an aside on
that is that jointly training
these two networks is
challenging and can be unstable.
So as we saw here, like
we're alternating between
training a discriminator
and training a generator.
This type of alternation is,
basically it's hard to learn
two networks at once and
there's also this issue
of depending on what our
loss landscape looks at,
it can affect our training dynamics.
So an active area of research
still is how can we choose
objectives with better loss
landscapes that can help
training and make it more stable?
Okay so now let's put this
all together and look at the
full GAN training algorithm.
So what we're going to do is
for each iteration of training
we're going to first train the generation,
train the discriminator network a bit
and then train the generator network.
So for k steps of training
the discriminator network
we'll sample a mini batch
of noise samples from our
noise prior Z and then
also sample a mini batch
of real samples from our training data X.
So what we'll do is we'll
pass the noise through our
generator, we'll get our fake images out.
So we have a mini batch of
fake images and mini batch
of real images.
And then we'll pick a gradient
step on the discriminator
using this mini batch, our
fake and our real images
and then update our
discriminator parameters.
And use this and do this a
certain number of iterations
to train the discriminator
for a bit basically.
And then after that we'll
go to our second step
which is training the generator.
And so here we'll sample just
a mini batch of noise samples.
We'll pass this through our
generator and then now we
want to do backpop on this
to basically optimize our
generator objective that we saw earlier.
So we want to have our
generator fool our discriminator
as much as possible.
And so we're going to alternate
between these two steps
of taking gradient steps
for our discriminator
and for the generator.
And I said for k steps up here,
for training the discriminator
and so this is kind
of a topic of debate.
Some people think just having
one iteration of discriminator
one type of discriminator,
one type of generator is best.
Some people think it's better
to train the discriminator
for a little bit longer before
switching to the generator.
There's no real clear rule
and it's something that
people have found different
things to work better
depending on the problem.
And one thing I want to point
out is that there's been
a lot of recent work that
alleviates this problem
and makes it so you don't
have to spend so much effort
trying to balance how the
training of these two networks.
It'll have more stable training
and give better results.
And so Wasserstein GAN
is an example of a paper
that was an important
work towards doing this.
Okay so looking at the whole
picture we've now trained,
we have our network setup,
we've trained both our
generator network and
our discriminator network
and now after training for generation,
we can just take our generator
network and use this to
generate new images.
So we just take noise Z and
pass this through and generate
fake images from here.
Okay and so now let's look
at some generated samples
from these GANs.
So here's an example of trained on MNIST
and then on the right on Faces.
And for each of these you can also see,
just for visualization
the closest, on the right,
the nearest neighbor from the
training set to the column
right next to it.
And so you can see that
we're able to generate
very realistic samples and
it never directly memorizes
the training set.
And here are some examples
from the original GAN paper
on CIFAR images.
And these are still fairly,
not such good quality yet,
these were, the original
work is from 2014,
so these are some older, simpler networks.
And these were using simple,
fully connected networks.
And so since that time
there's been a lot of work
on improving GANs.
One example of a work that
really took a big step
towards improving the quality
of samples is this work
from Alex Radford in ICLR
2016 on adding convolutional
architectures to GANs.
In this paper there was
a whole set of guidelines
on architectures for helping
GANs to produce better
samples.
So you can look at this for more details.
This is an example of a
convolutional architecture
that they're using which
is going from our input Z
noise vector Z and
transforming this all the way
to the output sample.
So now from this large
convolutional architecture
we'll see that the samples
from this model are really
starting to look very good.
So this is trained on
a dataset of bedrooms
and we can see all kinds of
very realistic fancy looking
bedrooms with windows and night
stands and other furniture
around there so these are
some really pretty samples.
And we can also try and
interpret a little bit of what
these GANs are doing.
So in this example here what
we can do is we can take
two points of Z, two
different random noise vectors
and let's just interpolate
between these points.
And each row across here
is an interpolation from
one random noise Z to
another random noise vector Z
and you can see that as it's changing,
it's smoothly interpolating
the image as well
all the way over.
And so something else that
we can do is we can see that,
well, let's try to analyze
further what these vectors Z
mean, and so we can try
and do vector math on here.
So what this experiment does is it says
okay, let's take some images of smiling,
samples of smiling women
images and then let's take some
samples of neutral women
and then also some samples
of neutral men.
And so let's try and do take
the average of the Z vectors
that produced each of
these samples and if we,
Say we take this, mean
vector for the smiling women,
subtract the mean vector
for the neutral women
and add the mean vector
for the neutral man,
what do we get?
And we get samples of smiling man.
So we can take the Z
vector produced there,
generate samples and get
samples of smiling men.
And we can have another example of this.
Of glasses man minus no glasses
man and plus glasses women.
And get women with glasses.
So here you can see that
basically the Z has this type
of interpretability that
you can use this to generate
some pretty cool examples.
Okay so this year, 2017 has really been
the year of the GAN.
There's been tons and tons of work on GANs
and it's really sort of
exploded and gotten some really
cool results.
So on the left here you
can see people working on
better training and generation.
So we talked about improving
the loss functions,
more stable training and this
was able to get really nice
generations here of different
types of architectures
on the bottom here really
crisp high resolution faces.
With GANs you can also do,
there's also been models on
source to try to domain
transfer and conditional GANs.
And so here, this is an
example of source to try to
get domain transfer where,
for example in the upper part
here we are trying to go
from source domain of horses
to an output domain of zebras.
So we can take an image
of horses and train a GAN
such that the output is
going to be the same thing
but now zebras in the same
image setting as the horses
and go the other way around.
We can transform apples into oranges.
And also the other way around.
We can also use this to
do photo enhancement.
So producing these, really
taking a standard photo
and trying to make really
nice, as if you had,
pretending that you have a
really nice expensive camera.
That you can get the nice blur effects.
On the bottom here we have scene changing,
so transforming an image
of Yosemite from the image
in winter time to the
image in summer time.
And there's really tons of applications.
So on the right here there's more.
There's also going from a text description
and having a GAN that's now
conditioned on this text
description and producing an image.
So there's something
here about a small bird
with a pink breast and crown
and now we're going to generate
images of this.
And there's also examples
down here of filling in edges.
So given conditions on some sketch that we have,
can we fill in a color version
of what this would look like.
Can we take a Google, a
map grid and put something
that looks like Google Earth on,
and turn it into something
that looks like Google Earth.
Go in and hallucinate all
of these buildings and trees
and so on.
And so there's lots of
really cool examples of this.
And there's also this
website for pics to pics
which did a lot of these
kind of conditional GAN type
examples.
I encourage you to go look
at for more interesting
applications that people
have done with GANs.
And in terms of research
papers there's also
there's a huge number of papers
about GANs this year now.
There's a website called
the GAN Zoo that kind of is
trying to compile a whole list of these.
And so here this has only
taken me from A through C
on the left here and
through like L on the right.
So it won't even fit on the slide.
There's tons of papers as
well that you can look at
if you're interested.
And then one last pointer
is also for tips and tricks
for training GANs, here's
a nice little website
that has pointers if you're
trying to train these GANs
in practice.
Okay, so summary of GANs.
GANs don't work with an
explicit density function.
Instead we're going to represent
this implicitly through
samples and they take a
game-theoretic approach to training
so we're going to learn to
generate from our training
distribution through a
two player game setup.
And the pros of GANs are
that they're really having
gorgeous state of the art
samples and you can do a lot
with these.
The cons are that they are
trickier and more unstable
to train, we're not
just directly optimizing
a one objective function
that we can just do backpop
and train easily.
Instead we have these two
networks that we're trying
to balance training with so
it can be a bit more unstable.
And we also can lose out
on not being able to do
some of the inference queries,
P of X, P of Z given X
that we had for example in our VAE.
And GANs are still an
active area of research,
this is a relatively new type
of model that we're starting
to see a lot of and you'll
be seeing a lot more of.
And so people are still working
now on better loss functions
more stable training, so Wasserstein GAN
for those of you who are
interested is basically
an improvement in this direction.
That now a lot of people are
also using and basing models
off of.
There's also other works
like LSGAN, Least Square's GAN,
Least Square's GAN and others.
So you can look into this more.
And a lot of times for these new models
in terms of actually implementing this,
they're not necessarily big changes.
They're different loss
functions that you can change
a little bit and get
like a big improvement
in training.
And so this is, some of
these are worth looking into
and you'll also get some
practice on your homework
assignment.
And there's also a lot of
work on different types of
conditional GANs and GANs
for all kinds of different
problem setups and applications.
Okay so a recap of today.
We talked about generative models.
We talked about three of the
most common kinds of generative
models that people are using
and doing research on today.
So we talked first about
pixelRNN and pixelCNN,
which is an explicit density model.
It optimizes the exact
likelihood and it produces good
samples but it's pretty
inefficient because of the
sequential generation.
We looked at VAE which
optimizes a variational or lower
bound on the likelihood
and this also produces
useful a latent representation.
You can do inference queries.
But the example quality
is still not the best.
So even though it has a
lot of promise, it's still
a very active area of
research and has a lot of
open problems.
And then GANs we talked
about is a game-theoretic
approach for training and
it's what currently achieves
the best state of the art examples.
But it can also be tricky
and unstable to train
and it loses out a bit
on the inference queries.
And so what you'll also
see is a lot of recent work
on combinations of these kinds of models.
So for example adversarial autoencoders.
Something like a VAE
trained with an additional
adversarial loss on top which
improves the sample quality.
There's also things like
pixelVAE is now a combination
of pixelCNN and VAE so
there's a lot of combinations
basically trying to take
the best of all these worlds
and put them together.
Okay so today we talked
about generative models
and next time we'll talk
about reinforcement learning.
Thanks.
