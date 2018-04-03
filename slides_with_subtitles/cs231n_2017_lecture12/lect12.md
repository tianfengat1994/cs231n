
- Good morning.
So, it's 12:03 so, I want to get started.
Welcome to Lecture 12, of CS-231N.
Today we are going to talk about
Visualizing and Understanding
convolutional networks.
This is always a super fun lecture to give
because we get to look a
lot of pretty pictures.
So, it's, it's one of my favorites.
As usual a couple administrative things.
So, hopefully your projects
are all going well,
because as a reminder your milestones
are due on Canvas tonight.
It is Canvas, right?
Okay, so want to double check, yeah.
Due on Canvas tonight, we are working on
furiously grading your midterms.
So, we'll hope to have those
midterms grades to you back
by on grade scope this week.
So, I know that was little confusion,
you all got registration
email's for grade scope
probably in the last week.
Something like that, we start
couple of questions on piazo.
So, we've decided to use grade
scope to grade the midterms.
So, don't be confused, if you
get some emails about that.
Another reminder is that assignment three
was released last week on Friday.
It will be due, a week from
this Friday, on the 26th.
This is, an assignment three,
is almost entirely brand new this year.
So, it we apologize for taking
a little bit longer than
expected to get it out.
But I think it's super cool.
A lot of that stuff, we'll
talk about in today's lecture.
You'll actually be implementing
on your assignment.
And for the assignment, you'll
get the choice of either
Pi torch or tensure flow.
To work through these different examples.
So, we hope that's really
useful experience for you guys.
We also saw a lot of activity
on HyperQuest over the weekend.
So that's, that's really awesome.
The leader board went up yesterday.
It seems like you guys are
really trying to battle it out
to show off your deep learning
neural network training skills.
So that's super cool.
And we because due to the high interest
in HyperQuest and due to
the conflicts with the,
with the Milestones submission time.
We decided to extend the deadline
for extra credit through Sunday.
So, anyone who does at
least 12 runs on HyperQuest
by Sunday will get little bit
of extra credit in the class.
Also those of you who are,
at the top of leader board
doing really well, will
get may be little bit
extra, extra credit.
So, I thanks for
participating we got lot of
interest and that was really cool.
Final reminder is about
the poster session.
So, we have the poster
session will be on June 6th.
That date is finalized,
I think that, I don't
remember the exact time.
But it is June 6th.
So that, we have some questions
about when exactly that poster session is
for those of you who are traveling
at the end of quarter
or starting internships
or something like that.
So, it will be June 6th.
Any questions on the admin notes.
No, totally clear.
So, last time we talked.
So, last time we had a pretty
jam packed lecture, when we
talked about lot of different
computer vision tasks, as a reminder.
We talked about semantic segmentation
which is this problem, where
you want to sign labels
to every pixel in the input image.
But does not differentiate the
object instances in those images.
We talked about classification
plus localization.
Where in addition to a class label
you also want to draw a box
or perhaps several boxes in the image.
Where the distinction here is that,
in a classification
plus localization setup.
You have some fix number of
objects that you are looking for
So, we also saw that this type of paradigm
can be applied to the things
like pose recognition.
Where you want to regress to
different numbers of joints
in the human body.
We also talked about the object detection
where you start with some fixed
set of category labels
that you are interested in.
Like dogs and cats.
And then the task is
to draw a boxes around
every instance of those objects
that appear in the input image.
And object detection
is really distinct from
classification plus localization
because with object
detection, we don't know
ahead of time, how many object instances
we're looking for in the image.
And we saw that there's
this whole family of methods
based on RCNN, Fast RCNN and faster RCNN,
as well as the single
shot detection methods
for addressing this problem
of object detection.
Then finally we talked
pretty briefly about
instance segmentation,
which is kind of combining
aspects of a semantic
segmentation and object detection
where the goal is to
detect all the instances
of the categories we care about,
as well as label the pixels
belonging to each instance.
So, in this case, we
detected two dogs and one cat
and for each of those instances we wanted
to label all the pixels.
So, these are we kind of
covered a lot last lecture
but those are really interesting
and exciting problems
that you guys might consider to
using in parts of your projects.
But today we are going to
shift gears a little bit
and ask another question.
Which is, what's really going on
inside convolutional networks.
We've seen by this point in the class
how to train convolutional networks.
How to stitch up different
types of architectures
to attack different problems.
But one question that you
might have had in your mind,
is what exactly is going
on inside these networks?
How did they do the things that they do?
What kinds of features
are they looking for?
And all this source of related questions.
So, so far we've sort of seen
ConvNets as a little bit of a black box.
Where some input image of raw pixels
is coming in on one side.
It goes to the many layers of convulsion
and pooling in different
sorts of transformations.
And on the outside, we end up
with some set of class scores
or some types of understandable
interpretable output.
Such as class scores or
bounding box positions
or labeled pixels or something like that.
But the question is.
What are all these other
layers in the middle doing?
What kinds of things in the input image
are they looking for?
And can we try again intuition for.
How ConvNets are working?
What types of things in the
image they are looking for?
And what kinds of techniques do we have
for analyzing this
internals of the network?
So, one relatively simple
thing is the first layer.
So, we've seen, we've
talked about this before.
But recalled that, the
first convolutional layer
consists of a filters that,
so, for example in AlexNet.
The first convolutional layer consists
of a number of convolutional filters.
Each convolutional of filter
has shape 3 by 11 by 11.
And these convolutional filters gets slid
over the input image.
We take inner products between
some chunk of the image.
And the weights of the
convolutional filter.
And that gives us our output of the
at, at after that first
convolutional layer.
So, in AlexNet then we
have 64 of these filters.
But now in the first layer
because we are taking
in a direct inner product
between the weights
of the convolutional layer
and the pixels of the image.
We can get some since for what
these filters are looking for
by simply visualizing the
learned weights of these filters
as images themselves.
So, for each of those
11 by 11 by 3 filters
in AlexNet, we can just
visualize that filter
as a little 11 by 11 image
with a three channels
give you the red, green and blue values.
And then because there
are 64 of these filters
we just visualize 64
little 11 by 11 images.
And we can repeat... So
we have shown here at the.
So, these are filters taken
from the prechain models,
in the pi torch model zoo.
And we are looking at the
convolutional filters.
The weights of the convolutional filters.
at the first layer of AlexNet, ResNet-18,
ResNet-101 and DenseNet-121.
And you can see, kind
of what all these layers
what this filters looking for.
You see the lot of things
looking for oriented edges.
Likes bars of light and dark.
At various angles, in various
angles and various positions
in the input, we can see opposing colors.
Like this are green and pink.
opposing colors or this orange
and blue opposing colors.
So, this, this kind of
connects back to what we
talked about with Hugh and Wiesel.
All the way in the first lecture.
That remember the human visual system
is known to the detect
things like oriented edges.
At the very early layers
of the human visual system.
And it turns out of that
these convolutional networks
tend to do something, somewhat similar.
At their first convolutional
layers as well.
And what's kind of interesting is that
pretty much no matter what type
of architecture you hook up
or whatever type of training
data you are train it on.
You almost always get
the first layers of your.
The first convolutional
weights of any pretty much
any convolutional network
looking at images.
Ends up looking something like this
with oriented edges and opposing colors.
Looking at that input image.
But this really only, sorry
what was that question?
Yes, these are showing the learned weights
of the first convolutional layer.
Oh, so that the question is.
Why does visualizing the
weights of the filters?
Tell you what the filter is looking for.
So this intuition comes from
sort of template matching
and inner products.
That if you imagine you have
some, some template vector.
And then you imagine you
compute a scaler output
by taking inner product
between your template vector
and some arbitrary piece of data.
Then, the input which
maximizes that activation.
Under a norm constraint on the input
is exactly when those
two vectors match up.
So, in that since that,
when, whenever you're taking
inner products, the thing
causes an inner product
to excite maximally
is a copy of the thing you are
taking an inner product with.
So, that, that's why we can
actually visualize these weights
and that, why that shows us,
what this first layer is looking for.
So, for these networks
the first layers always
was a convolutional layer.
So, generally whenever
you are looking at image.
Whenever you are thinking about image data
and training convolutional networks,
you generally put a convolutional layer
at the first, at the first stop.
Yeah, so the question is,
can we do this same type of procedure
in the middle open network.
That's actually the next slide.
So, good anticipation.
So, if we do, if we draw this exact same
visualization for the
intermediate convolutional layers.
It's actually a lot less interpretable.
So, this is, this is performing
exact same visualization.
So, remember for this using
the tiny ConvNets demo network
that's running on the course website
whenever you go there.
So, for that network,
the first layer is 7 by
7 convulsion 16 filters.
So, after the top visualizing
the first layer weights
for this network just like
we saw in a previous slide.
But now at the second layer weights.
After we do a convulsion
then there's some relu
and some other non-linearity perhaps.
But the second convolutional layer,
now receives the 16 channel input.
And does 7 by 7 convulsion
with 20 convolutional filters.
And we've actually,
so the problem is that
you can't really visualize
these directly as images.
So, you can try, so, here if you
this 16 by, so the input is
this has 16 dimensions in depth.
And we have these convolutional filters,
each convolutional filter is 7 by 7,
and is extending along the full depth
so has 16 elements.
Then we've 20 such of these
convolutional filters,
that are producing the output
planes of the next layer.
But the problem here is that
we can't, looking at the,
looking directly at the weights
of these filters, doesn't
really tell us much.
So, we, that's really done here is that,
now for this single 16 by 7
by 7 convolutional filter.
We can spread out those 167
by 7 planes of the filter
into a 167 by 7 grayscale images.
So, that's what we've done.
Up here, which is these little
tiny gray scale images here
show us what is, what are the weights
in one of the convolutional
filters of the second layer.
And now, because there are
20 outputs from this layer.
Then this second convolutional
layer, has 2o such of these
16 by 16 or 16 by 7 by 7 filters.
So if we visualize the weights
of those convolutional filters
as images, you can see that there are some
kind of spacial structures here.
But it doesn't really
give you good intuition
for what they are looking at.
Because these filters are not
looking, are not connected
directly to the input image.
Instead recall that the second
layer convolutional filters
are connected to the
output of the first layer.
So, this is giving visualization of,
what type of activation
pattern after the first
convulsion, would cause
the second layer convulsion
to maximally activate.
But, that's not very interpretable
because we don't have a good sense
for what those first layer
convulsions look like
in terms of image pixels.
So we'll need to develop some
slightly more fancy technique
to get a sense for what is going on
in the intermediate layers.
Question in the back.
Yeah. So the question is that
for... all the visualization
on this on the previous slide.
We've had the scale the weights
to the zero to 255 range.
So in practice those
weights could be unbounded.
They could have any range.
But to get nice visualizations
we need to scale those.
These visualizations also do not take
in to account the bias is in these layers.
So you should keep that in mind
when and not take these
HEPS visualizations
to, to literally.
Now at the last layer
remember when we looking at the last layer
of convolutional network.
We have these maybe 1000 class scores
that are telling us what
are the predicted scores
for each of the classes
in our training data set
and immediately before the last layer
we often have some fully connected layer.
In the case of Alex net
we have some 4096- dimensional
features representation
of our image that then
gets fed into that final
our final layer to predict
our final class scores.
And one another, another kind of route
for tackling the problem
of visual, visualizing
and understanding ConvNets
is to try to understand what's
happening at the last layer
of a convolutional network.
So what we can do
is how to take some,
some data set of images
run a bunch of, run a bunch of images
through our trained convolutional network
and recorded that 4096 dimensional vector
for each of those images.
And now go through and try to figure out
and visualize that last
layer, that last hidden layer
rather than those rather than
the first convolutional layer.
So, one thing you might imagine is,
is trying a nearest neighbor approach.
So, remember, way back
in the second lecture
we saw this graphic on the left
where we, where we had a
nearest neighbor classifier.
Where we were looking at
nearest neighbors in pixels
space between CIFAR 10 images.
And then when you look
at nearest neighbors
in pixel space between CIFAR 10 images
you see that you pull up images
that looks quite similar
to the query image.
So again on the left column
here is some CIFAR 10 image
from the CIFAR 10 data set
and then these, these next five columns
are showing the nearest
neighbors in pixel space
to those test set images.
And so for example
this white dog that you see here,
it's nearest neighbors are in pixel space
are these kinds of white blobby things
that may, may or may not be dogs,
but at least the raw pixels
of the image are quite similar.
So now we can do the same
type of visualization
computing and visualizing
these nearest neighbor images.
But rather than computing
the nearest neighbors in pixel space,
instead we can compute nearest neighbors
in that 4096 dimensional feature space.
Which is computed by the
convolutional network.
So here on the right
we see some examples.
So this, this first column shows us
some examples of images from the test set
of image that... Of the image
net classification data set
and now the, these
subsequent columns show us
nearest neighbors to those test set images
in the 4096, in the 4096th
dimensional features space
computed by Alex net.
And you can see here that
this is quite different
from the pixel space nearest neighbors,
because the pixels are
often quite different.
between the image in
it's nearest neighbors
and feature space.
However, the semantic
content of those images
tends to be similar in this feature space.
So for example, if you
look at this second layer
the query image is this elephant
standing on the left side of the image
with a screen grass behind him.
and now one of these, one of these...
it's third nearest
neighbor in the tough set
is actually an elephant standing
on the right side of the image.
So this is really interesting.
Because between this
elephant standing on the left
and this element stand,
elephant standing on the right
the pixels between those two images
are almost entirely different.
However, in the feature space
which is learned by the network
those two images and that
being very close to each other.
Which means that somehow
this, this last their features
is capturing some of those
semantic content of these images.
That's really cool and really exciting
and, and in general looking
at these kind of nearest
neighbor visualizations
is really quick and easy way to visualize
something about what's going on here.
Yes. So the question is that
through the... the standard
supervised learning procedure
for classific training,
classification network
There's nothing in the loss
encouraging these features
to be close together.
So that, that's true.
It just kind of a happy accident
that they end up being
close to each other.
Because we didn't tell the
network during training
these features should be close.
However there are sometimes
people do train networks
using things called
either contrastive loss
or a triplet loss.
Which actually explicitly make...
assumptions and constraints on the network
such that those last their features
end up having some metric
space interpretation.
But Alex net at least was not
trained specifically for that.
The question is, what is the nearest...
What is this nearest neighbor thing
have to do at the last layer?
So we're taking this image
we're running it through the network
and then the, the second to last
like the last hidden layer of the network
is of 4096th dimensional vector.
Because there's this, this is...
This is there, there are
these fully connected layers
at the end of the network.
So we are doing is...
We're writing down that
4096th dimensional vector
for each of the images
and then we are computing
nearest neighbors
according to that 4096th
dimensional vector.
Which is computed by,
computed by the network.
Maybe, maybe we can chat offline.
So another, another, another
another angle that we might have
for visualizing what's
going on in this last layer
is by some concept of
dimensionality reduction.
So those of you who have
taken CS229 for example
you've seen something like PCA.
Which let's you take some high
dimensional representation
like these 4096th dimensional features
and then compress it
down to two-dimensions.
So then you can visualize that
feature space more directly.
So, Principle Component Analysis or PCA
is kind of one way to do that.
But there's real another
really powerful algorithm
called t-SNE.
Standing for t-distributed
stochastic neighbor embeddings.
Which is slightly more powerful method.
Which is a non-linear
dimensionality reduction method
that people in deep often
use for visualizing features.
So here as an, just an
example of what t-SNE can do.
This visualization here is, is showing
a t-SNE dimensionality reduction
on the emnest data set.
So, emnest remember is this date set
of hand written digits
between zero and nine.
Each image is a gray scale image
20... 28 by 28 gray scale image
and now we're... So that
Now we've, we've used t-SNE
to take that 28 times 28
dimensional features space
of the raw pixels for m-nest
and now compress it
down to two- dimensions
ans then visualize each
of those m-nest digits
in this compress
two-dimensional representation
and when you do, when you run t-SNE
on the raw pixels and m-nest
You can see these natural
clusters appearing.
Which corresponds to the,
the digits of these m-nest
of, of these m-nest data set.
So now we can do a similar
type of visualization.
Where we apply this t-SNE
dimensionality reduction technique
to the features from the last layer
of our trained image net classifier.
So...To be a little bit more concrete here
what we've done
is that we take, a large set of images
we run them off convolutional network.
We record that final 4096th
dimensional feature vector
for, from the last layer
of each of those images.
Which gives us large collection
of 4096th dimensional vectors.
Now we apply t-SNE
dimensionality reduction
to compute, sort of compress
that 4096the dimensional
features space down into a
two-dimensional feature space
and now we, layout a grid in that
compressed two-dimensional feature space
and visualize what types of images appear
at each location in the grid
in this two-dimensional feature space.
So by doing this you get
some very close rough sense
of what the geometry
of this learned feature space looks like.
So these images are
little bit hard to see.
So I'd encourage you to check out
the high resolution versions online.
But at least maybe on
the left you can see that
there's sort of one
cluster in the bottom here
of, of green things, is a
different kind of flowers
and there's other types of clusters
for different types of dog breeds
and another types of
animals and, and locations.
So there's sort of
discontinuous semantic notion
in this feature space.
Which we can explore by looking through
this t-SNE dimensionality reduction
version of the, of the features.
Is there question?
Yeah. So the basic idea is that we're
we, we have an image
so now we end up with
three different pieces
of information about each image.
We have the pixels of the image.
We have the 4096th dimensional vector.
Then we use t-SNE to convert
the 4096th dimensional vector
into a two-dimensional coordinate
and then we take the
original pixels of the image
and place that at the
two-dimensional coordinate
corresponding to the
dimensionality reduced version
of the 4096th dimensional feature.
Yeah, little bit involved here.
Question in the front.
The question is
Roughly how much variants do
these two-dimension explain?
Well, I'm not sure of the exact number
and I get little bit muddy
when you're talking about t-SNE,
because it's a non-linear
dimensionality reduction technique.
So, I'd have to look offline
and I'm not sure of exactly
how much it explains.
Question?
Question is, can you do the same analysis
of upper layers of the network?
And yes, you can. But no,
I don't have those
visualizations here. Sorry.
Question?
The question is,
Shouldn't we have overlaps
of images once we do this
dimensionality reduction?
And yes, of course, you would.
So this is just kind of taking a,
nearest neighbor in
our, in our regular grid
and then picking an image
close to that grid point.
So, so... they, yeah.
this is not showing
you the kind of density
in different parts of the feature space.
So that's, that's another thing to look at
and again at the link
you, there's a couple more
visualizations of this nature that,
that address that a little bit.
Okay. So another, another thing
that you can do for some of
these intermediate features
is, so we talked a couple of slides ago
that visualizing the weights
of these intermediate layers
is not so interpretable.
But actually visualizing
the activation maps of those
intermediate layers
is kind of interpretable in some cases.
So for, so I, again an
example of Alex Net.
Remember the, the conv5
layers of Alex Net.
Gives us this 128 by...
The for...The conv5 features for any image
is now 128 by 13 by 13 dimensional tensor.
But we can think of that
as 128 different
13 by 132-D grids.
So now we can actually go and visualize
each of those 13 by 13 elements slices
of the feature map as a grayscale image
and this gives us some sense
for what types of things
in the input are each of those features
in that convolutional layer looking for.
So this is a, a really
cool interactive tool
by Jason Yasenski you can just download.
So it's run, so I don't have the video,
it has a video on his website.
But it's running a convolutional network
on the inputs stream of webcam
and then visualizing in real time
each of those slices of that
intermediate feature map
give you a sense of what it's looking for
and you can see that,
so here the input image
is this, this picture up in, settings...
of this picture of a person
in front of the camera
and most of these intermediate features
are kind of noisy, not much going on.
But there's a, but there's
this one highlighted
intermediate feature
where that is also shown larger here
that seems that it's activating
on the portions of the feature map
corresponding to the person's face.
Which is really interesting
and that kind of,
suggests that maybe this,
this particular slice of the feature map
of this layer of this particular network
is maybe looking for human
faces or something like that.
Which is kind of a nice, kind of a nice
and cool finding.
Question?
The question is, Are the
black activations dead relu's?
So you got to be... a little
careful with terminology.
We usually say dead relu to mean
something that's dead over
the entire training data set.
Here I would say that it's a
relu, that, it's not active
for this particular input.
Question?
The question is, If there's
no humans in image net
how can it recognize a human face?
There definitely are humans in image net
I don't think it's, it's one of the cat...
I don't think it's one of
the thousand categories
for the classification challenge.
But people definitely appear
in a lot of these images
and that can be useful
signal for detecting
other types of things.
So that's actually kind of nice results
because that shows that, it's
sort of can learn features
that are useful for the
classification task at hand.
That are even maybe a little bit different
from the explicit classification task
that we told it to perform.
So it's actually really cool results.
Okay, question?
So at each layer in the
convolutional network
our input image is of three,
it's like 3 by 224 by 224
and then it goes through
many stages of convolution.
And then, it, after
each convolutional layer
is some three dimensional
chunk of numbers.
Which are the outputs from that layer
of the convolutional network.
And that into the entire three
dimensional chunk of numbers
which are the output of the
previous convolutional layer,
we call, we call, like
an activation volume
and then one of those, one of those slices
is a, it's an activation map.
So the question is, If the image is K by K
will the activation map be K by K?
Not always because there
can be sub sampling
due to pool, straight
convolution and pooling.
But in general, the, the
size of each activation map
will be linear in the
size of the input image.
So another, another kind
of useful thing we can do
for visualizing
intermediate features is...
Visualizing what types of
patches from input images
cause maximal activation in different,
different features, different neurons.
So what we've done here
is that, we pick...
Maybe again the con five
layer from Alex Net?
And remember each of
these activation volumes
at the con, at the con
five in Alex net gives us
a 128 by 13 by 13 chunk of numbers.
Then we'll pick one of those 128 channels.
Maybe channel 17
and now what we'll do is run many images
through this convolutional network.
And then, for each of those images
record the con five features
and then look at the...
Right, so, then, then look at the, the...
The parts of that 17th feature map
that are maximally activated
over our data set of images.
And now, because again this
is a convolutional layer
each of those neurons in
the convolutional layer
has some small receptive
field in the input.
Each of those neurons is not
looking at the whole image.
They're only looking at
the sub set of the image.
Then what we'll do is,
is visualize the patches
from the, from this
large data set of images
corresponding to the maximal activations
of that, of that feature,
of that particular feature
in that particular layer.
And then we can sorts these out,
sort these patches by their activation
at that, at that particular layer.
So here is a, some examples from this...
Network called a, fully...
The network doesn't matter.
But these are some visualizations
of these kind of maximally
activating patches.
So, each, each row gives...
We've chosen one layer from or one neuron
from one layer of a network
and then each, and then,
the, they're sorted
of these are the patches from
some large data set of images.
That maximally activated this one neuron.
And these can give you a sense
for what type of features
these, these neurons might be looking for.
So for example, this top row
we see a lot of circly kinds
of things in the image.
Some eyes, some, mostly eyes.
But also this, kind of blue circly region.
So then, maybe this,
this particular neuron
in this particular layer of
this network is looking for
kind of blue circly things in the input.
Or maybe in the middle here
we have neurons that are looking for
text in different colors
or, or maybe curving, curving edges
of different colors and orientations.
Yeah, so, I've been a little bit loose
with terminology here.
So, I'm saying that a
neuron is one scaler value
in that con five activation map.
But because it's convolutional,
all the neurons in one channel
are all using the same weights.
So we've chosen one
channel and then, right?
So, you get a lot of neurons
for each convolutional filter
at any one layer.
So, we, we could have been,
so this patches could've
been drawn from anywhere
in the image due to the
convolutional nature of the thing.
And now at the bottom we also see
some maximally activating patches
for neurons from a higher up
layer in the same network.
And now because they are coming
from higher in the network
they have a larger receptive field.
So, they're looking at larger
patches of the input image
and we can also see
that they're looking for
maybe larger structures
in the input image.
So this, this second row is maybe looking,
it seems to be looking for human,
humans or maybe human faces.
We have maybe something looking for...
Parts of cameras or
different types of larger,
larger, larger object like
type things, types of things.
Another, another cool experiment we can do
which comes from Zeiler
and Fergus ECCV 2014 paper.
is this idea of an exclusion experiment.
So, what we want to do is figure out
which parts of the
input, of the input image
cause the network to make
it's classification decision.
So, what we'll do is,
we'll take our input image
in this case an elephant
and then we'll block
out some part of that,
some region in that input image
and just replace it with
the mean pixel value
from the data set.
And now, run that
occluded image throughout,
through the network and then record
what is the predicted probability
of this occluded image?
And now slide this occluded
patch over every position
in the input image and then
repeat the same process.
And then draw this heat map showing,
what was the predicted probability
output from the network
as a function of where did,
which part of the input
image did we occlude?
And the idea is that
if when we block out
some part of the image
if that causes the network
score to change drastically.
Then probably that part of the input image
was really important for
the classification decision.
So here we've shown...
I've shown three different examples of...
Of this occlusion type experiment.
So, maybe this example of
a Go-kart at the bottom,
you can see over here that
when we, so here, red,
the, the red corresponds
to a low probability
and the white and yellow
corresponds to a high probability.
So when we block out
the region of the image
corresponding to this Go-kart in front.
Then the predicted probability
for the Go-kart class drops a lot.
So that gives us some sense
that the network is actually
caring a lot about these,
these pixels in the input image
in order to make it's
classification decision.
Question?
Yes, the question is that,
what's going on in the background?
So maybe if the image is a
little bit too small to tell
but, there's, this is
actually a Go-kart track
and there's a couple other
Go-karts in the background.
So I think that, when
you're blocking out these
other Go-karts in the background,
that's also influencing the score
or maybe like the horizon is there
and maybe the horizon is an useful feature
for detecting Go-karts,
it's a little bit hard to tell sometimes.
But this is a pretty cool visualization.
Yeah, was there another question?
So the question is, sorry,
sorry, what was the first question?
So, the, so the question...
So for, for this example
we're taking one image
and then masking all parts of one image.
The second question
was, how is this useful?
It's not, maybe, you don't
really take this information
and then loop it directly
into the training process.
Instead, this is a way, a tool for humans
to understand, what types of computations
these train networks are doing.
So it's more for your understanding
than for improving performance per se.
So another, another related idea
is this concept of a Saliency Map.
Which is something that you
will see in your homeworks.
So again, we have the same question
of given an input image
of a dog in this case
and the predicted class label of dog
we want to know which
pixels in the input image
are important for classification.
We saw masking, is one way
to get at this question.
But Saliency Maps are another, another,
angle for attacking this problem.
And the question is, and
one relatively simple idea
from Karen Simonenian's
paper, a couple years ago.
Is, this is just computing the gradient
of the predicted class score
with respect to the
pixels of the input image.
And this will directly tell us
in this sort of, first
order approximation sense.
For each input, for each
pixel in the input image
if we wiggle that pixel a little bit
then how much will the
classification score
for the class change?
And this is another way
to get at this question
of which pixels in the input
matter for the classification.
And when we, and when we run
for example Saliency,
where computer Saliency map
for this dog, we see kind of a nice
outline of a dog in the image.
Which tells us that these
are probably the pixels
of that, network is actually
looking at, for this image.
And when we repeat this type of process
for different images, we get some sense
that the network is sort of
looking at the right regions.
Which is somewhat comforting.
Question?
The question is, do
people use Saliency Maps
for semantic segmentation?
The answer is yes.
That actually was...
Yeah, you guys are like really
on top of it this lecture.
So that was another component,
again in Karen's paper.
Where there's this idea
that maybe you can use these
Saliency Maps to perform
semantic segmentation
without direct, without any labeled data
for the, for these, for these segments.
So here they're using this
Grabcut Segmentation Algorithm
which I don't really want
to get into the details of.
But it's kind of an interactive
segmentation algorithm
that you can use.
So then when you combine this Saliency Map
with this Grabcut Segmentation Algorithm
then you can in fact,
sometimes segment out
the object in the image.
Which is really cool.
However I'd like to point out
that this is a little bit brittle
and in general if you,
this will probably work
much, much, much, worse than a network
which did have access to
supervision and training time.
So, I don't, I'm not sure
how, how practical this is.
But it is pretty cool
that it works at all.
But it probably works much
less than something trained
explicitly to segment with supervision.
So kind of another, another related idea
is this idea of, of
guided back propagation.
So again, we still want
to answer the question of
for one particular, for
one particular image.
Then now instead of
looking at the class score
we want to know, we want to
pick some intermediate neuron
in the network and ask again,
which parts of the input image
influence the score of that neuron,
that internal neuron in the network.
And, and then you could
imagine, again you could imagine
computing a Saliency Map
for this again, right?
That rather than computing the
gradient of the class scores
with respect to the pixels of the image.
You could compute the gradient
of some intermediate value
in the network with respect
to the pixels of the image.
And that would tell us again
which parts, which
pixels in the input image
influence that value of
that particular neuron.
And that would be using
normal back propagation.
But it turns out that
there is a slight tweak
that we can do to this
back propagation procedure
that ends up giving some
slightly cleaner images.
So that's this idea of
guided back propagation
that again comes from Zeiler
and Fergus's 2014 paper.
And I don't really want to get
into the details too much here
but, it, you just, it's
kind of weird tweak
where you change the way
that you back propagate
through relu non-linearities.
And you sort of, only, only back propagate
positive gradients through relu's
and you do not back propagate
negative gradients through the relu's.
So you're no longer
computing the true gradient
instead you're kind of only keeping track
of positive influences
on throughout the entire network.
So maybe you should read
through these, these papers
reference to your, if you
want a little bit more details
about why that's a good idea.
But empirically, when you
do guided back propagation
as appose to regular back propagation.
You tend to get much
cleaner, nicer images.
that tells you, which part,
which pixel of the input image
influence that particular neuron.
So, again we were seeing
the same visualization
we saw a few slides ago of the
maximally activating patches.
But now, in addition to visualizing
these maximally activating patches.
We've also performed
guided back propagation,
to tell us exactly which parts
of these patches influence
the score of that neuron.
So, remember for this example at the top,
we saw that, we thought this neuron
is may be looking for circly tight things,
in the input patch
because there're allot
of circly tight patches.
Well, when we look at
guided back propagation
We can see with that intuition
is somewhat confirmed
because it is indeed the circly
parts of that input patch
which are influencing
that, that neuron value.
So, this is kind of a useful
to all for synthesizing.
For understanding what these
different intermediates
are looking for.
But, one kind of interesting thing
about guided back propagation
or computing saliency maps.
Is that there's always a
function of fixed input image,
right, they're telling us
for a fixed input image,
which pixel or which parts
of that input image influence
the value of the neuron.
Another question you might answer is
is remove this reliance, on
that, on some input image.
And then instead just ask
what type of input in general
would cause this neuron to activate
and we can answer this question
using a technical Gradient ascent
so, remember we always use Gradient decent
to train our convolutional
networks by minimizing the loss.
Instead now, we want to fix the, fix
the weight of our trained
convolutional network
and instead synthesizing image
by performing Gradient ascent
on the pixels of the
image to try and maximize
the score of some intermediate
neuron or of some class.
So, in a process of Gradient ascent,
we're no longer optimizing
over the weights of the network
those weights remained fixed
instead we're trying to change
pixels of some input image
to cause this neuron,
or this neuron value,
or this class score to
maximally, to be maximized
but, instead but, in addition
we need some regularization term
so, remember we always a,
we before seeing regularization terms
to try to prevent the network weights
from over fitting to the training data.
Now, we need something kind of similar
to prevent the pixels
of our generated image
from over fitting to the peculiarities
of that particular network.
So, here we'll often incorporate
some regularization term
that, we're kind of, we
want a generated image
of two properties
one, we wanted to maximally activate some,
some score or some neuron value.
But, we also wanted to
look like a natural image.
we wanted to kind of have,
the kind of statistics
that we typically see in natural images.
So, these regularization
term in the subjective
is something to enforce a generated image
to look relatively natural.
And we'll see a couple
of different examples
of regualizers as we go through.
But, the general strategy for this
is actually pretty simple
and again informant allot
of things of this nature
on your assignment three.
But, what we'll do is start
with some initial image
either initializing to zeros
or to uniform or noise.
But, initialize your image in some way
and I'll repeat where
you forward your image
through 3D network and compute the score
or, or neuron value
that you're interested.
Now, back propagate to
compute the Gradient
of that neuron score with respect
to the pixels of the image
and then make a small Gradient ascent
or Gradient ascent update
to the pixels of the images itself.
To try and maximize that score.
And I'll repeat this
process over and over again,
until you have a beautiful image.
And, then we talked, we talked
about the image regularizer,
well a very simple, a very
simple idea for image regularizer
is simply to penalize L2
norm of a generated image
This is not so semantically meaningful,
it's just does something,
and this was one of the,
one of the earliest
regularizer that we've seen
in the literature for these
type of generating images type
of papers.
And, when you run this
on a trained network
you can see that now we're
trying to generate images
that maximize the dumble score
in the upper left hand
corner here for example.
And, then you can see that
the synthesized image,
it been, it's little
bit hard to see may be
but there're allot of
different dumble like shapes,
all kind of super impose
that different portions of the image.
or if we try to generate an image for cups
then we can may be see a
bunch of different cups
all kind of super imposed
the Dalmatian is pretty cool,
because now we can see kind of this black
and white spotted pattern
that's kind of
characteristics of Dalmatians
or for lemons we can see
these different kinds
of yellow splotches in the image.
And there's a couple
of more examples here,
I think may be the goose is kind of cool
or the kitfox are actually
may be looks like kitfox.
Question?
The question is, why are
these all rainbow colored
and in general getting true colors out
of this visualization is pretty tricky.
Right, because any, any actual image will
be bounded in the range zero to 255.
So, it really should be some kind
of constrained optimization problem
But, if, for using this generic
methods for Gradient ascent
then we, that's going to
be unconstrained problem.
So, you may be use like projector
Gradient ascent algorithm
or your rescaled image at the end.
So, the colors that you
see in this visualizations,
sometimes are you cannot
take them too seriously.
Question?
The question is what happens, if you let
the thing loose and don't
put any regularizer on it.
Well, then you tend to get
an image which maximize
the score which is confidently classified
as the class you wanted
but, usually it doesn't
look like anything.
It kind of look likes random noise.
So, that's kind of an
interesting property in itself
that will go into much more
detail in a future lecture.
But, that's why, that
kind of doesn't help you
so much for understanding what things
the network is looking for.
So, if we want to understand,
why the network thing makes its decisions
then it's kind of useful
to put regularizer
on there to generate an
image to look more natural.
A question in the back.
Yeah, so the question
is that we see a lot of
multi modality here, and
other ways to combat that.
And actually yes, we'll see that,
this is kind of first
step in the whole line
of work in improving these visualizations.
So, another, another kind
of, so then the angle here
is a kind of to improve the regularizer
to improve our visualized images.
And there's a another
paper from Jason Yesenski
and some of his collaborators where
they added some additional
impressive regularizers.
So, in addition to this
L2 norm constraint,
in addition we also periodically
during optimization,
and do some gauche and
blurring on the image,
we're also clip some,.
some small value, some small pixel values
all the way to zero, we're
also clip some of the,
some of the pixel values
of low Gradients to zero
So, you can see this is kind of
a projector Gradient ascent algorithm
where it reach periodically
we're projecting
our generated image onto some nicer set
of images with some nicer properties.
For example, special smoothness
with respect to the gauchian blurring
So, when you do this, you
tend to get much nicer images
that are much clear to see.
So, now these flamingos
look like flamingos
the ground beetle is starting
to look more beetle like
or this black swan maybe
looks like a black swan.
These billiard tables actually
look kind of impressive now,
where you can definitely see
this billiard table structure.
So, you can see that once you
add in nicer regularizers,
then the generated images become a bit,
a little bit cleaner.
And, now we can perform this procedure
not only for the final class course,
but also for these
intermediate neurons as well.
So, instead of trying to
maximize our billiard table score
for example instead we
can get maximize one
of the neurons from
some intermediate layer
Question.
So, the question is what's
with the for example here,
so those who remember
initializing our image randomly
so, these four images would
be different random
initialization of the input image.
And again, we can use these
same type of procedure
to visualize, to synthesis images
which maximally activate
intermediate neurons
of the network.
And, then you can get a sense from some
of these intermediate
neurons are looking for,
so may be at layer four there's neuron
that's kind of looking for spirally things
or there's neuron that's may be looking
for like chunks of caterpillars
it's a little bit harder to tell.
But, in generally as you
go larger up in the image
then you can see that
the one, the obviously
receptive fields of
these neurons are larger.
So, you're looking at the
larger patches in the image.
And they tend to be looking
for may be larger structures
or more complex patterns
in the input image.
That's pretty cool.
And, then people have
really gone crazy with this
and trying to, they basically
improve these visualization
by keeping on extra features
So, this was a cool paper kind
of explicitly trying to address this
multi modality, there's
someone asked question
about a few minutes ago.
So, here they were
trying to explicitly take
a count, take this multi
modality into account
in the optimization procedure
where they did indeed,
I think see the initial,
so they for each of the classes,
you run a clustering algorithm
to try to separate the
classes into different modes
and then initialize with something
that is close to one of those modes.
And, then when you do that,
you kind of account for
this multi modality.
so for intuition, on the
right here these eight images
are all of grocery stores.
But, the top row, is
kind of close up pictures
of produce on the shelf
and those are labeled as grocery stores
And the bottom row kind of shows
people walking around grocery stores
or at the checkout line
or something like that.
And, those are also labeled
those as grocery store,
but their visual appearance
is quiet different.
So, a lot of these classes
and that being sort
multi modal
And, if you can take, and
if you explicitly take
this more time mortality into account
when generating images, then
you can get nicer results.
And now, then when you look at some
of their example, synthesis
images for classes,
you can see like the
bell pepper, the card on,
strawberries, jackolantern now they end up
with some very beautifully
generated images.
And now, I don't want to get to much
into detail of the next slide.
But, then you can even go crazier.
and add an even stronger image prior
and generate some very
beautiful images indeed
So, these are all synthesized
images that are trying
to maximize the class score
or some image in a class.
But, the general idea is that rather
than optimizing directly the pixels
of the input image, instead they're trying
to optimize the FC6 representation
of that image instead.
And now they need to use some
feature inversion network
and I don't want to get
into the details here.
You should read the paper,
it's actually really cool
But, the point is that,
when you start adding additional priors
towards modeling natural images
and you can end generating
some quiet realistic images
they gave you some sense of
what the network is looking for
So, that's, that's sort of one cool thing
that we can do with this
strategy, but this idea
of trying to synthesis
images by using Gradients
on image pixels, is
actually super powerful.
And, another really cool
thing we can do with this,
is this concept of fooling image
So, what we can do is
pick some arbitrary image,
and then try to maximize the,
so, say we take it picture
of an elephant and then
we tell the network
that we want to, change the image
to maximize the score
of Koala bear instead
So, then what we were
doing is trying to change
that image of an elephant
to try and instead cause
the network to classify as a Koala bear.
And, what you might hope for is that,
maybe that elephant was
sort of thought more thing
into a Koala bear and
maybe he would sprout
little cute ears or something like that.
But, that's not what happens in practice,
which is pretty surprising.
Instead if you take this
picture of a elephant
and tell them that, tell them that
and try to change the
elephant image to instead
cause it to be classified as a koala bear
What you'll find is that, you is that
this second image on the right actually
is classified as koala bear
but it looks the same to us.
So that's pretty fishy
and pretty surprising.
So also on the bottom we've
taken this picture of a boat.
Schooner is the image in that class
and then we told the network
to classified as an iPod.
So now the second example looks just,
still looks like a boat to us
but the network thinks it's an iPod
and the difference is in
pixels between these two images
are basically nothing.
And if you magnify those differences
you don't really see any
iPod or Koala like features
on these differences,
they're just kind of like
random patterns of noise.
So the question is what's going here?
And like how can this possibly the case?
Well, we'll have a guest
lecture from Ian Goodfellow
in a week an half two weeks.
And he's going to go in much more detail
about this type of phenomenon
and that will be really exciting.
But I did want to mention it here
because it is on your homework.
Question?
Yeah, so that's something,
so the question is can we use
fooled images as training data
and I think, Ian's going
to go in much more detail
on all of these types of strategies.
Because that's literally,
that's really a whole lecture onto itself.
Question?
The question is why do we
care about any of this stuff?
Basically...
Okay, maybe that was a
mischaracterization, I am sorry.
Yeah, the question is
what is have in the...
understanding this intermediate neurons
how does that help our understanding
of the final classification.
So this is actually, this
whole field of trying
to visualize intermediates
is kind of in response
to a common criticism of deep learning.
So a common criticism of
deep learning is like,
you've got this big black box network
you trained it on gradient
ascent, you get a good number
and that's great but we
don't trust the network
because we don't understand as people
why it's making the
decisions, that's it's making.
So a lot of these type of
visualization techniques
were developed to try and address that
and try to understand as people
why the network are making
their various classification,
classification decisions a bit more.
Because if you contrast,
if you contrast a deep
convolutional neural network
with other machine running techniques.
Like linear models are much
easier to interpret in general
because you can look at
the weights and kind of
understand the interpretation
between how much each input
feature effect the decision or
if you look at something like
a random forest or decision tree.
Some other machine learning models
end up being a bit more interpretable
just by their very nature
then this sort of black box
convolutional networks.
So a lot of this is sort of
in response to that criticism
to say that, yes they are
these large complex models
but they are still doing some
interesting and interpretable
things under the hood.
They are not just totally going out
in randomly classifying things.
They are doing something meaningful
So another cool thing we can do with this
gradient based optimization of images
is this idea of DeepDream.
So this was a really cool blog post
that came out from
Google a year or two ago.
And the idea is that,
this is, so we talked
about scientific value,
this is almost entirely for fun.
So the point of this exercise is mostly
to generate cool images.
And aside, you also get
some sense for what features
images are looking at.
Or these networks are looking at.
So we can do is, we take our input image
we run it through the convolutional
network up to some layer
and now we back propagate
and set the gradient
of that, at that layer
equal to the activation value.
And now back propagate, back to the image
and update the image and
repeat, repeat, repeat.
So this has the interpretation
of trying to amplify
existing features that were
detected by the network
in this image. Right?
Because whatever features
existed on that layer
now we set the gradient
equal to the feature
and we just tell the network to amplify
whatever features you
already saw in that image.
And by the way you can also
see this as trying to maximize
the L2 norm of the features
at that layer of the image.
And it turns... And when you do this
the code ends up looking really simple.
So your code for many of
your homework assignments
will probably be about this complex
or maybe even a little bit a less so.
So the idea is that...
But there's a couple of tricks here
that you'll also see in your assignments.
So one trick is to jitter the image
before you compute your gradients.
So rather than running the
exact image through the network
instead you'll shift the
image over by two pixels
and kind of wrap the other
two pixels over here.
And this is a kind of regularizer
to prevent each of these [mumbling]
it regularizers a little bit to encourage
a little bit of extra special
smoothness in the image.
You also see they use L1
normalization of the gradients
that's kind of a useful trick sometimes
when doing this image generation problems.
You also see them clipping the
pixel values once in a while.
So again we talked about
images actually should be
between zero to 2.55
so this is a kind of
projected gradients decent
where we project on to the
space of actual valid images.
But now when we do all this
then we start, we might start
with some image of a sky
and then we get really
cool results like this.
So you can see that now
we've taken these tiny features on the sky
and they get amplified through
this, through this process.
And we can see things like this different
mutant animals start to pop up
or these kind of spiral shapes pop up.
Different kinds of houses and cars pop up.
So that's all, that's
all pretty interesting.
There's a couple patterns in particular
that pop up all the time
that people have named.
Right, so there's this Admiral
dog, that shows up allot.
There's the pig snail, the camel bird
this the dog fish.
Right, so these are
kind of interesting, but
actually this fact that
dog show up so much
in these visualization,
actually does tell us
something about the data on
which this network was trained.
Right, because this is a
network that was trained
for image net classification,
image that have thousand categories.
But 200 of those categories are dogs.
So, so it's kind of not
surprising in a sense
that when you do these
kind of visualizations
then network ends up hallucinating
a lot of dog like stuff
in the image often morphed
with other types of animals.
When you do this other
layers of the network
you get other types of results.
So here we're taking one
of these lower layers
in the network, the previous
example was relatively
high up in the network
and now again we have this
interpretation that lower layers
maybe computing edges and
swirls and stuff like that
and that's kind of borne out
when we running DeepDream
at a lower layer.
Or if you run this thing for a long time
and maybe add in some
multiscale processing
you can get some really,
really crazy images.
Right, so here they're doing a
kind of multiscale processing
where they start with a small image
run DeepDream on the small
image then make it bigger
and continue DeepDream on the larger image
and kind of repeat with
this multiscale processing
and then you can get,
and then maybe after you
complete the final scale
then you restart from the beginning
and you just go wild on this thing.
And you can get some really crazy images.
So these examples were all from networks
trained on image net
There's another data set from
MIT called MIT Places Data set
but instead of 1,000 categories of objects
instead it's 200 different types of scenes
like bedrooms and kitchens
like stuff like that.
And now if we repeat
this DeepDream procedure
using an network trained at MIT places.
We get some really cool
visualization as well.
So now instead of dogs,
slugs and admiral dogs
and that's kind of stuff,
instead we often get these
kind of roof shapes of these
kind of Japanese style building
or these different types of
bridges or mountain ranges.
They're like really, really
cool beautiful visualizations.
So the code for DeepDream is
online, released by Google
you can go check it out and
make your own beautiful pictures
So there's another kind of...
Sorry question?
So the question is, what
are taking gradient of?
So like I say, if you, because
like one over x squared
on the gradient of that is x.
So, if you send back
the volume of activation
as the gradient, that's equivalent to max,
that's equivalent to taking the
gradient with respect to the
like one over x squared some...
Some of the values.
So it's equivalent to maximizing the norm
of that of the features of that layer.
But in practice many implementation
you'll see not explicitly compute that
instead of send gradient back.
So another kind of useful,
another kind of useful thing
we can do is this concept
of feature inversion.
So this again gives us a
sense for what types of,
what types of elements
of the image are captured
at different layers of the network.
So what we're going to
do now is we're going to
take an image, run that
image through network
record the feature value
for one of those images
and now we're going to try
to reconstruct that image
from its feature representation.
And the question, and now
based on the how much, how much like
what that reconstructed image looks like
that'll give us some sense
for what type of information
about the image was captured
in that feature vector.
So again, we can do this
with gradient ascent
with some regularizer.
Where now rather than
maximizing some score
instead we want to minimize the distance
between this catch feature vector.
And between the computed
features of our generated image.
To try and again synthesize
a new image that matches
the feature back to
that we computed before.
And another kind of regularizer
that you frequently see here
is the total variation regularizer
that you also see on your homework.
So here with the total
variation regularizer
is panelizing differences
between adjacent pixels
on both of the left and
adjacent in left and right
and adjacent top to bottom.
To again try to encourage
special smoothness
in the generated image.
So now if we do this
idea of feature inversion
so this visualization here on the left
we're showing some original image.
The elephants or the fruits at the left.
And then we run that,
we run the image through a VGG-16 network.
Record the features of
that network at some layer
and then try to synthesize
a new image that matches
the recorded features of that layer.
And this is, this kind of
give us a sense for what
how much information is
stored in this images.
In these features of different layers.
So for example if we try
to reconstruct the image
based on the relu2_2 features
from VGC's, from VGG-16.
We see that the image gets
almost perfectly reconstructed.
Which means that we're
not really throwing away
much information about the raw
pixel values at that layer.
But as we move up into the
deeper parts of the network
and try to reconstruct
from relu4_3, relu5_1.
We see that our reconstructed image now,
we've kind of kept the general space,
the general spatial
structure of the image.
You can still tell that, that
it's a elephant or a banana
or a, or an apple.
But a lot of the low level details aren't
exactly what the pixel values were
and exactly what the colors were,
exactly what the textures were.
These are kind of low level details are
kind of lost at these higher
layers of this network.
So that gives us some sense that
maybe as we move up through
the flairs of the network
it's kind of throwing away
this low level information
about the exact pixels of the image
and instead is maybe trying
to keep around a little bit
more semantic information,
it's a little bit invariant
for small changes in color and
texture and things like that.
So we're building towards
a style transfer here
which is really cool.
So in addition to
understand style transfer,
in addition to feature inversion.
We also need to talk
about a related problem
called texture synthesis.
So in texture synthesis, this
is kind of an old problem
in computer graphics.
Here the idea is that
we're given some input
patch of texture.
Something like these little scales here
and now we want to build some model
and then generate a larger
piece of that same texture.
So for example, we might here
want to generate a large image
containing many scales that
kind of look like input.
And this is again a pretty old
problem in computer graphics.
There are nearest neighbor
approaches to textual synthesis
that work pretty well.
So, there's no neural networks here.
Instead, this kind of a simple algorithm
where we march through the generated image
one pixel at a time in scan line order.
And then copy...
And then look at a neighborhood
around the current pixel
based on the pixels that
we've already generated
and now compute a nearest
neighbor of that neighborhood
in the patches of the input image
and then copy over one
pixel from the input image.
So, maybe you don't need to
understand the details here
just the idea is that there's
a lot classical algorithms
for texture synthesis,
it's a pretty old problem
but you can do this without
neural networks basically.
And when you run this kind of
this kind of classical
texture synthesis algorithm
it actually works reasonably
well for simple textures.
But as we move to more complex textures
these kinds of simple methods
of maybe copying pixels
from the input patch directly
tend not to work so well.
So, in 2015, there was a really cool paper
that tried to apply
neural network features
to this problem of texture synthesis.
And ended up framing it as kind of
a gradient ascent procedure,
kind of similar to the feature map,
the various feature matching objectives
that we've seen already.
So, in order to perform
neural texture synthesis
they use this concept of a gram matrix.
So, what we're going to do,
is we're going to take our input texture
and in this case some pictures of rocks
and then take that input texture
and pass it through some
convolutional neural network
and pull out convolutional features
at some layer of the network.
So, maybe then this
convolutional feature volume
that we've talked about,
might be H by W by C
or sorry, C by H by W at
that layer of the network.
So, you can think of this
as an H by W spacial grid.
And at each point of the grid,
we have this C dimensional feature vector
describing the rough
appearance of that image
at that point.
And now, we're going to
use this activation map
to compute a descriptor of the
texture of this input image.
So, what we're going to do is take,
pick out two of these
different feature columns
in the input volume.
Each of these feature columns
will be a C dimensional vector.
And now take the outer product
between those two vectors
to give us a C by C matrix.
This C by C matrix now
tells us something about the co-occurrence
of the different features at
those two points in the image.
Right, so, if an element,
if like element IJ
in the C by C matrix is large
that means both elements I and
J of those two input vectors
were large and something like that.
So, this somehow captures
some second order statistics
about which features, in that feature map
tend to activate to together
at different spacial volumes...
At different spacial positions.
And now we're going to
repeat this procedure
using all different
pairs of feature vectors
from all different points
in this H by W grid.
Average them all out, and that gives us
our C by C gram matrix.
And this is then used a
descriptor to describe
kind of the texture of that input image.
So, what's interesting
about this gram matrix
is that it has now thrown
away all spacial information
that was in this feature volume.
Because we've averaged over
all pairs of feature vectors
at every point in the image.
Instead, it's just
capturing the second order
co-occurrence statistics between features.
And this ends up being a
nice descriptor for texture.
And by the way, this is
really efficient to compute.
So, if you have a C by H by
W three dimensional tensure
you can just reshape
it to see times H by W
and take that times its own transpose
and compute this all in one shot
so it's super efficient.
But you might be wondering why
you don't use an actual covariance matrix
or something like that instead
of this funny gram matrix
and the answer is that using covariance...
Using true covariance matrices also works
but it's a little bit
more expensive to compute.
So, in practice a lot of people
just use this gram matrix descriptor.
So then... Then there's this...
Now once we have this sort of
neural descriptor of texture
then we use a similar type
of gradient ascent procedure
to synthesize a new image
that matches the texture
of the original image.
So, now this looks kind of
like the feature reconstruction
that we saw a few slides ago.
But instead, I'm trying to
reconstruct the whole feature map
from the input image.
Instead, we're just going
to try and reconstruct
this gram matrix texture descriptor
of the input image instead.
So, in practice what this
looks like is that well...
You'll download some pretrained model,
like in feature inversion.
Often, people will use
the VGG networks for this.
You'll feed your... You'll
take your texture image,
feed it through the VGG network,
compute the gram matrix
and many different layers
of this network.
Then you'll initialize your new image
from some random initialization
and then it looks like
gradient ascent again.
Just like for these other
methods that we've seen.
So, you take that image, pass it through
the same VGG network,
Compute the gram matrix at various layers
and now compute loss
as the L2 norm between the gram matrices
of your input texture
and your generated image.
And then you back prop,
and compute pixel...
A gradient of the pixels
on your generated image.
And then make a gradient ascent step
to update the pixels of
the image a little bit.
And now, repeat this process many times,
go forward, compute your gram matrices,
compute your losses, back prop..
Gradient on the image and repeat.
And once you do this, eventually
you'll end up generating
a texture that matches your
input texture quite nicely.
So, this was all from Nip's 2015 paper
by a group in Germany.
And they had some really cool
results for texture synthesis.
So, here on the top,
we're showing four
different input textures.
And now, on the bottom, we're showing
doing this texture synthesis approach
by gram matrix matching.
Using, by computing the gram
matrix at different layers
at this pretrained convolutional network.
So, you can see that, if we
use these very low layers
in the convolutional network
then we kind of match the general...
We generally get splotches
of the right colors
but the overall spacial structure
doesn't get preserved so much.
And now, as we move to large
down further in the image
and you compute these gram
matrices at higher layers
you see that they tend to
reconstruct larger patterns
from the input image.
For example, these whole rocks
or these whole cranberries.
And now, this works pretty well
that now we can synthesize
these new images
that kind of match the
general spacial statistics
of your inputs.
But they are quite different pixel wise
from the actual input itself.
Question?
So, the question is, where
do we compute the loss?
And in practice, we
want to get good results
typically people will
compute gram matrices
at many different layers
and then the final loss
will be a sum of all those
potentially a weighted sum.
But I think for this visualization,
to try to pin point the
effect of the different layers
I think these were doing reconstruction
from just one layer.
So, now something really...
Then, then they had a
really brilliant idea
kind of after this paper
which is, what if we do this
texture synthesis approach
but instead of using an image
like rocks or cranberries
what if we set it equal
to a piece of artwork.
So then, for example, if you...
If you do the same texture
synthesis algorithm
by maximizing gram
matrices, but instead of...
But now we take, for example,
Vincent Van Gogh's Starry night
or the Muse by Picasso as our texture...
As our input texture,
and then run this same
texture synthesis algorithm.
Then we can see our generated images tend
to reconstruct interesting pieces
from those pieces of artwork.
And now, something really
interesting happens
when you combine this
idea of texture synthesis
by gram matrix matching
with feature inversion
by feature matching.
And then this brings us to
this really cool algorithm
called style transfer.
So, in style transfer, we're
going to take two images as input.
One, we're going to take a content image
that will guide like what
type of thing we want.
What we generally want
our output to look like.
Also, a style image that will tell us
what is the general texture or style
that we want our generated image to have
and then we will jointly
do feature recon...
We will generate a new image
by minimizing the feature
reconstruction loss
of the content image
and the gram matrix
loss of the style image.
And when we do these two things
we a get a really cool image that kind of
renders the content image
kind of in the artistic style
of the style image.
And now this is really cool.
And you can get these
really beautiful figures.
So again, what this kind of looks like
is that you'll take your style
image and your content image
pass them into your network
to compute your gram matrices
and your features.
Now, you'll initialize your output image
with some random noise.
Go forward, compute your losses
go backward, compute your
gradients on the image
and repeat this process over and over
doing gradient ascent on the
pixels of your generated image.
And after a few hundred iterations,
generally you'll get a beautiful image.
So, I have implementation of this online
on my Gethub, that a
lot of people are using.
And it's really cool.
So, you can, this is kind of...
Gives you a lot more control
over the generated image
as compared to DeepDream.
Right, so in DeepDream, you
don't have a lot of control
about exactly what types of
things are going to happen
coming out at the end.
You just kind of pick different
layers of the networks
maybe set different numbers of iterations
and then dog slugs pop up everywhere.
But with style transfer,
you get a lot more fine grain control
over what you want the
result to look like.
Right, by now, picking
different style images
with the same content image
you can generate whole
different types of results
which is really cool.
Also, you can play around with
the hyper parameters here.
Right, because we're doing
a joint reconstruct...
We're minimizing this
feature reconstruction loss
of the content image.
And this gram matrix reconstruction
loss of the style image.
If you trade off the constant,
the waiting between those
two terms and the loss.
Then you can get control
about how much we want
to match the content versus
how much we want to match the style.
There's a lot of other hyper
parameters you can play with.
For example, if you resize the style image
before you compute the gram matrix
that can give you some control
over what the scale of features are
that you want to reconstruct
from the style image.
So, you can see that here,
we've done this same reconstruction
the only difference is how
big was the style image
before we computed the gram matrix.
And this gives you another axis over
which you can control these things.
You can also actually do style transfer
with multiple style images
if you just match sort
of multiple gram matrices
at the same time.
And that's kind of a cool result.
We also saw this multi-scale process...
So, another cool thing you can do.
We talked about this multi-scale
processing for DeepDream
and saw how multi scale
processing in DeepDream
can give you some really
cool resolution results.
And you can do a similar type
of multi-scale processing
in style transfer as well.
So, then we can compute images like this.
That a super high resolution,
this is I think a 4k image
of our favorite school,
like rendered in the
style of Starry night.
But this is actually super
expensive to compute.
I think this one took four GPU's.
So, a little expensive.
We can also other style,
other style images.
And get some really cool results
from the same content image.
Again, at high resolution.
Another fun thing you can do is
you know, you can actually
do joint style transfer
and DeepDream at the same time.
So, now we'll have three
losses, the content loss
the style loss and this...
And this DeepDream loss that
tries to maximize the norm.
And get something like this.
So, now it's Van Gogh
with the dog slug's coming out everywhere.
[laughing]
So, that's really cool.
But there's kind of a problem
with this style transfer for algorithms
which is that they are pretty slow.
Right, you need to produce...
You need to compute a lot of
forward and backward passes
through your pretrained network
in order to complete these images.
And especially for these high
resolution results that we saw
in the previous slide.
Each forward and backward
pass of a 4k image
is going to take a lot of
compute and a lot of memory.
And if you need to do several
hundred of those iterations
generating these images could take many,
like tons of minutes
even on a powerful GPU.
So, it's really not so
practical to apply these things
in practice.
The solution is to now,
train another neural network
to do the style transfer for us.
So, I had a paper about this last year
and the idea is that we're
going to fix some style
that we care about at the beginning.
In this case, Starry night.
And now rather than running
a separate optimization procedure
for each image that we want to synthesize
instead we're going to train
a single feed forward network
that can input the content image
and then directly output
the stylized result.
And now the way that we train this network
is that we compute the same content
and style losses during training
of our feed forward network
and use that same gradient
to update the weights
of the feed forward network.
And now this thing takes
maybe a few hours to train
but once it's trained,
then in order to produce stylized images
you just need to do a single forward pass
through the trained network.
So, I have a code for this online
and you can see that it
ends up looking about...
Relatively comparable
quality in some cases
to this very slow optimization base method
but now it runs in real time
it's about a thousand times faster.
So, here you can see, this is
like a demo of it running live
off my webcam.
So, this is not running
live right now obviously,
but if you have a big GPU
you can easily run four different styles
in real time all simultaneously
because it's so efficient.
There was... There was
another group from Russia
that had a very similar out...
That had a very similar paper concurrently
and their results are about as good.
They also had this kind
of tweek on the algorithm.
So, this feed forward
network that we're training
ends up looking a lot like these...
These segmentation models that we saw.
So, these segmentation networks,
for semantic segmentation
we're doing down sampling
and then many, and then many layers
then some up sampling [mumbling]
With transposed convulsion in order
to down sample an up sample
to be more efficient.
The only difference is
that this final layer
produces a three channel output
for the RGB of that final image.
And inside this network,
we have batch normalization
in the various layers.
But in this paper, they introduce...
They swap out the batch normalization
for something else called
instance normalization
tends to give you much better results.
So, one drawback of these
types of methods is that
we're now training one new
style transfer network...
For every... For style
that we want to apply.
So that could be expensive
if now you need to keep a lot
of different trained networks around.
So, there was a paper from
Google that just came...
Pretty recently that addressed this
by using one feed forward trained network
to apply many different
styles to the input image.
So now, they can train one network
to apply many different
styles at test time
using one trained network.
So, here's it's going to
take the content images input
as well as the identity of
the style you want to apply
and then this is using one network
to apply many different types of styles.
And again, runs in real time.
That same algorithm can also
do this kind of style blending
in real time with one trained network.
So now, once you trained this network
on these four different styles
you can actually specify
a blend of these styles
to be applied at test
time which is really cool.
So, these kinds of real
time style transfer methods
are on various apps and
you can see these out
in practice a lot now these days.
So, kind of the summary
of what we've seen today
is that we've talked about
many different methods
for understanding CNN representations.
We've talked about some of
these activation based methods
like nearest neighbor,
dimensionality reduction,
maximal patches, occlusion images
to try to understand based
on the activation values
of what the features are looking for.
We also talked about a bunch
of gradient based methods
where you can use gradients
to synthesize new images
to understand your features
such as saliency maps
class visualizations, fooling images,
feature inversion.
And we also had fun by seeing
how a lot of these similar ideas
can be applied to things like
Style Transfer and DeepDream
to generate really cool images.
So, next time, we'll talk
about unsupervised learning
Autoencoders, Variational Autoencoders
and generative adversarial networks
so that should be a fun lecture.

