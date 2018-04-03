﻿
- Okay let's get started.
Alright, so welcome to lecture 14,
and today we'll be talking
about reinforcement learning.
So some administrative details first,
update on grades.
Midterm grades were released last night,
so see Piazza for more information
and statistics about that.
And we also have A2 and milestone grades
scheduled for later this week.
Also, about your projects, all teams
must register your projects.
So on Piazza we have a form posted,
so you should go there and
this is required, every team
should go and fill out
this form with information
about your project, that
we'll use for final grading
and the poster session.
And the Tiny ImageNet
evaluation servers are also now
online for those of you who are
doing the Tiny ImageNet challenge.
We also have a link to a
course survey on Piazza
that was released a few days ago,
so, please fill it out if
you guys haven't already.
We'd love to have your
feedback and know how
we can improve this class.
Okay, so the topic of today,
reinforcement learning.
Alright, so so far we've talked
about supervised learning,
which is about a type of
problem where we have data x
and then we have labels y
and our goal is to learn
a function that is mapping from x to y.
So, for example, the
classification problem
that we've been working with.
We also talked last lecture
about unsupervised learning,
which is the problem
where we have just data
and no labels, and our goal is to learn
some underlying, hidden
structure of the data.
So, an example of this
is the generative models
that we talked about last lecture.
And so today we're going
to talk about a different
kind of problem set-up, the
reinforcement learning problem.
And so here we have an agent
that can take actions in its environment,
and it can receive rewards
for for its action.
And its goal is going to be
to learn how to take actions
in a way that can maximize its reward.
And so we'll talk about this
in a lot more detail today.
So, the outline for today,
we're going to first
talk about the reinforcement
learning problem,
and then we'll talk about
Markov decision processes,
which is a formalism of the
reinforcement learning problem,
and then we'll talk
about two major classes
of RL algorithms, Q-learning
and policy gradients.
So, in the reinforcement
learning set up, what we have
is we have an agent and
we have an environment.
And so the environment
gives the agent a state.
In turn, the agent is
going to take an action,
and then the environment is
going to give back a reward,
as well as the next state.
And so this is going to
keep going on in this loop,
on and on, until the environment
gives back a terminal state,
which then ends the episode.
So, let's see some examples of this.
First we have here the cart-pole problem,
which is a classic problem
that some of you may have seen,
in, for example, 229 before.
And so this objective
here is that you want to
balance a pole on top of a movable cart.
Alright, so the state
that you have here is
your current description of the system.
So, for example, angular, angular speed
of your pole, your
position, and the horizontal
velocity of your cart.
And the actions you can
take are horizontal forces
that you apply onto the cart, right?
So you're basically trying
to move this cart around
to try and balance this pole on top of it.
And the reward that you're getting
from this environment
is one at each time step
if your pole is upright.
So you basically want to keep this pole
balanced for as long as you can.
Okay, so here's another example
of a classic RL problem.
Here is robot locomotion.
So we have here an example
of a humanoid robot,
as well as an ant robot model.
And our objective here is to
make the robot move forward.
And so the state that we have
describing our system is
the angle and the positions
of all the joints of our robots.
And then the actions that we can take are
the torques applied onto these joints,
right, and so these are
trying to make the robot
move forward and then
the reward that we get is
our forward movement as well
as, I think, in the time of,
in the case of the humanoid,
also, you can have something
like a reward of one for
each time step that this
robot is upright.
So, games are also
a big class of problems that
can be formulated with RL.
So, for example, here we have Atari games
which are a classic success
of deep reinforcement learning
and so here the objective
is to complete these games
with the highest possible score, right.
So, your agent is basically a player
that's trying to play these games.
And the state that you have is going to be
the raw pixels of the game state.
Right, so these are just the
pixels on the screen that you would see
as you're playing the game.
And then the actions that you have
are your game controls, so for example,
in some games maybe moving
left to right, up or down.
And then the score that you
have is your score increase
or decrease at each time step,
and your goal is going to be
to maximize your total score
over the course of the game.
And, finally, here we have
another example of a game here.
It's
Go, which is
something that was a
huge achievement of deep
reinforcement learning last year,
when Deep Minds AlphaGo beats Lee Sedol,
which is one of the
best Go players of the last few years,
and this is actually in the news again
for, as some of you may have
seen, there's another Go
competition going on now with
AlphaGo versus a top-ranked Go player.
And so the objective here is to
win the game, and our
state is the position
of all the pieces, the action
is where to put the next
piece down, and the reward
is, one, if you win at the end
of the game, and zero otherwise.
And we'll also talk about this one
in a little bit more detail, later.
Okay, so
how can we mathematically formalize
the RL problem, right?
This loop that we talked about earlier,
of environments giving agents states,
and then agents taking actions.
So, a Markov decision process is
the mathematical formulation
of the RL problem,
and an MDP satisfies the Markov property,
which is that the current state completely
characterizes the state of the world.
And an MDP here is defined
by tuple of objects,
consisting of S, which is
the set of possible states.
We have A, our set of possible actions,
we also have R, our
distribution of our reward,
given a state, action pair,
so it's a function
mapping from state action
to your reward.
You also have P, which is
a transition probability
distribution over your
next state, that you're
going to transition to given
your state, action pair.
And then finally we have a
Gamma, a discount factor,
which is basically
saying how much we value
rewards coming up soon versus later on.
So, the way the Markov
Decision Process works is that
at our initial time step t equals zero,
the environment is going to sample some
initial state as zero, from
the initial state distribution,
p of s zero.
And then, once it has that,
then from time t equals zero
until it's done, we're going
to iterate through this loop
where the agent is going to
select an action, a sub t.
The environment is going to
sample a reward from here,
so reward given your state and the
action that you just took.
It's also going to sample the next state,
at time t plus one, given
your probability distribution
and then the agent is going to receive
the reward, as well as the
next state, and then we're
going to through this process again,
and keep looping; agent
will select the next action,
and so on until the episode is over.
Okay, so
now based on this, we
can define a policy pi,
which is a function from
your states to your actions
that specifies what action
to take in each state.
And this can be either
deterministic or stochastic.
And our objective now is
to going to be to find
your optimal policy pi
star, that maximizes your
cumulative discounted reward.
So we can see here we have our
some of our future
rewards, which can be also
discounted by your discount factor.
So, let's look at an
example of a simple MDP.
And here we have Grid World, which is this
task where we have this grid of states.
So you can be in any of these
cells of your grid, which are your states.
And you can take actions from your states,
and so these actions are going to be
simple movements, moving to your right,
to your left, up or down.
And you're going to get a
negative reward for each
transition or each time step,
basically, that happens.
Each movement that you take,
and this can be something
like R equals negative one.
And so your objective is going to be
to reach one of the terminal states,
which are the gray states shown here,
in the least number of actions.
Right, so the longer
that you take to reach
your terminal state, you're going to keep
accumulating these negative rewards.
Okay, so if you look at
a random policy here,
a random policy would
consist of, basically,
at any given state or cell that you're in
just sampling randomly which direction
that you're going to move in next.
Right, so all of these
have equal probability.
On the other hand, an optimal policy that
we would like to have is
basically taking the action, the direction
that will move us closest
to a terminal state.
So you can see here,
if we're right next to one of the
terminal states we should
always move in the direction
that gets us to this terminal state.
And otherwise, if you're in
one of these other states,
you want to take the
direction that will take you
closest to one of these states.
Okay, so now given this
description of our MDP, what we want to do
is we want to find our
optimal policy pi star.
Right, our policy that's
maximizing the sum of the rewards.
And so this optimal policy
is going to tell us,
given any state that we're
in, what is the action that
we should take in order
to maximize the sum
of the rewards that we'll get.
And so one question is how do we
handle the randomness in the MDP, right?
We have randomness in
terms of our initial
state that we're sampling,
in therms of this transition probability
distribution that will give us
distribution of our
next states, and so on.
Also what we'll do is we'll
work, then, with maximizing
our expected sum of the rewards.
So, formally, we can write
our optimal policy pi star
as maximizing this expected
sum of future rewards
over policy's pi, where
we have our initial state
sampled from our state distribution.
We have our actions,
sampled from our policy, given the state.
And then we have our next states sampled
from our transition
probability distributions.
Okay, so
before we talk about
exactly how we're going
to find this policy,
let's first talk about a few definitions
that's going to be helpful
for us in doing so.
So, specifically, the value function
and the Q-value function.
So, as we follow the policy,
we're going to sample trajectories
or paths, right, for every episode.
And we're going to have
our initial state as zero,
a-zero, r-zero, s-one,
a-one, r-one, and so on.
We're going to have this trajectory
of states, actions, and
rewards that we get.
And so, how good is a state
that we're currently in?
Well, the value function at any state s,
is the expected cumulative reward
following the policy from
state s, from here on out.
Right, so it's going to be expected value
of our expected cumulative reward,
starting from our current state.
And then how good is a state, action pair?
So how good is taking action a in state s?
And we define this using
a Q-value function,
which is, the expected
cumulative reward from taking
action a in state s and
then following the policy.
Right, so then, the
optimal Q-value function
that we can get is going to be
Q star, which is the maximum
expected cumulative reward that we can get
from a given state action
pair, defined here.
So now we're going to
see one important thing
in reinforcement learning,
which is called the Bellman equation.
So let's consider this a Q-value function
from the optimal policy Q star,
which is then going to
satisfy this Bellman equation,
which is this identity shown here,
and what this means is that
given any state, action pair, s and a,
the value of this pair
is going to be the reward
that you're going to get, r,
plus the value of whatever
state that you end up in.
So, let's say, s prime.
And since we know that we
have the optimal policy,
then we also know that we're going to
play the best action that we can,
right, at our state s prime.
And so then, the value at state s prime
is just going to be the
maximum over our actions,
a prime, of Q star at s prime, a prime.
And so then we get this
identity here, for optimal Q-value.
Right, and then also, as always, we have
this expectation here,
because we have randomness over what state
that we're going to end up in.
And then we can also
infer, from here, that our
optimal policy, right, is going to consist
of taking the best action in any state,
as specified by Q star.
Q star is going to tell us
of the maximum
future reward that we can
get from any of our actions,
so we should just
take a policy that's following this
and just taking the action that's
going to lead to best reward.
Okay, so how can we solve
for this optimal policy?
So, one way we can solve for this is
something called a value
iteration algorithm,
where we're going to use
this Bellman equation
as an iterative update.
So at each step, we're going
to refine our approximation
of Q star by trying to
enforce the Bellman equation.
And so, under some
mathematical conditions,
we also know that this sequence Q, i
of our Q-function is going
to converge to our optimal
Q star as i approaches infinity.
And so this, this works well,
but what's the problem with this?
Well, an important problem
is that this is not scalable.
Right?
We have to compute
Q of s, a here for
every state, action pair
in order to make our iterative updates.
Right, but then this is a problem if,
for example, if we look at these
the state of, for example, an Atari game
that we had earlier, it's going to be
your screen of pixels.
And this is a huge state
space, and it's basically
computationally infeasible
to compute this for
the entire state space.
Okay, so what's the solution to this?
Well, we can use a function approximator
to estimate Q of s, a
so, for example, a neural network, right.
So, we've seen before that
any time, if we have some
really complex function that
don't know, that we want
to estimate, a neural network is
a good way to estimate this.
Okay, so this is going to take us to our
formulation of Q-learning
that we're going to look at.
And so, what we're going
to do is we're going
to use a function approximator
in order to estimate our
action value function.
Right?
And if this function approximator
is a deep neural network, which is
what's been used recently,
then this is going to be
called deep Q-learning.
And so this is something that
you'll hear around as one
of the common approaches
to deep reinforcement
learning that's in use.
Right, and so in this case,
we also have our function parameters
theta here, so our Q-value function
is determined by these weights,
theta, of our neural network.
Okay, so given this
function approximation,
how do we solve for our optimal policy?
So remember that we want to find
a Q-function that's satisfying
the Bellman equation.
Right, and so we want to
enforce this Bellman equation
to happen, so what we
can do when we have this
neural network approximating
our Q-function is that
we can train this where our loss function
is going to try and minimize
the error of our Bellman equation, right?
Or how far q of s, a is from its target,
which is the Y_i here,
the right hand side
of the Bellman equation
that we saw earlier.
So, we're basically going to take these
forward passes of our
loss function, trying
to minimize this error
and then our backward
pass, our gradient update,
is just going to be
you just take the gradient of this
loss, with respect to our
network parameter's theta.
Right, and so our goal is again to
have this effect as we're
taking gradient steps
of iteratively trying
to make our Q-function
closer to our target value.
So, any questions about this?
Okay.
So let's look at a case
study of an example where
one of the classic examples
of deep reinforcement learning
where this approach was applied.
And so we're going to look at
this problem that we saw earlier
of playing Atari games,
where our objective was
to complete the game
with the highest score
and remember our state is
going to be the raw pixel
inputs of the game state,
and we can take these actions
of moving left, right, up, down,
or whatever actions of
the particular game.
And our reward at each time
step, we're going to get
a reward of our score
increase or decrease that we
got at this time step, and
so our cumulative total
reward is this total reward
that we'll usually see
at the top of the screen.
Okay, so the network that
we're going to use for our
Q-function is going to
look something like this,
right, where we have our
Q-network, with weight's theta.
And then our input, our
state s, is going to be
our current game screen.
And in practice we're going to take
a stack of the last four
frames, so we have some history.
And so we'll take these raw pixel values,
we'll do some, you know, RGB
to gray-scale conversions,
some down-sampling, some cropping,
so, some pre-processing.
And what we'll get out of
this is this 84 by 84 by four
stack of the last four frames.
Yeah, question.
[inaudible question from audience]
Okay, so the question
is, are we saying here
that our network is
going to approximate our
Q-value function for
different state, action pairs,
for example, four of these?
Yeah, that's correct.
We'll see,
we'll talk about that in a few slides.
[inaudible question from audience]
So, no.
So, we don't have a Softmax
layer after the connected,
because here our goal
is to directly predict
our Q-value functions.
[inaudible question from audience]
Q-values.
[inaudible question from audience]
Yes, so it's more doing
regression to our Q-values.
Okay, so we have our input to this network
and then on top of this,
we're going to have
a couple of familiar convolutional layers,
and a fully-connected layer,
so here we have
an eight-by-eight
convolutions and we have some
four-by-four convolutions.
Then we have a FC 256 layer,
so this is just a standard kind of networK
that you've seen before.
And then, finally, our last
fully-connected layer has
a vector of outputs, which
is corresponding to your
Q-value for each action, right, given
the state that you've input.
And so, for example, if
you have four actions,
then here we have this
four-dimensional output
corresponding to Q of
current s, as well as a-one,
and then a-two, a-three, and a-four.
Right so this is going
to be one scalar value
for each of our actions.
And then the number of
actions that we have
can vary between,
for example, 4 to 18,
depending on the Atari game.
And one nice thing here is that
using this network structure,
a single feedforward
pass is able to compute
the Q-values for all functions
from the current state.
And so this is really efficient.
Right, so basically we
take our current state
in and then because we have
this output of an action
for each, or Q-value for each
action, as our output layer,
we're able to do one pass and
get all of these values out.
And then in order to train this,
we're just going to use our
loss function from before.
Remember, we're trying to
enforce this Bellman equation
and so, on our forward
pass, our loss function
we're going to try and
iteratively make our Q-value
close to our target value,
that it should have.
And then our backward pass is just
directly taking the gradient of this
loss function that we have and then taking
a gradient step based on that.
So one other thing that's used
here that I want to mention
is something called experience replay.
And so this addresses a
problem with just using
the plain two network
that I just described,
which is that learning from batches
of consecutive samples is bad.
And so the reason
because of this, right, is so for just
playing the game, taking samples
of state action rewards that we have
and just taking consecutive
samples of these
and training with these,
well all of these samples are correlated
and so this leads to
inefficient learning, first of all,
and also, because of this,
our current Q-network
parameters, right, this
determines the policy
that we're going to follow,
it determines our next
samples that we're going to get that
we're going to use for training.
And so this leads to problems where
you can have bad feedback loops.
So, for example, if
currently the maximizing
action that's going to take left,
well this is going to bias all of my
upcoming training examples to be dominated
by samples from the left-hand side.
And so this is a problem, right?
And so the way that we
are going to address these
problems is by using something called
experience replay, where
we're going to keep this
replay memory table of
transitions of state,
as state, action, reward, next state,
transitions that we have, and we're going
to continuously update this
table with new transitions
that we're getting as
game episodes are played,
as we're getting more experience.
Right, and so now what we can do
is that we can now train
our Q-network on random,
mini-batches of transitions
from the replay memory.
Right, so instead of
using consecutive samples,
we're now going to sample across these
transitions that we've accumulated
random samples of these,
and this breaks all of the,
these correlation problems
that we had earlier.
And then also, as another
side benefit is that
each of these transitions
can also contribute to potentially
multiple weight updates.
We're just sampling from this table and so
we could sample one multiple times.
And so, this is going to lead
also to greater data efficiency.
Okay, so let's put this all together
and let's look at the full algorithm
for deep Q-learning
with experience replay.
So we're going to start off with
initializing our replay memory
to some capacity that we
choose, N, and then we're also
going to initialize our
Q-network, just with our random weights
or initial weights.
And then we're going to play
M episodes, or full games.
This is going to be our training episodes.
And then what we're going to do
is we're going to initialize our state,
using the starting game screen pixels
at the beginning of each episode.
And remember, we go through
the pre-processing step
to get to our actual input state.
And then for each time step
of a game that we're currently playing,
we're going to, with a small probability,
select a random action,
so one thing that's
important in these algorithms
is to have sufficient exploration,
so we want to make sure that
we are sampling different
parts of the state space.
And then otherwise, we're going
to select from the greedy action
from the current policy.
Right, so most of the time
we'll take the greedy action
that we think is
a good policy of the type of
actions that we want to take
and states that we want to see,
and with a small probability
we'll sample something random.
Okay, so then we'll take this action,
a, t, and we'll observe the
next reward and the next state.
So r, t and s, t plus one.
And then we'll take this and
we'll store this transition
in our replay memory
that we're building up.
And then we're going to take,
we're going to train a
network a little bit.
So we're going to do experience replay
and we'll take a sample
of a random mini-batches
of transitions that we have
from the replay memory,
and then we'll perform
a gradient descent step on this.
Right, so this is going to
be our full training loop.
We're going to be
continuously playing this game
and then also sampling
minibatches, using
experienced replay to update
our weights of our Q-network and then
continuing in this fashion.
Okay, so let's see.
Let's see if I can,
is this playing?
Okay, so let's take a look
at this deep Q-learning algorithm
from Google DeepMind, trained
on an Atari game of Breakout.
Alright, so it's saying
here that our input
is just going to be our
state are raw game pixels.
And so here we're looking
at what's happening
at the beginning of training.
So we've just started training a bit.
And
right, so it's going to look to
it's learned to kind of hit the ball,
but it's not doing a very
good job of sustaining it.
But it is looking for the ball.
Okay, so now after some more training,
it looks like a couple hours.
Okay, so now it's learning
to do a pretty good job here.
So it's able to continuously follow
this ball and be able to
to remove most of the blocks.
Right, so after 240 minutes.
Okay, so here it's found
the pro strategy, right?
You want to get all the
way to the top and then
have it go by itself.
Okay, so
this is an example of using
deep Q-learning in order to
train an agent to be
able to play Atari games.
It's able to do this on many Atari games
and so you can check out
some more of this online.
Okay, so we've talked about Q-learning.
But there is a problem
with Q-learning, right?
It can be challenging
and what's the problem?
Well, the problem can be that
the Q-function is very complicated.
Right, so we have to, we're
saying that we want to learn
the value of every state action pair.
So, if, let's say you have
something, for example,
a robot grasping, wanting
to grasp an object.
Right, you're going to have a
really high dimensional state.
You have, I mean, let's
say you have all of your
even just joint, joint
positions, and angles.
Right, and so learning the
exact value of every state
action pair that you have, right,
can be really, really hard to do.
But on the other hand, your
policy can be much simpler.
Right, like what you want this robot to do
maybe just to have this simple motion
of just closing your hand, right?
Just, move your fingers in this
particular direction and keep going.
And so, that leads to the question of
can we just learn this policy directly?
Right, is it possible,
maybe, to just find the best
policy from a collection of policies,
without trying to go through this process
of estimating your Q-value
and then using that to infer your policy.
So, this is an approach that
oh,
so, okay, this is an approach that
we're going to call policy gradients.
And so, formally, let's define a
class of parametrized policies.
Parametrized by weights theta,
and so for each policy
let's define the value of the policy.
So, J, our value J,
given parameters theta,
is going to be, or expected
some cumulative sum of future
rewards that we care about.
So, the same reward that we've been using.
And so our goal then, under this setup
is that we want to find an optimal policy,
theta star, which is the maximum, right,
arg max over theta of J of theta.
So we want to find the
policy, the policy parameters
that gives our best expected reward.
So, how can we do this?
Any ideas?
Okay, well, what we can do
is just a gradient assent on
our policy parameters, right?
We've learned that given
some objective that we have,
some parameters we can
just use gradient asscent
and gradient assent in order
to continuously improve our parameters.
And so let's talk more
specifically about how
we can do this, which we're going to call
here the reinforce algorithm.
So, mathematically, we can write
out our expected future reward
over trajectories, and
so we're going to sample
these trajectories of experience, right,
like for example episodes of game play
that we talked about earlier.
S-zero, a-zero, r-zero, s-one,
a-one, r-one, and so on.
Using some policy pi of theta.
Right, and then so, for each trajectory
we can compute a reward
for that trajectory.
It's the cumulative reward that we
got from following this trajectory.
And then the value of a policy,
pi sub theta, is going
to be just the expected
reward of these
trajectories that we can get
from the following pi sub theta.
So that's here, this
expectation over trajectories
that we can get, sampling
trajectories from our policy.
Okay.
So, we want to do gradient ascent, right?
So let's differentiate this.
Once we differentiate
this, then we can just take
gradient steps, like normal.
So, the problem is that
now if we try and just
differentiate this exactly,
this is intractable, right?
So, the gradient of an
expectation is problematic
when p is dependent on
theta here, because here
we want to take this gradient
of p of tau, given theta,
but this is going to be,
we want to take this integral over tau.
Right, so this is intractable.
However, we can use a trick
here to get around this.
And this trick is taking this
gradient that we want, of p.
We can rewrite this
by just multiplying this by one,
by multiplying top and bottom,
both by p of tau given theta.
Right, and then if we look at these terms
that we have now here, in the
way that I've written this,
on the left and the right, this is
actually going to be equivalent to
p of tau times our gradient
with respect to theta, of log, of p.
Right, because the gradient
of the log of p is just going
to be one over p times gradient of p.
Okay, so if we then inject this back
into our expression that we
had earlier for this gradient,
we can see that, what this
will actually look like,
right, because now we
have a gradient of log p
times our probabilities of
all of these trajectories
and then taking this
integral here, over tau.
This is now going to be an expectation
over our trajectories tau,
and so what we've done here
is that we've taken a
gradient of an expectation
and we've transformed it into
an expectation of gradients.
Right, and so now we can use
sample trajectories that we can get
in order to estimate our gradient.
And so we do this using
Monte Carlo sampling,
and this is one of the
core ideas of reinforce.
Okay, so looking at this
expression that we want to compute,
can we compute these
quantities that we had here
without knowing the
transition probabilities?
Alright, so we have that
p of tau is going to be
the probability of a trajectory.
It's going to be the product of
all of our transition
probabilities of the next state
that we get, given our
current state and action
as well as our probability
of the actions that
we've taken under our policy pi.
Right, so we're going to
multiply all of these together,
and get our probability of our trajectory.
So this log of p of tau
that we want to compute
is going to be we just
take this log and this will
separate this out into a sum
of pushing the logs inside.
And then here, when we differentiate this,
we can see we want to
differentiate with respect
to theta, but this first
term that we have here,
log p of the state
transition probabilities
there's no theta term here, and so
the only place where we have
theta is the second term
that we have, of log of pi sub theta,
of our action, given our
state, and so this is the only
term that we keep
in our gradient estimate,
and so we can see here that
this doesn't depend on the
transition probabilities,
right, so we actually don't need to know
our transition probabilities
in order to computer
our gradient estimate.
And then, so, therefore
when we're sampling these,
for any given trajectory tau,
we can estimate J of theta
using this gradient estimate.
This is here shown for a single trajectory
from what we had earlier,
and then we can also sample
over multiple trajectories
to get the expectation.
Okay, so given this gradient
estimator that we've derived,
the interpretation that we can
make from this here, is that
if our reward for a trajectory
is high, if the reward that
we got from taking the
sequence of actions was good,
then let's push up the
probabilities of all
the actions that we've seen.
Right, we're just going to say that
these were good actions that we took.
And then if the reward is low,
we want to push down these probabilities.
We want to say these were bad actions,
let's try and not sample these so much.
Right and so we can see
that's what's happening here,
where we have pi of a, given s.
This is the likelihood of
the actions that we've taken
and then we're going to scale
this, we're going to take the
gradient and the gradient
is going to tell us how much
should we change the
parameters in order to increase
our likelihood of our action, a, right?
And then we're going to
take this and scale it by
how much reward we actually got from it,
so how good were these
actions, in reality.
Okay, so
this might seem simplistic to say that,
you know, if a trajectory
is good, then we're saying
here that all of its actions were good.
Right?
But, in expectation, this
actually averages out.
So we have an unbiased estimator here,
and so if you have many samples of this,
then we will get an accurate
estimate of our gradient.
And this is nice because we can just take
gradient steps and we know
that we're going to be
improving our loss
function and getting closer
to, at least some local optimum of our
policy parameters theta.
Alright, but there is a problem with this,
and the problem is that this also suffers
from high variance.
Because this credit
assignment is really hard.
Right, we're saying that
given a reward that we
got, we're going to say
all of the actions were good,
we're just going to hope
that this assignment of
which actions were actually
the best actions, that mattered,
are going to average out over time.
And so this is really hard
and we need a lot of samples
in order to have a good estimate.
Alright, so this leads to the
question of, is there anything
that we can do to reduce the variance
and improve the estimator?
And so variance reduction is
an important area of research
in policy gradients,
and in coming up with
ways in order to improve
the estimator and require fewer samples.
Alright, so let's look
at a couple of ideas
of how we can do this.
So given our gradient estimator,
so the first idea is that we can
push up the probabilities of an action
only by it's affect on future rewards
from that state, right?
So, now with instead of scaling
this likelihood, or
pushing up this likelihood
of this action by the total
reward of its trajectory,
let's look more
specifically at just the sum
of rewards coming from this time step
on to the end, right?
And so, this is basically saying that
how good an action is, is
only specified by how much
future reward it generates.
Which makes sense.
Okay, so a second idea
that we can also use
is using a discount factor in order
to ignore delayed effects.
Alright so here we've added
back in this discount factor,
that we've seen before,
which is saying that
we are, you know, our discount
factor's going to tell us
how much we care about just the
rewards that are coming up soon,
versus rewards that came much later on.
Right, so we were going to now
say how good or bad an action is,
looking more at the local neighborhood
of action set it generates
in the immediate near future
and down weighting the the
ones that come later on.
Okay so
these are some straightforward ideas
that are generally used in practice.
So, a third idea is this idea of using
a baseline in order to
reduce your variance.
And so, a problem with
just using the raw value
of your trajectories, is that
this isn't necessarily meaningful, right?
So, for example, if your
rewards are all positive,
then you're just going to keep pushing
up the probabilities of all your actions.
And of course, you'll push
them up to various degrees,
but what's really important
is whether a reward is better
or worse than what you're
expecting to be getting.
Alright, so in order to
address this, we can introduce
a baseline function that's
dependent on the state.
Right, so this baseline function tell us
what's, how much we, what's
our guess and what we expect
to get from this state, and then
our reward or our scaling
factor that we're going to use
to be pushing up or
down our probabilities,
can now just be our expected
sum of future rewards,
minus this baseline, so now
it's the relative of how
much better or worse is
the reward that we got
from what we expected.
And so how can we choose this baseline?
Well,
a very simple baseline, the
most simple you can use,
is just taking a moving average
of rewards that you've experienced so far.
So you can even do this
overall trajectories,
and this is just an
average of what rewards
have I been seeing as I've been training,
and as I've been playing these episodes?
Right, and so this gives
some idea of whether the
reward that I currently get
was relatively better or worse.
And so there's some variance
on this that you can use
but so far the variance
reductions that we've seen so far
are all used in what's typically
called &quot;vanilla REINFORCE&quot; algorithm.
Right, so looking at the
cumulative future reward,
having a discount factor,
and some simple baselines.
Now let's talk about how we can
think about this idea of baseline
and potentially choose better baselines.
Right, so if we're going to
think about what's a better
baseline that we can choose,
what we want to do is we want
to push up the probability
of an action from a state,
if the action was better than
the expected value of what we
should get from that state.
So, thinking about the value
of what we're going to expect
from the state, what
does this remind you of?
Does this remind you of anything
that we talked about
earlier in this lecture?
Yes.
[inaudible from audience]
Yeah, so the value functions, right?
The value functions that we
talked about with Q-learning.
So, exactly.
So Q-functions and value functions
and so, the intuition is that
well,
we're happy with an action,
taking an action in a state s, if
our Q-value of taking
a specific action from
this state is larger than
the value function or expected value
of the cumulative future reward
that we can get from this state.
Right, so this means that
this action was better than
other actions that we could've taken.
And on the contrary, we're
unhappy if this action,
if this value or this
difference is negative or small.
Right, so now if we plug
this in, in order to,
as our scaling factor of how much we want
to push up or down, our
probabilities of our actions,
then we can get this estimator here.
Right, so, it's going to be
exactly the same as before, but now where
we've had before our
cumulative expected reward,
with our various reduction,
variance reduction
techniques and baselines in,
here we can just plug in now
this difference of how much better our
current action was,
based on our Q-function
minus our value function from that state.
Right, but what we talked
about so far with our
REINFORCE algorithm, we don't know
what Q and V actually are.
So can we learn these?
And the answer is yes, using Q-learning.
What we've already talked about before.
So we can combine policy gradients
while we've just been talking
about, with Q-learning,
by training both an actor,
which is the policy,
as well as a critic, right, a Q-function,
which is going to tell us
how good we think a state is,
and an action in a state.
Right, so using this in approach,
an actor is going to
decide which action to take
and then the critic, or
Q-function, is going to tell
the actor how good its action
was and how it should adjust.
And so, and this also alleviates
a little bit of the task
of this critic compared
to the Q-learning problems
that we talked about earlier
of having to have this
learning a Q-value for
every state, action pair,
because here it only has to learn this
for the state-action pairs that
are generated by the policy.
It only needs to know this
where it matters for
computing this scaling factor.
Right, and then we can also,
as we're learning this,
incorporate all of the
Q-learning tricks that we saw
earlier, such as experience replay.
And so, now I'm also going to just
define this term that we saw earlier,
Q of s of a, how much,
how good was an action
in a given state, minus V of s?
Our expected value of
how good the state is
by this term advantage function.
Right, so the advantage
function is how much advantage
did we get from playing this action?
How much better the
action was than expected.
So, using this, we can
put together our full
actor-critic algorithm.
And so what this looks like,
is that we're going to start
off with by initializing
our policy parameters theta,
and our critic parameters
that we'll call phi.
And then for each, for
iterations of training,
we're going to sample M trajectories,
under the current policy.
Right, we're going to play
our policy and get these
trajectories as s-zero, a-zero,
r-zero, s-one and so on.
Okay, and then we're going to compute
the gradients that we want.
Right, so for each of these trajectories
and in each time step, we're going
to compute this advantage function,
and then we're going to
use this advantage function, right?
And then we're going to use
that in our gradient estimator
that we showed earlier, and accumulate our
gradient estimate that we have for here.
And then we're also going to train our
critic parameters phi
by exactly the same way,
so as we saw earlier,
basically trying to enforce
this value function, right,
to learn our value function,
which is going to be pulled
into, just minimizing
this advantage function and this will
encourage it to be closer
to this Bellman equation
that we saw earlier, right?
And so, this is basically
just iterating between
learning and optimizing
our policy function,
as well as our critic function.
And so then we're going to update the
gradients and then we're
going to go through and just
continuously repeat this process.
Okay, so now let's look at
some examples of REINFORCE
in action, and let's look
first here at something called
the Recurrent Attention Model,
which is something that,
which is a model also
referred to as hard attention,
but you'll see a lot in,
recently, in computer vision
tasks for various purposes.
Right, and so the idea behind this is
here, I've talked about the
original work on hard attention,
which is on image
classification, and your goal is
to still predict the image class,
but now you're going to do
this by taking a sequence
of glimpses around the image.
You're going to look at local
regions around the image
and you're basically going
to selectively focus on these
parts and build up information
as you're looking around.
Right, and so the reason
that we want to do this
is, well, first of all it
has some nice inspiration
from human perception in eye movement.
Let's say we're looking at a complex image
and we want to determine
what's in the image.
Well, you know, we might,
maybe look at a low-resolution
of it first, and then
look specifically at parts
of the image that will give us clues about
what's in this image.
And then,
this approach of just looking
at, looking around at an image
at local regions, is also
going to help you save
computational resources, right?
You don't need to process the full image.
In practice, what usually
happens is you look at a
low-resolution image
first, of a full image,
to decide how to get started,
and then you look at high-res
portions of the image after that.
And so this saves a lot
of computational resources
and you can think about,
then, benefits of this
to scalability, right,
being able to, let's say
process larger images more efficiently.
And then, finally, this
could also actually help
with actual classification performance,
because now you're able to
ignore clutter and irrelevant
parts of the image.
Right?
Like, you know, instead
of always putting through
your ConvNet, all the parts of your image,
you can use this to, maybe,
first prune out what are the
relevant parts that I
actually want to process,
using my ConvNet.
Okay, so what's the reinforcement learning
formulation of this problem?
Well, our state is going to be
the glimpses that we've
seen so far, right?
Our
what's the information that we've seen?
Our action is then going to be where
to look next in the image.
Right, so in practice,
this can be something like
the x, y-coordinates,
maybe centered around some
fixed-sized glimpse that
you want to look at next.
And then the reward for
the classification problem
is going to be one, at
the final time step,
if our image is correctly
classified, and zero otherwise.
And so, because this
glimpsing, taking these
glimpses around the image
is a non-differentiable operation,
this is why we need to use
reinforcement learning formulation,
and learn policies for how
to take these glimpse actions
and we can train this using REINFORCE.
So, given the state of glimpses so far,
the core of our model is going to be
this RNN that we're going
to use to model the state,
and then we're going to
use our policy parameters
in order to output the next action.
Okay, so what this model looks
like is we're going to take
an input image.
Right, and then we're going to
take a glimpse at this image.
So here, this glimpse is the red box here,
and this is all blank, zeroes.
And so we'll pass what
we see so far into some
neural network, and this can be any
kind of network depending on your task.
In the original experiments
that I'm showing here,
on MNIST, this is very
simple, so you can just
use a couple of small,
fully-connected layers,
but you can imagine
for more complex images
and other tasks you may want
to use fancier ConvNets.
Right, so you've passed this
into some neural network,
and then, remember I said
we're also going to be
integrating our state of,
glimpses that we've seen
so far, using a recurrent network.
So, I'm just going to
we'll see that later on, but
this is going to go through that,
and then it's going to output my
x, y-coordinates, of where
I'm going to see next.
And in practice, this is going to be
We want to output a
distribution over actions,
right, and so, what this is
going to be it's going to be
a gaussian distribution and
we're going to output the mean.
You can also output a mean and variance
of this distribution in practice.
The variance can also be fixed.
Okay, so we're going to take this
action that we're now going to sample
a specific x, y location
from our action distribution
and then we're going to put
this in to get the next,
extract the next glimpse from our image.
Right, so here we've moved
to the end of the two,
this tail part of the two.
And so now we're actually
starting to get some signal
of what we want to see, right?
Like, what we want to do is we
want to look at the relevant
parts of the image that are
useful for classification.
So we pass this through, again,
our neural network layers,
and then also through
our recurrent network, right,
that's taking this input
as well as this previous hidden
state, and we're going to
use this to get a,
so this is representing our policy,
and then we're going to use this to output
our distribution for the next
location that we want to glimpse at.
So we can continue doing this,
you can see in this next glimpse here,
we've moved a little bit more
toward the center of the two.
Alright, so it's probably learning that,
you know, once I've seen
this tail part of the two,
that looks like this,
maybe moving in this upper
left-hand direction will
get you more towards
a center, which will also have a value,
valuable information.
And then we can keep doing this.
And then finally, at the
end, at our last time step,
so we can have a fixed
number of time steps here,
in practice something like six or eight.
And then at the final time
step, since we want to do
classification, we'll have our standard
Softmax layer that will produce a
distribution of
probabilities for each class.
And then here the max class was a two,
so we can predict that this was a two.
Right, and so this is going
to be the set up of our
model and our policy, and then we have our
estimate for the gradient
of this policy that we've
said earlier we could compute by taking
trajectories from here
and using those to do back prop.
And so we can just do this
in order to train this model
and learn the parameters
of our policy, right?
All of the weights that you can see here.
Okay, so
here's an example of a
policies trained on MNIST,
and so you can see that, in general,
from wherever it's
starting, usually learns
to go closer to where the digit is,
and then looking at the relevant
parts of the digit, right?
So this is pretty cool and
this
you know, follows kind of
what you would expect, right,
if you were to
choose places to look next
in order to most efficiently determine
what digit this is.
Right, and so this idea of hard attention,
of recurrent attention
models, has also been used
in a lot of tasks in
computer vision in the last
couple of years, so you'll
see this, used, for example,
fine-grained image recognition.
So, I mentioned earlier that
one of the useful benefits of this
can be also to
both save on computational efficiency
as well as to ignore
clutter and irrelevant
parts of the image, and
when you have fine-grained
image classification problems,
you usually want both of these.
You want to keep high-resolution,
so that you can look
at, you know, important differences.
And then you also want to
focus on these differences
and ignore irrelevant parts.
Yeah, question.
[inaudible question from audience]
Okay, so yeah, so the question is
how is there is
computational efficiency,
because we also have this
recurrent neural network in place.
So that's true, it depends
on exactly what's your,
what is your problem, what
is your network, and so on,
but you can imagine that
if you had some really
hi- resolution image
and you don't want to process
the entire parts of this
image with some huge ConvNet
or some huge, you know,
network, now you can
get some savings by just
focusing on specific
smaller parts of the image.
You only process those parts of the image.
But, you're right, that
it depends on exactly
what problem set-up you have.
This has also been used
in image captioning,
so if we're going to produce
an caption for an image,
we can choose, you know,
we can have the image
use this attention model
to generate this caption
and what it usually ends up
learning is these policies
where it'll focus on
specific parts of the image,
in sequence, and as it
focuses on each part,
it'll generate some words
or the part of the caption
referring to that part of the image.
And then it's also been used,
also tasks such as visual
question answering,
where we ask a question about the image
and you want the model
to output some answer
to your question, for
example, I don't know,
how many chairs are around the table?
And so you can see how
this attention mechanism
might be a good type of model
for learning how to
answer these questions.
Okay, so that was an
example of policy gradients
in these hard attention models.
And so, now I'm going to
talk about one more example,
that also uses policy gradients,
which is learning how to play Go.
Right, so DeepMind had this agent
for playing Go, called AlphGo,
that's been in the news a lot
in the past, last year and this year.
So, sorry?
[inaudible comment from audience]
And yesterday, yes, that's correct.
So this is very exciting,
recent news as well.
So last year,
a first version of AlphaGo
was put into a
competition against one
of the best Go players
of recent years, Lee Sedol, and the agent
was able to beat him
four to one, in a game of five matches.
And actually, right now, just
there's another match with
Ke Jie, which is current
world number one, and
so it's best of three
in China right now.
And so the first game was yesterday.
AlphaGo won.
I think it was by just
half a point, and so,
so there's two more games to watch.
These are all live-stream, so
you guys, should also go
online and watch these games.
It's pretty interesting
to hear the commentary.
But, so what is this AlphaGo
agent, right, from DeepMind?
And it's based on a lot
of what we've talked
about so far in this lecture.
And what it is it's a mixed
of supervised learning
and reinforcement learning,
as well as a mix of some older
methods for Go, Monte Carlo Tree Search,
as well as recent deep RL approaches.
So, okay, so how does AlphaGo
beat the Go world champion?
Well, what it first does is
to train AlphaGo, what it
takes as input is going to be
a few featurization of the board.
So it's basically, right,
your board and the positions
of the pieces on the board.
That's your natural state representation.
And what they do in order
to improve performance
a little bit is that
they featurize this into
some
more channels of one is all
the different stone colors,
so this is kind of like your
configuration of your board.
Also some channels, for
example, where, which moves
are legal, some bias
channels, some various things
and then, given this state, right,
it's going to first
train a network
that's initialized with
supervised training
from professional Go games.
So, given the current board configuration
or features, featurization of this,
what's the correct next action to take?
Alright, so given
examples of professional games played,
you know, just collected over time,
we can just take all of
these professional Go moves,
train a standard, supervised mapping,
from board state to action to take.
Alright, so they take this,
which is a pretty good start,
and then they're going
to use this to initialize
a policy network.
Right, so policy network,
it's just going to take
the exact same structure of input is your
board state and your output is the
actions that you're going to take.
And this was the set-up
for the policy gradients
that we just saw, right?
So now we're going to just
continue training this
using policy gradients.
And it's going to do this
reinforcement learning training
by playing against itself for
random, previous iterations.
So self play, and the
reward it's going to get
is one, if it wins, and a
negative one if it loses.
And what we're also going to
do is we're also going to learn
a value network, so,
something like a critic.
And then, the final AlphaGo
is going to be combining
all of these together, so
policy and value networks
as well as with
a Monte Carlo Tree Search
algorithm, in order to select
actions by look ahead search.
Right, so after putting all this together,
a value of a node, of
where you are in play,
and what you do next, is
going to be a combination
of your value function, as well as
roll at outcome that you're
computing from standard
Monte Carlo Tree Search roll outs.
Okay, so, yeah, so this is basically
the various, the components of AlphaGo.
If you're interested in
reading more about this,
there's a nature paper about this in 2016,
and they trained this, I think, over,
the version of AlphaGo
that's being used in these
matches is, like, I think
a couple thousand CPUs
plus a couple hundred GPUs,
putting all of this together,
so it's a huge amount of
training that's going on, right.
And yeah, so you guys should,
follow the game this week.
It's pretty exciting.
Okay, so in summary,
today we've talked about
policy gradients, right,
which are general.
They, you're just directly
taking gradient descent or
ascent on your policy parameters,
so this works well for a
large class of problems,
but it also suffers from high variance,
so it requires a lot of samples,
and your challenge here
is sample efficiency.
We also talked about
Q-learning, which doesn't always
work, it's harder to
sometimes get it to work
because of this problem
that we talked earlier where
you are trying to compute this
exact state, action value
for many, for very high
dimensions, but when it does work,
for problems, for example,
the Atari we saw earlier,
then it's usually more sample efficient
than policy gradients.
Right, and one of the
challenges in Q-learning is that
you want to make sure that you're
doing sufficient exploration.
Yeah?
[inaudible question from audience]
Oh, so for Q-learning can
you do this process where
you're, okay, where you're
trying to start this off by
some supervised training?
So, I guess the direct
approach for Q-learning doesn't
do that because you're
trying to regress to these
Q-values, right, instead of
policy gradients over this
distribution, but I think there
are ways in which you can,
like, massage this
type of thing to also bootstrap.
Because I think bootstrapping
in general or like
behavior cloning is a good way to
warm start these policies.
Okay, so, right, so we've
talked about policy gradients
and Q-learning, and just
another look at some of these,
some of the guarantees that you have,
right, with policy gradients.
One thing we do know
that's really nice is that
this will always converge to
a local minimum of J of theta,
because we're just directly
doing gradient ascent,
and so this is often,
and this local minimum is
often just pretty good, right.
And in Q-learning, on the
other hand, we don't have any
guarantees because here
we're trying to approximate
this Bellman equation with
a complicated function
approximator and so, in this
case, this is the problem
with Q-learning being a
little bit trickier to train
in terms of applicability
to a wide range of problems.
Alright, so today you got basically very,
brief, kind of high-level
overview of reinforcement learning
and some major classes
of algorithms in RL.
And next time we're going to have a
guest lecturer from, Song
Han, who's done a lot
of pioneering work in model compression
and energy efficient deep learning,
and so he's going to talk some
of this, about some of this.
Thank you.
