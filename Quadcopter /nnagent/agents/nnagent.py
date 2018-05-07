

from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras
from math import *
import keras.backend as K
from agents.rpm import rpm
import numpy as np

def resdense(features):
    def unit(i):
        hfeatures = max(4,int(features/4))

        ident = i
        i = Dense(features,activation='tanh')(i)

        ident = Dense(hfeatures)(ident)
        ident = Dense(features)(ident)

        return add([ident,i])
    return unit


class nnagent(object):
    def __init__(self,
    	task,
    	discount_factor,
    	optimizer
    ):
        self.rpm = rpm(1000000) # 1M history
        

        self.inputdims = task.state_size
        # assume observation_space is continuous

        # if isinstance(action_space,Box): # if action space is continuous

        low = task.action_low
        high = task.action_high

        num_of_actions = task.action_size

        self.action_bias = (high+low)/2.
        self.action_multiplier = high - self.action_bias

        # say high,low -> [2,7], then bias -> 4.5
        # mult = 2.5. then [-1,1] multiplies 2.5 + bias 4.5 -> [2,7]

        self.is_continuous = True

        def clamper(env,actions):
            return np.clip(actions,a_max=env.action_high,a_min=env.action_low)

        self.clamper = clamper
        # else:
        #     num_of_actions = action_space.n

        #     self.action_bias = .5
        #     self.action_multiplier = .5 # map (-1,1) into (0,1)

        #     self.is_continuous = False

        self.outputdims = num_of_actions

        self.discount_factor = discount_factor
        self.optimizer = optimizer

        ids,ods = self.inputdims,self.outputdims
        self.actor = self.create_actor_network(ids,ods)
        self.critic, self.frozen_critic = self.create_critic_network(ids,ods)

        # print('inputdims:{}, outputdims:{}'.format(ids,ods))
        # print('actor network:')
        self.actor.summary()
        # print('critic network:')
        self.critic.summary()

        # target networks: identical copies of actor and critic
        self.actor_target = self.create_actor_network(ids,ods)
        self.critic_target, self.frozen_critic_target = self.create_critic_network(ids,ods)

        self.replace_weights(tau=1.)

        # now the dirty part: the actor trainer --------------------------------

        # explaination of this part is written in the train() method

        s_given = Input(shape=(self.inputdims,))
        a1_maybe = self.actor(s_given)
        q1_maybe = self.frozen_critic([s_given,a1_maybe])
        # frozen weight version of critic. so we can train only the actor

        actor_trainer = Model(inputs=s_given,outputs=q1_maybe)

        # use negative of q1_maybe as loss (so we can maximize q by minimizing the loss)
        def neg_q1(y_true,y_pred):
            return - y_pred # neat!

        actor_trainer.compile(optimizer=self.optimizer,loss=neg_q1)
        self.actor_trainer = actor_trainer
        # dirty part ended -----------------------------------------------------

    # (gradually) replace target network weights with online network weights
    def replace_weights(self,tau=0.002):
        theta_a,theta_c = self.actor.get_weights(),self.critic.get_weights()
        theta_a_targ,theta_c_targ = self.actor_target.get_weights(),self.critic_target.get_weights()

        # mixing factor tau : we gradually shift the weights...
        theta_a_targ = [theta_a[i]*tau + theta_a_targ[i]*(1-tau) for i in range(len(theta_a))]
        theta_c_targ = [theta_c[i]*tau + theta_c_targ[i]*(1-tau) for i in range(len(theta_c))]

        self.actor_target.set_weights(theta_a_targ)
        self.critic_target.set_weights(theta_c_targ)

    # a = actor(s) : predict actions given state
    def create_actor_network(self,inputdims,outputdims):
        inp = Input(shape=(inputdims,))
        i = inp
        i = resdense(32)(i)
        i = resdense(32)(i)
        i = resdense(64)(i)
        i = resdense(outputdims)(i)
        # map into (0,1)
        i = Activation('tanh')(i)
        # map into action_space
        i = Lambda(lambda x:x * self.action_multiplier + self.action_bias)(i)

        out = i
        model = Model(inputs=inp,outputs=out)
        model.compile(loss='mse',optimizer=self.optimizer)
        return model

    # q = critic(s,a) : predict q given state and action
    def create_critic_network(self,inputdims,actiondims):
        inp = Input(shape=(inputdims,))
        act = Input(shape=(actiondims,))
        # i = concatenate([inp,act])
        i = merge([inp,act],mode='concat')

        i = resdense(64)(i)
        i = resdense(32)(i)
        i = resdense(32)(i)
        i = resdense(1)(i)
        out = i
        model = Model(inputs=[inp,act],outputs=out)
        model.compile(loss='mse',optimizer=self.optimizer)

        # now we create a frozen_model,
        # that uses the same layers with weights frozen when trained.
        for i in model.layers:
            i.trainable = False # froze the layers

        frozen_model = Model(inputs=[inp,act],outputs=out)
        frozen_model.compile(loss='mse',optimizer=self.optimizer)

        return model,frozen_model

    def train(self,verbose=1):
        memory = self.rpm
        critic,frozen_critic = self.critic,self.frozen_critic
        actor = self.actor
        batch_size = 64

        if memory.size() > batch_size:
            #if enough samples in memory

            # sample randomly a minibatch from memory
            [s1,a1,r1,isdone,s2] = memory.sample_batch(batch_size)
            # print(s1.shape,a1.shape,r1.shape,isdone.shape,s2.shape)

            # a2_targ = actor_targ(s2) : what will you do in s2, Mr. old actor?
            a2 = self.actor_target.predict(s2)

            # q2_targ = critic_targ(s2,a2) : how good is action a2, Mr. old critic?
            q2 = self.critic_target.predict([s2,a2])

            # if a2 is q2-good, then what should q1 be?
            # Use Bellman Equation! (recursive definition of q-values)
            # if not last step of episode:
            #   q1 = (r1 + gamma * q2)
            # else:
            #   q1 = r1

            q1_target = r1 + (1-isdone) * self.discount_factor * q2
            # print(q1_target.shape)

            # train the critic to predict the q1_target, given s1 and a1.
            critic.fit([s1,a1],q1_target,
            batch_size=batch_size,
            epochs=1,
            verbose=verbose,
            shuffle=False
            )

            # now the critic can predict more accurate q given s and a.
            # thanks to the Bellman equation, and David Silver.

            # with a better critic, we can now improve our actor!

            if False: # the following part is for explaination purposes

                # a1_pred = actor(s1) : what will you do in s1, Mr. actor?
                a1_maybe = actor.predict(s1)
                # this action may not be optimal. now let's ask the critic.

                # what do you think of Mr. actor's action on s1, Mr. better critic?
                q1_maybe = critic.predict([s1,a1_maybe])

                # what should we do to the actor, to increase q1_maybe?
                # well, calculate the gradient of actor parameters
                # w.r.t. q1_maybe, then do gradient ascent.

                # so let's build a model that trains the actor to output higher q1_maybe values

                s_given = Input(shape=(self.inputdims,))
                a1_maybe = actor(s_given)
                q1_maybe = frozen_critic([s_given,a1_maybe])
                # frozen weight version of critic. so we only train the actor

                actor_trainer = Model(inputs=s_given,outputs=q1_maybe)

                # use negative of q1_maybe as loss (so we can maximize q by minimizing the loss)
                def neg_q1(y_true,y_pred):
                    return - y_pred # neat!

                actor_trainer.compile(optimizer=self.optimizer,loss=neg_q1)

            else: # the actor_trainer is already initialized in __init__
                actor_trainer = self.actor_trainer

                actor_trainer.fit(s1,
                np.zeros((batch_size,1)), # useless target label
                batch_size=batch_size,
                epochs=1,
                verbose=verbose,
                shuffle=False
                )

            # now both the actor and the critic have improved.
            self.replace_weights()

        else:
            pass
            # print('# no enough samples, not training')

    def feed_one(self,tup):
        self.rpm.add(tup)

    # gymnastics
    def play(self,env,max_steps=-1,realtime=False,render=True,noise_level=0., record=False): # play 1 episode
        max_steps = max_steps if max_steps > 0 else 5000
        steps = 0
        total_reward = 0

        # stack a little history to ensure markov property
        # LSTM will definitely be used here in the future...
        global que # python 2 quirk
        que = np.zeros((self.inputdims,),dtype='float32') # list of recent history actions

        def quein(observation):
            global que # python 2 quirk
            length = que.shape[0]
            que = np.hstack([que,observation])[-length:]

        # what the agent see as state is a stack of history observations.

        observation = env.reset()
        quein(observation) # quein o1
        lastque = que.copy() # s1
        if record:
            labels = ['time', 'reward','x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity', 'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity','psi_velocity']
            result = {x : [] for x in labels}

        while True and steps <= max_steps:
            steps +=1

            # add noise to our actions, since our policy by nature is deterministic
            exploration_noise = np.random.normal(loc=0.,scale=noise_level,size=(self.outputdims,))

            action = self.act(lastque) # a1
            action += exploration_noise
            action = self.clamper(env,action)

            # o2, r1,
            observation, reward, done = env.step(action)

            # d1
            isdone = 1 if done else 0
            total_reward += reward

            quein(observation) # quein o2
            nextque = que.copy() # s2

            # feed into replay memory
            self.feed_one((lastque,action,reward,isdone,nextque)) # s1,a1,r1,isdone,s2

            lastque = nextque
            if record:
            	to_write = [env.sim.time] + [reward] + list(env.sim.pose) + list(env.sim.v) + list(env.sim.angular_v)
            	for ii in range(len(labels)):
            		result[labels[ii]].append(to_write[ii])

            # if render and (steps%10==0 or realtime==True): env.render()
            if done :

                break

            verbose= 2 if steps==1 else 0
            self.train(verbose=verbose)

        print('episode done in',steps,'steps, total reward',total_reward)

        if record:
        	return total_reward, result
        else:
        	return total_reward 
    # one step of action, given observation
    def act(self,observation):
        actor = self.actor
        obs = np.reshape(observation,(1,len(observation)))
        actions = actor.predict([obs])[0]
        return actions
