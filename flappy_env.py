import tensorflow as tf 
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec , tensor_spec
from tf_agents.trajectories import time_step as ts 
import pygame 
import time 
import random 
from tf_agents.environments import utils
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
global WIN
WIN = pygame.display.set_mode((600,800))
bird_img = pygame.transform.scale((pygame.image.load('flap_bird.png')) ,(64,64))
pipe_img  = pygame.image.load('pipe.png')
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
pygame.font.init()
STAT_FONT = pygame.font.SysFont("comicsans" , 50)


collect_episodes_per_iteration = 10
num_iteration = 1000
class Bird:

	def __init__(self,x,y):

		self.x = x
		self.y = y 
		self.vel = 0
		self.tick = 0 
		self.height = self.y
		self.score = 0
		self.high_score = 0

	def render(self,win):

		win.blit(bird_img , (self.x , self.y))
		

	def jump(self):

		self.tick = 0
		self.vel = -10

	def move(self):
		self.tick+=1
		displacement = self.vel*self.tick + (1.5)*self.tick**2

		if displacement >16:
			displacement = 16

		if displacement<0:
			displacement-=2

		self.y +=displacement

		if self.y<2:
			self.y = 2

	def get_mask(self):
		return pygame.mask.from_surface(bird_img)

class Pipe:

	def __init__(self,x):
		self.x = x
		self.bottom = 0
		self.top = 0
		self.PIPE_TOP = pygame.transform.flip(pipe_img , False ,True)
		self.PIPE_BOTTOM = pipe_img
		self.passed = False
		self.gain = False
		self.GAP = 300
		self.set_height()

	def set_height(self):
		self.height = random.randrange(50,350)
		self.top = self.height - self.PIPE_TOP.get_height()
		self.bottom = self.height + self.GAP

	def render(self , win):

		win.blit(self.PIPE_TOP , (self.x , self.top))
		win.blit(self.PIPE_BOTTOM , (self.x , self.bottom))

	def move(self):
		self.x-=5

	def collision(self, bird):
		bird_mask = bird.get_mask()
		top_mask = pygame.mask.from_surface(self.PIPE_TOP)
		bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

		top_offset = (self.x - bird.x , self.top - round(bird.y))
		bottom_offset =(self.x - bird.x , self.bottom - round(bird.y))

		b_point = bird_mask.overlap(bottom_mask , bottom_offset)
		t_point = bird_mask.overlap(top_mask ,top_offset)

		if b_point or t_point:
			return True

		return False



class Flappy_bird_env(py_environment.PyEnvironment):

	def __init__(self):
		self.bird = Bird(300,400)
		self._action_spec = array_spec.BoundedArraySpec(shape = () , dtype = np.float32 , minimum = 0 ,maximum = 1 , name = 'action')
		self._observation_spec = array_spec.BoundedArraySpec(shape=(3,) , dtype = np.int32 , name = 'observation' )
		self._episode_ended = False
		self.stayed_alive = 0
		self.pipe_list = []
		# self.batch_size = 1
		pipe = Pipe(500)
		self.pipe_list.append(pipe)

	def render(self):
		win = pygame.display.set_mode((600,800))
		win.fill((0,255,0))
		self.bird.render(win)
		for pipe in self.pipe_list:
			pipe.render(win) 
		score_label = STAT_FONT.render('Score : '+str(self.bird.score) , 1,(255,255,255))
		win.blit(score_label , (600 - score_label.get_width()-400 , 40))
		hight_score_label = STAT_FONT.render('High Score : '+str(self.bird.high_score) , 1,(255,255,255))
		win.blit(hight_score_label , (600 - score_label.get_width()-400 , 80))
		pygame.display.update()

	def _reset(self):
		self.bird.x = 300
		self.bird.y = 400
		print('bird score : ' , self.bird.score)
		if self.bird.high_score < self.bird.score:
			self.bird.high_score = self.bird.score

		self.bird.score = 0
		self.pipe_list = []
		self.pipe_list.append(Pipe(500))
		self._current_time_step =  ts.restart(np.array([self.bird.y, abs(self.bird.y - self.pipe_list[0].height) , abs(self.bird.y - self.pipe_list[0].bottom )], dtype=np.int32))
		return self._current_time_step


	def _step(self, action):

	

		if action >0.5 :
			self.bird.jump()

		elif action == 0 :
			pass
		for pipe in self.pipe_list:
			if pipe.collision(self.bird):
				
				self._episode_ended = True

		if self._episode_ended:
			
			self._reset()

		self.bird.move()
		for pipe in self.pipe_list:
			pipe.move()

		if self.pipe_list[0].passed == True:
			self.pipe_list[0] = None
			self.pipe_list = self.pipe_list[1:]
		

		if self.pipe_list[0].x < self.bird.x - 40 and self.pipe_list[0].passed == False:
			self.pipe_list[0].passed = True
			self.bird.score+=1
			self.pipe_list.append(Pipe(700))
			# print('appended')	



		

		if len(self.pipe_list)	>=2:
			index  = 0 if self.pipe_list[0].passed == False else 1
		else:
			index = 0
		if self._episode_ended:
			self._episode_ended = False
			print('episodes ended ')
			self._current_time_step = ts.termination(np.array([self.bird.y, abs(self.bird.y - self.pipe_list[index].height) , abs(self.bird.y - self.pipe_list[index].bottom)] , dtype = np.int32 ) , reward = -1000 )
			return self._current_time_step
		
			# self._episode_ended = False


		

		if not self._episode_ended:
			if self.pipe_list[0].passed == True:
				self._current_time_step = ts.transition(np.array([self.bird.y, abs(self.bird.y - self.pipe_list[index].height) , abs(self.bird.y - self.pipe_list[index].bottom)] , dtype = np.int32) , reward = 1000, discount = 1)

			else:	
				self._current_time_step = ts.transition(np.array([self.bird.y, abs(self.bird.y - self.pipe_list[index].height ) , abs(self.bird.y - self.pipe_list[index].bottom)] , dtype = np.int32) , reward = 1, discount = 1)
			return self._current_time_step


	def action_spec(self):

		return self._action_spec

	def observation_spec(self):

		return self._observation_spec




py_env = Flappy_bird_env()
py_env = tf_py_environment.TFPyEnvironment(py_env)
# utils.validate_py_environment(py_env , episodes = 5)
# win = WIN
clock = pygame.time.Clock()
# print('validate_py_environment')
# while True:
# 	rand = 0
# 	clock.tick(30)
# 	for event in pygame.event.get():
# 			if event.type == pygame.QUIT:
# 				run = False
# 				pygame.quit()
# 				quit()
# 				break

# 			if event.type == pygame.KEYDOWN:
# 				if event.key == pygame.K_SPACE:
					
# 					rand = 1	
# 	py_env.render()	


# 	# rand = random.choice([0 ,0,0,0,0, 1])
# 	py_env._step(rand)

actor_net = actor_distribution_network.ActorDistributionNetwork(py_env.observation_spec() , tensor_spec.from_spec( py_env.action_spec()) , activation_fn = tf.keras.activations.tanh)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train_step_counter = tf.compat.v2.Variable(0)

tf_agent = reinforce_agent.ReinforceAgent(py_env.time_step_spec() , tensor_spec.from_spec( py_env.action_spec()) , actor_network = actor_net , optimizer = optimizer , normalize_returns = True , train_step_counter = train_step_counter)
tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

replay_buffers = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec = tf_agent.collect_data_spec , batch_size = py_env.batch_size , max_length = 2000)


def collect_episode(environment , policy , num_episodes):

	episode_counter = 0 
	environment.reset()

	while episode_counter < num_episodes:
		time_step = environment.current_time_step()
		action_step = policy.action(time_step)
		next_time_step = environment._step(action_step)
		traj = trajectory.from_transition(time_step , action_step , next_time_step)
		replay_buffers.add_batch(traj)

		if traj.is_boundary():
			episode_counter+=1



tf_agent.train = common.function(tf_agent.train)
tf_agent.train_step_counter.assign(0)

for _ in range(num_iteration):

	collect_episode(py_env , tf_agent.collect_policy , collect_episodes_per_iteration)
	experience = replay_buffers.gather_all()
	train_loss = tf_agent.train(experience)
	replay_buffers.clear()

	step = tf_agent.train_step_counter.numpy()
	if train_loss.loss == 0:
		break

	if step% 25:
		print('step = {0}: loss = {1}'.format(step, train_loss.loss))

	py_env_test = Flappy_bird_env()
	num_episodes = 10 
	for _ in range(num_episodes):
		time_step = py_env_test.reset()
		while not time_step.is_last():
			action_step = tf_agent.policy.action(time_step)
			print('action_step' , action_step.action)
			time_step = py_env_test.step(action_step.action)
			py_env_test.render()
			# clock.tick(30)


