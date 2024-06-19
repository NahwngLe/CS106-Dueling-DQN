import os 
import sys
# Declare SUMO_HOME in order to import traci
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit('please declare "$SUMO_HOME" variable')

import traci
import traci.constants as tc
import numpy as np
import random
import timeit


# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  
PHASE_NS_YELLOW = 1
PHASE_EW_GREEN = 2  
PHASE_EW_YELLOW = 3



class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = [] # store reward for ploting data
        self._training_epochs = training_epochs


    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1
        last_green_phase = -1
        cur_green_phase = 0

        while self._step < self._max_steps:
            penalty = 0
            # Flag represents whether the phase have been extended
            flag = 0
            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._get_waiting_times()
            if old_action == 1:
                penalty = 10
            reward = old_total_wait - current_total_wait - penalty

            # saving the data into the memory
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0: 
                if action == 0: # action 0: extend the last phase
                    self._set_green_phase(last_green_phase)
                    self._simulate(self._green_duration)
                    flag = 1
                elif action == 1: # action 1: change phase
                    self._set_yellow_phase(last_green_phase)
                    self._simulate(self._yellow_duration)
                    if last_green_phase == 0:
                        cur_green_phase = 2
                    elif last_green_phase == 2:
                        cur_green_phase = 0

            # execute the phase selected before
            if flag == 0:
                self._set_green_phase(cur_green_phase)
                self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait
            last_green_phase = cur_green_phase
            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward


        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time
    

    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            

    

    def _collect_speed(self):
        """
        Retrieve the speed of all car in the 50 meters away from traffic light area from all directions
        """
        total_speed = 0
        traci.junction.subscribeContext('TL', tc.CMD_GET_VEHICLE_VARIABLE, 50, [tc.VAR_SPEED, tc.VAR_ROAD_ID])
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        vehicle_info = traci.junction.getContextSubscriptionResults('TL')
        for vehicle_id, vehicle_data in vehicle_info.items():
            speed = vehicle_data[tc.VAR_SPEED]
            road_id = vehicle_data[tc.VAR_ROAD_ID]
            if road_id in incoming_roads:
                total_speed += speed
        return total_speed
    

    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state


    def _set_yellow_phase(self, last_green_phase):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = last_green_phase + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_green_phase(self, green_phase):
        """
        Activate the correct green light combination in sumo
        """
        traci.trafficlight.setPhase("TL", green_phase)



    def _get_waiting_times(self):
        """
        Retrieve the waiting time of all car in 50 meters away from traffic light area from all directions
        """
        total_waiting_time = 0
        traci.junction.subscribeContext('TL', tc.CMD_GET_VEHICLE_VARIABLE, 50, [tc.VAR_WAITING_TIME, tc.VAR_ROAD_ID])
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        vehicle_info = traci.junction.getContextSubscriptionResults('TL')
        for vehicle_id, vehicle_data in vehicle_info.items():
            waiting_time = vehicle_data[tc.VAR_WAITING_TIME]
            road_id = vehicle_data[tc.VAR_ROAD_ID]
            if road_id in incoming_roads:
                total_waiting_time += waiting_time
        return total_waiting_time
    

    def _get_state(self):
        """
        State la cac feature sau cua moi lane, 50 met tinh tu nut giao
        * number of vehicle 
        * percentage of occupied area
        * average speed
        """
        traci.junction.subscribeContext('TL', tc.CMD_GET_VEHICLE_VARIABLE, 50, [tc.VAR_SPEED, tc.VAR_ROAD_ID])
        Route_W_E = [0, 0, 0] 
        Route_N_S = [0, 0, 0]
        Route_E_W = [0, 0, 0]
        Route_S_N = [0, 0, 0]
        vehicle_info = traci.junction.getContextSubscriptionResults('TL')
        if vehicle_info is not None:
            for vehicle_id, vehicle_data in vehicle_info.items():
                speed = vehicle_data[tc.VAR_SPEED]
                road = vehicle_data[tc.VAR_ROAD_ID]
                if road == 'W2TL':
                    Route_W_E[0] += 1
                    Route_W_E[1] += 5
                    Route_W_E[2] += speed
                elif road == 'N2TL':
                    Route_N_S[0] += 1
                    Route_N_S[1] += 5
                    Route_N_S[2] += speed
                elif road == 'E2TL':
                    Route_E_W[0] += 1
                    Route_E_W[1] += 5
                    Route_E_W[2] += speed
                elif road == 'S2TL':
                    Route_S_N[0] += 1
                    Route_S_N[1] += 5
                    Route_S_N[2] += speed
            max_speed = 17.4
            # Set range for average speed
            if Route_W_E[0] != 0:
                Route_W_E[2] /= Route_W_E[0]
                Route_W_E[2] /= max_speed
                Route_W_E[2] *= 10
                Route_W_E[2] = round(Route_W_E[2] ,0)
            if Route_N_S[0] != 0:
                Route_N_S[2] /= Route_N_S[0]
                Route_N_S[2] /= max_speed
                Route_N_S[2] *= 10
                Route_N_S[2] = round(Route_N_S[2] ,0)
            if Route_E_W[0] != 0:
                Route_E_W[2] /= Route_E_W[0]
                Route_E_W[2] /= max_speed
                Route_E_W[2] *= 10
                Route_E_W[2] = round(Route_E_W[2] ,0)
            if Route_S_N[0] != 0:
                Route_S_N[2] /= Route_S_N[0]
                Route_S_N[2] /= max_speed
                Route_S_N[2] *= 10
                Route_S_N[2] = round(Route_S_N[2] ,0)
            # Set range for occupied area
            Route_W_E[1] /= 6
            Route_W_E[1] = round(Route_W_E[1], 0)
            Route_N_S[1] /= 6
            Route_N_S[1] = round(Route_N_S[1], 0)
            Route_E_W[1] /= 6
            Route_E_W[1] = round(Route_E_W[1], 0)
            Route_S_N[1] /= 6
            Route_S_N[1] = round(Route_S_N[1], 0)
            # Set range for number vehicle
            Route_W_E[0] /= 1.2
            Route_W_E[0] = round(Route_W_E[0], 0)
            Route_N_S[0] /= 1.2
            Route_N_S[0] = round(Route_N_S[0], 0)
            Route_E_W[0] /= 1.2
            Route_E_W[0] = round(Route_E_W[0], 0)
            Route_S_N[0] /= 1.2
            Route_S_N[0] = round(Route_S_N[0], 0)
        state = np.array([Route_W_E, Route_N_S, Route_E_W, Route_S_N])
        state = np.reshape(state, [12, ])
        # print(state)
        return state

    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)
        # print(batch)
        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN

    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode

    @property
    def reward_store(self):
        return self._reward_store