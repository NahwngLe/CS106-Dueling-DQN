import sys
import os 
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)
import traci
import traci.constants as tc

traci.start(["/Users/cicilian/sumo/bin/sumo-gui", "-c", "circle.sumocfg"])
# vehicle_info = traci.junction.getContextSubscriptionResults('TL')

for step in range(1000):    
    print("step:", step)
    traci.simulationStep()
    # if step >= 100:
    #     traci.junction.subscribeContext('TL', tc.CMD_GET_VEHICLE_VARIABLE, 50, [tc.VAR_SPEED, tc.VAR_ROAD_ID])
    #     # Route vector [number of vehicle, percentage occupied, average speed]
    #     Route_W_E = [0, 0, 0] 
    #     Route_N_S = [0, 0, 0]
    #     Route_E_W = [0, 0, 0]
    #     Route_S_N = [0, 0, 0]
    #     vehicle_info = traci.junction.getContextSubscriptionResults('TL')
    #     if vehicle_info is not None:
    #         for vehicle_id, vehicle_data in vehicle_info.items():
    #             speed = vehicle_data[tc.VAR_SPEED]
    #             road = vehicle_data[tc.VAR_ROAD_ID]
    #             print(vehicle_id, speed, road)
    #             if road == 'W2TL':
    #                 Route_W_E[0] += 1
    #                 Route_W_E[1] += 5
    #                 Route_W_E[2] += speed
    #             elif road == 'N2TL':
    #                 Route_N_S[0] += 1
    #                 Route_N_S[1] += 5
    #                 Route_N_S[2] += speed
    #             elif road == 'E2TL':
    #                 Route_E_W[0] += 1
    #                 Route_E_W[1] += 5
    #                 Route_E_W[2] += speed
    #             elif road == 'S2TL':
    #                 Route_S_N[0] += 1
    #                 Route_S_N[1] += 5
    #                 Route_S_N[2] += speed

    #         if Route_W_E[0] != 0:
    #             Route_W_E[2] /= Route_W_E[0]
    #         if Route_N_S[0] != 0:
    #             Route_N_S[2] /= Route_N_S[0]
    #         if Route_E_W[0] != 0:
    #             Route_E_W[2] /= Route_E_W[0]
    #         if Route_S_N[0] != 0:
    #             Route_S_N[2] /= Route_S_N[0]
    #         Route_W_E[1] /= 50
    #         Route_N_S[1] /= 50
    #         Route_E_W[1] /= 50
    #         Route_S_N[1] /= 50

    #         print("route we:", Route_W_E)
    #         print("route ns:", Route_N_S)
    #         print("route ew:", Route_E_W)
    #         print("route sn:", Route_S_N)

traci.close()
