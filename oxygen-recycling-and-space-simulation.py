#!/usr/bin/env python
# coding: utf-8

# # Oxygen Recycling

# In[1]:


# Importing necessary libraries from SimPy for the simulation
import simpy
import random
import pandas as pd

# Setting up the simulation environment
env = simpy.Environment()

# Defining the main components of the simulation: astronauts, oxygen recycling system, and events

class SpaceStation:
    def __init__(self, env):
        self.env = env
        # Resources
        self.oxygen_supply = simpy.Container(env, init=1000, capacity=1000)  # Oxygen supply in liters
        # Processes
        self.env.process(self.oxygen_consumption())
        self.env.process(self.oxygen_recycling())

    def oxygen_consumption(self):
        while True:
            yield self.env.timeout(1)  # Simulate oxygen consumption every hour
            oxygen_needed = random.randint(5, 10)  # Oxygen consumption per hour per astronaut
            print(f"Time {self.env.now}: Consuming {oxygen_needed} liters of oxygen.")
            yield self.oxygen_supply.get(oxygen_needed)

    def oxygen_recycling(self):
        while True:
            yield self.env.timeout(2)  # Simulate recycling process every 2 hours
            recycled_oxygen = random.randint(7, 12)  # Amount of oxygen recycled every 2 hours
            print(f"Time {self.env.now}: Recycling {recycled_oxygen} liters of oxygen.")
            yield self.oxygen_supply.put(recycled_oxygen)

# Starting the simulation
space_station = SpaceStation(env)
env.run(until=10)  # Run the simulation for 10 hours


# In[3]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'df' is your DataFrame with the appropriate data
plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['Consumption'], label='Oxygen Consumption', marker='o')
plt.plot(df['Time'], df['Recycling'], label='Oxygen Recycling', marker='x')
plt.xlabel('Time (hours)')
plt.ylabel('Oxygen (liters)')
plt.title('Oxygen Consumption and Recycling Over Time')
plt.legend()
plt.show()


# In[4]:


# It appears there was an issue executing the previous code block. Let's try generating the figures and charts again without errors.

# Re-defining the data since previous execution encountered an error
data = {
    "Time": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Consumption": [7, 5, 7, 10, 8, 10, 9, 10, 5],
    "Recycling": [0, 7, 0, 12, 0, 10, 0, 12, 0]
}

# Creating a DataFrame from the provided data
df = pd.DataFrame(data)

# Calculating the cumulative net oxygen level to understand the oxygen balance over time
df["Net Oxygen"] = df["Recycling"] - df["Consumption"]
df["Cumulative Net Oxygen"] = df["Net Oxygen"].cumsum()

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Oxygen consumption and recycling
ax1.bar(df["Time"] - 0.2, df["Consumption"], width=0.4, label="Oxygen Consumption", color='blue')
ax1.bar(df["Time"] + 0.2, df["Recycling"], width=0.4, label="Oxygen Recycling", color='green')

# Cumulative net oxygen
ax2 = ax1.twinx()
ax2.plot(df["Time"], df["Cumulative Net Oxygen"], label="Cumulative Net Oxygen", color='red', marker='o')

# Formatting the plot
ax1.set_xlabel("Time (Hours)")
ax1.set_ylabel("Oxygen Volume (Liters)")
ax2.set_ylabel("Cumulative Net Oxygen (Liters)", color='red')
ax1.set_title("Oxygen Consumption, Recycling, and Cumulative Net Oxygen Over Time")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Display the figure
plt.show()


# # Human physiology in space

# In[1]:


import simpy
import numpy as np

# Constants
MISSION_DURATION = 180 # days, example mission duration
RECOVERY_DURATION = 360 # days, example duration for recovery post-mission
SIMULATION_DURATION = MISSION_DURATION + RECOVERY_DURATION # total simulation time

# Initial Conditions
initial_BMD = 1.0 # arbitrary unit
initial_VO2 = 1.0 # arbitrary unit
BMD_loss_rate = -0.0077 # percentage per day, example based on pelvis loss
VO2_loss_rate = -0.001 # percentage per day, arbitrary example
BMD_recovery_rate = 0.0077 / 97 # per day, based on pelvis recovery half-life
VO2_recovery_rate = 0.001 # per day, arbitrary example for simplicity

# Environment setup
env = simpy.Environment()

def spaceflight(env):
    BMD = initial_BMD
    VO2 = initial_VO2
    
    for day in range(MISSION_DURATION):
        yield env.timeout(1) # simulate each day
        BMD += BMD_loss_rate * BMD
        VO2 += VO2_loss_rate * VO2
        print(f"Day {env.now}: BMD={BMD:.4f}, VO2={VO2:.4f}")
    
    for day in range(RECOVERY_DURATION):
        yield env.timeout(1) # simulate each day
        # Simulate recovery, ensuring BMD and VO2 do not exceed initial values
        BMD = min(BMD + BMD_recovery_rate * BMD, initial_BMD)
        VO2 = min(VO2 + VO2_recovery_rate * VO2, initial_VO2)
        print(f"Day {env.now}: BMD={BMD:.4f}, VO2={VO2:.4f}")

# Process setup
env.process(spaceflight(env))

# Run the simulation
env.run(until=SIMULATION_DURATION)


# In[2]:


import matplotlib.pyplot as plt
import numpy as np

# Days
days = np.arange(1, 540)

# BMD and VO2 values based on the provided data snippet
BMD = np.linspace(0.9923, 0.2559, len(days))
VO2 = np.concatenate([np.linspace(0.9990, 0.8352, 180), np.linspace(0.8360, 1.0000, len(days)-180)])

# Creating the figures
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Day')
ax1.set_ylabel('BMD', color=color)
ax1.plot(days, BMD, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('VO2', color=color)  # we already handled the x-label with ax1
ax2.plot(days, VO2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('BMD and VO2 Changes Over Time')
plt.show()


# In[3]:


# Let's create more detailed figures, focusing on specific segments: the initial decrease and the recovery phase.

# Segmenting the data for detailed analysis
initial_days = np.arange(1, 181)
recovery_days = np.arange(181, 540)

BMD_initial = BMD[:180]
BMD_recovery = BMD[180:]

VO2_initial = VO2[:180]
VO2_recovery = VO2[180:]

# Plotting Initial Decrease Phase
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(initial_days, BMD_initial, 'r-', label='BMD Decrease')
ax1.set_title('Initial Decrease Phase')
ax1.set_xlabel('Day')
ax1.set_ylabel('BMD')
ax1.legend(loc="upper right")

ax2.plot(initial_days, VO2_initial, 'b-', label='VO2 Decrease')
ax2.set_xlabel('Day')
ax2.set_ylabel('VO2')
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

# Plotting Recovery Phase
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(recovery_days, BMD_recovery, 'r--', label='BMD Recovery')
ax1.set_title('Recovery Phase')
ax1.set_xlabel('Day')
ax1.set_ylabel('BMD')
ax1.legend(loc="upper left")

ax2.plot(recovery_days, VO2_recovery, 'b--', label='VO2 Recovery')
ax2.set_xlabel('Day')
ax2.set_ylabel('VO2')
ax2.legend(loc="upper left")

plt.tight_layout()
plt.show()


# In[4]:


# For further detailed analysis, let's focus on a rate of change figure for both BMD and VO2 during the mission and recovery phase.

# Calculating rate of change for BMD and VO2
BMD_rate_change = np.diff(BMD) * 100 / BMD[:-1] # percentage change
VO2_rate_change = np.diff(VO2) * 100 / VO2[:-1] # percentage change
days_rate_change = days[:-1] # Adjusting days for rate of change calculation

# Plotting Rate of Change for BMD and VO2
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Day')
ax1.set_ylabel('BMD Rate of Change (%)', color=color)
ax1.plot(days_rate_change, BMD_rate_change, color=color, label='BMD Rate of Change')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('VO2 Rate of Change (%)', color=color)  # we already handled the x-label with ax1
ax2.plot(days_rate_change, VO2_rate_change, color=color, label='VO2 Rate of Change')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Rate of Change in BMD and VO2 Over Time')
plt.show()


# # Life Support Systems
# 

# # Part 1

# # Step 1: Setting Up the Simulation Environment¶

# In[5]:


import simpy

# Simulation parameters
SIM_DURATION = 1000  # Duration of simulation in time units (e.g., hours)

# Initialize the SimPy environment
env = simpy.Environment()


# # Step 2: Defining Constants and Initial Conditions

# In[6]:


# Constants for simulation
OXYGEN_CONSUMPTION_PER_HOUR = 0.84  # Volume of oxygen consumed by an astronaut per hour
CO2_PRODUCTION_PER_HOUR = 0.85     # Volume of CO2 produced by an astronaut per hour
WATER_RECOVERY_EFFICIENCY = 0.95   # Efficiency of water recovery system
ENERGY_CONSUMPTION_PER_SYSTEM_PER_HOUR = 5  # Energy consumption of the life support system per hour

# Initial conditions
INITIAL_OXYGEN_LEVEL = 1000  # Initial volume of oxygen available
INITIAL_CO2_LEVEL = 0        # Initial volume of CO2
INITIAL_WATER_LEVEL = 500    # Initial volume of water available
INITIAL_ENERGY_LEVEL = 2000  # Initial energy available


# # Step 3: Modeling Astronauts and Life Support Systems¶

# In[7]:


def astronaut(env, name, oxygen_store, co2_store):
    """Process simulation for astronaut's oxygen consumption and CO2 production."""
    while True:
        yield oxygen_store.get(OXYGEN_CONSUMPTION_PER_HOUR)
        yield co2_store.put(CO2_PRODUCTION_PER_HOUR)
        yield env.timeout(1)  # Wait for 1 time unit before the next cycle

def life_support_system(env, oxygen_store, co2_store, water_store, energy_store):
    """Process simulation for the life support system managing resources."""
    while True:
        # Simulate oxygen generation and CO2 removal here
        # Simulate water recovery and energy consumption
        yield env.timeout(1)  # System checks and balances every time unit


# # Step 4: Creating Resources and Starting Processes

# In[8]:


# Create resource containers
oxygen_store = simpy.Container(env, capacity=2000, init=INITIAL_OXYGEN_LEVEL)
co2_store = simpy.Container(env, capacity=1000, init=INITIAL_CO2_LEVEL)
water_store = simpy.Container(env, capacity=1000, init=INITIAL_WATER_LEVEL)
energy_store = simpy.Container(env, capacity=5000, init=INITIAL_ENERGY_LEVEL)

# Start process for each astronaut and the life support system
env.process(astronaut(env, "Astronaut 1", oxygen_store, co2_store))
env.process(life_support_system(env, oxygen_store, co2_store, water_store, energy_store))


# # Step 5: Running the Simulation¶

# In[9]:


# Run the simulation
env.run(until=SIM_DURATION)

# Print final resource levels
print(f"Final Oxygen Level: {oxygen_store.level}")
print(f"Final CO2 Level: {co2_store.level}")
print(f"Final Water Level: {water_store.level}")
print(f"Final Energy Level: {energy_store.level}")


# # Part 2

# Simulate Processes:
# Oxygen and CO2 cycling: Based on the astronauts' metabolic rates and the efficiency of life support systems. Water recovery: Simulate the cycle of water use and purification. Energy use: Model the energy consumption of each component, adjusting for inefficiencies.
# 
# Implement Failures and Redundancies:
# Define random events or scheduled maintenance that can affect the performance of life support systems.

# In[10]:


import simpy
import random

# Constants
OXYGEN_CONSUMPTION_RATE = 1.0  # volume per cycle, per astronaut
CO2_PRODUCTION_RATE = 0.85     # volume per cycle, per astronaut
WATER_RECOVERY_EFFICIENCY = 0.95  # efficiency of water recovery system
ENERGY_CONSUMPTION_PER_SYSTEM = 5  # arbitrary units per cycle, per system

# Environment setup
env = simpy.Environment()

# Astronaut behavior
def astronaut(env, name, oxygen_store, co2_store):
    while True:
        # Consume oxygen and produce CO2
        oxygen_store.get(OXYGEN_CONSUMPTION_RATE)
        co2_store.put(CO2_PRODUCTION_RATE)
        yield env.timeout(1)  # Simulate one cycle (e.g., one hour)

# Life support system behavior
def life_support_system(env, oxygen_store, co2_store, water_store, energy_store):
    while True:
        # Check and balance oxygen and CO2 levels
        if co2_store.level > 1:
            co2_store.get(1)
            oxygen_store.put(0.89)  # Algae conversion rate
        # Simulate water recovery process
        water_store.put(water_store.level * WATER_RECOVERY_EFFICIENCY)
        # Energy consumption
        energy_store.get(ENERGY_CONSUMPTION_PER_SYSTEM)
        yield env.timeout(1)  # Simulate one cycle

# Stores for resources
oxygen_store = simpy.Container(env, init=100)
co2_store = simpy.Container(env, init=0)
water_store = simpy.Container(env, init=100)
energy_store = simpy.Container(env, init=500)  # Initial energy supply

# Start processes
env.process(astronaut(env, "Astronaut 1", oxygen_store, co2_store))
env.process(life_support_system(env, oxygen_store, co2_store, water_store, energy_store))

# Run simulation
env.run(until=100)  # Simulate for 100 cycles

print(f"Final Oxygen Level: {oxygen_store.level}")
print(f"Final CO2 Level: {co2_store.level}")
print(f"Final Water Level: {water_store.level}")
print(f"Final Energy Level: {energy_store.level}")


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt

# Sample data creation
# Let's create some sample data resembling what your simulation might produce
data = {
    "Time": [1, 2, 3, 4, 5],  # Example time points
    "Oxygen_Level": [100, 95, 90, 85, 80],  # Example oxygen levels over time
    "CO2_Level": [0, 5, 10, 15, 20],  # Example CO2 levels over time
    # Add similar entries for water and energy if you have those data
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

# Plotting Oxygen Level over Time
plt.figure(figsize=(10, 6))
plt.plot(df['Time'], df['Oxygen_Level'], label='Oxygen Level', marker='o')
plt.xlabel('Time (cycles)')
plt.ylabel('Oxygen Level')
plt.title('Oxygen Level Over Time')
plt.legend()
plt.grid(True)
plt.show()


# # Spacecraft

# In[13]:


# Simulation parameters (based on typical values)
oxygen_consumption_rate = 0.84  # kg per day
co2_production_rate = 1.0  # kg per day
oxygen_recycling_efficiency = 0.75  # 75% efficiency in recycling CO2 to O2

# Convert rates from per day to per minute for the simulation
minutes_in_day = 1440
oxygen_consumption_rate_per_min = oxygen_consumption_rate / minutes_in_day
co2_production_rate_per_min = co2_production_rate / minutes_in_day

# Initialize the SimPy environment
env = simpy.Environment()

# Resources
oxygen_supply = simpy.Container(env, init=100, capacity=1000)  # Starting with 100 kg of O2
co2_level = simpy.Container(env, init=0, capacity=1000)  # Starting level of CO2

def astronaut_breathing(env, name, oxygen_supply, co2_level):
    while True:
        yield oxygen_supply.get(oxygen_consumption_rate_per_min)
        yield co2_level.put(co2_production_rate_per_min)
        print(f"{env.now:.2f} min: Astronaut {name} consumed oxygen and produced CO2.")
        yield env.timeout(1)  # Simulating a minute

def oxygen_recycling_system(env, co2_level, oxygen_supply):
    while True:
        yield co2_level.get(co2_production_rate_per_min * oxygen_recycling_efficiency)
        recycled_oxygen = co2_production_rate_per_min * oxygen_recycling_efficiency
        yield oxygen_supply.put(recycled_oxygen)
        print(f"{env.now:.2f} min: Recycled {recycled_oxygen:.2f} kg of CO2 into oxygen.")
        yield env.timeout(1)  # Recycling process takes a minute

# Start processes
env.process(astronaut_breathing(env, "John Doe", oxygen_supply, co2_level))
env.process(oxygen_recycling_system(env, co2_level, oxygen_supply))

# Run the simulation for a specified time
simulation_time = 60  # Simulate for 60 minutes
env.run(until=simulation_time)


# In[14]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming the simulation ran for 60 minutes with constant rates
simulation_minutes = np.arange(0, 60)  # 0 to 59 minutes

# Oxygen consumption and CO2 production rates (kg per minute)
# These are based on the daily rates provided earlier, converted to per minute for the simulation duration
oxygen_consumption_rate_per_min = 0.84 / 1440  # kg per day to kg per min
co2_production_rate_per_min = 1.0 / 1440  # kg per day to kg per min

# Calculating total oxygen consumed and CO2 produced over time
oxygen_consumed = oxygen_consumption_rate_per_min * simulation_minutes
co2_produced = co2_production_rate_per_min * simulation_minutes

# Plotting Oxygen Consumption over Time
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(simulation_minutes, oxygen_consumed, label='Oxygen Consumed', color='blue')
plt.xlabel('Time (minutes)')
plt.ylabel('Oxygen Consumed (kg)')
plt.title('Oxygen Consumption over Time')
plt.legend()

# Plotting CO2 Production over Time
plt.subplot(1, 2, 2)
plt.plot(simulation_minutes, co2_produced, label='CO2 Produced', color='green')
plt.xlabel('Time (minutes)')
plt.ylabel('CO2 Produced (kg)')
plt.title('CO2 Production over Time')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




