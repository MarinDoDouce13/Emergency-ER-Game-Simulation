import numpy as np
import pandas as pd
import random

# Define the Room class
class Room:
    def __init__(self, room_type):
        """
        Initialize a Room object.

        Parameters:
            room_type (str): 'A', 'B', or 'C' indicating room type.
            duration (int): Number of hours a patient must stay.
        """
        self.room_type = room_type  # Room type (A, B, or C)
        self.is_free = True  # Initially, the room is available
        self.remaining_time = 0  # Time left for the current patient
        self.patient_acuity = None  # Acuity level of the patient

    def assign_patient(self, acuity, treatment_time):
        """
        Assign a patient to the room if it's free.
        Returns True if successful, False if the room is occupied.
        """
        if self.is_free:
            self.is_free = False  # Room is now occupied
            self.remaining_time = treatment_time  # Set treatment time
            self.patient_acuity = acuity  # Set patient acuity
            return True
        return False  # Room is occupied, cannot assign patient

    def advance_time(self):
        """
        Simulates time progression. Reduces remaining time for the patient.
        Frees the room if treatment is completed.
        """
        if not self.is_free:
            self.remaining_time -= 1  # Reduce time left
            if self.remaining_time == 0:
                self.is_free = True  # Room becomes available
                returned_acuity = self.patient_acuity  # Get patient acuity
                self.patient_acuity = None  # Reset patient acuity

                return returned_acuity  # Return the acuity of the patient that left
        return None  # Room is free or patient still being treated
    
    def __repr__(self):
        """
        String representation of the room's current status.
        """
        return f"Room({self.room_type}, Free={self.is_free}, Remaining Time={self.remaining_time})"
    

def random_token_distribution(total_tokens, max_tokens=50):
    """
    Distribute the given total number of tokens randomly over 24 turns,
    with each turn having a number of tokens centered around the average number of tokens per turn.
    The total number of tokens will still be equally distributed as integers across 24 turns,
    and the total will not exceed the max_tokens constraint. Negative values are not allowed.

    Parameters:
    - total_tokens: The total number of tokens to distribute over 24 turns.
    - max_tokens: The maximum total number of tokens to be distributed (default 50).

    Returns:
    - A list of 24 integers representing the randomly distributed tokens at each turn.
    """
    # Ensure total_tokens doesn't exceed max_tokens
    if total_tokens > max_tokens:
        total_tokens = max_tokens

    # Calculate the average tokens per turn
    average_per_turn = total_tokens / 24
    
    # Generate random variations using Poisson distribution (only positive values)
    tokens_per_turn = np.random.poisson(average_per_turn, 24)

    # Ensure the total number of tokens is exactly equal to total_tokens
    difference = total_tokens - sum(tokens_per_turn)
    
    # Distribute the difference across the turns (adding or subtracting one token as needed)
    for i in range(abs(difference)):
        tokens_per_turn[i % 24] += np.sign(difference)  # Adjust one token at a time to balance the total
    
    return tokens_per_turn

    averages = [21, 38, 41]  # Averages to test
    for avg in averages:
        tokens_distribution = random_token_distribution(avg)
        
        print(f"\nTesting for average of {avg} tokens per turn:")
        print("Tokens distributed across 24 turns:", tokens_distribution)
        print("Average tokens per turn:", np.mean(tokens_distribution))
        print("Total tokens distributed:", sum(tokens_distribution))


def simulate_game(x_A, x_B, x_C):
    """
    Simulate a single game of EMERGENCY with the given room allocation.
    
    Parameters:
        x_A (int): Number of High Acuity (A) rooms staffed
        x_B (int): Number of Medium Acuity (B) rooms staffed
        x_C (int): Number of Low Acuity (C) rooms staffed
        verbose (bool): If True, prints detailed simulation output
        
    Returns:
        int: Net revenue for this game run
    """
    # Initialize rooms
    rooms_A = [Room('A') for _ in range(x_A)]
    rooms_B = [Room('B') for _ in range(x_B)]
    rooms_C = [Room('C') for _ in range(x_C)]

    # Initialize wating room, dict 
    waiting_room = [0, 0, 0]

    # Initalize revenue
    total_revenue = 0

    # Initialize the patient distribution
    patients_A = random_token_distribution(21)
    patients_B = random_token_distribution(38)
    patients_C = random_token_distribution(41)

    completed_patients = [0, 0, 0]

    penalty_LWBS = 0
    penalty_harmed = 0

    patient_waiting = [0, 0, 0]

    # Loop through 24 hours
    for hour in range(24):
        # Get the number of patients for each acuity level and add to the waiting room
        waiting_room[0] += patients_A[hour]
        waiting_room[1] += patients_B[hour]
        waiting_room[2] += patients_C[hour]

        # Assing the patients to the rooms based on their acuity level
        for room in rooms_A:
            # if there are free rooms and  A patients in the waiting room assign them to the room
            if room.is_free and waiting_room[0] > 0:
                room.assign_patient('A', 4)
                waiting_room[0] -= 1
            
            # if there are 2 free rooms and B patients in the waiting room assign them to the room
            if sum([1 for room in rooms_A if room.is_free]) > 2 and waiting_room[1] > 0:
                room.assign_patient('B', 3)
                waiting_room[1] -= 1
        
        for room in rooms_B:
            # if there are free rooms and  B patients in the waiting room assign them to the room
            if room.is_free and waiting_room[1] > 0:
                room.assign_patient('B', 3)
                waiting_room[1] -= 1
            
            # if there are 2 free rooms and C patients in the waiting room assign them to the room
            if sum([1 for room in rooms_B if room.is_free]) > 2 and waiting_room[2] > 0:
                room.assign_patient('C', 2)
                waiting_room[2] -= 1

        for room in rooms_C:
            # if there are free rooms and  C patients in the waiting room assign them to the room
            if room.is_free and waiting_room[2] > 0:
                room.assign_patient('C', 2)
                waiting_room[2] -= 1

        # Advance time by 1 hour
        for room in rooms_A + rooms_B + rooms_C:
            acuity = room.advance_time()
            if acuity == 'A':
                completed_patients[0] += 1
            elif acuity == 'B':
                completed_patients[1] += 1
            elif acuity == 'C':
                completed_patients[2] += 1
        
        # Add to the patient waiting list
        patient_waiting[0] += waiting_room[0]
        patient_waiting[1] += waiting_room[1]
        patient_waiting[2] += waiting_room[2]

        # Determine the left without being seen penalty and the harmed patient penalty
        for i in range(waiting_room[0]):
            if np.random.randint(0, 20) == 0:
                penalty_harmed += 1

        lwbs_turn_count = 0
        for i in range(waiting_room[1]):
            if np.random.randint(0, 20) == 0:
                lwbs_turn_count += 1
        # Remove the patients that left without being seen from the waiting room
        waiting_room[1] -= lwbs_turn_count
        penalty_LWBS += lwbs_turn_count

        lwbs_turn_count = 0
        for i in range(waiting_room[2]):
            if np.random.randint(0, 20) == 0:
                penalty_harmed += 1
        # Remove the patients that left without being seen from the waiting room
        waiting_room[2] -= lwbs_turn_count


    # Calculate the costs
    # From the room staffing
    cost_A = x_A * 3900
    cost_B = x_B * 3000
    cost_C = x_C * 1600

    # From LWBS and harmed patients
    penalty_cost = (penalty_LWBS *200 + penalty_harmed * 10000)

    # From the patient waiting
    waiting_cost_A = patient_waiting[0] * 250
    waiting_cost_B = patient_waiting[1] * 100
    waiting_cost_C = patient_waiting[2] * 25

    total_cost = cost_A + cost_B + cost_C + penalty_cost + waiting_cost_A + waiting_cost_B + waiting_cost_C

    # Calculate the revenue
    revenue_A = completed_patients[0] * 1000
    revenue_B = completed_patients[1] * 600
    revenue_C = completed_patients[2] * 250

    total_revenue = revenue_A + revenue_B + revenue_C

    net_revenue = total_revenue - total_cost

    return net_revenue


def simulate_multiple_games(x_A, x_B, x_C, num_games=10):
    """
    Run multiple simulations and return the average revenue.
    
    Parameters:
        x_A (int): Number of High Acuity (A) rooms staffed
        x_B (int): Number of Medium Acuity (B) rooms staffed
        x_C (int): Number of Low Acuity (C) rooms staffed
        num_games (int): Number of simulations to run
        
    Returns:
        float: Average net revenue over multiple simulations
    """
    net_revenues = [simulate_game(x_A, x_B, x_C) for _ in range(num_games)]

    # Calculate the average net revenue
    print(f"Net Revenues: {net_revenues}")
    print(f"Average Net Revenue: {np.mean(net_revenues)}")
    return np.mean(net_revenues)


def generate_neighbors(a, b, c, max_value=16):
    """
    Generate neighboring solutions by slightly adjusting a, b, or c.
    
    Parameters:
        a (int): Current value of A rooms.
        b (int): Current value of B rooms.
        c (int): Current value of C rooms.
        max_value (int): The maximum allowed sum of a, b, and c.
        
    Returns:
        list of tuples: List of neighboring (a, b, c) combinations.
    """
    neighbors = []
    
    # Try adjusting a, b, or c by ±1 while maintaining the sum <= max_value
    for i, param in enumerate([a, b, c]):
        for delta in [-1, 1]:
            # Modify the current parameter and check if the sum is valid
            new_a, new_b, new_c = a, b, c
            if i == 0:
                new_a += delta
            elif i == 1:
                new_b += delta
            else:
                new_c += delta
                
            # Ensure the sum does not exceed max_value and parameters are non-negative
            if 0 <= new_a <= max_value and 0 <= new_b <= max_value and 0 <= new_c <= max_value and new_a + new_b + new_c <= max_value:
                neighbors.append((new_a, new_b, new_c))
    
    return neighbors

def hill_climbing(max_value=16, num_games=10, max_iterations=1000, restarts=20):
    """
    Hill climbing algorithm with multiple restarts to explore more solutions.
    
    Parameters:
        max_value (int): Maximum value for a, b, and c (each between 0 and max_value).
        num_games (int): Number of games to simulate for each configuration.
        max_iterations (int): Maximum number of iterations per run.
        restarts (int): Number of times to restart after reaching a local optimum.
        
    Returns:
        tuple: Best combination of a, b, c and its corresponding average revenue.
    """
    global_best_combination = None
    global_best_revenue = -float('inf')

    for restart in range(restarts):
        # Start with a random solution
        a = random.randint(0, max_value)
        b = random.randint(0, max_value - a)  
        c = max_value - a - b  
        
        best_combination = (a, b, c)
        best_revenue = simulate_multiple_games(a, b, c, num_games)
        
        iterations = 0
        
        while iterations < max_iterations:
            neighbors = generate_neighbors(a, b, c, max_value)
            best_neighbor = None
            best_neighbor_revenue = -float('inf')
            
            # Evaluate neighbors and choose the one with the highest revenue
            for neighbor in neighbors:
                a, b, c = neighbor
                avg_revenue = simulate_multiple_games(a, b, c, num_games)
                if avg_revenue > best_neighbor_revenue:
                    best_neighbor = neighbor
                    best_neighbor_revenue = avg_revenue
            
            # If a better neighbor is found, move to that neighbor
            if best_neighbor_revenue > best_revenue:
                best_combination = best_neighbor
                best_revenue = best_neighbor_revenue
                a, b, c = best_combination
            else:
                # No improvement, stop the search
                break
            
            iterations += 1
        
        # Update the global best solution if a better one is found
        if best_revenue > global_best_revenue:
            global_best_combination = best_combination
            global_best_revenue = best_revenue

        print(f"Restart {restart + 1}/{restarts}: Best so far -> {best_combination} with revenue {best_revenue}")

    return global_best_combination, global_best_revenue

# Run the hill climbing algorithm
best_combination, best_revenue = hill_climbing()
print(f"Best combination: {best_combination}")
print(f"Best average revenue: {best_revenue}")