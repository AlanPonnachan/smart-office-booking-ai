import pandas as pd
import numpy as np
import random
from faker import Faker
import networkx as nx
import re
from google import genai
from datetime import datetime, timedelta
import json
import matplotlib # Important for backend use
matplotlib.use('Agg') # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import os # For creating directories and paths
import networkx as nx # Ensure networkx is imported
import logging


# --- (Your provided data generation code: employees_df, hierarchy_df, seats_df, booking_history_df) ---
# --- Data Generation ---
fake = Faker()

# Generate 100 employees
num_employees = 100
employee_ids = range(1, num_employees + 1)
names = [fake.name() for _ in range(num_employees)]
departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Product', 'Design']
teams = {
    'Engineering': ['Frontend', 'Backend', 'DevOps', 'QA', 'Mobile'],
    'Marketing': ['Content', 'Digital', 'Brand', 'Events'],
    'Sales': ['Enterprise', 'SMB', 'Partnerships'],
    'HR': ['Recruitment', 'People Ops', 'Training'],
    'Finance': ['Accounting', 'Payroll', 'Analysis'],
    'Product': ['Product Management', 'UX Research'],
    'Design': ['UI Design', 'Graphic Design']
}

# Generate employee data
employee_data = []
for i in range(num_employees):
    dept = random.choice(departments)
    team = random.choice(teams[dept])
    seniority = random.choice(['Junior', 'Mid-level', 'Senior', 'Lead', 'Manager'])

    # Generate work days (3-5 days per week)
    work_days = random.sample(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                             random.randint(3, 5))

    # Generate preferences
    preferences = {
        'window_seat': random.random() > 0.5,
        'quiet_zone': random.random() > 0.7,
        'near_team': random.random() > 0.3,
        'near_manager': random.random() > 0.6,
        'preferred_floor': random.randint(1, 3),
        'preferred_zone': random.choice(['A', 'B', 'C', 'D']),
        'preference_strength': random.randint(1, 10)  # How important preferences are
    }

    employee_data.append({
        'employee_id': i + 1,
        'name': names[i],
        'department': dept,
        'team': team,
        'seniority': seniority,
        'work_days': work_days,
        'preferences': preferences
    })

employees_df = pd.DataFrame(employee_data)

# Generate team hierarchy
hierarchy_data = []
managers = {}

# Assign managers for each team
for dept in departments:
    for team_name in teams[dept]:
        # Filter employees by team
        team_members = [e for e in employee_data if e['team'] == team_name]
        if team_members:
            # Select a manager (prefer someone with 'Manager' or 'Lead' seniority)
            potential_managers = [e for e in team_members if e['seniority'] in ['Manager', 'Lead']]
            if not potential_managers:
                potential_managers = team_members

            manager = random.choice(potential_managers)
            managers[team_name] = manager['employee_id']

            # Create hierarchy relationships
            for member in team_members:
                if member['employee_id'] != manager['employee_id']:
                    hierarchy_data.append({
                        'employee_id': member['employee_id'],
                        'manager_id': manager['employee_id'],
                        'team': team_name,
                        'department': dept
                    })

hierarchy_df = pd.DataFrame(hierarchy_data)

# Generate office layout
floors = 3
zones_per_floor = 4  # A, B, C, D
rows_per_zone = 5
seats_per_row = 6

seat_data = []
seat_id = 1

for floor in range(1, floors + 1):
    for zone in ['A', 'B', 'C', 'D']:
        for row in range(1, rows_per_zone + 1):
            for seat in range(1, seats_per_row + 1):
                # Determine seat attributes
                is_window = (row == 1)
                is_quiet_zone = (zone == 'D')
                is_near_kitchen = (zone == 'A' and row == rows_per_zone)
                is_near_meeting_rooms = (zone == 'B')

                seat_data.append({
                    'seat_id': seat_id,
                    'floor': floor,
                    'zone': zone,
                    'row': row,
                    'seat_number': seat,
                    'is_window': is_window,
                    'is_quiet_zone': is_quiet_zone,
                    'is_near_kitchen': is_near_kitchen,
                    'is_near_meeting_rooms': is_near_meeting_rooms,
                    'fixed_assignment': None  # For special needs or executives
                })
                seat_id += 1

seats_df = pd.DataFrame(seat_data)

# Assign some fixed seats for special needs
special_needs_count = int(num_employees * 0.05)  # 5% have special needs
if special_needs_count > 0:
    special_needs_employees = random.sample(range(1, num_employees + 1), special_needs_count)
    special_seats = random.sample(range(1, len(seat_data) + 1), special_needs_count)

    for i in range(special_needs_count):
        seats_df.loc[seats_df['seat_id'] == special_seats[i], 'fixed_assignment'] = special_needs_employees[i]


# Generate booking history for the past 4 weeks
booking_history = []
current_date = pd.Timestamp.now()
start_date = current_date - pd.Timedelta(days=28)  # 4 weeks ago

for day_offset in range(28):
    date = start_date + pd.Timedelta(days=day_offset)
    weekday = date.strftime('%A')

    # Only generate bookings for weekdays
    if weekday not in ['Saturday', 'Sunday']:
        # Find employees who work on this day
        for employee in employee_data:
            if weekday in employee['work_days']:
                # 90% chance they booked a seat that day
                if random.random() < 0.9:
                    # Find a suitable seat
                    prefs = employee['preferences']
                    potential_seats = seats_df.copy()

                    # Apply preference filters
                    if prefs['window_seat']:
                        potential_seats = potential_seats[potential_seats['is_window'] == True]
                    if prefs['quiet_zone']:
                        potential_seats = potential_seats[potential_seats['is_quiet_zone'] == True]
                    if prefs['preferred_floor']:
                        potential_seats = potential_seats[potential_seats['floor'] == prefs['preferred_floor']]
                    if prefs['preferred_zone']:
                        potential_seats = potential_seats[potential_seats['zone'] == prefs['preferred_zone']]

                    # If no seats match all preferences, use the original pool
                    if len(potential_seats) == 0:
                        potential_seats = seats_df

                    # Select a random seat from potentials
                    chosen_seat = random.choice(potential_seats['seat_id'].tolist())

                    # 95% chance they showed up
                    attended = random.random() < 0.95

                    booking_history.append({
                        'date': date,
                        'employee_id': employee['employee_id'],
                        'seat_id': chosen_seat,
                        'booking_time': date - pd.Timedelta(days=random.randint(1, 7)),  # Booked 1-7 days in advance
                        'attended': attended,
                        'cancelled': False
                    })

booking_history_df = pd.DataFrame(booking_history)

def data_intake(employees_df, hierarchy_df, seats_df, booking_history_df):
    """Loads, validates, and preprocesses the data."""

    # Basic Validation (example - expand as needed)
    if employees_df.isnull().values.any():
        print("Warning: Missing values in employees_df")
    if not pd.api.types.is_numeric_dtype(employees_df['employee_id']):
        print("Error: employee_id should be numeric")

    # Preprocessing
    # 1. One-hot encode work_days
    work_days_dummies = pd.get_dummies(employees_df['work_days'].apply(pd.Series).stack()).groupby(level=0).sum()
    # Rename the one-hot encoded columns to avoid conflicts
    work_days_dummies.columns = ["workday_" + col for col in work_days_dummies.columns]
    employees_df = pd.concat([employees_df, work_days_dummies], axis=1)

    # 2. Calculate team size
    team_sizes = hierarchy_df.groupby('team')['employee_id'].count().reset_index()
    team_sizes.rename(columns={'employee_id': 'team_size'}, inplace=True)
    hierarchy_df = pd.merge(hierarchy_df, team_sizes, on='team', how='left')

    # 3. Convert dates to datetime
    booking_history_df['date'] = pd.to_datetime(booking_history_df['date'])
    booking_history_df['booking_time'] = pd.to_datetime(booking_history_df['booking_time'])

    # Return Processed Data
    processed_data = {
        'employees': employees_df,
        'hierarchy': hierarchy_df,
        'seats': seats_df,
        'booking_history': booking_history_df,
    }
    return processed_data






def get_real_time_occupancy(employees_df, seats_df, date, real_time_occupancy_df=None):
    """
    Gets real-time occupancy data (dummy or provided) and updates seats_df.
    """
    if real_time_occupancy_df is None:
        # Generate dummy real-time occupancy data
        today = date.strftime('%Y-%m-%d')
        occupied_seats = []
        for employee in employees_df.itertuples():
             if pd.Timestamp.now().strftime('%A') in employee.work_days:
                if random.random() < 0.70:  # 70% occupancy
                    available_seats = seats_df[seats_df['fixed_assignment'].isnull()]['seat_id'].tolist()
                    if available_seats:
                        chosen_seat = random.choice(available_seats)
                        occupied_seats.append({
                            'seat_id': chosen_seat,
                            'employee_id': employee.employee_id,
                            'date': today
                        })
        real_time_occupancy_df = pd.DataFrame(occupied_seats)

    # Convert date to datetime
    if 'date' in real_time_occupancy_df.columns: #handle empty dataframe
        real_time_occupancy_df['date'] = pd.to_datetime(real_time_occupancy_df['date'])

    # Update seats_df with real-time occupancy
    seats_df['is_available'] = True  # Reset availability
    for occupancy in real_time_occupancy_df.itertuples():
        seats_df.loc[seats_df['seat_id'] == occupancy.seat_id, 'is_available'] = False

    return seats_df, real_time_occupancy_df


def data_fusion(processed_data, seats_df, real_time_occupancy_df, hierarchy_df): #add hierarchy_df
    """Combines and enriches data for analysis."""

    employees_df = processed_data['employees']
    booking_history_df = processed_data['booking_history']

    # Combine Data
    employee_hierarchy = pd.merge(employees_df, hierarchy_df, on='employee_id', how='left')

    # # Create Derived Features: Seat Distances
    # seat_distances = {}
    # for i in range(len(seats_df)):
    #     for j in range(i + 1, len(seats_df)):
    #         seat1 = seats_df.iloc[i]
    #         seat2 = seats_df.iloc[j]
    #         distance = np.sqrt((seat1['floor'] - seat2['floor'])**2 +
    #                            (ord(seat1['zone']) - ord(seat2['zone']))**2 +
    #                            (seat1['row'] - seat2['row'])**2 +
    #                            (seat1['seat_number'] - seat2['seat_number'])**2)
    #         seat_distances[(seat1['seat_id'], seat2['seat_id'])] = distance
    #         seat_distances[(seat2['seat_id'], seat1['seat_id'])] = distance

    # Return Fused Data
    fused_data = {
        'employee_hierarchy': employee_hierarchy,
        'seats': seats_df,
        'booking_history': booking_history_df,
        #'seat_distances': seat_distances,
        'real_time_occupancy': real_time_occupancy_df,
        'hierarchy': hierarchy_df, # Return hierarchy
    }
    return fused_data


def process_preferences(employee_hierarchy_df):
    """
    Processes employee preferences into a usable format.

    Args:
        employee_hierarchy_df: DataFrame with employee and hierarchy data,
                              including a 'preferences' column (dictionary).

    Returns:
        A dictionary where keys are employee_ids and values are dictionaries
        of processed preferences.
    """
    preference_vectors = {}
    for employee in employee_hierarchy_df.itertuples():
        emp_id = employee.employee_id
        prefs = employee.preferences  # Access the preferences dictionary

        # Create a numerical representation of preferences
        vector = {
            'window_seat': int(prefs['window_seat']),  # Boolean -> 0 or 1
            'quiet_zone': int(prefs['quiet_zone']),    # Boolean -> 0 or 1
            'preferred_floor': prefs['preferred_floor'],
            'preferred_zone': ord(prefs['preferred_zone']) - ord('A'),  # A=0, B=1, C=2, D=3
            'near_team': int(prefs['near_team']),        #Boolean-> 0 or 1
            'near_manager': int(prefs['near_manager']),    #Boolean-> 0 or 1
            'preference_strength': prefs['preference_strength'] #keep preference strength
        }
        preference_vectors[emp_id] = vector
    return preference_vectors


def build_improved_team_graph(employee_hierarchy_df):
    """
    Builds an improved graph representing team, managerial, and department relationships.
    The function uses the merged DataFrame (employee_hierarchy_df) that comes from merging
    employees_df and hierarchy data.
    
    Relationships:
    - Team: Employees in the same team (team_x) are connected (weight=0.7).
    - Manager-Subordinate: A non-null manager_id connects manager and employee (weight=0.9).
    - Department: Employees in the same department (department_x) get a weak connection (weight=0.3).
    """
    graph = nx.Graph()

    # Create nodes for all employees using basic info from employees_df columns in the merged DataFrame.
    # Note: 'department_x', 'team_x', and 'seniority' come from employees_df.
    for idx, row in employee_hierarchy_df.iterrows():
        emp_id = row['employee_id']
        graph.add_node(emp_id,
                       name=row['name'],
                       department=row['department_x'],
                       team=row['team_x'],
                       seniority=row['seniority'])
    
    # 1. Add edges between employees in the same team.
    teams = employee_hierarchy_df.groupby('team_x')
    for team_name, group in teams:
        team_members = group['employee_id'].tolist()
        for i in range(len(team_members)):
            for j in range(i + 1, len(team_members)):
                # Add team edge with moderate weight.
                graph.add_edge(team_members[i], team_members[j], weight=0.7, relation='team')
    
    # 2. Add manager-subordinate edges using non-null manager_id.
    # Since not all employees have hierarchy info, we iterate over rows that have a valid manager_id.
    hierarchy_info = employee_hierarchy_df[employee_hierarchy_df['manager_id'].notnull()]
    for idx, row in hierarchy_info.iterrows():
        subordinate = row['employee_id']
        manager = int(row['manager_id'])  # manager_id is float; convert to int for consistency
        # Ensure that the manager node exists (they might be in the employees table even if missing hierarchy details)
        if manager in graph.nodes:
            # If an edge already exists (say via team membership), update the weight.
            if graph.has_edge(manager, subordinate):
                graph[manager][subordinate]['weight'] = max(graph[manager][subordinate]['weight'], 0.9)
                graph[manager][subordinate]['relation'] = 'team+manager'
            else:
                graph.add_edge(manager, subordinate, weight=0.9, relation='manager')
    
    # 3. Add weak department-level ties.
    departments = employee_hierarchy_df.groupby('department_x')
    for dept_name, group in departments:
        dept_members = group['employee_id'].tolist()
        for i in range(len(dept_members)):
            for j in range(i + 1, len(dept_members)):
                # Only add a department edge if there is no connection already.
                if not graph.has_edge(dept_members[i], dept_members[j]):
                    graph.add_edge(dept_members[i], dept_members[j], weight=0.3, relation='department')
    
    return graph


def prepare_static_context(employees_df, hierarchy_df, seats_df, booking_history_df):
    """
    Prepares a static textual context for the LLM.  This includes:
        - Office layout information.
        - Summarized employee preferences.
        - A summary of the team structure (using the graph).
        - General information about booking history (but not specific bookings).
    """

    # --- Office Layout (as before) ---
    context = """
    Office Layout:
    - The office has {num_floors} floors.
    - Each floor has {num_zones} zones: {zones}.
    - Each zone has {num_rows} rows and {num_seats_per_row} seats per row.
    - Total number of seats: {num_seats}.
    - Some seats have fixed assignments for employees with special needs.

    """.format(
        num_floors=seats_df['floor'].nunique(),
        num_zones=len(seats_df['zone'].unique()),
        zones=", ".join(seats_df['zone'].unique()),
        num_rows=seats_df['row'].nunique(),
        num_seats_per_row=seats_df['seat_number'].nunique(),
        num_seats=len(seats_df)
    )

    # --- Employee Preferences (Summarized) ---
    context += "Employee Preferences:\n"
    # Instead of just percentages, provide more descriptive summaries:
    context += "- Employees have preferences for window seats, quiet zones, \n  proximity to their team and manager, preferred floor, and preferred zone.\n"
    context += "- Each employee has a preference strength score (1-10) indicating the importance of their preferences.\n"
    # Example of a more descriptive preference summary:
    pref_summary = employees_df['preferences'].apply(pd.Series)  # Expand the dicts
    window_pref = pref_summary['window_seat'].mean()
    quiet_pref = pref_summary['quiet_zone'].mean()
    context += f"- Approximately {int(window_pref * 100)}% of employees prefer window seats.\n"
    context += f"- Approximately {int(quiet_pref * 100)}% of employees prefer quiet zones.\n"


    # --- Team Structure (Using the Graph) ---
    context += "\nTeam Structure:\n"
    # 1. Basic team info:
    context += "- Employees are organized into departments and teams.\n"
    for dept in employees_df['department'].unique():
        context += f"  - {dept}: "
        teams_in_dept = employees_df[employees_df['department'] == dept]['team'].unique()
        context += ", ".join(teams_in_dept) + "\n"

    # 2.  Graph information (summarized):
    #     We can't include the *entire* graph (too much text).
    #     Instead, provide a summary.
    team_graph = build_improved_team_graph(
        pd.merge(employees_df, hierarchy_df, on='employee_id', how='left')  # Use the MERGED data

    )
    num_teams = len(hierarchy_df['team'].unique()) # Number of unique teams.
    avg_degree = sum(dict(team_graph.degree()).values()) / team_graph.number_of_nodes() if team_graph.number_of_nodes() >0 else 0 # Calculate Average degree
    context += f"- There are {num_teams} teams.\n"
    context += f"- On average, each employee is directly connected to {avg_degree:.2f} other employees within their team and/or department.\n"
    context += "- Manager-subordinate relationships have a stronger connection weight than team relationships.\n"

    # --- General Booking History Information ---
    context += "\nGeneral Booking History Information:\n"
    context += "- Historical booking data is available, including booking dates, attendance, and cancellations.\n"
    # Add some overall statistics (but not specific bookings)
    overall_attendance_rate = booking_history_df['attended'].mean()
    overall_cancellation_rate = booking_history_df['cancelled'].mean()
    context += f"- Overall historical attendance rate: {overall_attendance_rate:.2f}\n"
    context += f"- Overall historical cancellation rate: {overall_cancellation_rate:.2f}\n"

    return context


def prepare_dynamic_context(fused_data, target_date, team_graph, employee_id=None):
    """
    Prepares a dynamic textual context FOR A SPECIFIC DATE, including:
        - Seat availability (counts).
        - Relevant historical booking data (for the target day of the week).
        - OPTIONALLY: Employee-specific information and team graph details.
        - List of currently OCCUPIED seats.
    """
    target_date_str = target_date.strftime('%Y-%m-%d')
    target_weekday = target_date.strftime('%A')

    context = f"Dynamic Context for {target_date_str} ({target_weekday}):\n\n"

    # --- Real-time Occupancy (using is_available and explicit list) ---
    available_seats_count = fused_data['seats']['is_available'].sum()
    occupied_seats_count = len(fused_data['seats']) - available_seats_count
    context += f"Seat Availability:\n"
    context += f"- Total seats: {len(fused_data['seats'])}\n"
    context += f"- Available seats: {available_seats_count}\n"
    context += f"- Occupied seats: {occupied_seats_count}\n"

    # List occupied seats with employee IDs
    occupied_seats_df = fused_data['real_time_occupancy']
    if not occupied_seats_df.empty:
        context += "\nCurrently Occupied Seats:\n"
        for idx, row in occupied_seats_df.iterrows():
            context += f"  - Seat ID: {row['seat_id']}, Employee ID: {row['employee_id']}\n"
    else:
        context += "\nCurrently, all seats are available.\n"

    # --- Relevant Historical Booking Data (for the specific *weekday*) ---
    relevant_history = fused_data['booking_history']
    relevant_history = relevant_history[relevant_history['date'].dt.strftime('%A') == target_weekday]
    context += f"\nRelevant Historical Booking Data (Bookings on {target_weekday}s):\n"

    if not relevant_history.empty: # Check if any history exists.
        attendance_rate = relevant_history['attended'].mean()
        context += f"- Average attendance rate on {target_weekday}s: {attendance_rate:.2f}\n"
        cancellation_rate = relevant_history['cancelled'].mean()
        context += f"- Average cancellation rate on {target_weekday}s: {cancellation_rate:.2f}\n"
    else:
        context += f"- No historical booking data available for {target_weekday}s.\n"
    # --- Employee-Specific Context (if employee_id is provided) ---
    if employee_id:
        employee_data = fused_data['employee_hierarchy'][fused_data['employee_hierarchy']['employee_id'] == employee_id]
        if not employee_data.empty:
            employee = employee_data.iloc[0]
            context += f"\nEmployee-Specific Information (Employee ID: {employee_id}):\n"
            context += f"- Name: {employee['name']}\n"
            context += f"- Department: {employee['department_x']}\n"
            context += f"- Team: {employee['team_x']}\n"
            context += f"- Seniority: {employee['seniority']}\n"
            context += f"- Preferences: {employee['preferences']}\n"
            if 'manager_id' in employee_data.columns and not pd.isna(employee['manager_id']):
                context += f"- Manager ID: {int(employee['manager_id'])}\n"

            # --- Team Graph Context WITH WEIGHTS ---
            if team_graph:
                if employee_id in team_graph:
                    neighbors = list(team_graph.neighbors(employee_id))
                    if neighbors:
                        context += f"- Team Members (from graph):\n"
                        for neighbor in neighbors:
                            weight = team_graph[employee_id][neighbor]['weight']
                            relation = team_graph[employee_id][neighbor].get('relation', 'team')
                            context += f"  - Employee {neighbor} (Relationship: {relation}, Weight: {weight:.2f})\n"
                    manager_id = employee['manager_id']
                    if manager_id and not pd.isna(manager_id):
                        manager_id = int(manager_id)
                        if manager_id not in neighbors:
                            context += f"- Manager: Employee {manager_id}\n"
                else:
                    context += "- No team information found in graph for this employee.\n"
            else:
                context += "- Team graph not provided.\n"

        else:
            context += f"\nEmployee-Specific Information: Employee ID {employee_id} not found.\n"

    return context


def beh_rel_data(processed_preferences,team_graph):

    context = """
            Employee preferences:
            
            Here is the list of processed_preferences:
            {list_processed_preferences}

            Details about representation:
            for each employee id:
                'window_seat':   # Boolean -> 0 or 1
                'quiet_zone':     # Boolean -> 0 or 1
                'preferred_floor': 
                'preferred_zone':  # A=0, B=1, C=2, D=3
                'near_team':      #Boolean-> 0 or 1
                'near_manager':    #Boolean-> 0 or 1
                'preference_strength':  #keep preference strength

            """.format(list_processed_preferences=list(processed_preferences.items()))
    
    context += """
              Team clustering graph:

              Here is the list of team graph:
              {list_team_graph}

              Details about representation:
              for each employee id:
              (employee_id, neighbour employee_id, weight:, relation:)
""".format(list_team_graph=list(team_graph.edges(data=True)))
    

    return context


def extract_seat_data(text, date):
    # Regex pattern
    pattern = r"\((\d+): (\d+), ([^)]+)\)"

    # Extract matches
    matches = re.findall(pattern, text)

    # Create DataFrame and add the date column
    df = pd.DataFrame(matches, columns=["employee_id", "seat_id", "Reason"])
    
    # Convert numeric columns to integers
    df["employee_id"] = df["employee_id"].astype(int)
    df["seat_id"] = df["seat_id"].astype(int)

    # Add date column
    df["Date"] = date

    return df


def predict_seat_occupancy(static_context, dynamic_context, fused_seats_df_string,behavior_rel_context,date):
    """
    Predicts the occupancy status of all seats for a SPECIFIC DATE using an LLM.
    """
    date_str = date
    prompt = f"""
    {static_context}

    {dynamic_context}

    {behavior_rel_context}

    {fused_seats_df_string}

    

    Based on the above all informations, predict where each reamining employees who doesn't have assigned or occupied seats should be assigned to which currently available seats.
on {date_str}.  Assume today is {date_str}.
Provide the output as a simple list,  in the format:

(employee_id: which seat_id assigned, reason for this seat assignment in detail )

you have to list all (employee_id:seat_id, resason) corresponding to all employees for the completion of output. means for some employees will be already occcupied and need not to predict for a seat, here you don't need to predict but while giving output from the data just show the occupied seat by employee.
In the output you generate should be only like this:
(employee_id: which seat_id assigned, reason for this seat assignment in detail )
In the output I don't want to see any other text

    """

    client = genai.Client(api_key='Replace_with_your_api_key')

    response = client.models.generate_content(
    model='gemini-2.0-flash', contents=prompt
)
    
   
    predictions_df = extract_seat_data(response.text, date)

    return predictions_df


def prediction_week(processed_data, prediction_days=7):
    """
    Performs processing and analysis, including LLM-based predictions.
    Now makes predictions for multiple days.
    """

    # --- LLM-Based Predictions (for multiple days) ---
    today = datetime.today()

    static_context = prepare_static_context(processed_data['employees'], processed_data['hierarchy'], processed_data['seats'], processed_data['booking_history'])

    

    daily_predictions = {}
    for i in range(prediction_days):
        predict_date = today + timedelta(days=i)

        # --- Test get_real_time_occupancy (with dummy data) ---
        seats_updated, real_time_occupancy = get_real_time_occupancy(
            processed_data['employees'], processed_data['seats'],date
        )

        fused_data = data_fusion(processed_data, seats_updated, real_time_occupancy, processed_data['hierarchy'])

        # 1. Preference Processing (PP) -  already have it
        processed_preferences = process_preferences(fused_data['employee_hierarchy'])

        # 2. Team Clustering (TC) - already have it
        team_graph = build_improved_team_graph(fused_data['employee_hierarchy'])

        behavior_rel_context = beh_rel_data(processed_preferences,team_graph)

        fused_seats_df = fused_data['seats']
        fused_seats_df_string = fused_seats_df.to_string()
            
        dynamic_context = prepare_dynamic_context(fused_data, predict_date, team_graph=team_graph)
        predictions_df = predict_seat_occupancy(static_context, dynamic_context, fused_seats_df_string,behavior_rel_context,predict_date.strftime('%Y-%m-%d'))
        daily_predictions[predict_date.strftime('%Y-%m-%d')] = predictions_df


  
    return daily_predictions


def recommend_seat(static_context, dynamic_context_employee,fused_seats_df_string, employee_id, seat_id,date_booking, action):
    """
    Recommends a seat to an employee or confirms/denies a booking action.
    """
    prompt = f"""{static_context}

    {dynamic_context_employee}

    {fused_seats_df_string}

    Employee {employee_id} is requesting to {action} seat_id-{seat_id} seat on date- {date_booking}.
    Based on all available information, what is your recommendation?

    Provide your response in the following format:

    recommended_seat: [seat_id or 'none']
    action_status: [confirmed/denied]
    reason: [brief explanation]

    If action_status=confirmed that means user requested seat_id is confirmed. If action_status=denied that means user requested seat_id is denied( for this case  you must find a recommend_seat )
    In the output I don't want to see any other text that the format.
    """
    
    client = genai.Client(api_key='Replace_with_your_api_key')

    response = client.models.generate_content(
    model='gemini-2.0-flash', contents=prompt
)

    pattern = r"\{[^{}]*\}"

    # Find match
    match = re.search(pattern, response.text)
    if match:
        json_content = match.group()


    recommendation = json.loads(json_content)
    return recommendation



# --- Graph Plotting Functions (Modified to Save Files) ---

def _save_plot(fig, relative_path_os, static_folder):
    """Helper function to save a matplotlib figure."""
    try:
        # Construct full path using OS-specific separators for saving
        full_path = os.path.join(static_folder, relative_path_os)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        fig.savefig(full_path, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        logging.info(f"Saved graph image to: {full_path}")

        # **** FIX: Convert OS path to URL path (use forward slashes) ****
        relative_url_path = relative_path_os.replace(os.sep, '/')
        # -----------------------------------------------------------------

        return relative_url_path # Return path suitable for url_for
    except Exception as e:
        logging.error(f"Failed to save plot to {relative_path_os}: {e}", exc_info=True)
        plt.close(fig)
        return None

def plot_team_graph(graph, static_folder, filename="team_graph.png"):
    """ Plots the team graph and saves it to a file. """
    if graph is None or graph.number_of_nodes() == 0: return None

    # **** FIX: Define path relative to static folder root ****
    # Use os.path.join for creating the path for saving initially
    relative_path_os = os.path.join('img', 'graphs', filename)
    # ---------------------------------------------------------

    fig = plt.figure(figsize=(12, 10))
    try:
        # ... (plotting code remains the same) ...
        pos = nx.kamada_kawai_layout(graph); teams = set(nx.get_node_attributes(graph, 'team').values())
        if not teams: node_colors = 'skyblue'
        else: team_colors = {team: plt.cm.tab10(i % 10) for i, team in enumerate(teams)}; node_colors = [team_colors.get(graph.nodes[n].get('team'), 'gray') for n in graph.nodes]
        nx.draw_networkx_nodes(graph, pos, node_size=400, node_color=node_colors, alpha=0.8)
        edges = graph.edges(data=True)
        if edges: edge_widths = [data.get('weight', 0.5) * 2 for _, _, data in edges]; nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.4, edge_color='gray')
        labels = {n: graph.nodes[n].get('name', str(n)) for n in graph.nodes if graph.degree(n) > 2}; nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
        plt.title("Team Relationship Graph"); plt.axis('off')
    except Exception as e:
        logging.error(f"Error generating team graph plot: {e}", exc_info=True); plt.close(fig); return None

    # Pass the OS-specific path for saving
    return _save_plot(fig, relative_path_os, static_folder)


def plot_strong_connections(graph, static_folder, min_weight=0.8, filename="strong_connections.png"):
    """ Shows only strong connections and saves the plot. """
    if graph is None or graph.number_of_nodes() == 0: return None
    # **** FIX: Define path relative to static folder root ****
    relative_path_os = os.path.join('img', 'graphs', filename)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(8, 6))
    try:
        # ... (plotting code remains the same) ...
        strong_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('weight', 0) >= min_weight]
        if not strong_edges: plt.title("Strongest Employee Connections (None Found)"); plt.text(0.5, 0.5, f'No connections >= {min_weight}', ha='center', va='center'); plt.axis('off')
        else:
            subgraph = graph.edge_subgraph(strong_edges).copy()
            if subgraph.number_of_nodes() > 0: pos = nx.spring_layout(subgraph, k=0.5, iterations=50); labels = {n: graph.nodes[n].get('name', str(n)) for n in subgraph.nodes}; nx.draw(subgraph, pos, labels=labels, node_size=500, node_color='tomato', edge_color='gray', font_size=8); plt.title(f"Strong Connections (Weight >= {min_weight})"); plt.axis('off')
            else: plt.title(f"Strong Connections (Weight >= {min_weight}) - Error"); plt.axis('off')
    except Exception as e:
        logging.error(f"Error generating strong connections plot: {e}", exc_info=True); plt.close(fig); return None
    # Pass the OS-specific path for saving
    return _save_plot(fig, relative_path_os, static_folder)


def plot_cross_team_connections(graph, static_folder, filename="cross_team.png"):
    """ Shows only cross-team connections and saves the plot. """
    if graph is None or graph.number_of_nodes() == 0: return None
    # **** FIX: Define path relative to static folder root ****
    relative_path_os = os.path.join('img', 'graphs', filename)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    try:
        # ... (plotting code remains the same) ...
        if not all('team' in graph.nodes[n] for n in graph.nodes): plt.title("Cross-Team Connections (Data Missing)"); plt.text(0.5, 0.5, "Missing 'team' attribute.", ha='center', va='center'); plt.axis('off')
        else:
            cross_team_edges = [(u, v) for u, v in graph.edges() if graph.nodes[u].get('team') != graph.nodes[v].get('team')]
            if not cross_team_edges: plt.title("Cross-Team Connections (None Found)"); plt.text(0.5, 0.5, 'No connections between teams.', ha='center', va='center'); plt.axis('off')
            else:
                subgraph = graph.edge_subgraph(cross_team_edges).copy()
                if subgraph.number_of_nodes() > 0:
                    pos = nx.spring_layout(subgraph, k=0.6, iterations=50); teams = set(nx.get_node_attributes(subgraph, 'team').values()); team_colors = {team: plt.cm.tab10(i % 10) for i, team in enumerate(teams)}; node_colors = [team_colors.get(subgraph.nodes[n].get('team'), 'gray') for n in subgraph.nodes]; labels = {n: subgraph.nodes[n].get('name', str(n)) for n in subgraph.nodes}
                    nx.draw(subgraph, pos, labels=labels, node_size=400, node_color=node_colors, edge_color='lightgray', font_size=8, alpha=0.9); plt.title("Cross-Team Connections"); plt.axis('off')
                else: plt.title("Cross-Team Connections - Error"); plt.axis('off')
    except Exception as e:
        logging.error(f"Error generating cross-team connections plot: {e}", exc_info=True); plt.close(fig); return None
    # Pass the OS-specific path for saving
    return _save_plot(fig, relative_path_os, static_folder)


# --- Functions requiring arguments (plot_team_subgraph, plot_manager_subgraph) ---
# These are harder to display all at once. You might:
# 1. Plot ONE example for demonstration.
# 2. Modify the Flask route/template later to allow selection.
# Let's skip these for the initial implementation on the graphs page for simplicity.
# You can call them directly elsewhere if needed for specific analysis.

# Example of how you *could* modify one if you wanted to show a specific team:
def plot_specific_team_subgraph(graph, team_name, static_folder, filename=None):
    if graph is None or graph.number_of_nodes() == 0: return None
    if not filename: filename = f"team_{team_name.lower().replace(' ','_')}.png"
    relative_path = os.path.join('img', 'graphs', filename)
    fig = plt.figure(figsize=(8, 6))
    try:
        subgraph_nodes = [n for n, attr in graph.nodes(data=True) if attr.get('team') == team_name]
        if not subgraph_nodes:
             logging.info(f"No nodes found for team '{team_name}'.")
             plt.title(f"Team: {team_name} (No Members Found)")
             plt.axis('off')
        else:
             subgraph = graph.subgraph(subgraph_nodes)
             pos = nx.kamada_kawai_layout(subgraph)
             labels = {n: subgraph.nodes[n].get('name', str(n)) for n in subgraph.nodes}
             nx.draw(subgraph, pos, labels=labels, node_size=500, node_color='skyblue', edge_color='gray', font_size=8)
             plt.title(f"Team: {team_name} Network")
             plt.axis('off')
    except Exception as e:
        logging.error(f"Error generating subgraph plot for team {team_name}: {e}", exc_info=True)
        plt.close(fig)
        return None
    return _save_plot(fig, relative_path, static_folder)