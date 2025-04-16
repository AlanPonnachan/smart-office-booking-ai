# --- app.py ---

import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from datetime import datetime, timedelta
from functools import wraps
import backend_logic as be # Assuming your backend logic is in backend_logic.py
#from dotenv import load_dotenv
import logging
import pickle # For admin predictions if kept
import json
import networkx as nx # Import networkx
from google import genai
import asyncio
import re

# --- Basic Setup ---
#load_dotenv() # Load environment variables from .env file
# Setup basic logging with timestamp and level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
# IMPORTANT: Use a strong, random secret key, preferably set via environment variable
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'replace_this_with_a_real_secret_key_3498fjn')

# --- Simple Admin Config ---
# In a real app, get this from a database or secure config
# Ensure these IDs exist in your generated employees_df or handle them separately
ADMIN_EMPLOYEE_IDS = [1, 2, 99] # Example admin IDs (Adjust to valid IDs from your data)

# --- File to store latest predictions IF admin feature is kept ---
LATEST_PREDICTIONS_FILE = 'latest_predictions.pkl'

# --- Global Variables for Data & Context (Loaded Once) ---
processed_data = {}
employees_df = pd.DataFrame()
hierarchy_df = pd.DataFrame()
seats_df_base = pd.DataFrame() # The original, unmodified seat layout
booking_history_df = pd.DataFrame()
# In-memory store for real-time bookings made via web app: { 'YYYY-MM-DD': {seat_id: employee_id, ...} }
real_time_bookings_store = {}
static_context_global = ""    # Store prepared static context
team_graph_global = None      # Store prepared team graph


# --- Define static folder path ---
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
GRAPH_SAVE_DIR = os.path.join(STATIC_FOLDER, 'img', 'graphs')


# --- Create graph directory at startup ---
try:
    os.makedirs(GRAPH_SAVE_DIR, exist_ok=True)
    logging.info(f"Ensured graph save directory exists: {GRAPH_SAVE_DIR}")
except OSError as e:
    logging.error(f"Could not create graph save directory '{GRAPH_SAVE_DIR}': {e}")


    
# --- Initialize the client ONCE at the module level ---
GENERATIVE_AI_CLIENT = None
try:
    api_key = 'Replace_with_your_api_key'
    if api_key:
        # Ensure an event loop exists in the main thread for initialization
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # No event loop running
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        GENERATIVE_AI_CLIENT = genai.Client(api_key=api_key)
        print("Google Generative AI Client initialized successfully.")
    else:
        print("Warning: GOOGLE_API_KEY not set. LLM recommendations will fail.")
        # Optionally raise an error or handle appropriately
except Exception as client_init_error:
    print(f"ERROR: Failed to initialize Google Generative AI Client: {client_init_error}")
    GENERATIVE_AI_CLIENT = None # Ensure it's None on failure



# --- Initial Data Loading and Context Preparation ---
try:
    logging.info("Loading base data (employees, hierarchy, seats, history)...")
    # Use the data generation directly from backend_logic for consistency
    employees_df = be.employees_df
    hierarchy_df = be.hierarchy_df
    seats_df_base = be.seats_df
    booking_history_df = be.booking_history_df

    if not all(df.empty for df in [employees_df, hierarchy_df, seats_df_base, booking_history_df]):
        # Perform data intake using copies to avoid modifying originals if needed elsewhere
        processed_data = be.data_intake(employees_df.copy(), hierarchy_df.copy(), seats_df_base.copy(), booking_history_df.copy())
        logging.info("Base data loaded and processed via data_intake.")

        logging.info("Preparing static context and team graph for LLM...")
        # Prepare static context once using data from processed_data
        static_context_global = be.prepare_static_context(
            processed_data['employees'], processed_data['hierarchy'],
            processed_data['seats'], processed_data['booking_history']
        )
        # Build team graph once using processed employee/hierarchy data
        if 'employee_hierarchy' not in processed_data:
             processed_data['employee_hierarchy'] = pd.merge(processed_data['employees'], processed_data['hierarchy'], on='employee_id', how='left')

        team_graph_global = be.build_improved_team_graph(processed_data['employee_hierarchy'])

        logging.info("Static context and team graph prepared.")
    else:
        logging.error("One or more base dataframes are empty. Check data generation in backend_logic.py.")
        # Assign empty/error states
        processed_data = {'employees': pd.DataFrame(), 'hierarchy': pd.DataFrame(), 'seats': pd.DataFrame(), 'booking_history': pd.DataFrame()}
        static_context_global = "Error: Base data missing."
        team_graph_global = nx.Graph() # Initialize empty graph

    # Initialize real-time store (for web app bookings)
    real_time_bookings_store = {}
    logging.info("Initial data loading and context preparation complete.")

except Exception as e:
    logging.error(f"FATAL: Error during initial data loading or context preparation: {e}", exc_info=True)
    # Ensure fallback states
    processed_data = {}
    employees_df = pd.DataFrame()
    hierarchy_df = pd.DataFrame()
    seats_df_base = pd.DataFrame()
    booking_history_df = pd.DataFrame()
    real_time_bookings_store = {}
    static_context_global = "Error: Initialization failed."
    team_graph_global = nx.Graph() # Initialize empty graph

# --- Helper Functions & Decorators ---

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
    
    client = GENERATIVE_AI_CLIENT

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

    client = client = GENERATIVE_AI_CLIENT

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

    static_context = be.prepare_static_context(processed_data['employees'], processed_data['hierarchy'], processed_data['seats'], processed_data['booking_history'])

    

    daily_predictions = {}
    for i in range(prediction_days):
        predict_date = today + timedelta(days=i)

        # --- Test get_real_time_occupancy (with dummy data) ---
        seats_updated, real_time_occupancy = be.get_real_time_occupancy(
            processed_data['employees'], processed_data['seats'],predict_date
        )

        fused_data = be.data_fusion(processed_data, seats_updated, real_time_occupancy, processed_data['hierarchy'])

        # 1. Preference Processing (PP) -  already have it
        processed_preferences = be.process_preferences(fused_data['employee_hierarchy'])

        # 2. Team Clustering (TC) - already have it
        team_graph = be.build_improved_team_graph(fused_data['employee_hierarchy'])

        behavior_rel_context = be.beh_rel_data(processed_preferences,team_graph)

        fused_seats_df = fused_data['seats']
        fused_seats_df_string = fused_seats_df.to_string()
            
        dynamic_context = be.prepare_dynamic_context(fused_data, predict_date, team_graph=team_graph)
        predictions_df = predict_seat_occupancy(static_context, dynamic_context, fused_seats_df_string,behavior_rel_context,predict_date.strftime('%Y-%m-%d'))
        daily_predictions[predict_date.strftime('%Y-%m-%d')] = predictions_df


  
    return daily_predictions






def login_required(f):
    """Decorator to ensure user is logged in as a regular employee."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'employee_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('show_login'))
        if session.get('user_type') != 'employee':
             flash('Access denied. Please log in as an employee.', 'danger')
             return redirect(url_for('admin_dashboard') if session.get('user_type') == 'admin' else url_for('show_login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator to ensure user is logged in as an admin."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'employee_id' not in session or session.get('user_type') != 'admin':
            flash('Admin access required.', 'danger')
            return redirect(url_for('show_login'))
        return f(*args, **kwargs)
    return decorated_function

def get_employee_info(employee_id):
    """Fetches basic employee info for display from the global employees_df."""
    if employees_df.empty:
        logging.warning(f"Attempted get_employee_info({employee_id}), but employees_df is empty.")
        return None
    try:
        employee_data = employees_df[employees_df['employee_id'] == employee_id]
        if not employee_data.empty: return employee_data.iloc[0].to_dict()
    except Exception as e: logging.error(f"Error in get_employee_info({employee_id}): {e}", exc_info=True)
    logging.warning(f"Employee info not found for ID: {employee_id}")
    return None

def apply_booking_update(date_str, seat_id, employee_id, action):
    """
    Applies a CONFIRMED booking/cancellation to the in-memory web app booking store.
    Logs potential inconsistencies if LLM confirmed an action that seems contradictory to store state.
    Returns True if update was logically applied, False otherwise.
    """
    global real_time_bookings_store
    if date_str not in real_time_bookings_store: real_time_bookings_store[date_str] = {}
    logging.debug(f"Applying booking update: Date={date_str}, Seat={seat_id}, Emp={employee_id}, Action={action}")
    current_occupant = real_time_bookings_store[date_str].get(seat_id)

    if action == 'book':
        if current_occupant is not None and current_occupant != employee_id:
            logging.error(f"[apply_booking_update] CRITICAL: Race condition or LLM error? Trying to book seat {seat_id} for {employee_id} but store shows {current_occupant} for {date_str}.")
            real_time_bookings_store[date_str][seat_id] = employee_id # Overwrite based on LLM confirmation
            logging.warning(f"[apply_booking_update] Overwrote web booking for seat {seat_id} based on LLM confirmation.")
            return True
        elif current_occupant == employee_id:
             logging.info(f"[apply_booking_update] Redundant book action for seat {seat_id} by {employee_id} on {date_str}. State unchanged.")
             return True # Already booked by same user
        else: # Seat is free in web store, proceed with booking
            real_time_bookings_store[date_str][seat_id] = employee_id
            logging.info(f"[apply_booking_update] Stored web booking: Emp {employee_id} booked seat {seat_id} for {date_str}")
            return True
    elif action == 'cancel':
        if current_occupant is not None: # Check if there's a web booking to cancel
            if current_occupant == employee_id:
                del real_time_bookings_store[date_str][seat_id]
                logging.info(f"[apply_booking_update] Removed web booking: Emp {employee_id} cancelled seat {seat_id} for {date_str}")
                return True
            else: # LLM confirmed cancel, but store shows different user for web booking
                 logging.error(f"[apply_booking_update] Inconsistency: LLM confirmed cancel for seat {seat_id} (Emp {employee_id}) but web store shows occupant {current_occupant} on {date_str}.")
                 del real_time_bookings_store[date_str][seat_id] # Remove based on LLM's authority
                 logging.warning(f"[apply_booking_update] Removed web booking for seat {seat_id} based on LLM cancellation confirmation despite occupant mismatch.")
                 return True
        else: # LLM confirmed cancel, but seat wasn't in web store
            logging.warning(f"[apply_booking_update] LLM confirmed cancel for seat {seat_id} on {date_str}, but it wasn't found in web store. State unchanged.")
            return True # Cancellation of non-existent web booking is okay
    else: # Invalid action
        logging.error(f"Invalid action '{action}' passed to apply_booking_update.")
        return False

def get_current_seat_state_df(date_str, target_date):
    """
    Creates a DataFrame reflecting the current seat availability for a given date,
    combining simulated occupancy, web app bookings, and fixed assignments.
    Handles potential duplicate seat assignments from simulation.
    """
    if seats_df_base.empty or employees_df.empty:
        logging.error("Cannot get current seat state: Base seats_df or employees_df is empty.")
        return pd.DataFrame()

    try:
        logging.debug(f"Generating current state for {date_str}...")

        # 1. Get simulated state from backend logic function
        seats_simulated_df = pd.DataFrame() # Initialize
        sim_occupancy_df = pd.DataFrame() # Initialize
        try:
            if hasattr(be, 'get_real_time_occupancy'):
                 seats_simulated_df, sim_occupancy_df = be.get_real_time_occupancy(
                     employees_df.copy(), seats_df_base.copy(), target_date
                 )
                 logging.debug(f"Simulated occupancy fetched for {date_str}. Sim Occ Count: {len(sim_occupancy_df)}, Avail after sim: {seats_simulated_df['is_available'].sum()}")
            else:
                 logging.warning("'get_real_time_occupancy' function not found in backend_logic. Using only web bookings + fixed.")
                 seats_simulated_df = seats_df_base.copy()
                 seats_simulated_df['is_available'] = True
        except Exception as sim_err:
            logging.error(f"Error calling be.get_real_time_occupancy for {date_str}: {sim_err}", exc_info=True)
            seats_simulated_df = seats_df_base.copy()
            seats_simulated_df['is_available'] = True

        # Ensure necessary columns exist
        if 'is_available' not in seats_simulated_df.columns: seats_simulated_df['is_available'] = True
        seats_simulated_df['occupant_employee_id'] = pd.NA

        # Add occupants from the simulation result (sim_occupancy_df)
        if not sim_occupancy_df.empty:
            # **** FIX: Handle potential duplicate seat_ids from simulation ****
            # Keep the first employee assigned if duplicates exist
            sim_occupancy_df_unique = sim_occupancy_df.drop_duplicates(subset=['seat_id'], keep='first')
            # Create map from the unique DataFrame
            sim_map = sim_occupancy_df_unique.set_index('seat_id')['employee_id']
            # -----------------------------------------------------------------

            unavailable_sim_indices = seats_simulated_df[~seats_simulated_df['is_available']].index
            if not unavailable_sim_indices.empty:
                # Apply the map using the unique index
                seats_simulated_df.loc[unavailable_sim_indices, 'occupant_employee_id'] = \
                    seats_simulated_df.loc[unavailable_sim_indices, 'seat_id'].map(sim_map).fillna(
                        seats_simulated_df.loc[unavailable_sim_indices, 'occupant_employee_id']
                    )
                logging.debug(f"Populated occupant IDs for {len(unavailable_sim_indices)} simulated unavailable seats using unique map.")
            else:
                logging.debug("No seats marked unavailable by simulation.")
        else:
            logging.debug("Simulation occupancy DataFrame (sim_occupancy_df) was empty.")

        # Start final state from simulation result (now with simulated occupants)
        seats_current_df = seats_simulated_df.copy()

        # 2. Layer on web app bookings (takes precedence over simulation)
        web_app_bookings = real_time_bookings_store.get(date_str, {})
        logging.debug(f"Applying {len(web_app_bookings)} web app bookings for {date_str}...")
        for seat_id, emp_id in web_app_bookings.items():
            idx = seats_current_df.index[seats_current_df['seat_id'] == seat_id]
            if not idx.empty:
                base_fixed_assignee = seats_current_df.loc[idx[0], 'fixed_assignment']
                if base_fixed_assignee is None or pd.isna(base_fixed_assignee):
                    seats_current_df.loc[idx[0], 'is_available'] = False
                    seats_current_df.loc[idx[0], 'occupant_employee_id'] = emp_id
                    #logging.debug(f"Applied web booking: Seat {seat_id} -> Emp {emp_id}") # Can be noisy
            else:
                logging.warning(f"[get_current_seat_state_df] Seat ID {seat_id} from web store not found for {date_str}")

        # 3. Apply fixed assignments (final override)
        fixed_indices = seats_current_df[seats_current_df['fixed_assignment'].notna()].index
        if not fixed_indices.empty:
            logging.debug(f"Applying {len(fixed_indices)} fixed assignments for {date_str}...")
            seats_current_df.loc[fixed_indices, 'is_available'] = False
            seats_current_df.loc[fixed_indices, 'occupant_employee_id'] = seats_current_df.loc[fixed_indices, 'fixed_assignment']

        # Final type conversions
        seats_current_df['occupant_employee_id'] = pd.to_numeric(seats_current_df['occupant_employee_id'], errors='coerce').fillna(0).astype(int)
        seats_current_df['is_available'] = seats_current_df['is_available'].astype(bool)
        seats_current_df['fixed_assignment'] = seats_current_df['fixed_assignment'].fillna('')

        logging.info(f"Generated final current state for {date_str}. Available seats: {seats_current_df['is_available'].sum()}")
        return seats_current_df

    except Exception as e:
        logging.error(f"Error generating current seat state for {date_str}: {e}", exc_info=True)
        return pd.DataFrame()


# --- Routes ---

# --- Authentication & Basic Pages ---
@app.route('/')
def show_login():
    if 'employee_id' in session:
        user_type = session.get('user_type')
        if user_type == 'employee': return redirect(url_for('show_booking_page'))
        elif user_type == 'admin': return redirect(url_for('admin_dashboard'))
        else: session.clear()
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    employee_id_str = request.form.get('employee_id')
    user_type_select = request.form.get('user_type')
    if not employee_id_str or not employee_id_str.isdigit():
        flash('Please enter a valid Employee ID.', 'danger'); return redirect(url_for('show_login'))
    employee_id = int(employee_id_str)
    employee_info = get_employee_info(employee_id)
    is_configured_admin = employee_id in ADMIN_EMPLOYEE_IDS
    if not employee_info and not is_configured_admin:
        flash('Invalid Employee ID.', 'danger'); return redirect(url_for('show_login'))
    actual_role = 'admin' if is_configured_admin else 'employee'
    if user_type_select == 'employee':
        if actual_role == 'admin': flash('Admin users should log in via the "Admin" option.', 'warning'); return redirect(url_for('show_login'))
        elif not employee_info: flash('Employee record not found for this ID.', 'danger'); return redirect(url_for('show_login'))
        else: session['employee_id'] = employee_id; session['user_type'] = 'employee'; logging.info(f"Employee {employee_id} logged in."); return redirect(url_for('show_booking_page'))
    elif user_type_select == 'admin':
        if actual_role == 'admin': session['employee_id'] = employee_id; session['user_type'] = 'admin'; logging.info(f"Admin {employee_id} logged in."); return redirect(url_for('admin_dashboard'))
        else: flash('This Employee ID does not have admin privileges.', 'danger'); return redirect(url_for('show_login'))
    else: flash('Invalid user type selected.', 'danger'); return redirect(url_for('show_login'))

@app.route('/logout')
def logout():
    employee_id = session.get('employee_id'); session.clear()
    if employee_id: logging.info(f"User {employee_id} logged out.")
    flash('You have been logged out.', 'success'); return redirect(url_for('show_login'))

# --- Employee Booking Page ---
@app.route('/booking')
@login_required
def show_booking_page():
    employee_id = session['employee_id']
    employee_info = get_employee_info(employee_id)
    if not employee_info:
        flash('Could not retrieve your employee information. Please log in again.', 'danger'); session.clear(); return redirect(url_for('show_login'))
    today = datetime.today()
    available_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    floors = sorted(seats_df_base['floor'].unique().tolist()) if not seats_df_base.empty else [1]
    return render_template('booking.html', employee_info=employee_info, available_dates=available_dates, initial_date=available_dates[0], floors=floors)

# --- Admin Routes ---
@app.route('/admin')
@admin_required
def admin_dashboard():
    admin_id = session['employee_id']; last_prediction_status = "Prediction feature unavailable or file missing."
    if hasattr(be, 'prediction_week'):
        if os.path.exists(LATEST_PREDICTIONS_FILE):
            try: file_mod_time = os.path.getmtime(LATEST_PREDICTIONS_FILE); last_prediction_time = datetime.fromtimestamp(file_mod_time).strftime('%Y-%m-%d %H:%M:%S'); last_prediction_status = f"Last generated: {last_prediction_time}"
            except Exception as e: logging.warning(f"Could not read timestamp from {LATEST_PREDICTIONS_FILE}: {e}"); last_prediction_status = "Prediction file exists, timestamp unreadable."
    return render_template('admin.html', admin_id=admin_id, last_prediction_status=last_prediction_status)

# --- NEW ROUTE for Graphs ---
@app.route('/admin/graphs')
@admin_required
def admin_graphs():
    """Generates and displays graph visualizations."""
    admin_id = session['employee_id']
    graph_image_paths = {} # Store relative paths for the template

    if team_graph_global is None or team_graph_global.number_of_nodes() == 0:
        flash('Team graph data is not available or empty. Cannot generate visualizations.', 'warning')
        return render_template('admin_graphs.html', admin_id=admin_id, graph_images=graph_image_paths)

    logging.info(f"Admin {admin_id} requesting graph generation.")

    # Generate and save graphs, storing the relative paths
    # Pass the app's static folder path to the backend functions
    try:
        # Plot Overall Team Graph
        path_team = be.plot_team_graph(team_graph_global, STATIC_FOLDER)
        if path_team: graph_image_paths['Overall Team Graph'] = path_team

        # Plot Strong Connections
        path_strong = be.plot_strong_connections(team_graph_global, STATIC_FOLDER)
        if path_strong: graph_image_paths['Strong Connections (Weight >= 0.8)'] = path_strong

        # Plot Cross-Team Connections
        path_cross = be.plot_cross_team_connections(team_graph_global, STATIC_FOLDER)
        if path_cross: graph_image_paths['Cross-Team Connections'] = path_cross

        # Example: Plot a specific team subgraph (if function exists and you choose a team)
        # team_to_plot = 'Engineering' # Or get from employees_df
        # if hasattr(be, 'plot_specific_team_subgraph'):
        #     path_specific_team = be.plot_specific_team_subgraph(team_graph_global, team_to_plot, STATIC_FOLDER)
        #     if path_specific_team: graph_image_paths[f'Team: {team_to_plot}'] = path_specific_team

        logging.info(f"Generated {len(graph_image_paths)} graph images.")
        if not graph_image_paths:
             flash('Could not generate any graph images. Check server logs.', 'danger')

    except Exception as e:
        logging.error(f"Error occurred during graph generation: {e}", exc_info=True)
        flash('An error occurred while generating graph images. Check server logs.', 'danger')

    # Render the template, passing the dictionary of image paths
    return render_template('admin_graphs.html', admin_id=admin_id, graph_images=graph_image_paths)



@app.route('/api/admin/trigger_predictions', methods=['POST'])
@admin_required
def trigger_predictions():
    if not hasattr(be, 'prediction_week'): return jsonify({"success": False, "message": "Prediction function unavailable."}), 501
    admin_id = session['employee_id']; logging.info(f"Admin {admin_id} triggered prediction calculation.")
    # if not os.getenv("GOOGLE_API_KEY"): return jsonify({"success": False, "message": "Prediction service not configured."}), 500
    if not processed_data or 'employees' not in processed_data or processed_data['employees'].empty: return jsonify({"success": False, "message": "Base data missing."}), 500
    try:
        ##
        
        logging.info("Calling backend_logic.prediction_week...") 
        predictions_dict = prediction_week(processed_data, prediction_days=7)
        logging.info("backend_logic.prediction_week completed.")
        try:
            with open(LATEST_PREDICTIONS_FILE, 'wb') as f: pickle.dump(predictions_dict, f)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S'); logging.info(f"Admin {admin_id} saved predictions to {LATEST_PREDICTIONS_FILE} at {timestamp}.")
            return jsonify({"success": True, "message": f"Predictions generated and saved at {timestamp}."})
        except Exception as e: logging.error(f"Failed to save prediction results: {e}", exc_info=True); return jsonify({"success": False, "message": "Failed to save predictions."}), 500
    except Exception as e: logging.error(f"Failed during prediction_week: {e}", exc_info=True); return jsonify({"success": False, "message": f"Error during prediction: {e}"}), 500


# --- Core API Endpoints ---

@app.route('/api/office_layout')
@login_required
def get_office_layout():
    if seats_df_base.empty: return jsonify({"error": "Office layout data unavailable"}), 500
    try: layout_info = {"num_floors": int(seats_df_base['floor'].nunique()), "zones": sorted(list(seats_df_base['zone'].unique())), "floors": sorted(list(seats_df_base['floor'].unique().tolist()))}; return jsonify(layout_info)
    except Exception as e: logging.error(f"Error generating office layout API response: {e}", exc_info=True); return jsonify({"error": "Could not retrieve office layout"}), 500

# Modified to use updated get_current_seat_state_df
@app.route('/api/availability')
@login_required
def get_availability():
    """ Provides seat availability reflecting combined state (simulated + web + fixed). """
    date_str = request.args.get('date')
    if not date_str: return jsonify({"error": "Date parameter is required"}), 400
    try: target_date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError: return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
    try:
        # Use the updated helper function to get the combined state
        seats_current_df = get_current_seat_state_df(date_str, target_date) # Pass both str and datetime
        if seats_current_df.empty and not seats_df_base.empty: raise RuntimeError("Failed to generate current seat state.")
        elif seats_current_df.empty and seats_df_base.empty: return jsonify({"error": "Base seat data unavailable"}), 500
        # Select columns, ensuring NaN fixed assignments become empty strings for JSON
        seats_current_df['fixed_assignment'] = seats_current_df['fixed_assignment'].fillna('')
        seats_to_send = seats_current_df[[
            'seat_id', 'floor', 'zone', 'row', 'seat_number', 'is_window', 'is_quiet_zone',
            'is_near_kitchen', 'is_near_meeting_rooms', 'fixed_assignment',
            'is_available', 'occupant_employee_id'
        ]].to_dict('records')
        return jsonify({"seats": seats_to_send})
    except Exception as e:
        logging.error(f"Error in get_availability for {date_str}: {e}", exc_info=True)
        return jsonify({"error": "Could not retrieve seat availability due to internal error."}), 500

# Modified to use updated get_current_seat_state_df for context
@app.route('/api/book', methods=['POST'])
@login_required
def handle_book_action():
    """Handles booking or cancelling a seat BY CONSULTING THE LLM using combined state."""
    employee_id = session['employee_id']
    data = request.get_json()

    # 1. Input Validation
    if not data or not all(k in data for k in ['seat_id', 'date', 'action']): return jsonify({"success": False, "message": "Missing data"}), 400
    try: seat_id = int(data['seat_id']); date_str = data['date']; action = data['action'].lower(); target_date = datetime.strptime(date_str, '%Y-%m-%d')
    except (ValueError, TypeError) as e: return jsonify({"success": False, "message": f"Invalid input: {e}"}), 400
    if action not in ['book', 'cancel']: return jsonify({"success": False, "message": "Invalid action"}), 400

    logging.info(f"Handling '{action}' request: Seat={seat_id}, Date={date_str}, Emp={employee_id}.")
    if not hasattr(be, 'recommend_seat'): return jsonify({"success": False, "message": "Recommendation service unavailable."}), 501

    # 2. Prepare Data for LLM (using combined state)
    try:
        # Get combined current state DF using the updated function
        current_seats_df = get_current_seat_state_df(date_str, target_date)
        if current_seats_df.empty: raise RuntimeError("Could not retrieve current seat state.")
        fused_seats_df_string = current_seats_df.to_string()

        # Get web app occupancy DF (if needed specifically by data_fusion)
        web_app_occupancy_list = [{'seat_id': s_id, 'employee_id': e_id, 'date': date_str} for s_id, e_id in real_time_bookings_store.get(date_str, {}).items()]
        web_app_occupancy_df = pd.DataFrame(web_app_occupancy_list)
        if not web_app_occupancy_df.empty: web_app_occupancy_df['date'] = pd.to_datetime(web_app_occupancy_df['date'])

        hierarchy_data = processed_data.get('hierarchy', pd.DataFrame())
        # Pass the combined state DF as the primary 'seats' data
        fused_data = be.data_fusion(processed_data, current_seats_df, web_app_occupancy_df, hierarchy_data) # Adjust data_fusion if needed
        dynamic_context_employee = be.prepare_dynamic_context(fused_data, target_date, team_graph_global, employee_id)
        if not static_context_global or "Error" in static_context_global: raise RuntimeError("Static context unavailable.")
    except Exception as e:
        logging.error(f"Error preparing data for LLM call (Action: {action}, Seat: {seat_id}, Date: {date_str}): {e}", exc_info=True)
        return jsonify({"success": False, "message": "Internal error preparing recommendation request."}), 500

    # 3. Call LLM
    try:
        logging.debug(f"Calling LLM recommend_seat for action={action}, seat={seat_id}, date={date_str}")
        recommendation = recommend_seat(static_context_global, dynamic_context_employee, fused_seats_df_string, employee_id, seat_id, date_str, action)
        logging.info(f"LLM Recvd: Action={action}, Seat={seat_id}, Date={date_str}, Result: {recommendation}")
    except Exception as e:
        logging.error(f"LLM recommendation call failed (Action: {action}, Seat: {seat_id}, Date: {date_str}): {e}", exc_info=True)
        return jsonify({"success": False, "message": f"Could not get booking recommendation: {e}"}), 500

    # 4. Process LLM Response & Update State
    try:
        action_status = recommendation.get("action_status"); llm_reason = recommendation.get("reason", "No reason provided."); recommended_seat_str = recommendation.get("recommended_seat", "none")
        if action_status == "confirmed":
            update_success = apply_booking_update(date_str, seat_id, employee_id, action) # Apply to web store
            if update_success:
                logging.info(f"LLM Confirmed & Store Updated: Action={action}, Seat={seat_id}, Date={date_str}, Emp={employee_id}")
                return jsonify({"success": True, "action_status": "confirmed", "message": llm_reason, "seat_id": seat_id, "action": action})
            else: logging.error(f"Internal Error: apply_booking_update failed post-LLM confirm! Action={action}, Seat={seat_id}, Date={date_str}, Emp={employee_id}"); return jsonify({"success": False, "message": "Booking confirmed but failed to save state."}), 500
        elif action_status == "denied":
            logging.info(f"LLM Denied: Action={action}, Seat={seat_id}, Date={date_str}, Emp={employee_id}. Reason: {llm_reason}")
            recommended_seat_id = None
            if recommended_seat_str and str(recommended_seat_str).lower() != "none":
                try: recommended_seat_id = int(recommended_seat_str)
                except (ValueError, TypeError): logging.warning(f"LLM gave invalid recommended_seat: '{recommended_seat_str}'")
            return jsonify({"success": False, "action_status": "denied", "message": llm_reason, "recommended_seat": recommended_seat_id, "original_seat_id": seat_id, "original_action": action}), 200 # OK status, logical denial
        else: logging.error(f"LLM returned unexpected status: '{action_status}' (Action={action}, Seat={seat_id}, Date={date_str})"); return jsonify({"success": False, "message": "Invalid recommendation response."}), 500
    except Exception as e:
        logging.error(f"Error processing LLM response/update (Action: {action}, Seat={seat_id}, Date={date_str}): {e}", exc_info=True)
        return jsonify({"success": False, "message": "Internal error processing recommendation."}), 500

# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True) # Set debug=False in production!