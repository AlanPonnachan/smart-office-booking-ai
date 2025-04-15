// --- static/js/booking.js ---

$(document).ready(function() {
    // --- State Variables ---
    let currentDate = initialDate; // Passed from Flask template (usually today)
    let currentFloor = initialFloor; // Passed from Flask template
    let seatDataCache = {}; // Cache fetched seat data: { 'YYYY-MM-DD': [seat_objects] }
    let isProcessingAction = false; // Flag to prevent double clicks

    // --- UI Elements ---
    const seatMapDiv = $('#seat-map');
    const currentFloorDisplay = $('#current-floor-display');
    const currentDateDisplay = $('#current-date-display');
    // Optional: Add a div in your HTML to show status messages
    // <div id="booking-status" class="status-message"></div>
    const statusDiv = $('#booking-status');

    // --- Initial Load ---
    updateDisplayInfo();
    fetchAndDisplaySeats();

    // --- Event Listeners ---

    // Date Selection
    $('.date-btn').on('click', function() {
        if (isProcessingAction) return; // Prevent changing date during an action
        const newDate = $(this).data('date');
        if (newDate === currentDate) return; // No change

        $('.date-btn').removeClass('active');
        $(this).addClass('active');
        currentDate = newDate;
        updateDisplayInfo();
        fetchAndDisplaySeats(); // Fetch data for the new date
    });

    // Floor Selection
    $('#floor-select').on('change', function() {
        if (isProcessingAction) return; // Prevent changing floor during an action
        currentFloor = $(this).val();
        updateDisplayInfo();
        // Re-render using cached data for the current date, or fetch if not cached
        fetchAndDisplaySeats();
    });

    // Seat Click (Event Delegation)
    seatMapDiv.on('click', '.seat', function() {
        if (isProcessingAction) {
            alert("Please wait for the current booking action to complete.");
            return; // Prevent new action while one is processing
        }

        const seatElement = $(this);
        const seatId = seatElement.data('seat-id');

        // --- Determine Action based on Seat State ---
        const isAvailable = seatElement.hasClass('available');
        const isMyBooking = seatElement.hasClass('my-booking');
        const isFixed = seatElement.hasClass('fixed');
        const isOccupied = seatElement.hasClass('occupied');

        let action = null;
        let confirmationMessage = '';

        if (isFixed) {
            alert(`Seat ${seatId} has a fixed assignment and cannot be booked or cancelled.`);
            return;
        } else if (isAvailable) {
            action = 'book';
            confirmationMessage = `Request to book seat ${seatId} for ${currentDate}? The system will check availability and rules.`;
        } else if (isMyBooking) {
            action = 'cancel';
            confirmationMessage = `Request to cancel your booking for seat ${seatId} on ${currentDate}?`;
        } else if (isOccupied) {
            // Occupied by someone else - provide info, no action
            const occupant = seatElement.find('.employee-id-display').text();
            const occupantId = occupant.split(':').pop().trim(); // Extract ID if possible
            const alertOccupant = occupantId && occupantId !== '?' ? `Employee ${occupantId}` : 'someone else';
            alert(`Seat ${seatId} is currently occupied by ${alertOccupant} on ${currentDate}.`);
            return;
        } else {
            // Should not happen if classes are set correctly
            console.warn("Seat clicked with unknown state:", seatElement.attr('class'));
            return;
        }

        // --- Confirm and Execute Action ---
        if (confirm(confirmationMessage)) {
            handleBookingAction(seatId, action);
        }
    });

    // --- Core Functions ---

    /** Updates the displayed date and floor information */
    function updateDisplayInfo() {
        currentFloorDisplay.text(currentFloor);
        currentDateDisplay.text(currentDate);
    }

    /** Clears status messages */
    function clearStatus() {
        if(statusDiv.length) statusDiv.text('').removeClass('info success error');
    }

    /** Displays status messages */
    function showStatus(message, type = 'info') {
         clearStatus();
         if(statusDiv.length) statusDiv.text(message).addClass(type);
    }


    /** Fetches seat availability for the current date and renders the map */
    async function fetchAndDisplaySeats() {
        clearStatus(); // Clear previous status messages
        const cacheKey = currentDate; // Cache key is just the date now
        seatMapDiv.html('Loading seats...'); // Show loading indicator

        // Try using cached data first
        if (seatDataCache[cacheKey]) {
            console.log("Using cached data for Date:", cacheKey, "Floor:", currentFloor);
            renderSeats(seatDataCache[cacheKey], currentFloor);
            return;
        }

        // Fetch new data if not in cache
        console.log("Fetching data for Date:", cacheKey, "Floor:", currentFloor);
        try {
            // API call no longer includes 'mode'
            const response = await fetch(`/api/availability?date=${currentDate}`);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({})); // Try to get error msg
                throw new Error(`HTTP error! Status: ${response.status} - ${errorData.message || 'Server error'}`);
            }
            const data = await response.json();
            if (!data || !data.seats) {
                 throw new Error("Invalid data format received from server.");
            }

            seatDataCache[cacheKey] = data.seats; // Cache the fetched data
            renderSeats(data.seats, currentFloor); // Render the seats for the selected floor

        } catch (error) {
            console.error('Error fetching seat data:', error);
            seatMapDiv.html(`<p class="error">Could not load seat data: ${error.message}. Please try refreshing.</p>`);
        }
    }

    /**
     * Renders the seat map HTML based on seat data for a specific floor.
     * @param {Array} allSeats - Array of seat objects for the current date.
     * @param {string|number} floor - The floor number to render.
     */
    function renderSeats(allSeats, floor) {
        seatMapDiv.empty(); // Clear previous map content

        const seatsOnFloor = allSeats.filter(seat => seat.floor == floor);

        if (seatsOnFloor.length === 0) {
            seatMapDiv.html('<p>No seats found for this floor.</p>');
            return;
        }

        // --- Group seats by zone, then row for structured layout ---
        const zones = {};
        seatsOnFloor.forEach(seat => {
            if (!zones[seat.zone]) zones[seat.zone] = {};
            if (!zones[seat.zone][seat.row]) zones[seat.zone][seat.row] = [];
            zones[seat.zone][seat.row].push(seat);
        });

        const sortedZoneNames = Object.keys(zones).sort();

        // --- Create Grid Layout (Adjust columns based on zone count) ---
        const numZones = sortedZoneNames.length;
        let gridCols = numZones <= 2 ? numZones : (numZones <= 4 ? 2 : 4); // Example: max 4 cols
        seatMapDiv.css('grid-template-columns', `repeat(${gridCols}, 1fr)`);

        // --- Generate HTML for each zone, row, and seat ---
        sortedZoneNames.forEach(zoneName => {
            const zoneDiv = $('<div class="zone"></div>').append(`<div class="zone-title">Zone ${zoneName}</div>`);
            const sortedRowNumbers = Object.keys(zones[zoneName]).map(Number).sort((a, b) => a - b);

            sortedRowNumbers.forEach(rowNum => {
                const rowDiv = $('<div class="row"></div>');
                const seatsInRow = zones[zoneName][rowNum].sort((a, b) => a.seat_number - b.seat_number);

                seatsInRow.forEach(seat => {
                    const seatDiv = $(`<div class="seat" data-seat-id="${seat.seat_id}"></div>`);
                    seatDiv.append(`<span class="seat-id-display">S${seat.seat_id}</span>`);

                    let seatClass = 'available'; // Default class
                    let occupantInfo = '';

                    // Determine seat status and apply appropriate class
                    if (seat.fixed_assignment) {
                        seatClass = 'fixed';
                        // In real app, might map fixed_assignment ID to name if needed
                        occupantInfo = `Fixed: ${seat.fixed_assignment}`;
                    } else if (!seat.is_available) {
                         // Check if occupied by the current user
                        if (seat.occupant_employee_id == employeeId) { // employeeId is global from template
                            seatClass = 'my-booking';
                            occupantInfo = `Me: ${seat.occupant_employee_id}`;
                        } else {
                            seatClass = 'occupied';
                            occupantInfo = `E: ${seat.occupant_employee_id || '?'}`; // Show employee ID
                        }
                    } // else: it remains 'available'

                    seatDiv.addClass(seatClass);
                    if (occupantInfo) {
                         seatDiv.append(`<span class="employee-id-display">${occupantInfo}</span>`);
                    }

                    // --- Add Tooltip for seat details ---
                    let titleText = `Seat ${seat.seat_id} (F${seat.floor}-Z${seat.zone}-R${seat.row}-N${seat.seat_number})`;
                    if (seat.is_window) titleText += ' | Window';
                    if (seat.is_quiet_zone) titleText += ' | Quiet';
                    // Add other attributes if needed
                    if (occupantInfo) titleText += ` | ${occupantInfo}`;
                    else if (seatClass === 'available') titleText += ' | Available';
                    seatDiv.attr('title', titleText);

                    rowDiv.append(seatDiv);
                });
                zoneDiv.append(rowDiv);
            });
            seatMapDiv.append(zoneDiv);
        });
    }

    /**
     * Handles the API request to book or cancel a seat, processing the LLM response.
     * @param {number} seatId - The ID of the seat to act upon.
     * @param {'book' | 'cancel'} action - The requested action.
     */
    async function handleBookingAction(seatId, action) {
        console.log(`Requesting to ${action} seat ${seatId} for ${currentDate}`);
        isProcessingAction = true; // Set flag
        showStatus(`Processing request to ${action} seat ${seatId}...`, 'info');

        const cacheKey = currentDate; // Cache key for potential invalidation

        try {
            const response = await fetch('/api/book', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    seat_id: seatId,
                    date: currentDate,
                    action: action
                }),
            });

            // Try to parse JSON regardless of response.ok to get potential error messages
            const result = await response.json();

            console.log("API Response:", result); // Log the full response for debugging

            // --- Process Response based on LLM Decision ---
            if (response.ok && result.success && result.action_status === "confirmed") {
                // ACTION CONFIRMED BY LLM & Server
                showStatus(`Success: ${result.message || `Seat ${seatId} ${action === 'book' ? 'booked' : 'cancelled'}.`}`, 'success');
                alert(`Request successful: ${result.message || `Seat ${seatId} ${action === 'book' ? 'booked' : 'cancelled'}.`}`);
                delete seatDataCache[cacheKey]; // Invalidate cache for this date
                fetchAndDisplaySeats(); // Refresh the seat map display

            } else if (result.action_status === "denied") {
                // ACTION DENIED BY LLM
                console.log("Booking denied by recommendation system.", result);
                let denialMessage = `Request denied: ${result.message || 'The system could not fulfill your request.'}`;
                const recommendedSeat = result.recommended_seat; // null or integer

                if (recommendedSeat !== null && recommendedSeat !== undefined) {
                    // Offer the recommendation
                    // Use setTimeout to allow the current processing status to clear before confirm dialog
                    setTimeout(() => {
                         if (confirm(`${denialMessage}\n\nThe system recommends seat ${recommendedSeat} instead. Would you like to try booking seat ${recommendedSeat}?`)) {
                            // User accepted recommendation
                            showStatus(`Attempting recommended seat ${recommendedSeat}...`, 'info');
                             // Make a NEW request for the recommended seat (always 'book')
                            handleBookingAction(recommendedSeat, 'book');
                             // Note: isProcessingAction is still true until this new call completes or fails
                             return; // Exit current handler flow
                        } else {
                            // User declined recommendation
                            showStatus(`Request denied for seat ${seatId}. Recommendation declined.`, 'error');
                            alert(`Request for seat ${seatId} was denied. You chose not to book the recommended seat.`);
                            isProcessingAction = false; // Reset flag as this path ends here
                            // Optionally refresh view if needed, although no state changed from this action
                            // fetchAndDisplaySeats();
                        }
                    }, 100); // Short delay before confirm

                } else {
                    // Denied without a recommendation
                    showStatus(`Request denied for seat ${seatId}. ${result.message}`, 'error');
                    alert(`Request denied: ${result.message}`);
                    isProcessingAction = false; // Reset flag
                     // Refresh view as underlying state might have changed even if denied
                     delete seatDataCache[cacheKey]; // Invalidate cache
                     fetchAndDisplaySeats();
                }
                // If a recommendation was offered and accepted, isProcessingAction remains true
                // If denied or recommendation declined, reset the flag here IF not already done in setTimeout callback
                if (recommendedSeat === null || recommendedSeat === undefined) {
                     isProcessingAction = false;
                }


            } else {
                 // Handle other errors (e.g., server error 500, network issue, unexpected JSON)
                 let errorMsg = result.message || `Failed to ${action} seat ${seatId}.`;
                 if (!response.ok) { errorMsg += ` (Server Status: ${response.status})`; }
                 throw new Error(errorMsg);
            }

             // Reset flag only if not handling a recommendation acceptance which calls handleBookingAction again
             // This is tricky with the async nature of confirm in setTimeout. A more robust approach might use promises.
             // For now, resetting it here if not denied with accepted recommendation
             if(result.action_status !== 'denied' || result.recommended_seat === null || result.recommended_seat === undefined){
                 //isProcessingAction = false; // Moved to finally block
             }


        } catch (error) {
            console.error('Error performing booking action:', error);
            showStatus(`Error: ${error.message}`, 'error');
            alert(`Error: ${error.message}`);
            // Force refresh on error just in case state is inconsistent
            delete seatDataCache[cacheKey];
            fetchAndDisplaySeats();
            //isProcessingAction = false; // Reset flag on error, moved to finally
        } finally {
            // Ensure the flag is always reset UNLESS a recursive call for recommendation is pending
            // This simple flag isn't perfect for the recursive case, but helps prevent basic double-clicks.
            // A better solution involves disabling UI elements or more complex state management.
             // Let's reset it here for now, acknowledging the slight race condition possibility with the confirm dialog.
             isProcessingAction = false;
             console.log("isProcessingAction reset to false");
        }
    }

}); // End of document ready