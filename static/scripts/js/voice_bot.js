document.addEventListener('DOMContentLoaded', () => {
    console.log('voice_bot.js loaded, initializing...');
  
    // DOM elements
    const controlButton = document.getElementById('controlButton');
    const audioPlayer = document.getElementById('audioPlayer');
    const transcriptDiv = document.getElementById('transcriptContent');
    const statusDiv = document.getElementById('status-message');
    const transcriptPanel = document.getElementById('transcriptPanel');
  
    // Check for missing DOM elements
    if (!controlButton || !audioPlayer || !transcriptDiv || !statusDiv || !transcriptPanel) {
        console.error('Missing required DOM elements:', {
            controlButton: !!controlButton,
            audioPlayer: !!audioPlayer,
            transcriptDiv: !!transcriptDiv,
            statusDiv: !!statusDiv,
            transcriptPanel: !!transcriptPanel
        });
        alert('Error: Required elements not found. Please check the HTML.');
        return;
    }
  
    // Ensure button is disabled initially
    controlButton.disabled = true;
    controlButton.textContent = 'Initializing...';
    statusDiv.textContent = 'Initializing, please wait...';
  
    // Check Firebase and auth
    if (!window.firebase || !window.auth) {
        console.error('Firebase not initialized. Ensure Firebase SDK is loaded.');
        statusDiv.textContent = 'Authentication service unavailable. Please try again later.';
        controlButton.disabled = true;
        return;
    }
  
    // Access server-side data
    const { uid, patientName } = window.appData || {};
    console.log('Server data:', { uid, patientName });
  
    // Initialize SpeechRecognition for English
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.error('SpeechRecognition not supported in this browser');
        statusDiv.textContent = 'Your browser does not support speech recognition. Please use a modern browser like Chrome.';
        controlButton.disabled = true;
        return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    console.log(`SpeechRecognition language set to: ${recognition.lang}`);
    recognition.continuous = true; // Enable continuous recording to prevent auto-stop on pause
    recognition.interimResults = false;
  
    let idToken = sessionStorage.getItem('idToken');
    let recognizing = false;
    let conversationComplete = false;
    let userInitiatedStop = false; // Flag to track if stop was initiated by user
    let currentTranscript = ''; // Store transcript during recording
  
    // Update transcript display
    function updateTranscript(speaker, text) {
        console.log(`Updating transcript - Speaker: ${speaker}, Text: ${text}`);
        const entry = document.createElement('p');
        entry.className = speaker.toLowerCase();
        entry.innerHTML = `<strong>${speaker}:</strong> <span>${text}</span>`;
        transcriptDiv.appendChild(entry);
  
        // Limit to maxMessages AI-User pairs (2 messages per pair)
        const maxMessages = 3;
        const messages = transcriptDiv.getElementsByTagName('p');
        console.log(`Current number of messages: ${messages.length}`);
        if (messages.length > maxMessages * 2) {
            transcriptDiv.removeChild(messages[0]);
            console.log('Removed oldest message to maintain limit');
        }
        transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
        console.log('Transcript updated and scrolled to bottom');
    }
  
    // Toggle button and visual state
    function toggleButtonState(isRecognizing) {
        recognizing = isRecognizing;
        controlButton.textContent = isRecognizing ? 'Stop Recording' : 'Start Recording';
        controlButton.classList.toggle('stop', isRecognizing);
        statusDiv.textContent = isRecognizing ? 'Recording...' : '';
        // Toggle recording animation on transcript panel
        if (isRecognizing) {
            transcriptPanel.classList.add('recording');
        } else {
            transcriptPanel.classList.remove('recording');
        }
        console.log(`Button state updated, recognizing: ${isRecognizing}, disabled: ${controlButton.disabled}`);
    }
  
    // Function to handle redirection if specified in the response
    function handleRedirection(data) {
        if (data.redirect && conversationComplete) {
            console.log(`Redirecting to: ${data.redirect}`);
            // Redirect to /login instead of /dashboard
            fetch('/logout', {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${idToken}`,
                    'Content-Type': 'application/json'
                }
            }).then(response => {
                if (response.ok) {
                    window.location.href = '/login';
                } else {
                    console.error('Logout failed:', response.statusText);
                    statusDiv.textContent = 'Failed to log out. Please try logging in again.';
                    window.location.href = '/login';
                }
            }).catch(error => {
                console.error('Logout error:', error);
                statusDiv.textContent = 'Error logging out: ' + error.message;
                window.location.href = '/login';
            });
        } else if (data.redirect) {
            console.log(`Redirect requested to ${data.redirect}, but conversation is not complete yet.`);
        }
    }
  
    // Function to initiate conversation on page load with retry logic
    async function initiateConversation(retryCount = 3, delay = 2000) {
        if (!idToken) {
            console.error('No idToken available for initiating conversation');
            statusDiv.textContent = 'Authentication failed. Please try again.';
            return;
        }
        for (let attempt = 1; attempt <= retryCount; attempt++) {
            try {
                console.log(`Initiating conversation with empty transcript (Attempt ${attempt}/${retryCount})...`);
                const response = await fetch('/start_conversation', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${idToken}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ transcript: '' })
                });
  
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
                }
  
                const data = await response.json();
                console.log('Initiation response:', data);
                if (data.success) {
                    updateTranscript('AI', data.response);
                    if (data.audio_url) {
                        console.log('Playing initial greeting audio:', data.audio_url);
                        audioPlayer.src = data.audio_url;
                        await new Promise((resolve, reject) => {
                            audioPlayer.onended = () => {
                                console.log('Initial greeting audio finished playing.');
                                resolve();
                            };
                            audioPlayer.onerror = (error) => {
                                console.error('Initial greeting audio playback failed:', error);
                                statusDiv.textContent = 'Failed to play AI greeting audio. Please try refreshing the page.';
                                reject(error);
                            };
                            audioPlayer.onloadedmetadata = () => {
                                console.log(`Audio duration: ${audioPlayer.duration} seconds`);
                            };
                            audioPlayer.play().then(() => {
                                console.log('Playing initial greeting audio');
                            }).catch((error) => {
                                console.error('Initial greeting audio playback failed:', error);
                                statusDiv.textContent = 'Failed to play AI greeting audio. Please try refreshing the page.';
                                reject(error);
                            });
                        });
                        // Add a small delay to ensure the user hears the audio
                        await new Promise(resolve => setTimeout(resolve, 500));
                    }
                    handleRedirection(data);
                    return;
                } else {
                    throw new Error(data.error || 'Error initiating conversation');
                }
            } catch (error) {
                console.error(`Initiation fetch error (Attempt ${attempt}/${retryCount}):`, error);
                if (attempt === retryCount) {
                    statusDiv.textContent = `Failed to initiate conversation after ${retryCount} attempts: ${error.message}. Please refresh the page or try again later.`;
                } else {
                    console.log(`Retrying in ${delay}ms...`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }
    }
  
    // Function to process the transcript with the server
    async function processTranscript(transcript) {
        try {
            console.log('Sending transcript to /start_conversation...');
            // Add processing animation
            transcriptPanel.classList.add('processing');
            const response = await fetch('/start_conversation', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${idToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ transcript })
            });
  
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }
  
            const data = await response.json();
            console.log('Received server response:', data);
            if (data.success) {
                // Remove processing animation
                transcriptPanel.classList.remove('processing');
                // Update the response message to suggest logging in
                let responseText = data.response;
                if (responseText.includes("Redirecting you to the dashboard")) {
                    responseText = responseText.replace(
                        "Redirecting you to the dashboard for further details.",
                        "Please log in to access your dashboard and view further details."
                    );
                }
                updateTranscript('AI', responseText);
  
                if (data.audio_url) {
                    console.log('Playing audio:', data.audio_url);
                    audioPlayer.src = data.audio_url;
                    // Use a Promise to wait for the audio to finish playing
                    await new Promise((resolve, reject) => {
                        audioPlayer.onended = () => {
                            console.log('Response audio finished playing.');
                            resolve();
                        };
                        audioPlayer.onerror = (error) => {
                            console.error('Audio playback failed:', error);
                            statusDiv.textContent = 'Failed to play AI response audio.';
                            reject(error);
                        };
                        audioPlayer.onloadedmetadata = () => {
                            console.log(`Audio duration: ${audioPlayer.duration} seconds`);
                        };
                        audioPlayer.play().then(() => {
                            console.log('Playing response audio');
                        }).catch((error) => {
                            console.error('Audio playback failed:', error);
                            statusDiv.textContent = 'Failed to play AI response audio.';
                            reject(error);
                        });
                    });
                    // Add a small delay to ensure the user hears the audio
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
  
                // Check for doctor assignment message and redirect only at the end
                if (responseText.includes("You have been assigned to") && responseText.includes("Please log in to view details.")) {
                    conversationComplete = true;
                    console.log('Doctor assigned, setting flag to redirect to login.');
                    handleRedirection(data);
                }
            } else {
                console.error('Server error:', data.error);
                statusDiv.textContent = data.error || 'Error processing transcript. Please try again.';
                transcriptPanel.classList.remove('processing');
            }
        } catch (error) {
            console.error('Fetch error:', error);
            statusDiv.textContent = `Failed to process transcript: ${error.message}. Please check server logs or try again later.`;
            transcriptPanel.classList.remove('processing');
        }
    }
  
    // Initialize authentication with timeout
    const authTimeout = setTimeout(() => {
        console.error('Authentication timeout: onAuthStateChanged did not fire within 10 seconds.');
        statusDiv.textContent = 'Voice bot not initialized: Authentication timed out. Please refresh the page.';
        controlButton.disabled = true;
        controlButton.textContent = 'Initialization Failed';
    }, 10000); // 10 seconds timeout
  
    try {
        window.auth.onAuthStateChanged(async (user) => {
            clearTimeout(authTimeout); // Clear timeout on success or failure
            try {
                if (user) {
                    idToken = await user.getIdToken(true);
                    sessionStorage.setItem('idToken', idToken);
                    console.log('ID token retrieved:', idToken.substring(0, 20) + '...');
                    controlButton.disabled = false;
                    controlButton.textContent = 'Start Recording';
                    statusDiv.textContent = '';
                    attachRecognitionHandlers();
                    initiateConversation();
                } else {
                    console.error('No user authenticated. Redirecting to login.');
                    statusDiv.textContent = 'Please log in to continue.';
                    window.location.href = '/login';
                    controlButton.disabled = true;
                    controlButton.textContent = 'Login Required';
                }
            } catch (error) {
                console.error('Error in onAuthStateChanged:', error);
                statusDiv.textContent = 'Initialization failed: ' + error.message + '. Please refresh the page.';
                controlButton.disabled = true;
                controlButton.textContent = 'Initialization Failed';
            }
        });
    } catch (error) {
        console.error('Error setting up onAuthStateChanged:', error);
        clearTimeout(authTimeout);
        statusDiv.textContent = 'Failed to set up authentication: ' + error.message + '. Please refresh the page.';
        controlButton.disabled = true;
        controlButton.textContent = 'Initialization Failed';
    }
  
    // Attach SpeechRecognition event handlers
    function attachRecognitionHandlers() {
        recognition.onstart = () => {
            toggleButtonState(true);
            userInitiatedStop = false; // Reset flag when recording starts
            currentTranscript = ''; // Reset transcript when recording starts
            console.log('Speech recognition started');
        };
  
        recognition.onend = () => {
            if (!userInitiatedStop) {
                // If stop was not user-initiated, restart recognition to keep it continuous
                console.log('Speech recognition ended unexpectedly, restarting...');
                try {
                    recognition.start();
                } catch (error) {
                    console.error('Error restarting recognition:', error);
                    toggleButtonState(false);
                    statusDiv.textContent = 'Failed to restart recording: ' + error.message;
                }
            } else {
                // If stop was user-initiated, update the button state and process the transcript
                toggleButtonState(false);
                console.log('Speech recognition stopped by user');
                // Process the collected transcript only if there is one
                if (currentTranscript.trim()) {
                    updateTranscript('User', currentTranscript);
                    if (!idToken) {
                        console.error('No idToken available');
                        statusDiv.textContent = 'Authentication error. Please log in again.';
                        return;
                    }
                    processTranscript(currentTranscript);
                    currentTranscript = ''; // Reset after processing
                }
            }
        };
  
        recognition.onresult = async (event) => {
            let transcriptPiece = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    transcriptPiece += event.results[i][0].transcript;
                }
            }
            if (transcriptPiece.trim()) {
                console.log('Transcribed piece:', transcriptPiece);
                // Append to the current transcript instead of processing immediately
                currentTranscript += (currentTranscript ? ' ' : '') + transcriptPiece;
                console.log('Current transcript:', currentTranscript);
            }
        };
  
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            if (event.error === 'not-allowed') {
                statusDiv.textContent = 'Microphone access denied. Please allow access in browser settings.';
                toggleButtonState(false);
                userInitiatedStop = true; // Mark as user-initiated to prevent restart
            } else if (event.error === 'no-speech') {
                console.log('No speech detected, continuing to listen...');
                // Do not toggle button state, let recording continue
            } else {
                statusDiv.textContent = `Speech recognition error: ${event.error}`;
                toggleButtonState(false);
                userInitiatedStop = true; // Mark as user-initiated to prevent restart
            }
        };
    }
  
    // Button click handler
    controlButton.addEventListener('click', async () => {
        console.log('Button click event triggered, recognizing:', recognizing, 'disabled:', controlButton.disabled);
        if (!idToken) {
            console.warn('idToken not available, cannot proceed with recording');
            statusDiv.textContent = 'Please wait for authentication to complete.';
            return;
        }
  
        if (recognizing) {
            userInitiatedStop = true; // Mark stop as user-initiated
            recognition.stop();
            console.log('Stop recording clicked');
        } else {
            try {
                console.log('Start recording clicked');
                recognition.start();
            } catch (error) {
                console.error('Error starting recognition:', error);
                statusDiv.textContent = 'Failed to start recording: ' + error.message;
            }
        }
    });
  
    console.log('Voice bot initialization complete');
});