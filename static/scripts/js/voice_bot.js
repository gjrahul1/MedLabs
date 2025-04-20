document.addEventListener('DOMContentLoaded', () => {
    console.log('voice_bot.js loaded, initializing...');
  
    // DOM elements
    const voiceCircle = document.getElementById('voiceCircle');
    const controlButton = document.getElementById('controlButton');
    const audioPlayer = document.getElementById('audioPlayer');
    const transcriptDiv = document.getElementById('transcriptContent');
    const statusDiv = document.getElementById('status-message');
  
    // Check for missing DOM elements
    if (!voiceCircle || !controlButton || !audioPlayer || !transcriptDiv || !statusDiv) {
        console.error('Missing required DOM elements:', {
            voiceCircle: !!voiceCircle,
            controlButton: !!controlButton,
            audioPlayer: !!audioPlayer,
            transcriptDiv: !!transcriptDiv,
            statusDiv: !!statusDiv
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
        console.error('Firebase not initialized. Ensure firebaseConfig.js is loaded.');
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
    recognition.continuous = false;
    recognition.interimResults = false;
  
    let idToken = sessionStorage.getItem('idToken');
    let recognizing = false;
    let conversationComplete = false;
  
    // Initialize authentication
    window.auth.onAuthStateChanged(async (user) => {
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
            }
        } catch (error) {
            console.error('Error in onAuthStateChanged:', error);
            statusDiv.textContent = 'Initialization failed: ' + error.message + '. Please refresh the page.';
            controlButton.disabled = true;
        }
    });
  
    // Update transcript display
    function updateTranscript(speaker, text) {
        const entry = document.createElement('p');
        entry.className = speaker.toLowerCase();
        entry.innerHTML = `<strong>${speaker}:</strong> ${text}`;
        transcriptDiv.appendChild(entry);
        transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
    }
  
    // Toggle button and visual state
    function toggleButtonState(isRecognizing) {
        recognizing = isRecognizing;
        controlButton.textContent = isRecognizing ? 'Stop Recording' : 'Start Recording';
        controlButton.classList.toggle('stop', isRecognizing);
        voiceCircle.classList.toggle('recording', isRecognizing);
        statusDiv.textContent = isRecognizing ? 'Recording...' : '';
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
                                voiceCircle.classList.remove('vibrating');
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
                                voiceCircle.classList.add('vibrating');
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
                            voiceCircle.classList.remove('vibrating');
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
                            voiceCircle.classList.add('vibrating');
                        }).catch((error) => {
                            console.error('Audio playback failed:', error);
                            statusDiv.textContent = 'Failed to play AI response audio.';
                            reject(error);
                        });
                    });
                    // Add a small delay to ensure the user hears the audio
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
  
                if (responseText.includes("Please log in to access your dashboard")) {
                    conversationComplete = true;
                    console.log('Conversation completed, setting flag to redirect to login.');
                }
                handleRedirection(data);
            } else {
                console.error('Server error:', data.error);
                statusDiv.textContent = data.error || 'Error processing transcript. Please try again.';
            }
        } catch (error) {
            console.error('Fetch error:', error);
            statusDiv.textContent = `Failed to process transcript: ${error.message}. Please check server logs or try again later.`;
        }
    }
  
    // Attach SpeechRecognition event handlers
    function attachRecognitionHandlers() {
        recognition.onstart = () => {
            toggleButtonState(true);
            console.log('Speech recognition started');
        };
  
        recognition.onend = () => {
            toggleButtonState(false);
            console.log('Speech recognition stopped');
        };
  
        recognition.onresult = async (event) => {
            let transcript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    transcript += event.results[i][0].transcript;
                }
            }
            if (transcript.trim()) {
                console.log('Transcribed text:', transcript);
                updateTranscript('User', transcript);
  
                if (!idToken) {
                    console.error('No idToken available');
                    statusDiv.textContent = 'Authentication error. Please log in again.';
                    recognition.stop();
                    return;
                }
  
                await processTranscript(transcript);
            }
        };
  
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            toggleButtonState(false);
            if (event.error === 'not-allowed') {
                statusDiv.textContent = 'Microphone access denied. Please allow access in browser settings.';
            } else if (event.error === 'no-speech') {
                statusDiv.textContent = 'No speech detected. Please try again.';
            } else {
                statusDiv.textContent = `Speech recognition error: ${event.error}`;
            }
        };
    }
  
    // Button click handler
    controlButton.addEventListener('click', async () => {
        console.log('Button clicked, recognizing:', recognizing);
        if (!idToken) {
            statusDiv.textContent = 'Please wait for authentication to complete.';
            return;
        }
  
        if (recognizing) {
            recognition.stop();
        } else {
            try {
                recognition.start();
            } catch (error) {
                console.error('Error starting recognition:', error);
                statusDiv.textContent = 'Failed to start recording: ' + error.message;
            }
        }
    });
  
    console.log('Voice bot initialization complete');
  });