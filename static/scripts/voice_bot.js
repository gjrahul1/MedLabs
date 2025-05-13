console.log("voice_bot.js loaded:", new Date().toISOString());

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let userStopped = false;
let audioQueue = [];
let responseQueue = [];
let isPlaying = false;
let statusElement = null;
let transcriptContentElement = null;
let transcriptPanel = null;
let controlButton = null;
let isProcessing = false;
let hasUserInteracted = false; // Track user interaction for autoplay

window.appData = {
    sessionId: null,
    transcript: "",
    medicalData: {
        symptoms: [],
        severity: [],
        duration: [],
        triggers: []
    },
    currentState: "INITIAL",
    authToken: null,
    uid: null,
    conversationStarted: false
};

// Generate a new sessionId on page load/refresh
function generateSessionId() {
    return Date.now().toString();
}

// Initialize the voice bot
function initializeVoiceBot() {
    console.log("Voice bot initializing:", new Date().toISOString());

    window.appData.sessionId = generateSessionId();
    console.log("Generated new sessionId:", window.appData.sessionId);

    statusElement = document.getElementById('status-message');
    transcriptContentElement = document.getElementById('transcriptContent');
    transcriptPanel = document.getElementById('transcriptPanel');
    controlButton = document.getElementById('controlButton');

    if (!statusElement) {
        console.error("Status element not found. Ensure <div id='status-message'> exists in the HTML.");
    }
    if (!transcriptContentElement) {
        console.error("Transcript content element not found. Ensure <div id='transcriptContent'> exists in the HTML.");
    }
    if (!transcriptPanel) {
        console.error("Transcript panel not found. Ensure <div id='transcriptPanel'> exists in the HTML.");
    }
    if (!controlButton) {
        console.error("Control button not found. Ensure <button id='controlButton'> exists in the HTML.");
        return;
    }

    controlButton.disabled = false;
    controlButton.style.display = 'none'; // Hide the button initially
    updateButtonStates();
    console.log("Control button initialized: text='Start Recording', class='start'");

    updateStatus("Click 'Play' to start the conversation");
    updateTranscript('ai', "What symptoms are you experiencing?");
    // Create a "Play" button to initiate audio playback
    const playButton = document.createElement('button');
    playButton.textContent = 'Play';
    playButton.className = 'control-button start';
    playButton.style.position = 'fixed';
    playButton.style.bottom = '20px';
    playButton.style.left = '50%';
    playButton.style.transform = 'translateX(-50%)';
    playButton.onclick = async () => {
        hasUserInteracted = true;
        playButton.remove(); // Remove the Play button
        controlButton.style.display = 'block'; // Show the Start Recording button
        updateStatus("Conversation started, click 'Start Recording' to respond");
        if (!window.appData.conversationStarted) {
            window.appData.conversationStarted = true;
            console.log("Starting conversation on first user interaction");
            startConversation();
        }
    };
    document.body.appendChild(playButton);

    setupSpeechRecognition();
    console.log("Waiting for user interaction to enable recording");

    window.voiceBotInitialized = true;
    console.log("Voice bot initialized successfully");
}

// Update status display
function updateStatus(status) {
    if (statusElement) {
        statusElement.textContent = `Status: ${status}`;
    }
    console.log("Status:", status);
    if (transcriptPanel) {
        if (status === "Recording...") {
            transcriptPanel.classList.add('recording');
        } else {
            transcriptPanel.classList.remove('recording');
        }
        if (status === "Processing...") {
            transcriptPanel.classList.add('processing');
        } else {
            transcriptPanel.classList.remove('processing');
        }
    }
}

// Update transcript display with optional playing indicator
function updateTranscript(speaker, message, isPlaying = false) {
    if (transcriptContentElement) {
        const messageElement = document.createElement('div');
        messageElement.className = speaker === 'user' ? 'user' : 'ai';
        messageElement.textContent = `${speaker === 'user' ? 'You' : 'AI'}: ${message}${isPlaying ? ' (Playing...)' : ''}`;
        transcriptContentElement.appendChild(messageElement);
        transcriptContentElement.scrollTop = transcriptContentElement.scrollHeight;
        console.log(`Transcript updated - ${speaker}: ${message}${isPlaying ? ' (Playing...)' : ''}`);
        // Force DOM repaint to ensure visibility
        transcriptContentElement.style.display = 'none';
        transcriptContentElement.offsetHeight; // Trigger reflow
        transcriptContentElement.style.display = 'block';
    } else {
        console.error("Cannot update transcript: transcriptContentElement is null");
    }
}

// Check microphone permissions
async function checkMicPermission() {
    try {
        const permissionStatus = await navigator.permissions.query({ name: 'microphone' });
        if (permissionStatus.state === 'denied') {
            updateStatus("Microphone access denied. Please allow microphone access to proceed.");
            return false;
        } else if (permissionStatus.state === 'prompt') {
            updateStatus("Please grant microphone access to continue.");
        }
        return true;
    } catch (error) {
        console.error("Error checking microphone permission:", error);
        updateStatus("Error checking microphone permission. Please ensure microphone access is enabled.");
        return false;
    }
}

// Update button states
function updateButtonStates() {
    if (controlButton) {
        controlButton.textContent = isRecording ? "Stop Recording" : "Start Recording";
        controlButton.classList.remove('start', 'stop');
        controlButton.classList.add(isRecording ? 'stop' : 'start');
        controlButton.disabled = false;
        console.log(`Control button updated: text='${controlButton.textContent}', disabled=${controlButton.disabled}, class='${controlButton.className}'`);
    }
}

// Setup media recorder for capturing audio
async function setupSpeechRecognition() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
        mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
                console.log("Audio chunk captured, size:", event.data.size);
            } else {
                console.warn("Empty audio chunk received");
            }
        };

        mediaRecorder.onstop = async () => {
            console.log("MediaRecorder stopped, processing audio...");
            if (audioChunks.length === 0) {
                console.error("No audio data captured");
                updateStatus("Error: No audio data captured. Please try again.");
                isRecording = false;
                updateButtonStates();
                if (!isPlaying && !isProcessing && !userStopped) {
                    startRecognition();
                }
                return;
            }
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            console.log("Audio blob created, size:", audioBlob.size);
            audioChunks = []; // Reset chunks for next recording
            await sendAudioToBackend(audioBlob);
        };

        mediaRecorder.onerror = (event) => {
            console.error("MediaRecorder error:", event.error);
            updateStatus(`Recording error: ${event.error}`);
            isRecording = false;
            updateButtonStates();
            if (!isPlaying && !isProcessing && !userStopped) {
                startRecognition();
            }
        };

        console.log("MediaRecorder setup complete with MIME type:", mimeType);
    } catch (error) {
        console.error("Error setting up media recorder:", error);
        updateStatus("Error accessing microphone. Please ensure microphone access is enabled.");
    }
}

// Start recording
function startRecognition() {
    if (mediaRecorder && !isRecording) {
        userStopped = false;
        audioChunks = []; // Clear previous audio chunks
        try {
            mediaRecorder.start(1000); // Capture audio in 1-second chunks
            isRecording = true;
            updateStatus("Recording...");
            updateButtonStates();
            console.log("Recording started: controlButton text='Stop Recording', class='stop'");
            hasUserInteracted = true; // User interaction enables autoplay
        } catch (error) {
            console.error("Error starting MediaRecorder:", error);
            updateStatus("Error starting recording: " + error.message);
            isRecording = false;
            updateButtonStates();
        }
    } else if (!mediaRecorder) {
        console.error("MediaRecorder not initialized");
        updateStatus("Error: MediaRecorder not initialized. Please check microphone setup.");
    }
}

// Stop recording
function stopRecognition() {
    if (mediaRecorder && isRecording) {
        userStopped = true;
        try {
            mediaRecorder.stop();
            isRecording = false;
            updateStatus("Processing...");
            updateButtonStates();
            console.log("Recording stopped manually: controlButton text='Start Recording', class='start'");
        } catch (error) {
            console.error("Error stopping MediaRecorder:", error);
            updateStatus("Error stopping recording: " + error.message);
            isRecording = false;
            updateButtonStates();
        }
    }
}

// Send audio to backend for transcription using Whisper
async function sendAudioToBackend(audioBlob) {
    if (audioBlob.size < 1024) {
        console.error("Audio blob too small:", audioBlob.size);
        updateStatus("Error: Audio too small or empty. Please try again.");
        if (!isPlaying && !isProcessing && !userStopped) {
            startRecognition();
        }
        return;
    }

    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    try {
        console.log("Sending audio to backend for transcription, blob size:", audioBlob.size);
        const response = await fetch('/transcribe', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${window.appData.authToken}`
            },
            body: formData,
            signal: AbortSignal.timeout(15000) // 15-second timeout
        });

        console.log("Transcription response status:", response.status);
        const responseData = await response.json();
        console.log("Transcription response data:", responseData);

        if (!response.ok) {
            throw new Error(responseData.error || "Failed to transcribe audio");
        }

        const transcript = responseData.transcript?.trim();
        if (!transcript || transcript.length < 2) {
            throw new Error("Transcription too short or empty");
        }

        console.log("Transcription received from backend:", transcript);
        window.appData.transcript = transcript;
        updateTranscript('user', transcript); // Display user input in transcript panel
        userStopped = false; // Reset userStopped after processing transcription
        await startConversation(transcript);
    } catch (error) {
        console.error("Error sending audio to backend:", error);
        updateStatus(`Error: ${error.message}`);
        window.appData.transcript = "Error in transcription";
        updateTranscript('user', "Error: Unable to transcribe audio");
        userStopped = false; // Reset userStopped even on error
        if (!isPlaying && !isProcessing && !userStopped) {
            startRecognition();
        }
    }
}

// Start conversation with the server
async function startConversation(transcript = "") {
    if (isProcessing) {
        console.log("Request already in progress, skipping...");
        return;
    }

    isProcessing = true;
    try {
        // Clear queues to prevent multiple audios from playing
        audioQueue = [];
        responseQueue = [];
        console.log("Cleared audio and response queues to prevent overlap");

        const payload = {
            sessionId: window.appData.sessionId,
            transcript: transcript,
            medicalData: window.appData.medicalData,
            currentState: window.appData.currentState || "INITIAL"
        };
        console.log("Sending conversation request:", payload);

        const response = await fetch('/start_conversation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${window.appData.authToken}`
            },
            body: JSON.stringify(payload)
        });

        const responseData = await response.json();
        if (!response.ok) {
            throw new Error(responseData.error || "Failed to start conversation");
        }

        console.log("Conversation response:", responseData);
        window.appData.medicalData = responseData.medical_data;
        window.appData.currentState = responseData.nextState;
        updateStatus("Ready");
        // Only update transcript if this isn't the initial prompt (already displayed)
        if (transcript !== "") {
            updateTranscript('ai', responseData.response, true); // Indicate audio is playing
        }

        if (responseData.response) {
            responseQueue.push(responseData.response);
        } else {
            throw new Error("No response text received from backend");
        }

        // Preload audio if URL is provided
        if (responseData.audio_url) {
            console.log("Audio URL received:", responseData.audio_url);
            // Validate audio URL before adding to queue
            try {
                const audioResponse = await fetch(responseData.audio_url, { method: 'HEAD' });
                if (audioResponse.ok) {
                    // Preload the audio
                    const audio = new Audio(responseData.audio_url);
                    audio.crossOrigin = "anonymous";
                    audio.preload = "auto";
                    audio.load();
                    console.log("Audio preloaded:", responseData.audio_url);
                    audioQueue.push(responseData.audio_url);
                    console.log("Audio URL validated successfully, added to queue");
                } else {
                    console.warn("Audio URL inaccessible, status:", audioResponse.status);
                    audioQueue.push(null);
                }
            } catch (error) {
                console.error("Error validating audio URL:", error.message);
                console.error("Falling back to local audio generation due to validation error");
                audioQueue.push(null);
            }
        } else {
            console.warn("No audio URL received, falling back to local audio generation");
            audioQueue.push(null); // Push null to indicate local generation
        }

        // Handle redirect after audio playback
        if (responseData.redirect) {
            console.log("Redirect received in responseData:", responseData.redirect);
            // Disable further interactions
            controlButton.disabled = true;
            controlButton.style.display = 'none';
            updateStatus("Redirecting to dashboard...");

            // Play the audio if available, then redirect
            if (audioQueue.length > 0) {
                const audio = new Audio(audioQueue[0]);
                audio.crossOrigin = "anonymous";
                try {
                    await audio.play();
                    audio.onended = () => {
                        console.log("Redirect audio finished, redirecting to dashboard");
                        window.location.href = "http://127.0.0.1:5000/dashboard";
                    };
                } catch (error) {
                    console.error("Error playing redirect audio:", error);
                    // Fallback to local audio if backend audio fails
                    await generateLocalAudio(responseData.response);
                    console.log("Redirecting to dashboard after local audio");
                    window.location.href = "http://127.0.0.1:5000/dashboard";
                }
            } else {
                // If no audio, redirect immediately
                console.log("No audio to play, redirecting to dashboard immediately");
                window.location.href = "http://127.0.0.1:5000/dashboard";
            }

            // Fallback: Redirect after 5 seconds if audio playback doesn't complete
            setTimeout(() => {
                console.log("Fallback redirect triggered after 5 seconds");
                window.location.href = "http://127.0.0.1:5000/dashboard";
            }, 5000);
        } else {
            // Ensure playNextAudio is called immediately if no redirect
            if (!isPlaying && !isRecording && !userStopped) {
                console.log("Triggering playNextAudio immediately after receiving response");
                await playNextAudio();
            } else {
                console.log("Conditions not met for immediate playback: isPlaying=", isPlaying, "isRecording=", isRecording, "userStopped=", userStopped);
            }
        }
    } catch (error) {
        console.error("Error in startConversation:", error);
        updateStatus("Error: " + error.message);
    } finally {
        isProcessing = false;
    }
}

// Generate audio locally using SpeechSynthesis API with a female voice
function generateLocalAudio(text) {
    return new Promise((resolve, reject) => {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'en-US';

            // Select a female voice if available
            const voices = window.speechSynthesis.getVoices();
            console.log("Available voices:", voices.map(voice => ({
                name: voice.name,
                lang: voice.lang,
                default: voice.default
            })));

            // Try to find a female voice by name
            let femaleVoice = voices.find(voice => 
                voice.name.toLowerCase().includes('zira') || // Microsoft Zira (female voice)
                voice.name.toLowerCase().includes('samantha') ||
                voice.name.toLowerCase().includes('google us english') ||
                voice.name.toLowerCase().includes('female') ||
                voice.name.toLowerCase().includes('jenny') ||
                voice.name.toLowerCase().includes('aria')
            );

            if (femaleVoice) {
                utterance.voice = femaleVoice;
                console.log("Using female voice for SpeechSynthesis:", femaleVoice.name);
            } else {
                // Fallback to default voice if no female voice is found
                const defaultVoice = voices.find(voice => voice.default) || voices[0];
                utterance.voice = defaultVoice;
                console.log("No female voice found, using default voice:", defaultVoice ? defaultVoice.name : "none");
            }

            utterance.onend = () => {
                console.log("Local audio playback completed for text:", text);
                resolve();
            };
            utterance.onerror = (event) => {
                console.error("Speech synthesis error:", event.error);
                reject(new Error(`Speech synthesis error: ${event.error}`));
            };
            window.speechSynthesis.speak(utterance);
        } else {
            reject(new Error("Speech synthesis not supported in this browser"));
        }
    });
}

// Ensure voices are loaded before using SpeechSynthesis
function loadVoices() {
    return new Promise(resolve => {
        let voices = window.speechSynthesis.getVoices();
        if (voices.length > 0) {
            resolve(voices);
        } else {
            window.speechSynthesis.onvoiceschanged = () => {
                voices = window.speechSynthesis.getVoices();
                resolve(voices);
            };
        }
    });
}

// Play the next audio in the queue with fallback to local generation
async function playNextAudio() {
    if (audioQueue.length === 0 || responseQueue.length === 0) {
        console.log("Audio queue empty, stopping playback");
        isPlaying = false;
        if (!isRecording && !isProcessing && !userStopped) {
            startRecognition();
        }
        return;
    }

    if (!hasUserInteracted) {
        console.warn("User interaction required for audio playback due to browser autoplay policy");
        updateStatus("Please interact with the page to enable audio playback (e.g., click Start Recording)");
        isPlaying = false;
        return;
    }

    isPlaying = true;
    const audioUrl = audioQueue.shift();
    const responseText = responseQueue.shift();

    updateStatus("Playing audio response...");

    if (audioUrl) {
        // Try playing the backend-generated audio (Google TTS)
        const audio = new Audio(audioUrl);
        audio.crossOrigin = "anonymous"; // Attempt to handle CORS
        try {
            console.log("Attempting to play backend audio:", audioUrl);
            const playPromise = audio.play();
            await playPromise;
            console.log("Audio playback started successfully");
            audio.onended = () => {
                console.log("Backend audio playback completed");
                isPlaying = false;
                playNextAudio();
            };
            audio.onerror = async (error) => {
                console.error("Backend audio playback failed:", error);
                console.error("Error details:", error.message, error.name);
                await fallbackToLocalAudio(responseText);
            };
        } catch (error) {
            console.error("Backend audio playback failed:", error);
            console.error("Error details:", error.message, error.name);
            await fallbackToLocalAudio(responseText);
        }
    } else {
        // If no audioUrl, directly fallback to local generation
        await fallbackToLocalAudio(responseText);
    }
}

// Fallback to local audio generation
async function fallbackToLocalAudio(responseText) {
    try {
        console.log("Falling back to local audio generation for text:", responseText);
        await loadVoices();
        await generateLocalAudio(responseText);
        console.log("Local audio playback completed successfully");
        isPlaying = false;
        playNextAudio();
    } catch (localError) {
        console.error("Local audio generation failed:", localError);
        updateStatus("Error: Unable to play audio response locally");
        isPlaying = false;
        playNextAudio();
    }
}

// Firebase authentication state change handler
firebase.auth().onAuthStateChanged(user => {
    if (user) {
        user.getIdToken().then(token => {
            window.appData.authToken = token;
            window.appData.uid = user.uid;
            console.log("Firebase idToken updated:", token.substring(0, 20) + "...");
            console.log("User authenticated:", user.uid);
            initializeVoiceBot();
        });
    } else {
        console.log("User not authenticated");
        window.location.href = '/login';
    }
});

// Add event listeners
document.addEventListener('DOMContentLoaded', () => {
    controlButton = document.getElementById('controlButton');
    if (controlButton) {
        controlButton.removeEventListener('click', toggleRecording);
        controlButton.addEventListener('click', toggleRecording);
    }

    checkMicPermission().then(hasPermission => {
        if (hasPermission && !isRecording && !isPlaying && !isProcessing) {
            console.log("Waiting for user to start recording");
        }
    });

    // Ensure user interaction for autoplay
    document.addEventListener('click', () => {
        if (!hasUserInteracted) {
            hasUserInteracted = true;
            console.log("User interaction detected, enabling audio playback");
            // Retry playing any queued audio
            if (!isPlaying && !isRecording && !userStopped && audioQueue.length > 0) {
                console.log("Retrying audio playback after user interaction");
                playNextAudio();
            }
        }
    });
});

// Toggle recording function for clarity
function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

// Alias toggleRecording functions for clarity
function startRecording() {
    startRecognition();
}

function stopRecording() {
    stopRecognition();
}