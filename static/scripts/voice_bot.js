document.addEventListener('DOMContentLoaded', () => {
  // Local flag to ensure initializeBot is called only once
  let hasInitializedBot = false;

  console.log('DOMContentLoaded fired at:', new Date().toISOString());
  if (hasInitializedBot) {
    console.log('initializeBot already called, skipping duplicate DOMContentLoaded event');
    return;
  }
  hasInitializedBot = true;

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
    statusDiv.textContent = 'Error: Required elements not found. Please refresh the page.';
    return;
  }

  // Initialize button state
  controlButton.disabled = true;
  statusDiv.textContent = 'Loading dependencies...';

  // Function to load external scripts dynamically with a timeout
  async function loadScripts() {
    const scripts = [
      'https://cdn.jsdelivr.net/npm/axios@1.6.8/dist/axios.min.js',
      'https://cdn.jsdelivr.net/npm/dayjs@1.11.10/dayjs.min.js'
    ];

    for (const src of scripts) {
      try {
        console.log(`Loading script: ${src}`);
        await new Promise((resolve, reject) => {
          const script = document.createElement('script');
          script.src = src;
          script.onload = () => {
            console.log(`Successfully loaded script: ${src}`);
            resolve();
          };
          script.onerror = () => {
            console.error(`Failed to load script: ${src}`);
            reject(new Error(`Failed to load script: ${src}`));
          };
          document.head.appendChild(script);

          // Set a timeout for script loading
          setTimeout(() => {
            reject(new Error(`Timeout loading script: ${src}`));
          }, 10000); // 10-second timeout for each script
        });
      } catch (error) {
        console.error('Script loading error:', error.message);
        statusDiv.textContent = 'Failed to load required scripts: ' + error.message + '. Please check your network and refresh the page.';
        return false;
      }
    }
    console.log('All scripts loaded successfully');
    return true;
  }

  // Check server-side authentication status
  async function checkServerAuthStatus() {
    try {
      const idToken = sessionStorage.getItem('idToken');
      if (!idToken) {
        console.log('No idToken in sessionStorage, user likely not authenticated.');
        return false;
      }
      const response = await axios.get('/check-auth', {
        headers: { 'Authorization': `Bearer ${idToken}` }
      });
      console.log('Server auth status:', response.data.authenticated);
      return response.data.authenticated;
    } catch (error) {
      console.error('Error checking server auth status:', error.message);
      return false;
    }
  }

  // Request microphone permission explicitly
  async function requestMicrophonePermission() {
    try {
      const permissionStatus = await navigator.permissions.query({ name: 'microphone' });
      if (permissionStatus.state === 'denied') {
        console.error('Microphone permission denied by user.');
        statusDiv.textContent = 'Microphone access denied. Please allow microphone permissions to use voice features.';
        controlButton.disabled = true;
        return false;
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());
      console.log('Microphone permission granted');
      return true;
    } catch (error) {
      console.error('Microphone permission denied or error:', error.message);
      statusDiv.textContent = 'Microphone access denied. Please allow microphone permissions to use voice features.';
      controlButton.disabled = true;
      return false;
    }
  }

  // Convert Blob to Base64 for temporary storage
  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result.split(',')[1]); // Extract base64 part
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  // Convert Base64 to Blob for uploading
  function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
  }

  // Start recording audio using MediaRecorder
  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const chunks = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
          console.log('Audio data received, chunk size:', e.data.size);
        } else {
          console.warn('Empty audio chunk received');
        }
      };

      mediaRecorder.onstop = () => {
        console.log('MediaRecorder stopped, total chunks:', chunks.length);
      };

      mediaRecorder.onerror = (e) => {
        console.error('MediaRecorder error:', e);
      };

      mediaRecorder.start();
      console.log('Recording started at:', new Date().toISOString());
      return { mediaRecorder, stream, chunks, startTime: Date.now() };
    } catch (error) {
      console.error('Error starting recording:', error);
      statusDiv.textContent = `Failed to start recording: ${error.message}. Please ensure microphone access is granted.`;
      return null;
    }
  }

  // Stop recording and return the audio blob
  async function stopRecording(recordingContext) {
    const { mediaRecorder, stream, chunks, startTime } = recordingContext;
    return new Promise((resolve, reject) => {
      if (!mediaRecorder || mediaRecorder.state === 'inactive') {
        console.error('MediaRecorder is not active or missing');
        stream.getTracks().forEach(track => track.stop());
        reject(new Error('MediaRecorder is not active'));
        return;
      }

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(chunks, { type: 'audio/webm' });
        stream.getTracks().forEach(track => track.stop());
        console.log('Recording stopped, audio blob size:', audioBlob.size);
        resolve(audioBlob);
      };

      mediaRecorder.onerror = (e) => {
        console.error('Recording error:', e);
        stream.getTracks().forEach(track => track.stop());
        reject(e);
      };

      // Ensure minimum recording duration of 1 second
      const elapsedTime = Date.now() - startTime;
      if (elapsedTime < 1000) {
        console.log(`Recording duration (${elapsedTime}ms) is too short, waiting to reach at least 1000ms`);
        setTimeout(() => {
          mediaRecorder.stop();
        }, 1000 - elapsedTime);
      } else {
        mediaRecorder.stop();
      }
      console.log('Recording stopped at:', new Date().toISOString());
    });
  }

  // Retry uploading audio to the server
  async function retryUploadAudio(audioBlob, sessionId, maxRetries = 3) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`Attempt ${attempt} to upload audio to server`);
        const formData = new FormData();
        formData.append('audio', audioBlob, `${sessionId}.webm`);
        formData.append('session_id', sessionId);
        formData.append('medicalData', JSON.stringify(sessionState.medicalData));
        formData.append('currentState', sessionState.currentState);

        const response = await axios.post('/start_conversation', formData, {
          headers: {
            'Authorization': `Bearer ${idToken}`,
            'Content-Type': 'multipart/form-data'
          },
          timeout: 30000
        });

        const data = response.data;
        if (!data.success) {
          throw new Error(data.error || 'Server error');
        }
        return data;
      } catch (error) {
        console.error(`Upload attempt ${attempt} failed:`, error.response ? error.response.data : error.message);
        if (attempt === maxRetries) {
          throw error;
        }
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds before retrying
      }
    }
  }

  // Debounce function to prevent rapid successive calls
  function debounce(func, wait) {
    let timeout;
    return function (...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        console.log(`Debounced function called after ${wait}ms delay`);
        func.apply(this, args);
      }, wait);
    };
  }

  // Play audio with retry and re-generation on failure
  async function playAudio(url, retries = 2) {
    for (let attempt = 1; attempt <= retries + 1; attempt++) {
      try {
        console.log(`Attempt ${attempt} to play audio from:`, url);
        audioPlayer.src = url;
        const playPromise = audioPlayer.play();
        await playPromise;
        console.log('Audio playback started successfully');
        return new Promise((resolve) => {
          audioPlayer.onended = () => {
            console.log('Audio playback ended');
            resolve();
          };
        });
      } catch (error) {
        console.error(`Attempt ${attempt} failed to play audio:`, error.message, error.stack);
        if (error.name === "NotSupportedError" && attempt <= retries) {
          console.warn("NotSupportedError detected, attempting to re-generate audio...");
          const cacheKey = url.split('/').pop().replace('.mp3', '');
          const newUrl = await regenerateAudio(cacheKey);
          if (newUrl) {
            url = newUrl;
            continue;
          }
        }
        if (attempt === retries + 1) {
          console.error('Audio playback failed after retries:', error);
          throw new Error('Failed to play audio after retries: ' + error.message);
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  }

  // Function to re-generate audio by calling the server
  async function regenerateAudio(cacheKey) {
    try {
      console.log(`Requesting re-generation of audio for cache key: ${cacheKey}`);
      const idToken = sessionStorage.getItem('idToken');
      if (!idToken) {
        throw new Error('No authentication token available');
      }
      const response = await axios.post('/regenerate-audio', {
        cacheKey: cacheKey
      }, {
        headers: {
          'Authorization': `Bearer ${idToken}`,
          'Content-Type': 'application/json'
        }
      });

      const data = response.data;
      if (!data.success) {
        throw new Error(data.error || 'Failed to re-generate audio');
      }

      console.log('Audio re-generated successfully:', data.audio_url);
      return data.audio_url;
    } catch (error) {
      console.error('Error re-generating audio:', error.message);
      return null;
    }
  }

  // Update transcript display
  function updateTranscript(speaker, text) {
    const entry = document.createElement('p');
    entry.className = speaker.toLowerCase();
    entry.innerHTML = `<strong>${speaker}:</strong> <span>${text}</span>`;
    transcriptDiv.appendChild(entry);
    if (transcriptDiv.children.length > 6) {
      transcriptDiv.removeChild(transcriptDiv.firstChild);
    }
    transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
  }

  // Toggle button state
  function toggleButtonState(isRecognizing) {
    recognizing = isRecognizing;
    controlButton.textContent = isRecognizing ? 'Stop Recording' : 'Start Recording';
    controlButton.classList.toggle('stop', isRecognizing);
    statusDiv.textContent = isRecognizing ? 'Recording...' : '';
    transcriptPanel.classList.toggle('recording', isRecognizing);
  }

  // Main initialization function
  async function initializeBot() {
    try {
      // Load external scripts
      console.log('Starting to load scripts...');
      const scriptsLoaded = await loadScripts();
      if (!scriptsLoaded) {
        console.error('Script loading failed, aborting initialization');
        return;
      }

      // Use the global Firebase objects initialized in firebaseConfig.js
      if (!window.firebase || !window.auth || !window.db || !window.storage) {
        console.error('Firebase not initialized. Ensure firebaseConfig.js is loaded and initializes Firebase correctly.');
        statusDiv.textContent = 'Firebase initialization failed. Please refresh the page.';
        return;
      }
      console.log('Firebase initialized successfully');

      // Check for SpeechRecognition API support
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        console.error('SpeechRecognition not supported');
        statusDiv.textContent = 'Your browser does not support speech recognition. Please use Chrome.';
        controlButton.disabled = true;
        return;
      }
      console.log('SpeechRecognition API supported');

      // Request microphone permission before initializing recognition
      const hasMicPermission = await requestMicrophonePermission();
      if (!hasMicPermission) {
        console.error('Microphone permission not granted, aborting');
        return;
      }
      console.log('Microphone permission granted successfully');

      // Initialize SpeechRecognition
      const recognition = new SpeechRecognition();
      recognition.lang = 'en-US';
      recognition.continuous = true;
      recognition.interimResults = false;
      console.log('SpeechRecognition initialized');

      // Session state
      const sessionState = {
        session_id: generateSessionId(),
        currentState: 'INITIAL',
        medicalData: { symptoms: [], severity: [], duration: [], triggers: [] },
        conversationComplete: false,
        hasInitiatedConversation: false
      };
      console.log('Session state initialized:', sessionState);

      let idToken = sessionStorage.getItem('idToken');
      let recognizing = false;
      let currentTranscript = ''; // Accumulate all transcripts in a session
      let lastProcessedTranscript = ''; // Track the last processed transcript to prevent duplicates
      let retryCount = 0;
      const maxRetries = 3;
      let hasInitiatedConversation = false; // Global flag to prevent multiple initiations
      let isInitiatingConversation = false; // Lock to prevent concurrent initiation
      let hasAuthListenerAttached = false; // Flag to ensure single auth listener attachment
      let recordingContext = null; // For recording audio
      let uid = null; // Store the patient's UID

      // Function to generate a unique session ID
      function generateSessionId() {
        if (crypto.randomUUID) {
          return crypto.randomUUID();
        } else {
          const timestamp = Date.now();
          const random = Math.random().toString(36).substring(2);
          return `${timestamp}-${random}`;
        }
      }

      // Ensure session_id is initialized
      if (!sessionState.session_id) {
        sessionState.session_id = generateSessionId();
        console.log('Initial session_id set to:', sessionState.session_id);
      }

      // Initiate conversation with lock
      async function initiateConversation() {
        console.log('Attempting to initiate conversation at:', new Date().toISOString());
        if (isInitiatingConversation) {
          console.log('Already initiating conversation, skipping concurrent attempt');
          return;
        }
        if (sessionState.hasInitiatedConversation || hasInitiatedConversation) {
          console.log('Conversation already initiated, skipping. Session flag:', sessionState.hasInitiatedConversation, 'Global flag:', hasInitiatedConversation);
          return;
        }
        isInitiatingConversation = true;
        try {
          sessionState.hasInitiatedConversation = true;
          hasInitiatedConversation = true;
          console.log('Initiating conversation for the first time');

          // Ensure session_id is initialized
          if (!sessionState.session_id) {
            sessionState.session_id = generateSessionId();
            console.log('Generated new session_id:', sessionState.session_id);
          }

          const payload = {
            transcript: '',
            session_id: sessionState.session_id
          };
          console.log('Sending request to /start_conversation with payload:', JSON.stringify(payload));

          const response = await axios.post('/start_conversation', payload, {
            headers: { 
              'Authorization': `Bearer ${idToken}`,
              'Content-Type': 'application/json'
            }
          });

          const data = response.data;
          if (!data.success) {
            throw new Error(data.error || 'Server error');
          }

          updateTranscript('AI', data.response);
          if (data.audio_url) {
            console.log('Playing audio from URL:', data.audio_url);
            try {
              await playAudio(data.audio_url);
            } catch (error) {
              console.error('Failed to initiate conversation:', error);
              statusDiv.textContent = 'Error initiating conversation: Failed to play audio after retries. Please try again.';
            }
          } else {
            console.warn('No audio_url in response, displaying text response');
            statusDiv.textContent = 'Audio unavailable, displaying text response.';
          }
          console.log('Initial prompt sent successfully:', data.response);
        } catch (error) {
          console.error('Failed to initiate conversation:', error.response ? error.response.data : error.message);
          statusDiv.textContent = 'Error initiating conversation: ' + (error.response ? (error.response.data.error || error.message) : error.message) + '. Please try again.';
          // Reset flags to allow retry
          sessionState.hasInitiatedConversation = false;
          hasInitiatedConversation = false;
        } finally {
          isInitiatingConversation = false;
        }
      }

      // Authentication handler with loop prevention
      const authTimeout = setTimeout(() => {
        statusDiv.textContent = 'Authentication timed out. Please refresh the page.';
        controlButton.disabled = true;
      }, 10000);

      let isRedirecting = false;
      let isProcessingAuth = false;

      const debouncedAuthHandler = debounce(async (user) => {
        console.log('Auth state change detected at:', new Date().toISOString());
        if (isProcessingAuth) {
          console.log('Already processing auth state change, skipping this event');
          return;
        }
        isProcessingAuth = true;
        try {
          clearTimeout(authTimeout);
          if (user) {
            console.log('User authenticated, refreshing token...');
            idToken = await user.getIdToken(true);
            sessionStorage.setItem('idToken', idToken);
            uid = user.uid; // Store the patient's UID
            controlButton.disabled = false;
            statusDiv.textContent = '';
            if (window.location.pathname === '/further_patient_registration' && !hasInitiatedConversation) {
              console.log('Conditions met for initiating conversation');
              await initiateConversation();
            } else {
              console.log('Skipping conversation initiation: ', {
                currentPath: window.location.pathname,
                hasInitiated: hasInitiatedConversation
              });
            }
          } else {
            console.log('No user authenticated, checking server auth status...');
            const isServerAuthenticated = await checkServerAuthStatus();
            if (isServerAuthenticated) {
              console.log('Server confirms user is authenticated, attempting to sign in client-side');
              if (window.location.pathname === '/further_patient_registration' && !hasInitiatedConversation) {
                console.log('Conditions met for initiating conversation (server auth)');
                await initiateConversation();
              } else {
                console.log('Skipping conversation initiation (server auth): ', {
                  currentPath: window.location.pathname,
                  hasInitiated: hasInitiatedConversation
                });
              }
              return;
            }
            if (window.location.pathname !== '/login' && !isRedirecting) {
              isRedirecting = true;
              console.log('User not authenticated, redirecting to /login');
              statusDiv.textContent = 'Please log in to continue.';
              window.location.href = '/login';
            } else {
              console.log('Already on /login or redirecting, skipping redirect');
            }
          }
        } catch (error) {
          console.error('Auth error:', error.response ? error.response.data : error.message);
          statusDiv.textContent = 'Authentication failed. Please refresh the page.';
          controlButton.disabled = true;
          if (window.location.pathname !== '/login' && !isRedirecting) {
            isRedirecting = true;
            window.location.href = '/login';
          }
        } finally {
          isProcessingAuth = false;
          console.log('Auth processing complete at:', new Date().toISOString());
        }
      }, 2000);

      if (!hasAuthListenerAttached) {
        console.log('Attaching onAuthStateChanged listener at:', new Date().toISOString());
        window.auth.onAuthStateChanged((user) => {
          console.log('onAuthStateChanged fired with user:', user ? user.uid : 'null');
          debouncedAuthHandler(user);
        });
        hasAuthListenerAttached = true;
      } else {
        console.log('Auth listener already attached, skipping attachment');
      }

      // Speech recognition handlers
      recognition.onstart = () => {
        console.log('Speech recognition started at:', new Date().toISOString());
        toggleButtonState(true);
        currentTranscript = ''; // Reset transcript at the start of a new recording session
      };

      recognition.onend = () => {
        console.log('Speech recognition ended at:', new Date().toISOString());
        toggleButtonState(false);
      };

      recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            transcript += event.results[i][0].transcript + ' ';
          }
        }
        if (transcript.trim()) {
          console.log('Recognized transcript at:', new Date().toISOString(), 'Transcript:', transcript);
          currentTranscript += transcript; // Accumulate transcripts within the same recording session
        } else {
          console.warn('No transcript recognized');
        }
      };

      recognition.onerror = (event) => {
        console.error('Recognition error at:', new Date().toISOString(), 'Error:', event.error);
        statusDiv.textContent = `Speech recognition error: ${event.error}. Please ensure microphone access is granted.`;
        toggleButtonState(false);
        if (event.error === 'no-speech') {
          statusDiv.textContent = 'No speech detected. Please try speaking again.';
        } else if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
          statusDiv.textContent = 'Microphone access denied. Please allow microphone permissions to use voice features.';
          controlButton.disabled = true;
        }
      };

      controlButton.addEventListener('click', async () => {
        if (recognizing) {
          console.log('Stopping speech recognition at:', new Date().toISOString());
          recognizing = false;
          recognition.stop();
          if (!recordingContext) {
            console.error('No recording context available. Recording failed to start.');
            statusDiv.textContent = 'Error: No recording context available. Please try again.';
            transcriptPanel.classList.remove('processing');
            return;
          }
          try {
            console.log('Stopping recording at:', new Date().toISOString());
            // Ensure at least 1 second of recording
            const elapsedTime = Date.now() - recordingContext.startTime;
            if (elapsedTime < 1000) {
              console.log(`Recording duration (${elapsedTime}ms) is too short, waiting to reach at least 1000ms`);
              await new Promise(resolve => setTimeout(resolve, 1000 - elapsedTime));
            }
            const audioBlob = await stopRecording(recordingContext);
            recordingContext = null;

            if (!audioBlob || audioBlob.size === 0) {
              console.error('Audio recording failed: audioBlob is empty or invalid. Size:', audioBlob ? audioBlob.size : 'null');
              statusDiv.textContent = 'Error: Failed to record audio. Please try again.';
              transcriptPanel.classList.remove('processing');
              return;
            }

            console.log('Audio recorded successfully, size:', audioBlob.size, 'bytes');

            // Temporarily store audio in localStorage as base64
            const audioBase64 = await blobToBase64(audioBlob);
            localStorage.setItem(`patient_audio_${sessionState.session_id}`, audioBase64);
            console.log('Audio temporarily stored in localStorage');

            // Send the audio directly to /start_conversation
            transcriptPanel.classList.add('processing');
            const startTime = Date.now();
            console.log('Sending audio to /start_conversation at:', new Date().toISOString());

            let data;
            try {
              data = await retryUploadAudio(audioBlob, sessionState.session_id);
              // If successful, clear the localStorage entry
              localStorage.removeItem(`patient_audio_${sessionState.session_id}`);
              console.log('Audio successfully uploaded, cleared from localStorage');
            } catch (error) {
              console.error('Failed to upload audio after retries:', error.response ? error.response.data : error.message);
              statusDiv.textContent = 'Error uploading audio: ' + (error.response ? (error.response.data.error || error.message) : error.message) + '. Audio saved locally, will retry on next interaction.';
              transcriptPanel.classList.remove('processing');
              return;
            }

            const endTime = Date.now();
            console.log(`Received response from /start_conversation after ${endTime - startTime}ms at:`, new Date().toISOString());
            console.log('Full server response:', JSON.stringify(data, null, 2));

            // Log the GCS URI for the patient's audio
            if (data.gcs_uri) {
              console.log('Patient audio stored at:', data.gcs_uri);
            } else {
              console.error('No gcs_uri in response. Audio upload failed.');
              statusDiv.textContent = 'Error: Audio upload failed, but audio is saved locally. Please try again.';
            }

            // Update transcript with the patient's input (currentTranscript) and AI's response
            if (currentTranscript.trim()) {
              updateTranscript('User', currentTranscript);
            } else {
              console.warn('No transcript captured from speech recognition, using server transcript if available');
              if (data.transcript) {
                updateTranscript('User', data.transcript);
              } else {
                updateTranscript('User', 'Speech not recognized');
              }
            }
            updateTranscript('AI', data.response);

            // Update session state
            sessionState.medicalData = data.medical_data;
            sessionState.currentState = data.nextState || sessionState.currentState;

            // Play the AI's response audio
            if (data.audio_url) {
              console.log('Playing audio from URL:', data.audio_url);
              try {
                await playAudio(data.audio_url);
              } catch (error) {
                console.error('Failed to play audio:', error);
                statusDiv.textContent = 'Failed to play audio, displaying text response.';
              }
            } else {
              console.warn('No audio_url in response, displaying text response');
              statusDiv.textContent = 'Audio unavailable, displaying text response.';
            }

            // Handle conversation completion and redirect
            if (data.conversationComplete && data.redirect) {
              console.log('Conversation complete, redirecting to:', data.redirect);
              try {
                await axios.get('/logout?confirm=yes', {
                  headers: { 'Authorization': `Bearer ${idToken}` }
                });
                console.log('Logout request sent successfully');
              } catch (logoutError) {
                console.warn('Logout failed, proceeding with redirect:', logoutError);
              }
              setTimeout(() => {
                console.log('Executing redirect to:', data.redirect);
                window.location.href = data.redirect;
              }, 500);
            } else if (data.response.includes("You have been assigned to")) {
              console.log('Doctor assignment detected in response, forcing redirect to dashboard');
              setTimeout(() => {
                console.log('Executing forced redirect to: /dashboard');
                window.location.href = '/dashboard';
              }, 500);
            } else {
              console.log('Redirect or conversationComplete missing:', { redirect: data.redirect, conversationComplete: data.conversationComplete });
            }
          } catch (error) {
            console.error('Error processing conversation:', error.response ? error.response.data : error.message, error.stack);
            statusDiv.textContent = 'Error processing conversation: ' + (error.response ? (error.response.data.error || error.message) : error.message) + '. Please try again.';
            if (retryCount < maxRetries) {
              retryCount++;
              console.log('Retrying conversation processing, attempt:', retryCount);
              setTimeout(() => {
                controlButton.click(); // Simulate a retry by re-triggering the click event
              }, 2000);
            }
          } finally {
            transcriptPanel.classList.remove('processing');
            currentTranscript = ''; // Reset transcript after processing
          }
        } else {
          try {
            const permissionStatus = await navigator.permissions.query({ name: 'microphone' });
            if (permissionStatus.state === 'denied') {
              statusDiv.textContent = 'Microphone access denied. Please allow microphone permissions to use voice features.';
              return;
            }
            console.log('Starting speech recognition at:', new Date().toISOString());
            recognizing = true;
            recognition.start();

            // Start recording audio
            console.log('Starting audio recording at:', new Date().toISOString());
            recordingContext = await startRecording();
            if (!recordingContext) {
              console.error('Failed to start recording: recordingContext is null');
              statusDiv.textContent = 'Error: Failed to start audio recording. Please try again.';
              recognizing = false;
              recognition.stop();
              return;
            }
          } catch (error) {
            console.error('Start recognition error at:', new Date().toISOString(), 'Error:', error);
            statusDiv.textContent = `Failed to start recording: ${error.message}. Please ensure microphone access is granted.`;
            recognizing = false;
          }
        }
      });

      // Retry any locally stored audio on page load
      async function retryStoredAudio() {
        const storedAudio = localStorage.getItem(`patient_audio_${sessionState.session_id}`);
        if (storedAudio) {
          console.log('Found locally stored audio, attempting to upload');
          try {
            const audioBlob = base64ToBlob(storedAudio, 'audio/webm');
            const data = await retryUploadAudio(audioBlob, sessionState.session_id);
            localStorage.removeItem(`patient_audio_${sessionState.session_id}`);
            console.log('Successfully uploaded locally stored audio, cleared from localStorage');
            // Optionally update UI with the response
            updateTranscript('AI', data.response);
            if (data.audio_url) {
              await playAudio(data.audio_url);
            }
          } catch (error) {
            console.error('Failed to upload locally stored audio:', error.response ? error.response.data : error.message);
            statusDiv.textContent = 'Failed to upload locally stored audio. Please try again later.';
          }
        }
      }

      retryStoredAudio();

      console.log('Initialization complete');
    } catch (error) {
      console.error('Error during initialization:', error.message, error.stack);
      statusDiv.textContent = 'Initialization failed: ' + (error.message || 'Unknown error') + '. Please refresh the page.';
    }
  }

  initializeBot();
});