// Global flag to ensure initializeBot is called only once
let hasInitializedBot = false;

document.addEventListener('DOMContentLoaded', () => {
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
        sessionId: generateSessionId(),
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

      // Play audio with retry
      async function playAudio(url, retries = 2) {
        try {
          audioPlayer.src = url;
          console.log('Attempting to play audio from:', url);
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
          console.error('Audio playback error:', error.message, error.stack);
          if (retries > 0) {
            console.warn(`Audio playback failed, retrying (${retries} left):`, error);
            await new Promise(resolve => setTimeout(resolve, 1000));
            return playAudio(url, retries - 1);
          }
          console.error('Audio playback failed after retries:', error);
          throw new Error('Failed to play audio after retries');
        }
      }

      // Process transcript with server (debounced to prevent rapid calls)
      const debouncedProcessTranscript = debounce(async (transcript) => {
        if (transcript === lastProcessedTranscript) {
          console.log('Duplicate transcript detected, skipping:', transcript);
          return;
        }
        lastProcessedTranscript = transcript;
        console.log('Processing transcript:', transcript);
      
        if (!idToken) {
          statusDiv.textContent = 'Authentication required. Please log in.';
          return;
        }
      
        const maxTimeoutRetries = 2; // Retry up to 2 times on timeout
        let timeoutRetryCount = 0;
      
        while (timeoutRetryCount <= maxTimeoutRetries) {
          try {
            transcriptPanel.classList.add('processing');
            const startTime = Date.now();
            console.log('Sending request to /start_conversation at:', new Date().toISOString());
      
            const response = await axios.post('/start_conversation', {
              transcript,
              sessionId: sessionState.sessionId,
              medicalData: sessionState.medicalData,
              currentState: sessionState.currentState
            }, {
              headers: { 'Authorization': `Bearer ${idToken}` },
              timeout: 30000 // Increased timeout to 30 seconds
            });
      
            const endTime = Date.now();
            console.log(`Received response from /start_conversation after ${endTime - startTime}ms at:`, new Date().toISOString());
            console.log('Full server response:', JSON.stringify(response.data, null, 2));
      
            const data = response.data;
            if (!data.success) {
              throw new Error(data.error || 'Server error');
            }
      
            sessionState.medicalData = data.medicalData;
            sessionState.currentState = data.nextState || sessionState.currentState;
            updateTranscript('AI', data.response);
      
            if (data.audio_url) {
              console.log('Playing audio from URL:', data.audio_url);
              await playAudio(data.audio_url);
            } else {
              console.warn('No audio_url in response');
            }
      
            if (data.redirect && data.conversationComplete) {
              console.log('Conversation complete, redirecting to dashboard...');
              console.log('Redirect URL:', data.redirect);
              console.log('Conversation complete flag:', data.conversationComplete);
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
            } else {
              console.log('Redirect or conversationComplete missing:', {
                redirect: data.redirect,
                conversationComplete: data.conversationComplete
              });
              if (data.response.includes("You have been assigned to")) {
                console.log('Doctor assignment detected in response, forcing redirect to dashboard');
                setTimeout(() => {
                  console.log('Executing forced redirect to: /dashboard');
                  window.location.href = '/dashboard';
                }, 500);
              }
            }
      
            break; // Exit the retry loop on success
      
          } catch (error) {
            console.error('Process transcript error:', error.response ? error.response.data : error.message, error.stack);
            if (error.code === 'ECONNABORTED' && error.message.includes('timeout') && timeoutRetryCount < maxTimeoutRetries) {
              timeoutRetryCount++;
              console.log(`Timeout occurred, retrying (${timeoutRetryCount}/${maxTimeoutRetries})...`);
              await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds before retrying
              continue;
            }
            statusDiv.textContent = 'Error processing response: ' + (error.message || 'Unknown error') + '. Please try again.';
            if (retryCount < maxRetries) {
              retryCount++;
              console.log('Retrying transcript processing, attempt:', retryCount);
              setTimeout(() => debouncedProcessTranscript(transcript), 2000);
            }
            break; // Exit the retry loop on non-timeout errors or max retries reached
          } finally {
            transcriptPanel.classList.remove('processing');
          }
        }
      }, 1000);

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
          const response = await axios.post('/start_conversation', {
            transcript: null,
            sessionId: sessionState.sessionId,
            medicalData: sessionState.medicalData,
            currentState: sessionState.currentState
          }, {
            headers: { 'Authorization': `Bearer ${idToken}` }
          });
      
          const data = response.data;
          if (!data.success) {
            throw new Error(data.error || 'Server error');
          }
      
          updateTranscript('AI', data.response);
          if (data.audio_url) {
            console.log('Playing audio from URL:', data.audio_url);
            await playAudio(data.audio_url);
          } else {
            console.warn('No audio_url in response, displaying text response');
            statusDiv.textContent = 'Audio unavailable, displaying text response.';
          }
          console.log('Initial prompt sent successfully:', data.response);
        } catch (error) {
          console.error('Failed to initiate conversation:', error.response ? error.response.data : error.message);
          statusDiv.textContent = 'Error initiating conversation: ' + (error.message || 'Unknown error') + '. Please try again.';
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
        if (currentTranscript.trim()) {
          updateTranscript('User', currentTranscript);
          debouncedProcessTranscript(currentTranscript);
          currentTranscript = '';
        }
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
          } catch (error) {
            console.error('Start recognition error at:', new Date().toISOString(), 'Error:', error);
            statusDiv.textContent = `Failed to start recording: ${error.message}. Please ensure microphone access is granted.`;
          }
        }
      });

      console.log('Initialization complete');
    } catch (error) {
      console.error('Error during initialization:', error.message, error.stack);
      statusDiv.textContent = 'Initialization failed: ' + (error.message || 'Unknown error') + '. Please refresh the page.';
    }
  }

  initializeBot();
});