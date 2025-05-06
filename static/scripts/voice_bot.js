(function () {
  let hasInitializedBot = false;
  let authListenerAttached = false;

  function initializeBot() {
    if (hasInitializedBot) {
      console.log('initializeBot already called, skipping');
      return;
    }
    hasInitializedBot = true;

    console.log('voice_bot.js loaded, initializing at:', new Date().toISOString());

    const controlButton = document.getElementById('controlButton');
    const audioPlayer = document.getElementById('audioPlayer');
    const transcriptDiv = document.getElementById('transcriptContent');
    const statusDiv = document.getElementById('status-message');
    const transcriptPanel = document.getElementById('transcriptPanel');

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

    controlButton.disabled = true;
    statusDiv.textContent = 'Loading dependencies...';

    let sessionState = null;

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
            setTimeout(() => {
              reject(new Error(`Timeout loading script: ${src}`));
            }, 10000);
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

    function blobToBase64(blob) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(',')[1]);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    }

    function base64ToBlob(base64, mimeType) {
      const byteCharacters = atob(base64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      return new Blob([byteArray], { type: mimeType });
    }

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

    async function retryUploadAudio(sessionId, transcript, maxRetries = 3) {
      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
          console.log(`Attempt ${attempt} to send transcript to server`);
          const payload = {
            sessionId: sessionId,
            transcript: transcript,
            medicalData: sessionState.medicalData,
            currentState: sessionState.currentState
          };
          console.log('Sending payload with transcript:', transcript);

          const response = await axios.post('/start_conversation', payload, {
            headers: {
              'Authorization': `Bearer ${sessionStorage.getItem('idToken')}`,
              'Content-Type': 'application/json'
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
          await new Promise(resolve => setTimeout(resolve, 2000));
        }
      }
    }

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

    function toggleButtonState(isRecognizing) {
      recognizing = isRecognizing;
      controlButton.textContent = isRecognizing ? 'Stop Recording' : 'Start Recording';
      controlButton.classList.toggle('stop', isRecognizing);
      statusDiv.textContent = isRecognizing ? 'Recording...' : '';
      transcriptPanel.classList.toggle('recording', isRecognizing);
    }

    async function initializeBotInternal() {
      try {
        console.log('Starting to load scripts...');
        const scriptsLoaded = await loadScripts();
        if (!scriptsLoaded) {
          console.error('Script loading failed, aborting initialization');
          return;
        }

        if (!window.firebase || !window.auth || !window.db || !window.storage) {
          console.error('Firebase not initialized. Ensure firebaseConfig.js is loaded and initializes Firebase correctly.');
          statusDiv.textContent = 'Firebase initialization failed. Please refresh the page.';
          return;
        }
        console.log('Firebase initialized successfully');

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
          console.error('SpeechRecognition not supported');
          statusDiv.textContent = 'Your browser does not support speech recognition. Please use Chrome.';
          controlButton.disabled = true;
          return;
        }
        console.log('SpeechRecognition API supported');

        const hasMicPermission = await requestMicrophonePermission();
        if (!hasMicPermission) {
          console.error('Microphone permission not granted, aborting');
          return;
        }
        console.log('Microphone permission granted successfully');

        const recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.continuous = true;
        recognition.interimResults = false;
        console.log('SpeechRecognition initialized');

        sessionState = {
          session_id: generateSessionId(),
          currentState: 'INITIAL',
          medicalData: { symptoms: [], severity: [], duration: [], triggers: [] },
          conversationComplete: false,
          hasInitiatedConversation: false
        };
        console.log('Session state initialized:', sessionState);

        let idToken = sessionStorage.getItem('idToken');
        let recognizing = false;
        let currentTranscript = '';
        let lastProcessedTranscript = '';
        let retryCount = 0;
        const maxRetries = 3;
        let hasInitiatedConversation = false;
        let isInitiatingConversation = false;
        let uid = null;
        let recordingContext = null;

        function generateSessionId() {
          if (crypto.randomUUID) {
            return crypto.randomUUID();
          } else {
            const timestamp = Date.now();
            const random = Math.random().toString(36).substring(2);
            return `${timestamp}-${random}`;
          }
        }

        async function checkConversationStarted() {
          try {
            const convoSnapshot = await window.db.collection('conversations')
              .where('session_id', '==', sessionState.session_id)
              .where('state', '==', 'INITIAL')
              .orderBy('timestamp', 'desc')
              .limit(1)
              .get();
            return !convoSnapshot.empty;
          } catch (error) {
            console.error('Error checking conversation status:', error);
            return false;
          }
        }

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
          const hasStarted = await checkConversationStarted();
          if (hasStarted) {
            console.log('Conversation already started in Firestore, skipping initiation');
            sessionState.hasInitiatedConversation = true;
            hasInitiatedConversation = true;
            return;
          }
          isInitiatingConversation = true;
          try {
            sessionState.hasInitiatedConversation = true;
            hasInitiatedConversation = true;
            console.log('Initiating conversation for the first time');

            if (!sessionState.session_id) {
              sessionState.session_id = generateSessionId();
              console.log('Generated new session_id:', sessionState.session_id);
            }

            const payload = {
              transcript: '',
              sessionId: sessionState.session_id
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
            sessionState.hasInitiatedConversation = false;
            hasInitiatedConversation = false;
          } finally {
            isInitiatingConversation = false;
          }
        }

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
              uid = user.uid;
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

        if (!authListenerAttached) {
          console.log('Attaching onAuthStateChanged listener at:', new Date().toISOString());
          window.auth.onAuthStateChanged((user) => {
            console.log('onAuthStateChanged fired with user:', user ? user.uid : 'null');
            debouncedAuthHandler(user);
          });
          authListenerAttached = true;
        } else {
          console.log('Auth listener already attached, skipping attachment');
        }

        recognition.onstart = () => {
          console.log('Speech recognition started at:', new Date().toISOString());
          toggleButtonState(true);
          currentTranscript = '';
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
            currentTranscript = transcript.trim();
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

              // Wait for SpeechRecognition to complete
              await new Promise(resolve => {
                recognition.onend = () => {
                  console.log('Speech recognition ended, resolving');
                  resolve();
                };
                setTimeout(resolve, 2000); // Timeout after 2 seconds
              });

              if (!currentTranscript.trim()) {
                console.warn('No transcript captured after recognition');
                statusDiv.textContent = 'No speech detected. Please try speaking again.';
                transcriptPanel.classList.remove('processing');
                return;
              }

              transcriptPanel.classList.add('processing');
              const startTime = Date.now();
              console.log('Sending transcript to /start_conversation at:', new Date().toISOString());

              let data;
              try {
                data = await retryUploadAudio(sessionState.session_id, currentTranscript);
                console.log('Transcript successfully sent');
              } catch (error) {
                console.error('Failed to send transcript after retries:', error.response ? error.response.data : error.message);
                statusDiv.textContent = 'Error sending transcript: ' + (error.response ? (error.response.data.error || error.message) : error.message) + '. Please try again.';
                transcriptPanel.classList.remove('processing');
                return;
              }

              const endTime = Date.now();
              console.log(`Received response from /start_conversation after ${endTime - startTime}ms at:`, new Date().toISOString());
              console.log('Full server response:', JSON.stringify(data, null, 2));

              updateTranscript('User', currentTranscript);
              updateTranscript('AI', data.response);

              sessionState.medicalData = data.medical_data;
              sessionState.currentState = data.nextState || sessionState.currentState;

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
              } else if (data.redirect && data.response.includes("You have been assigned to")) {
                console.log('Doctor assignment detected in response, forcing redirect to:', data.redirect);
                setTimeout(() => {
                  console.log('Executing forced redirect to:', data.redirect);
                  window.location.href = data.redirect;
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
                  controlButton.click();
                }, 2000);
              }
            } finally {
              transcriptPanel.classList.remove('processing');
              currentTranscript = '';
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

        async function retryStoredAudio() {
          console.log('retryStoredAudio skipped: audio functionality removed');
        }

        retryStoredAudio();

        console.log('Initialization complete');
      } catch (error) {
        console.error('Error during initialization:', error.message, error.stack);
        statusDiv.textContent = 'Initialization failed: ' + (error.message || 'Unknown error') + '. Please refresh the page.';
      }
    }

    document.addEventListener('DOMContentLoaded', () => {
      console.log('DOMContentLoaded fired at:', new Date().toISOString());
      initializeBotInternal();
    });
  }

  initializeBot();
})();