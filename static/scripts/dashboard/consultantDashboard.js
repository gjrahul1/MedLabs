(function() {
  const auth = window.auth;
  const db = window.db;
  const storage = window.storage;

  console.log("Firebase services initialized:", { auth, db, storage });

  function getGreeting() {
    const hour = new Date().getHours();
    return hour < 12 ? "Good Morning" : hour < 18 ? "Good Afternoon" : "Good Evening";
  }

  async function getAuthToken() {
    const user = auth.currentUser;
    if (!user) {
      console.error("No current user found");
      return null;
    }
    try {
      const token = await user.getIdToken(true);
      console.log("Retrieved auth token");
      return token;
    } catch (error) {
      console.error("Error getting auth token:", error);
      return null;
    }
  }

  async function fetchWithAuth(url, options = {}) {
    const token = await getAuthToken();
    if (!token) throw new Error("No authentication token available");
    const headers = options.headers ? { ...options.headers } : {};
    headers['Authorization'] = `Bearer ${token}`;
    const method = options.method ? options.method.toUpperCase() : "GET";
    if (method !== "GET" && !(options.body instanceof FormData) && options.body) {
      headers['Content-Type'] = 'application/json';
    }
    return fetch(url, { ...options, headers, credentials: 'include' });
  }

  async function loadConsultantData(consultantId) {
    console.log("Loading consultant data for ID:", consultantId);
    const consultantRef = db.doc(`consultant_registrations/${consultantId}`);
    try {
      const consultantSnap = await consultantRef.get();
      if (!consultantSnap.exists) {
        console.error("Consultant ID unmatched in consultant_registrations");
        alert("Consultant ID unmatched in records!");
        return;
      }
      const consultantData = consultantSnap.data();
      console.log("Consultant data fetched from Firestore:", consultantData);

      const consultantIdElement = document.getElementById("consultant-id");
      const headerTitle = document.querySelector(".dashboard-header h1");

      if (!consultantIdElement || !headerTitle) {
        console.error("DOM elements missing:", {
          consultantIdElement: !!consultantIdElement,
          headerTitle: !!headerTitle
        });
        return;
      }

      consultantIdElement.textContent = consultantId || "N/A";
      headerTitle.textContent = `${getGreeting()}, ${consultantData.full_name || 'Consultant'}!`;
    } catch (error) {
      console.error("Error loading consultant data:", error);
    }
  }

  async function loadPatientList(consultantId) {
    const dropdown = document.getElementById("patient-dropdown");
    const searchInput = document.getElementById("patient-search");
    if (!dropdown || !searchInput) {
      console.error("Patient dropdown or search input not found");
      return;
    }

    dropdown.innerHTML = '<option value="">Select a patient...</option>';
    try {
      const patientsQuery = await db.collection('patient_registrations')
        .where('consultant_id', '==', consultantId)
        .get();

      if (patientsQuery.empty) {
        console.log("No patients found for this consultant");
        dropdown.innerHTML += '<option value="">No patients assigned</option>';
        return;
      }

      const patients = [];
      for (const doc of patientsQuery.docs) {
        const patientData = doc.data();
        patients.push({
          uid: doc.id,
          full_name: patientData.full_name || doc.id
        });
      }

      patients.forEach(patient => {
        const option = document.createElement("option");
        option.value = patient.uid;
        option.textContent = patient.full_name;
        dropdown.appendChild(option);
      });
      console.log("Loaded patients:", patients);

      // Add search functionality
      searchInput.addEventListener('input', () => {
        const filter = searchInput.value.toLowerCase();
        const options = dropdown.options;
        for (let i = 1; i < options.length; i++) { // Skip the first "Select a patient..." option
          const text = options[i].textContent.toLowerCase();
          options[i].style.display = text.includes(filter) ? '' : 'none';
        }
      });
    } catch (error) {
      console.error("Error loading patient list:", error);
      dropdown.innerHTML += '<option value="">Error loading patients</option>';
    }
  }

  async function loadHealthCondition(uid) {
    const container = document.getElementById("health-condition");
    const detailsDiv = document.getElementById("patient-details");
    const generalSeverity = document.getElementById("general-severity");
    const indicatorsContainer = document.getElementById("indicators-container");
    const conditionList = document.getElementById("condition-list");
    const trendChartCanvas = document.getElementById("condition-trend-chart");
    if (!container || !detailsDiv || !generalSeverity || !indicatorsContainer || !conditionList || !trendChartCanvas) {
      console.error("Health condition elements not found");
      return;
    }

    detailsDiv.style.display = 'block';
    conditionList.innerHTML = '<li>Loading health condition...</li>';

    try {
      const historyRef = db.collection("medical_histories").doc(`${uid}`);
      const historySnap = await historyRef.get();
      if (!historySnap.exists) {
        conditionList.innerHTML = '<li class="no-data">No health condition data available.</li>';
        generalSeverity.className = 'severity-fill';
        indicatorsContainer.innerHTML = '';
        return;
      }

      const historyData = historySnap.data();
      const summaryLines = (historyData.summary || 'No summary available').split('\n').filter(line => line.trim());
      let formattedLines = [];
      let generalSeverityClass = 'mild';
      let indicatorsHTML = '';
      const conditionMap = {
        'stomach pain': { label: 'Stomach', icon: 'fas fa-stomach', system: 'digestive' },
        'sepsis': { label: 'Sepsis', icon: 'fas fa-virus', system: 'systemic' },
        'headache': { label: 'Head', icon: 'fas fa-brain', system: 'neurological' },
        'photophobia': { label: 'Eyes', icon: 'fas fa-eye', system: 'neurological' },
        'lung abscess': { label: 'Lungs', icon: 'fas fa-lungs', system: 'respiratory' }
      };
      const conditions = [];

      summaryLines.forEach(line => {
        line = line.replace(/\*+/g, '').trim();
        let severityClass = 'mild';
        if (line.toLowerCase().includes('mild')) {
          severityClass = 'mild';
        } else if (line.toLowerCase().includes('moderate')) {
          severityClass = 'moderate';
        } else if (line.toLowerCase().includes('severe')) {
          severityClass = 'severe';
          if (generalSeverityClass !== 'severe') {
            generalSeverityClass = 'severe';
          }
        }

        if (line.includes('Initial complaint')) {
          const dateMatch = line.match(/\((.*?)\)/);
          const date = dateMatch ? dateMatch[1] : '';
          line = line.replace(/\((.*?)\)/, '');
          formattedLines.push(`<li><strong>Initial complaint:</strong> ${line}${date ? `<br><strong>Date:</strong> ${date}` : ''}</li>`);

          for (const [condition, info] of Object.entries(conditionMap)) {
            if (line.toLowerCase().includes(condition)) {
              indicatorsHTML += `
                <div class="map-indicator" id="${condition}-indicator">
                  <i class="${info.icon} indicator-icon"></i>
                  <span class="indicator-label">${info.label}</span>
                  <div class="severity-bar">
                    <div class="severity-fill ${severityClass}" id="${condition}-severity"></div>
                  </div>
                </div>
              `;
              conditions.push({ name: condition, severity: severityClass, date: date || new Date().toISOString().split('T')[0] });
            }
          }
        } else if (line.includes('Diagnosis')) {
          formattedLines.push(`<li><strong>Diagnosis:</strong> ${line.split(':')[1].trim()}</li>`);
          for (const [condition, info] of Object.entries(conditionMap)) {
            if (line.toLowerCase().includes(condition)) {
              indicatorsHTML += `
                <div class="map-indicator" id="${condition}-indicator">
                  <i class="${info.icon} indicator-icon"></i>
                  <span class="indicator-label">${info.label}</span>
                  <div class="severity-bar">
                    <div class="severity-fill ${severityClass}" id="${condition}-severity"></div>
                  </div>
                </div>
              `;
              conditions.push({ name: condition, severity: severityClass, date: new Date().toISOString().split('T')[0] });
            }
          }
        } else if (line.includes('Treatment')) {
          formattedLines.push(`<li><strong>Treatment:</strong> ${line.split(':')[1].trim()}</li>`);
        } else if (line.includes('Prognosis')) {
          formattedLines.push(`<li><strong>Prognosis:</strong> ${line.split(':')[1].trim()}</li>`);
        } else {
          formattedLines.push(`<li>${line}</li>`);
        }
      });

      // Set indicators
      generalSeverity.className = `severity-fill ${generalSeverityClass}`;
      indicatorsContainer.innerHTML = indicatorsHTML;
      conditionList.innerHTML = formattedLines.join('');

      // Fetch historical summaries for trends
      const historicalSummaries = await db.collection(`medical_histories/${uid}/summaries`)
        .orderBy('timestamp', 'asc')
        .get();

      const trendData = {};
      conditions.forEach(condition => {
        trendData[condition.name] = { labels: [], severities: [] };
      });

      historicalSummaries.forEach(doc => {
        const summary = doc.data().summary.split('\n').filter(line => line.trim());
        const date = doc.data().timestamp ? new Date(doc.data().timestamp.seconds * 1000).toLocaleDateString() : new Date().toLocaleDateString();
        conditions.forEach(condition => {
          let severityValue = 0;
          summary.forEach(line => {
            line = line.replace(/\*+/g, '').trim().toLowerCase();
            if (line.includes(condition.name.toLowerCase())) {
              if (line.includes('mild')) severityValue = 1;
              else if (line.includes('moderate')) severityValue = 2;
              else if (line.includes('severe')) severityValue = 3;
            }
          });
          trendData[condition.name].labels.push(date);
          trendData[condition.name].severities.push(severityValue);
        });
      });

      // Plot trends using Chart.js
      const ctx = trendChartCanvas.getContext('2d');
      new Chart(ctx, {
        type: 'line',
        data: {
          labels: trendData[conditions[0]?.name]?.labels || [],
          datasets: conditions.map(condition => ({
            label: condition.name,
            data: trendData[condition.name].severities,
            borderColor: condition.severity === 'mild' ? '#48bb78' : condition.severity === 'moderate' ? '#ecc94b' : '#f56565',
            fill: false,
            tension: 0.1
          }))
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              max: 3,
              ticks: {
                stepSize: 1,
                callback: value => ['None', 'Mild', 'Moderate', 'Severe'][value]
              }
            }
          },
          plugins: {
            legend: {
              position: 'top'
            }
          }
        }
      });
    } catch (error) {
      console.error("Error loading health condition:", error);
      conditionList.innerHTML = '<li class="error">Failed to load health condition data.</li>';
      generalSeverity.className = 'severity-fill';
      indicatorsContainer.innerHTML = '';
    }
  }

  async function loadPrescriptions(consultantId) {
    const container = document.getElementById("prescription-summary");
    if (!container) {
      console.error("Prescriptions section container not found");
      return;
    }

    container.innerHTML = '<div class="content-placeholder">Loading prescriptions...</div>';

    try {
      const querySnapshot = await db.collection('prescriptions')
        .where('consultant_id', '==', consultantId)
        .orderBy('timestamp', 'desc')
        .get();

      if (querySnapshot.empty) {
        container.innerHTML = '<p class="no-data">No prescriptions found.</p>';
        return;
      }

      let prescriptionsHTML = '';
      let index = 0;
      for (const doc of querySnapshot.docs) {
        const data = doc.data();
        const patientRef = await db.doc(`patient_registrations/${data.uid}`).get();
        const patientName = patientRef.exists ? patientRef.data().full_name : "Unknown";

        const metadata = `
          <div class="metadata">
            <div class="metadata-item">
              <span class="label">Patient Name:</span>
              <span class="value">${patientName}</span>
            </div>
            <div class="metadata-item">
              <span class="label">Date:</span>
              <span class="value">${new Date(data.timestamp?.seconds * 1000 || Date.now()).toLocaleDateString()}</span>
            </div>
            <div class="metadata-item">
              <span class="label">Consultant:</span>
              <span class="value">${data.consultant_id || "Not assigned"}</span>
            </div>
          </div>
        `;

        const summaryLines = (data.professional_summary || 'No professional summary available').split('\n')
          .filter(line => line.trim())
          .map(line => {
            line = line.replace(/\*+/g, '').trim();
            if (line.startsWith('**') && line.includes(':**')) {
              const heading = line.replace(/^(\*\*[^:]+:\*\*)/, '$1').replace(/\*\*/g, '');
              return `<div class="summary-heading">${heading}</div>`;
            }
            return `<div class="summary-text">${line}</div>`;
          })
          .join('');

        prescriptionsHTML += `
          <div class="report-card" data-language="${data.language || 'english'}" style="animation: fadeIn 0.5s ease-in-out ${index * 0.1}s forwards;">
            ${metadata}
            <div class="summary-container professional-summary">${summaryLines}</div>
          </div>
        `;
        index++;
      }
      container.innerHTML = prescriptionsHTML;
    } catch (error) {
      console.error("Error loading prescriptions:", error);
      container.innerHTML = '<p class="error">Failed to load prescriptions.</p>';
    }
  }

  async function loadLabRecords(consultantId) {
    const container = document.getElementById("lab-records-summary");
    if (!container) {
      console.error("Lab records section container not found");
      return;
    }

    container.innerHTML = '<div class="content-placeholder">Loading lab records...</div>';

    try {
      const querySnapshot = await db.collection('lab_records')
        .where('consultant_id', '==', consultantId)
        .orderBy('timestamp', 'desc')
        .get();

      if (querySnapshot.empty) {
        container.innerHTML = '<p class="no-data">No lab records found.</p>';
        return;
      }

      let labRecordsHTML = '';
      let index = 0;
      for (const doc of querySnapshot.docs) {
        const data = doc.data();
        const patientRef = await db.doc(`patient_registrations/${data.uid}`).get();
        const patientName = patientRef.exists ? patientRef.data().full_name : "Unknown";

        const metadata = `
          <div class="metadata">
            <div class="metadata-item">
              <span class="label">Patient Name:</span>
              <span class="value">${patientName}</span>
            </div>
            <div class="metadata-item">
              <span class="label">Date:</span>
              <span class="value">${new Date(data.timestamp?.seconds * 1000 || Date.now()).toLocaleDateString()}</span>
            </div>
            <div class="metadata-item">
              <span class="label">Consultant:</span>
              <span class="value">${data.consultant_id || "Not assigned"}</span>
            </div>
          </div>
        `;

        const summaryLines = (data.professional_summary || 'No professional summary available').split('\n')
          .filter(line => line.trim())
          .map(line => {
            line = line.replace(/\*+/g, '').trim();
            if (line.startsWith('**') && line.includes(':**')) {
              const heading = line.replace(/^(\*\*[^:]+:\*\*)/, '$1').replace(/\*\*/g, '');
              return `<div class="summary-heading">${heading}</div>`;
            }
            return `<div class="summary-text">${line}</div>`;
          })
          .join('');

        labRecordsHTML += `
          <div class="report-card" data-language="${data.language || 'english'}" style="animation: fadeIn 0.5s ease-in-out ${index * 0.1}s forwards;">
            ${metadata}
            <div class="summary-container professional-summary">${summaryLines}</div>
          </div>
        `;
        index++;
      }
      container.innerHTML = labRecordsHTML;
    } catch (error) {
      console.error("Error loading lab records:", error);
      container.innerHTML = '<p class="error">Failed to load lab records.</p>';
    }
  }

  async function updateMedicalHistory(uid) {
    try {
      const initialScreening = await db.collection('initial_screenings').doc(`initial_screening_${uid}`).get();
      const prescriptions = await db.collection('prescriptions').where('uid', '==', uid).orderBy('timestamp', 'desc').get();
      const labRecords = await db.collection('lab_records').where('uid', '==', uid).orderBy('timestamp', 'desc').get();

      let historyText = "Patient Medical History:\n\nInitial Screening:\n";
      if (initialScreening.exists) {
        const screeningData = initialScreening.data();
        historyText += `- Symptoms: ${screeningData.symptoms || 'N/A'}\n`;
        historyText += `  Severity: ${screeningData.severity || 'N/A'}\n`;
        historyText += `  Duration: ${screeningData.duration || 'N/A'}\n`;
        historyText += `  Triggers: ${screeningData.triggers || 'N/A'}\n`;
        historyText += `  Date: ${screeningData.timestamp ? new Date(screeningData.timestamp.seconds * 1000).toLocaleDateString() : 'N/A'}\n`;
      } else {
        historyText += "No initial screening data available.\n";
      }

      historyText += "\nPrescriptions:\n";
      if (prescriptions.empty) {
        historyText += "No prescriptions available.\n";
      } else {
        for (const doc of prescriptions.docs) {
          const data = doc.data();
          historyText += `- Date: ${new Date(data.timestamp?.seconds * 1000 || Date.now()).toLocaleDateString()}\n`;
          historyText += `  Professional Summary: ${data.professional_summary}\n`;
        }
      }

      historyText += "\nLab Records:\n";
      if (labRecords.empty) {
        historyText += "No lab records available.\n";
      } else {
        for (const doc of labRecords.docs) {
          const data = doc.data();
          historyText += `- Date: ${new Date(data.timestamp?.seconds * 1000 || Date.now()).toLocaleDateString()}\n`;
          historyText += `  Professional Summary: ${data.professional_summary}\n`;
        }
      }

      const response = await fetchWithAuth('/process-medical-history', {
        method: 'POST',
        body: JSON.stringify({ uid, historyText }),
        headers: { 'Content-Type': 'application/json' }
      });

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || "Failed to process medical history");
      }

      const historyRef = db.collection('medical_histories').doc(`${uid}`);
      await historyRef.set({
        uid: uid,
        summary: data.summary,
        timestamp: firestore.SERVER_TIMESTAMP
      }, { merge: true });

      // Store historical summary in a subcollection
      await db.collection(`medical_histories/${uid}/summaries`).add({
        summary: data.summary,
        timestamp: firestore.SERVER_TIMESTAMP
      });

      console.log(`Updated medical history for UID: ${uid}`);
    } catch (error) {
      console.error("Error updating medical history:", error);
    }
  }

  async function handlePatientSelection(uid) {
    if (!uid) {
      document.getElementById("patient-details").style.display = 'none';
      return;
    }

    await Promise.all([
      loadHealthCondition(uid),
      updateMedicalHistory(uid)
    ]);
  }

  async function initializeDashboard() {
    try {
      const user = auth.currentUser;
      if (!user) {
        console.log("No user authenticated, redirecting to login");
        window.location.href = "/login";
        return;
      }

      const consultantId = window.consultantId;
      console.log("Consultant authenticated with ID:", consultantId);

      // Load initial data
      await loadConsultantData(consultantId);
      await loadPatientList(consultantId);

      // Load reports (Prescriptions and Lab Records) only when the Reports section is active
      const reportsSection = document.getElementById("reports");
      const reportsMenuItem = document.querySelector('.menu-item[data-section="reports"]');
      if (reportsSection.classList.contains('active')) {
        await Promise.all([
          loadPrescriptions(consultantId),
          loadLabRecords(consultantId)
        ]);
      }

      // Sidebar navigation
      document.querySelectorAll('.menu-item').forEach(item => {
        item.addEventListener('click', async () => {
          document.querySelectorAll('.menu-item').forEach(i => i.classList.remove('active'));
          item.classList.add('active');

          const section = item.getAttribute('data-section');
          document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));

          const targetSection = document.getElementById(section);
          if (targetSection) {
            targetSection.classList.add('active');

            // Load reports when the Reports section is activated
            if (section === 'reports') {
              await Promise.all([
                loadPrescriptions(consultantId),
                loadLabRecords(consultantId)
              ]);
            }
          }
        });
      });

      const logoutBtn = document.getElementById("logout-btn");
      if (!logoutBtn) {
        console.error("Logout button (logout-btn) not found in the DOM after dashboard load");
        return;
      }

      logoutBtn.addEventListener("click", async () => {
        console.log("Logout button clicked, current user:", auth.currentUser);
        if (confirm("Are you sure you want to log out?")) {
          if (!auth.currentUser) {
            console.warn("No active user session found, redirecting to /login");
            window.location.href = "/login";
            return;
          }
          try {
            console.log("Calling server logout");
            const logoutResponse = await fetchWithAuth('/logout?confirm=yes', { method: 'GET', redirect: 'follow' });
            console.log("Server logout response status:", logoutResponse.status);
            if (logoutResponse.status === 302 || logoutResponse.ok) {
              console.log("Server logout successful");
              await auth.signOut();
              console.log("Client-side sign-out successful, redirecting to /login");
              window.location.href = "/login";
            } else {
              throw new Error(`Server logout failed with status ${logoutResponse.status}`);
            }
          } catch (error) {
            console.error("Logout error:", error);
            alert("Failed to log out. Please try again or contact support. Error: " + error.message);
            window.location.href = "/login";
          }
        }
      });
    } catch (error) {
      console.error("Dashboard initialization error:", error);
      window.location.href = "/";
    }
  }

  // Expose handlePatientSelection globally
  window.handlePatientSelection = handlePatientSelection;

  // Initialize on auth state change to ensure login state is resolved
  document.addEventListener('DOMContentLoaded', () => {
    console.log("Waiting for auth state change...");
    auth.onAuthStateChanged((user) => {
      console.log("Auth state changed:", user ? `User: ${user.email}, UID: ${user.uid}` : "No user");
      if (!user) {
        console.log("No user authenticated, redirecting to login");
        window.location.href = "/login";
        return;
      }
      console.log("User authenticated, initializing dashboard");
      initializeDashboard();
    });
  });

  window.initializeDashboard = initializeDashboard;
})();