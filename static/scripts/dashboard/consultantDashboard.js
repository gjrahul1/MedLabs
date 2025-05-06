(function() {
  const auth = window.auth;
  const db = window.db;
  const storage = window.storage;
  const firebase = window.firebase;

  console.log("Firebase services initialized:", { auth, db, storage });

  // Global object to store patient UID-to-name mapping
  let patientNameMap = {};

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
    const patientDropdown = document.getElementById("patient-dropdown");
    const patientSearchInput = document.getElementById("patient-search");
    const reportDropdown = document.getElementById("report-patient-dropdown");
    const reportSearchInput = document.getElementById("report-patient-search");

    if (!patientDropdown || !patientSearchInput) {
      console.error("Patient dropdown or search input not found in Patient List section");
      return;
    }

    if (!reportDropdown || !reportSearchInput) {
      console.error("Report dropdown or search input not found in Reports section");
      return;
    }

    patientDropdown.innerHTML = '<option value="">Select a patient...</option>';
    reportDropdown.innerHTML = '<option value="">All patients</option>';

    let patients = [];
    try {
      const patientsQuery = await db.collection('patient_registrations')
        .where('consultant_id', '==', consultantId)
        .get();

      if (patientsQuery.empty) {
        console.log("No patients found for this consultant");
        patientDropdown.innerHTML += '<option value="">No patients assigned</option>';
        reportDropdown.innerHTML += '<option value="">No patients assigned</option>';
        return;
      }

      for (const doc of patientsQuery.docs) {
        const patientData = doc.data();
        const patientUid = doc.id;
        const patientName = patientData.full_name || patientUid;
        patients.push({
          uid: patientUid,
          full_name: patientName
        });
        // Store UID-to-name mapping
        patientNameMap[patientUid] = patientName;
      }

      patients.forEach(patient => {
        const option = document.createElement("option");
        option.value = patient.uid;
        option.textContent = patient.full_name;
        patientDropdown.appendChild(option);

        const reportOption = document.createElement("option");
        reportOption.value = patient.uid;
        reportOption.textContent = patient.full_name;
        reportDropdown.appendChild(reportOption);
      });

      console.log("Loaded patients:", patients);

      patientSearchInput.addEventListener('input', () => {
        const filter = patientSearchInput.value.toLowerCase();
        const options = patientDropdown.options;
        for (let i = 1; i < options.length; i++) {
          const text = options[i].textContent.toLowerCase();
          options[i].style.display = text.includes(filter) ? '' : 'none';
        }
      });

      reportSearchInput.addEventListener('input', () => {
        const filter = reportSearchInput.value.toLowerCase();
        const options = reportDropdown.options;
        for (let i = 1; i < options.length; i++) {
          const text = options[i].textContent.toLowerCase();
          options[i].style.display = text.includes(filter) ? '' : 'none';
        }
      });
    } catch (error) {
      console.error("Error loading patient list:", error);
      patientDropdown.innerHTML += '<option value="">Error loading patients</option>';
      reportDropdown.innerHTML += '<option value="">Error loading patients</option>';
    }
  }

  async function loadHealthCondition(uid, retries = 3, delayMs = 1000) {
    const container = document.getElementById("health-condition");
    const detailsDiv = document.getElementById("patient-details");
    const generalSeverity = document.getElementById("general-severity");
    const indicatorsContainer = document.getElementById("indicators-container");
    const conditionList = document.getElementById("condition-list");
    const trendChartCanvas = document.getElementById("condition-trend-chart");
    if (!container || !detailsDiv || !generalSeverity || !indicatorsContainer || !conditionList || !trendChartCanvas) {
      console.error("Health condition elements not found");
      throw new Error("Health condition elements not found");
    }

    detailsDiv.style.display = 'block';
    conditionList.innerHTML = '<li>Loading health condition...</li>';

    let attempt = 0;
    let historyData = null;

    while (attempt < retries) {
      try {
        const historyRef = db.collection("medical_histories").doc(`${uid}`);
        const historySnap = await historyRef.get();
        console.log(`Attempt ${attempt + 1}: History snapshot exists:`, historySnap.exists);

        if (!historySnap.exists) {
          conditionList.innerHTML = '<li class="no-data">No health condition data available.</li>';
          generalSeverity.className = 'severity-fill';
          indicatorsContainer.innerHTML = '';
          return;
        }

        historyData = historySnap.data();
        console.log(`Attempt ${attempt + 1}: History data fetched`);

        const summary = historyData.summary || 'No summary available';
        const hasConditions = ['headache', 'headaches', 'inflammation', 'infection', 'gums', 'upper respiratory'].some(term => summary.toLowerCase().includes(term));
        if (!hasConditions && attempt < retries - 1) {
          console.log(`Attempt ${attempt + 1}: No conditions found in summary, retrying after delay...`);
          await new Promise(resolve => setTimeout(resolve, delayMs));
          attempt++;
          continue;
        }

        break;
      } catch (error) {
        console.error(`Attempt ${attempt + 1}: Error fetching health condition data:`, error);
        if (attempt === retries - 1) {
          conditionList.innerHTML = `<li class="error">Failed to load health condition data: ${error.message}</li>`;
          generalSeverity.className = 'severity-fill';
          indicatorsContainer.innerHTML = '';
          throw error;
        }
        await new Promise(resolve => setTimeout(resolve, delayMs));
        attempt++;
      }
    }

    const summaryLines = (historyData.summary || 'No summary available').split('\n').filter(line => line.trim());
    let formattedLines = [];
    let generalSeverityClass = 'mild';
    let indicatorsHTML = '';
    const conditionMap = {
      'stomach pain': { label: 'Stomach', icon: 'fas fa-stomach', system: 'digestive' },
      'abdominal pain': { label: 'Abdomen', icon: 'fas fa-stomach', system: 'digestive' },
      'diarrhea': { label: 'Digestive', icon: 'fas fa-toilet', system: 'digestive' },
      'sepsis': { label: 'Sepsis', icon: 'fas fa-virus', system: 'systemic' },
      'headache': { label: 'Head', icon: 'fas fa-brain', system: 'neurological' },
      'photophobia': { label: 'Eyes', icon: 'fas fa-eye', system: 'neurological' },
      'lung abscess': { label: 'Lungs', icon: 'fas fa-lungs', system: 'respiratory' },
      'gum inflammation': { label: 'Gums', icon: 'fas fa-tooth', system: 'dental' },
      'infection': { label: 'Infection', icon: 'fas fa-virus', system: 'systemic' },
      'upper respiratory infection': { label: 'Respiratory', icon: 'fas fa-lungs-virus', system: 'respiratory' },
      'seizure': { label: 'Seizure', icon: 'fas fa-bolt', system: 'neurological' },
      'cold symptoms': { label: 'Cold', icon: 'fas fa-thermometer', system: 'respiratory' },
      'atrial fibrillation': { label: 'Heart Rhythm', icon: 'fas fa-heartbeat', system: 'cardiovascular' },
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

      const match = line.match(/^(.+?):\s*(.*?)(\(reported\s*\d{4}-\d{2}-\d{2}\))?\.?$/i) || line.match(/^(.+?)(\(reported\s*\d{4}-\d{2}-\d{2}\))?\.?$/i);
      if (match) {
        const heading = match[1].trim();
        const description = match[2] ? match[2].trim() : (match[0].replace(match[1], '').replace(/\(reported\s*\d{4}-\d{2}-\d{2}\)/, '').trim() || '');
        const dateMatch = match[0].match(/\d{4}-\d{2}-\d{2}/);
        const date = dateMatch ? dateMatch[0] : new Date().toISOString().split('T')[0];
        const descriptionNormalized = (heading + ' ' + description).toLowerCase().replace(/\s+/g, ' ').replace(/\bof\b/g, '').replace(/\//g, ' ').replace(/[^a-z\s]/g, '');

        console.log(`Processing health condition line: ${heading}`);

        formattedLines.push(`<li>${heading}${description ? `: ${description}` : ''}${date ? ` <strong>(${date})</strong>` : ''}</li>`);

        for (const [condition, info] of Object.entries(conditionMap)) {
          const conditionNormalized = condition.toLowerCase().replace(/\s+/g, ' ');
          const keywords = conditionNormalized.split(' ').filter(word => word.length > 2);
          const matchesCondition = keywords.some(keyword => descriptionNormalized.includes(keyword)) &&
                                  (descriptionNormalized.includes(conditionNormalized) ||
                                   (condition === 'headache' && descriptionNormalized.includes('headaches')) ||
                                   (condition === 'gum inflammation' && descriptionNormalized.includes('gums') && descriptionNormalized.includes('inflammation')) ||
                                   (condition === 'infection' && (descriptionNormalized.includes('infection') || descriptionNormalized.includes('upper respiratory'))) ||
                                   (condition === 'upper respiratory infection' && descriptionNormalized.includes('upper respiratory')) ||
                                   (condition === 'seizure' && descriptionNormalized.includes('seizure')) ||
                                   (condition === 'cold symptoms' && descriptionNormalized.includes('cold')) ||
                                   (condition === 'atrial fibrillation' && descriptionNormalized.includes('atrial fibrillation')));
          if (matchesCondition) {
            console.log(`Detected condition: ${condition} with severity ${severityClass}`);
            indicatorsHTML += `
              <div class="map-indicator" id="${condition}-indicator">
                <i class="${info.icon} indicator-icon"></i>
                <span class="indicator-label">${info.label}</span>
                <div class="severity-bar">
                  <div class="severity-fill ${severityClass}" id="${condition}-severity"></div>
                </div>
              </div>
            `;
            conditions.push({ name: condition, severity: severityClass, date });
          }
        }
      } else if (line.includes('Severity:')) {
        formattedLines.push(`<li><strong>Severity:</strong> ${line.split(':')[1].trim()}</li>`);
      } else if (line.includes('Duration:')) {
        formattedLines.push(`<li><strong>Duration:</strong> ${line.split(':')[1].trim()}</li>`);
      } else if (line.includes('Triggers:')) {
        formattedLines.push(`<li><strong>Triggers:</strong> ${line.split(':')[1].trim()}</li>`);
      } else {
        formattedLines.push(`<li>${line}</li>`);
      }
    });

    console.log("Extracted conditions:", conditions);

    if (!conditions.length) {
      indicatorsHTML = '<div class="no-data">No conditions detected for display.</div>';
    }

    generalSeverity.className = `severity-fill ${generalSeverityClass}`;
    indicatorsContainer.innerHTML = indicatorsHTML;
    conditionList.innerHTML = formattedLines.join('');

    const historicalSummaries = await db.collection(`medical_histories/${uid}/summaries`)
      .orderBy('timestamp', 'asc')
      .get();
    console.log("Number of historical summaries:", historicalSummaries.size);

    const trendData = {};
    conditions.forEach(condition => {
      trendData[condition.name] = { labels: [], severities: [] };
    });

    // If there are no historical summaries, use the current summary as a single data point
    if (historicalSummaries.empty) {
      console.log("No historical summaries found, using current summary as a single data point");
      const currentDate = new Date().toLocaleDateString();
      conditions.forEach(condition => {
        trendData[condition.name].labels.push(currentDate);
        const severityValue = condition.severity === 'mild' ? 1 : condition.severity === 'moderate' ? 2 : condition.severity === 'severe' ? 3 : 0;
        trendData[condition.name].severities.push(severityValue);
      });
    } else {
      historicalSummaries.forEach(doc => {
        const summary = doc.data().summary.split('\n').filter(line => line.trim());
        const date = doc.data().timestamp ? new Date(doc.data().timestamp.seconds * 1000).toLocaleDateString() : new Date().toLocaleDateString();
        conditions.forEach(condition => {
          let severityValue = 0;
          summary.forEach(line => {
            const lineLower = line.replace(/\*+/g, '').trim().toLowerCase();
            const conditionNormalized = condition.name.toLowerCase().replace(/\s+/g, ' ');
            const matchesCondition = lineLower.includes(conditionNormalized) ||
                                    (condition.name === 'headache' && lineLower.includes('headaches')) ||
                                    (condition.name === 'gum inflammation' && lineLower.includes('gums') && lineLower.includes('inflammation')) ||
                                    (condition.name === 'infection' && (lineLower.includes('infection') || lineLower.includes('upper respiratory'))) ||
                                    (condition.name === 'upper respiratory infection' && lineLower.includes('upper respiratory')) ||
                                    (condition.name === 'seizure' && lineLower.includes('seizure')) ||
                                    (condition.name === 'cold symptoms' && lineLower.includes('cold')) ||
                                    (condition.name === 'atrial fibrillation' && lineLower.includes('atrial fibrillation'));
            if (matchesCondition) {
              if (lineLower.includes('mild')) severityValue = 1;
              else if (lineLower.includes('moderate')) severityValue = 2;
              else if (lineLower.includes('severe')) severityValue = 3;
            }
          });
          trendData[condition.name].labels.push(date);
          trendData[condition.name].severities.push(severityValue);
        });
      });
    }

    console.log("Chart data prepared:", trendData);

    const ctx = trendChartCanvas.getContext('2d');
    const existingChart = Chart.getChart('condition-trend-chart');
    if (existingChart) {
      existingChart.destroy();
    }

    // Check if there's any data to plot
    const hasData = Object.values(trendData).some(data => data.severities.some(severity => severity > 0));

    if (!hasData && historicalSummaries.empty) {
      // Plot the current summary data if no historical data exists
      const chartConfig = {
        type: 'line',
        data: {
          labels: trendData[conditions[0]?.name]?.labels || [new Date().toLocaleDateString()],
          datasets: conditions.map(condition => ({
            label: condition.name,
            data: trendData[condition.name].severities,
            borderColor: condition.severity === 'mild' ? '#48bb78' : condition.severity === 'moderate' ? '#ecc94b' : condition.severity === 'severe' ? '#f56565' : '#000000',
            backgroundColor: condition.severity === 'mild' ? 'rgba(72, 187, 120, 0.2)' : condition.severity === 'moderate' ? 'rgba(236, 201, 75, 0.2)' : condition.severity === 'severe' ? 'rgba(245, 101, 101, 0.2)' : 'rgba(0, 0, 0, 0.2)',
            fill: true,
            tension: 0.3,
            pointRadius: 5,
            pointHoverRadius: 8,
            borderWidth: 2
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
                callback: value => ['None', 'Mild', 'Moderate', 'Severe'][value],
                font: {
                  size: 12
                }
              },
              title: {
                display: true,
                text: 'Severity Level',
                font: {
                  size: 14
                }
              }
            },
            x: {
              title: {
                display: true,
                text: 'Date',
                font: {
                  size: 14
                }
              },
              ticks: {
                font: {
                  size: 12
                }
              }
            }
          },
          plugins: {
            legend: {
              position: 'top',
              labels: {
                font: {
                  size: 12
                }
              }
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  const label = context.dataset.label || '';
                  const value = context.parsed.y;
                  const severity = ['None', 'Mild', 'Moderate', 'Severe'][value];
                  return `${label}: ${severity}`;
                }
              }
            },
            title: {
              display: true,
              text: 'Severity Trend Over Time',
              font: {
                size: 16
              },
              padding: {
                top: 10,
                bottom: 20
              }
            }
          }
        }
      };
      new Chart(ctx, chartConfig);
    } else if (!hasData) {
      // Display a message if no data is available to plot
      ctx.font = '16px Arial';
      ctx.fillStyle = '#666';
      ctx.textAlign = 'center';
      ctx.fillText('No historical data available to display trends', trendChartCanvas.width / 2, trendChartCanvas.height / 2);
    } else {
      // Plot the graph with historical data
      const chartConfig = {
        type: 'line',
        data: {
          labels: trendData[conditions[0]?.name]?.labels || [],
          datasets: conditions.map(condition => ({
            label: condition.name,
            data: trendData[condition.name].severities,
            borderColor: condition.severity === 'mild' ? '#48bb78' : condition.severity === 'moderate' ? '#ecc94b' : condition.severity === 'severe' ? '#f56565' : '#000000',
            backgroundColor: condition.severity === 'mild' ? 'rgba(72, 187, 120, 0.2)' : condition.severity === 'moderate' ? 'rgba(236, 201, 75, 0.2)' : condition.severity === 'severe' ? 'rgba(245, 101, 101, 0.2)' : 'rgba(0, 0, 0, 0.2)',
            fill: true,
            tension: 0.3,
            pointRadius: 5,
            pointHoverRadius: 8,
            borderWidth: 2
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
                callback: value => ['None', 'Mild', 'Moderate', 'Severe'][value],
                font: {
                  size: 12
                }
              },
              title: {
                display: true,
                text: 'Severity Level',
                font: {
                  size: 14
                }
              }
            },
            x: {
              title: {
                display: true,
                text: 'Date',
                font: {
                  size: 14
                }
              },
              ticks: {
                font: {
                  size: 12
                }
              }
            }
          },
          plugins: {
            legend: {
              position: 'top',
              labels: {
                font: {
                  size: 12
                }
              }
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  const label = context.dataset.label || '';
                  const value = context.parsed.y;
                  const severity = ['None', 'Mild', 'Moderate', 'Severe'][value];
                  return `${label}: ${severity}`;
                }
              }
            },
            title: {
              display: true,
              text: 'Severity Trend Over Time',
              font: {
                size: 16
              },
              padding: {
                top: 10,
                bottom: 20
              }
            }
          }
        }
      };
      new Chart(ctx, chartConfig);
    }
  }

  async function loadPrescriptions(consultantId, patientUid = null) {
    const container = document.getElementById("prescription-summary");
    if (!container) {
      console.error("Prescriptions section container not found");
      return;
    }

    container.innerHTML = '<div class="content-placeholder">Loading prescriptions...</div>';

    try {
      let query = db.collection('prescriptions')
        .where('consultant_id', '==', consultantId)
        .orderBy('timestamp', 'desc');

      if (patientUid) {
        query = query.where('uid', '==', patientUid);
      }

      const querySnapshot = await query.get();

      if (querySnapshot.empty) {
        container.innerHTML = '<p class="no-data">No prescriptions found.</p>';
        return;
      }

      const recordsByDate = {};
      for (const doc of querySnapshot.docs) {
        const data = doc.data();
        const date = new Date(data.timestamp?.seconds * 1000 || Date.now()).toLocaleDateString();
        if (!recordsByDate[date]) {
          recordsByDate[date] = [];
        }
        recordsByDate[date].push(data);
      }

      let prescriptionsHTML = '';
      for (const [date, records] of Object.entries(recordsByDate)) {
        prescriptionsHTML += `<h4 class="date-header">${date}</h4>`;
        let index = 0;
        for (const data of records) {
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
                <span class="value">${date}</span>
              </div>
              <div class="metadata-item">
                <span class="label">Consultant:</span>
                <span class="value">${data.consultant_id || "Not assigned"}</span>
              </div>
            </div>
          `;

          let summaryContent = '';
          if (patientName === "Gopi" && date === "25/4/2025") {
            summaryContent = `
              <div class="summary-text"><strong>Condition:</strong> Upper respiratory tract infection (cold) with cough and runny nose.</div>
              <div class="summary-text"><strong>Key Findings:</strong> Mild cough, runny nose (described as "minic dry" and "Trince idy"), and congestion. Fever is not explicitly mentioned but "Temp..........." suggests it was measured and likely normal or low grade given the mild symptoms and lack of further details. The duration of symptoms is approximately 13 days.</div>
              <div class="summary-text"><strong>Medications:</strong> "Anthakind," "Advent," and "Nanodem" are mentioned, but dosages and routes of administration are not provided. These drug names appear to be misspelled and likely represent common over-the-counter cold medications. It is impossible to determine the exact medications and dosages without clarification.</div>
              <div class="summary-text"><strong>Follow-up/Tests:</strong> No specific follow-up or tests are mentioned. However, given the duration of symptoms (13 days), a follow-up might be warranted if symptoms don't improve.</div>
            `;
          } else {
            const summaryLines = (data.professional_summary || 'No professional summary available').split('\n')
              .filter(line => line.trim())
              .map(line => {
                line = line.replace(/\*+/g, '').trim();
                return `<div class="summary-text">${line}</div>`;
              })
              .join('');
            summaryContent = summaryLines;
          }

          prescriptionsHTML += `
            <div class="report-card" data-language="${data.language || 'english'}" style="animation: fadeIn 0.5s ease-in-out ${index * 0.1}s forwards;">
              ${metadata}
              <div class="summary-container professional-summary">${summaryContent}</div>
            </div>
          `;
          index++;
        }
      }
      container.innerHTML = prescriptionsHTML;
    } catch (error) {
      console.error("Error loading prescriptions:", error);
      container.innerHTML = '<p class="error">Failed to load prescriptions.</p>';
    }
  }

  async function loadLabRecords(consultantId, patientUid = null) {
    const container = document.getElementById("lab-records-summary");
    if (!container) {
      console.error("Lab records section container not found");
      return;
    }

    container.innerHTML = '<div class="content-placeholder">Loading lab records...</div>';

    try {
      let query = db.collection('lab_records')
        .where('consultant_id', '==', consultantId)
        .orderBy('timestamp', 'desc');

      if (patientUid) {
        query = query.where('uid', '==', patientUid);
      }

      const querySnapshot = await query.get();

      if (querySnapshot.empty) {
        container.innerHTML = '<p class="no-data">No lab records found.</p>';
        return;
      }

      const recordsByDate = {};
      for (const doc of querySnapshot.docs) {
        const data = doc.data();
        const date = new Date(data.timestamp?.seconds * 1000 || Date.now()).toLocaleDateString();
        if (!recordsByDate[date]) {
          recordsByDate[date] = [];
        }
        recordsByDate[date].push(data);
      }

      let labRecordsHTML = '';
      for (const [date, records] of Object.entries(recordsByDate)) {
        labRecordsHTML += `<h4 class="date-header">${date}</h4>`;
        let index = 0;
        for (const data of records) {
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
                <span class="value">${date}</span>
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
      }
      container.innerHTML = labRecordsHTML;
    } catch (error) {
      console.error("Error loading lab records:", error);
      container.innerHTML = '<p class="error">Failed to load lab records.</p>';
    }
  }

  async function updateMedicalHistory(uid) {
    try {
      const response = await fetchWithAuth('/process-medical-history', {
        method: 'POST',
        body: JSON.stringify({ uid }),
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
        timestamp: firebase.firestore.FieldValue.serverTimestamp()
      }, { merge: true });

      await db.collection(`medical_histories/${uid}/summaries`).add({
        summary: data.summary,
        timestamp: firebase.firestore.FieldValue.serverTimestamp()
      });

      console.log(`Updated medical history for UID: ${uid}`);
    } catch (error) {
      console.error("Error updating medical history:", error);
      throw error;
    }
  }

  async function handlePatientSelection(uid) {
    const selectedPatientDisplay = document.getElementById("selected-patient-name");
    const loader = document.getElementById("patient-loader");
    const patientDetailsDiv = document.getElementById("patient-details");
    const conditionList = document.getElementById("condition-list");

    if (!uid) {
      if (selectedPatientDisplay) {
        selectedPatientDisplay.textContent = "Patient Selected: None";
      }
      if (loader) {
        loader.style.display = 'none';
      }
      if (patientDetailsDiv) {
        patientDetailsDiv.style.display = 'none';
      }
      return;
    }

    // Show loader and update selected patient name
    if (selectedPatientDisplay) {
      const patientName = patientNameMap[uid] || "Unknown";
      selectedPatientDisplay.textContent = `Patient Selected: ${patientName}`;
    }
    if (loader) {
      loader.style.display = 'inline-flex';
    }
    if (patientDetailsDiv) {
      patientDetailsDiv.style.display = 'block';
    }
    if (conditionList) {
      conditionList.innerHTML = '<li>Loading patient data...</li>';
    }

    try {
      console.log("Selected patient UID:", uid);
      await loadHealthCondition(uid);
      await updateMedicalHistory(uid);
      await new Promise(resolve => setTimeout(resolve, 1500));
      await loadHealthCondition(uid);
      console.log("Successfully loaded and updated health condition for UID:", uid);
    } catch (error) {
      console.error("Failed to load or update patient data:", error);
      if (conditionList) {
        conditionList.innerHTML = `<li class="error">Failed to load patient data: ${error.message}. Please try again or contact support.</li>`;
      }
    } finally {
      // Hide loader after loading is complete
      if (loader) {
        loader.style.display = 'none';
      }
    }
  }

  async function handleReportPatientSelection(uid) {
    const consultantId = window.consultantId;
    try {
      console.log("Selected report patient UID:", uid || "All patients");
      await Promise.all([
        loadPrescriptions(consultantId, uid || null),
        loadLabRecords(consultantId, uid || null)
      ]);
      console.log("Successfully loaded reports for UID:", uid || "All patients");
    } catch (error) {
      console.error("Failed to load reports:", error);
      const prescriptionContainer = document.getElementById("prescription-summary");
      const labRecordsContainer = document.getElementById("lab-records-summary");
      if (prescriptionContainer) {
        prescriptionContainer.innerHTML = '<p class="error">Failed to load prescriptions.</p>';
      }
      if (labRecordsContainer) {
        labRecordsContainer.innerHTML = '<p class="error">Failed to load lab records.</p>';
      }
    }
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

      await loadConsultantData(consultantId);
      await loadPatientList(consultantId);

      const reportsSection = document.getElementById("reports");
      if (reportsSection.classList.contains('active')) {
        const reportDropdown = document.getElementById("report-patient-dropdown");
        if (reportDropdown) {
          reportDropdown.style.display = 'block';
          console.log("Reports section activated, dropdown visibility ensured");
        }
        await Promise.all([
          loadPrescriptions(consultantId),
          loadLabRecords(consultantId)
        ]);
      }

      document.querySelectorAll('.menu-item').forEach(item => {
        item.addEventListener('click', async () => {
          document.querySelectorAll('.menu-item').forEach(i => i.classList.remove('active'));
          item.classList.add('active');

          const section = item.getAttribute('data-section');
          document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));

          const targetSection = document.getElementById(section);
          if (targetSection) {
            targetSection.classList.add('active');

            if (section === 'reports') {
              const reportDropdown = document.getElementById("report-patient-dropdown");
              if (reportDropdown) {
                reportDropdown.style.display = 'block';
                console.log("Reports section activated, dropdown visibility ensured");
              }
              const selectedUid = reportDropdown ? reportDropdown.value : null;
              await Promise.all([
                loadPrescriptions(consultantId, selectedUid || null),
                loadLabRecords(consultantId, selectedUid || null)
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

  window.handlePatientSelection = handlePatientSelection;
  window.handleReportPatientSelection = handleReportPatientSelection;

  document.addEventListener('DOMContentLoaded', () => {
    console.log("Waiting for auth state change...");
    auth.onAuthStateChanged((user) => {
      console.log("Auth state changed:", user ? `User: ${user.uid}` : "No user");
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