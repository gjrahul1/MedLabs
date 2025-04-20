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
      console.log("Retrieved auth token:", token);
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

  async function loadPatientData(uid) {
    console.log("Loading patient data for UID:", uid);
    const patientRef = db.doc(`patient_registrations/${uid}`);
    try {
      const patientSnap = await patientRef.get();
      if (!patientSnap.exists) {
        console.error("UID unmatched in patient_registrations");
        alert("UID unmatched in patient records!");
        return;
      }
      const patientData = patientSnap.data();
      console.log("Patient data fetched from Firestore:", patientData);

      const patientIdElement = document.getElementById("patient-id");
      const nextVisitElement = document.getElementById("next-visit");
      const consultantNameElement = document.getElementById("consultant-name");
      const headerTitle = document.querySelector(".dashboard-header h1");

      if (!patientIdElement || !nextVisitElement || !consultantNameElement || !headerTitle) {
        console.error("DOM elements missing:", {
          patientIdElement: !!patientIdElement,
          nextVisitElement: !!nextVisitElement,
          consultantNameElement: !!consultantNameElement,
          headerTitle: !!headerTitle
        });
        return;
      }

      patientIdElement.textContent = uid || "N/A";
      nextVisitElement.textContent = patientData.next_visit ? new Date(patientData.next_visit.seconds * 1000).toLocaleDateString() : "Not scheduled";
      consultantNameElement.textContent = patientData.consultant_id
        ? await db.doc(`consultant_registrations/${patientData.consultant_id}`).get()
            .then(snap => {
              if (snap.exists) {
                const consultantName = snap.data().full_name;
                console.log("Consultant name fetched:", consultantName);
                return consultantName;
              } else {
                console.warn(`No consultant found for ID: ${patientData.consultant_id}`);
                return "No Consultant Assigned";
              }
            })
            .catch(error => {
              console.error("Error fetching consultant name:", error);
              return "No Consultant Assigned";
            })
        : "No Consultant Assigned";

      // Update header with greeting and patient name
      headerTitle.textContent = `${getGreeting()}, ${patientData.full_name || 'Patient'}!`;

      // Show notification if no consultant is assigned
      if (!patientData.consultant_id) {
        document.getElementById("doctor-notification").style.display = "block";
      }
    } catch (error) {
      console.error("Error loading patient data:", error);
    }
  }

  async function loadInitialScreening(uid) {
    const container = document.querySelector("#home .screening-container");
    if (!container) {
      console.error("Screening container not found");
      return;
    }

    container.innerHTML = `
      <h2 class="section-title">Initial Screening Details</h2>
      <div class="content-placeholder">Loading initial screening details...</div>
    `;

    try {
      const screeningDoc = await db.collection("initial_screenings").doc(`initial_screening_${uid}`).get();
      if (!screeningDoc.exists) {
        container.innerHTML = `
          <h2 class="section-title">Initial Screening Details</h2>
          <p class="no-data">No initial screening data available.</p>
        `;
        return;
      }

      const data = screeningDoc.data();
      const screeningHTML = `
        <h2 class="section-title">Initial Screening Details</h2>
        <div class="screening-cards">
          <div class="screening-card">
            <div class="card-header"><i class="fas fa-heartbeat card-icon"></i> Symptoms</div>
            <div class="card-content">${data.symptoms || 'N/A'}</div>
          </div>
          <div class="screening-card">
            <div class="card-header"><i class="fas fa-exclamation-triangle card-icon"></i> Severity</div>
            <div class="card-content">
              <span class="severity-badge ${data.severity ? data.severity.toLowerCase() : 'unknown'}">
                ${data.severity || 'Unknown'}
              </span>
            </div>
          </div>
          <div class="screening-card">
            <div class="card-header"><i class="fas fa-clock card-icon"></i> Duration</div>
            <div class="card-content">${data.duration || 'N/A'}</div>
          </div>
          <div class="screening-card">
            <div class="card-header"><i class="fas fa-bolt card-icon"></i> Triggers</div>
            <div class="card-content">${data.triggers || 'N/A'}</div>
          </div>
          <div class="screening-card full-width">
            <div class="card-header"><i class="fas fa-calendar-alt card-icon"></i> Date</div>
            <div class="card-content">${new Date(data.timestamp?.seconds * 1000).toLocaleDateString()}</div>
          </div>
        </div>
      `;

      container.innerHTML = screeningHTML;

      // Add fade-in animation to screening cards
      container.querySelectorAll('.screening-card').forEach((card, index) => {
        card.style.animation = `fadeIn 0.5s ease-in-out ${index * 0.1}s forwards`;
      });
    } catch (error) {
      console.error("Error loading initial screening:", error);
      container.innerHTML = `
        <h2 class="section-title">Initial Screening Details</h2>
        <p class="error">Failed to load initial screening data.</p>
      `;
    }
  }

  async function loadPrescriptions(uid) {
    const container = document.getElementById("prescriptions");
    if (!container) {
      console.error("Prescriptions section container not found");
      return;
    }

    container.innerHTML = `
      <h2 class="section-title">Prescriptions</h2>
      <div class="content-placeholder">Loading prescriptions...</div>
    `;

    try {
      const querySnapshot = await db.collection('prescriptions')
        .where('uid', '==', uid)
        .orderBy('timestamp', 'desc')
        .get();

      if (querySnapshot.empty) {
        container.innerHTML = `
          <h2 class="section-title">Prescriptions</h2>
          <p class="no-data">No prescriptions found.</p>
        `;
        return;
      }

      let prescriptionsHTML = `
        <h2 class="section-title">Prescriptions</h2>
        <div class="report-container">
      `;
      let index = 0;
      for (const doc of querySnapshot.docs) {
        const data = doc.data();
        const consultantName = data.consultant_id
          ? await db.doc(`consultant_registrations/${data.consultant_id}`).get()
              .then(snap => snap.exists ? snap.data().full_name : "Unknown")
              .catch(() => "Unknown")
          : "Not assigned";

        const metadata = `
          <div class="metadata">
            <div class="metadata-item">
              <span class="label">Patient Name:</span>
              <span class="value">${data.patient_name || 'Unknown'}</span>
            </div>
            <div class="metadata-item">
              <span class="label">Date:</span>
              <span class="value">${new Date(data.timestamp?.seconds * 1000 || Date.now()).toLocaleDateString()}</span>
            </div>
            <div class="metadata-item">
              <span class="label">Consultant:</span>
              <span class="value">${consultantName}</span>
            </div>
          </div>
        `;

        // Parse markdown summary using marked library
        const regionalSummaryHTML = marked.parse(data.summary || 'No summary available');
        const professionalSummaryHTML = marked.parse(data.professional_summary || 'No professional summary available');

        prescriptionsHTML += `
          <div class="report-card" data-language="${data.language || 'english'}" style="animation: fadeIn 0.5s ease-in-out ${index * 0.1}s forwards;">
            ${metadata}
            <div class="summary-container regional-summary">${regionalSummaryHTML}</div>
            <div class="summary-container professional-summary" style="display: none;">${professionalSummaryHTML}</div>
          </div>
        `;
        index++;
      }
      prescriptionsHTML += "</div>";
      container.innerHTML = prescriptionsHTML;
    } catch (error) {
      console.error("Error loading prescriptions:", error);
      container.innerHTML = `
        <h2 class="section-title">Prescriptions</h2>
        <p class="error">Failed to load prescriptions.</p>
      `;
    }
  }

  async function loadLabRecords(uid) {
    const container = document.getElementById("lab-records");
    if (!container) {
      console.error("Lab records section container not found");
      return;
    }

    container.innerHTML = `
      <h2 class="section-title">Lab Reports</h2>
      <div class="content-placeholder">Loading lab records...</div>
    `;

    try {
      const querySnapshot = await db.collection('lab_records')
        .where('uid', '==', uid)
        .orderBy('timestamp', 'desc')
        .get();

      if (querySnapshot.empty) {
        container.innerHTML = `
          <h2 class="section-title">Lab Reports</h2>
          <p class="no-data">No lab records found.</p>
        `;
        return;
      }

      let labRecordsHTML = `
        <h2 class="section-title">Lab Reports</h2>
        <div class="report-container">
      `;
      let index = 0;
      for (const doc of querySnapshot.docs) {
        const data = doc.data();
        const consultantName = data.consultant_id
          ? await db.doc(`consultant_registrations/${data.consultant_id}`).get()
              .then(snap => snap.exists ? snap.data().full_name : "Unknown")
              .catch(() => "Unknown")
          : "Not assigned";

        const metadata = `
          <div class="metadata">
            <div class="metadata-item">
              <span class="label">Patient Name:</span>
              <span class="value">${data.patient_name || 'Unknown'}</span>
            </div>
            <div class="metadata-item">
              <span class="label">Date:</span>
              <span class="value">${new Date(data.timestamp?.seconds * 1000 || Date.now()).toLocaleDateString()}</span>
            </div>
            <div class="metadata-item">
              <span class="label">Consultant:</span>
              <span class="value">${consultantName}</span>
            </div>
          </div>
        `;

        // Parse markdown summary using marked library
        const regionalSummaryHTML = marked.parse(data.summary || 'No summary available');
        const professionalSummaryHTML = marked.parse(data.professional_summary || 'No professional summary available');

        labRecordsHTML += `
          <div class="report-card" data-language="${data.language || 'english'}" style="animation: fadeIn 0.5s ease-in-out ${index * 0.1}s forwards;">
            ${metadata}
            <div class="summary-container regional-summary">${regionalSummaryHTML}</div>
            <div class="summary-container professional-summary" style="display: none;">${professionalSummaryHTML}</div>
          </div>
        `;
        index++;
      }
      labRecordsHTML += "</div>";
      container.innerHTML = labRecordsHTML;
    } catch (error) {
      console.error("Error loading lab records:", error);
      container.innerHTML = `
        <h2 class="section-title">Lab Reports</h2>
        <p class="error">Failed to load lab records.</p>
      `;
    }
  }

  async function uploadImage(uid, file, category) {
    console.log("Uploading file:", file.name, "for UID:", uid, "Category:", category);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("category", category);
    formData.append("uid", uid);

    try {
      const patientRef = db.doc(`patient_registrations/${uid}`);
      const patientSnap = await patientRef.get();
      if (patientSnap.exists) {
        const patientData = patientSnap.data();
        const consultantId = patientData.consultant_id;
        if (consultantId) {
          formData.append("consultantId", consultantId);
          console.log("Added consultantId:", consultantId, "to upload request");
        } else {
          console.warn("No consultant_id found for UID:", uid);
        }
      } else {
        console.warn("Patient data not found for UID:", uid);
      }
    } catch (error) {
      console.error("Error fetching consultant_id:", error);
    }

    try {
      const response = await fetchWithAuth('/upload-image', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Upload failed with status ${response.status}: ${errorText}`);
        throw new Error(`Upload failed with status ${response.status}: ${errorText}`);
      }
      const data = await response.json();
      if (!data.success) throw new Error(data.error || "Upload failed");

      console.log("File path received:", data.filePath);
      return data.filePath;
    } catch (error) {
      console.error("Upload error:", error);
      throw error;
    }
  }

  async function sendDataToBackend(uid, languageText, imagePath, category) {
    try {
      const patientRef = db.doc(`patient_registrations/${uid}`);
      const patientSnap = await patientRef.get();
      const patientData = patientSnap.data();

      const formData = new FormData();
      formData.append("languageText", languageText);
      formData.append("filePath", imagePath);
      formData.append("category", category);
      formData.append("uid", uid);
      if (patientData?.consultant_id) {
        formData.append("consultantId", patientData.consultant_id);
        console.log("Added consultantId:", patientData.consultant_id, "to process-upload request");
      } else {
        console.warn("No consultant_id found for UID:", uid);
      }

      console.log("Sending data to /process-upload:", {
        languageText,
        filePath: imagePath,
        category,
        uid,
        consultantId: patientData?.consultant_id
      });

      const response = await fetchWithAuth("/process-upload", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      console.log("Response from /process-upload:", data);

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${data.error || 'Unknown error'}`);
      }
      if (data.success) {
        // Update preferred language in local storage
        localStorage.setItem('preferredLanguage', languageText);
        alert(`${category === 'prescriptions' ? 'Prescription' : 'Lab Record'} processed successfully!`);
        // Reload summaries to reflect the new language
        category === "prescriptions" ? await loadPrescriptions(uid) : await loadLabRecords(uid);
      } else {
        throw new Error(data.error || "Processing failed");
      }
    } catch (error) {
      console.error("Error sending data to backend:", error);
      alert("Failed to process: " + error.message);
      throw error;
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

      const uid = user.uid;
      console.log("User authenticated with UID:", uid);

      // Check for registration prompt
      const patientRef = db.doc(`patient_registrations/${uid}`);
      const patientSnap = await patientRef.get();
      if (patientSnap.exists && !patientSnap.data().consultant_id) {
        document.getElementById("registration-prompt").style.display = "block";
      }

      await Promise.all([
        loadPatientData(uid),
        loadInitialScreening(uid),
        loadPrescriptions(uid),
        loadLabRecords(uid)
      ]);

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

      const elements = {
        openUploadBtn: document.getElementById("openUpload"),
        fileInput: document.getElementById("imageUpload"),
        sendBtn: document.getElementById("sendBtn"),
        languageInput: document.getElementById("languageInput"),
        prescriptionBtn: document.getElementById("prescriptionBtn"),
        labRecordBtn: document.getElementById("labRecordBtn"),
        toggleLanguageBtn: document.getElementById("toggle-language"),
        chatInputArea: document.querySelector(".chat-input-area")
      };

      if (Object.values(elements).some(el => !el)) {
        console.error("Missing required DOM elements:", elements);
        return;
      }

      let selectedFile = null;
      let selectedCategory = "prescriptions";
      let preferredLanguage = localStorage.getItem('preferredLanguage') || 'kannada'; // Default to Kannada
      let displayLanguage = localStorage.getItem('displayLanguage') || 'regional'; // Default to regional

      // Update toggle button text based on initial display language
      elements.toggleLanguageBtn.textContent = displayLanguage === 'regional' ? 'Switch to English' : 'Switch to Regional Language';

      // Function to toggle summary display
      function toggleSummaries() {
        const cards = document.querySelectorAll('.report-card');
        cards.forEach(card => {
          const regionalSummary = card.querySelector('.regional-summary');
          const professionalSummary = card.querySelector('.professional-summary');
          if (displayLanguage === 'regional') {
            regionalSummary.style.display = 'block';
            professionalSummary.style.display = 'none';
          } else {
            regionalSummary.style.display = 'none';
            professionalSummary.style.display = 'block';
          }
        });
      }

      // Initial display of summaries
      toggleSummaries();

      // Language toggle button event listener
      elements.toggleLanguageBtn.addEventListener('click', async () => {
        displayLanguage = displayLanguage === 'regional' ? 'english' : 'regional';
        localStorage.setItem('displayLanguage', displayLanguage);
        elements.toggleLanguageBtn.textContent = displayLanguage === 'regional' ? 'Switch to English' : 'Switch to Regional Language';
        toggleSummaries();
      });

      // Sidebar navigation (moved from inline script in HTML)
      document.querySelectorAll('.menu-item').forEach(item => {
        item.addEventListener('click', () => {
          document.querySelectorAll('.menu-item').forEach(i => i.classList.remove('active'));
          item.classList.add('active');

          const section = item.getAttribute('data-section');
          document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));

          const targetSection = document.getElementById(section);
          if (targetSection) {
            targetSection.classList.add('active');
          }

          // Hide chat input area when Prescriptions or Lab Reports is selected
          if (section === 'prescriptions' || section === 'lab-records') {
            elements.chatInputArea.style.display = 'none';
            elements.prescriptionBtn.classList.remove('active');
            elements.labRecordBtn.classList.remove('active');
          } else {
            elements.chatInputArea.style.display = 'flex';
            elements.prescriptionBtn.classList.remove('active');
            elements.labRecordBtn.classList.remove('active');
          }
        });
      });

      elements.openUploadBtn.addEventListener("click", () => elements.fileInput.click());

      elements.fileInput.addEventListener("change", (e) => {
        selectedFile = e.target.files[0];
        if (selectedFile && !selectedFile.type.startsWith("image/")) {
          alert("Only image files are allowed.");
          selectedFile = null;
        } else if (selectedFile) {
          if (selectedCategory === "prescriptions") {
            elements.prescriptionBtn.classList.add("active");
            elements.labRecordBtn.classList.remove("active");
            alert("Image will be sent as a Prescription");
          } else if (selectedCategory === "lab_records") {
            elements.labRecordBtn.classList.add("active");
            elements.prescriptionBtn.classList.remove("active");
            alert("Image will be sent as a Lab Record");
          }
        }
      });

      elements.prescriptionBtn.addEventListener("click", () => {
        selectedCategory = "prescriptions";
        elements.prescriptionBtn.classList.add("active");
        elements.labRecordBtn.classList.remove("active");
      });

      elements.labRecordBtn.addEventListener("click", () => {
        selectedCategory = "lab_records";
        elements.labRecordBtn.classList.add("active");
        elements.prescriptionBtn.classList.remove("active");
      });

      elements.sendBtn.addEventListener("click", async () => {
        const languageText = elements.languageInput.value.trim();
        if (!selectedFile) return alert("Please select an image first.");
        if (!languageText) return alert("Please specify a language");

        try {
          const imagePath = await uploadImage(uid, selectedFile, selectedCategory);
          await sendDataToBackend(uid, languageText, imagePath, selectedCategory);
          elements.languageInput.value = "";
          elements.fileInput.value = "";
          selectedFile = null;
        } catch (error) {
          alert("Error processing upload: " + error.message);
        }
      });

      elements.languageInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          elements.sendBtn.click();
        }
      });
    } catch (error) {
      console.error("Dashboard initialization error:", error);
      window.location.href = "/login";
    }
  }

  // Initialize on auth state change to ensure login state is resolved
  document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded, checking Firebase initialization");
    if (typeof window.auth === 'undefined' || typeof window.db === 'undefined' || typeof window.storage === 'undefined') {
      console.error("Firebase services not initialized. Ensure firebaseConfig.js is loaded correctly.");
      window.location.href = "/login";
      return;
    }

    console.log("Waiting for auth state change...");
    let hasRedirected = false; // Prevent multiple redirects
    auth.onAuthStateChanged((user) => {
      console.log("Auth state changed:", user ? `User: ${user.email}, UID: ${user.uid}` : "No user");
      if (!user && !hasRedirected) {
        console.log("No user authenticated, redirecting to login");
        hasRedirected = true;
        window.location.href = "/login";
        return;
      }
      if (user && !hasRedirected) {
        console.log("User authenticated, initializing dashboard");
        initializeDashboard();
      }
    });
  });

  window.initializeDashboard = initializeDashboard;
})();