(function() {
  const auth = window.auth;
  const db = window.db;
  const storage = window.storage;

  console.log("Firebase services initialized:", { auth, db, storage });

  function getGreeting() {
    const hour = new Date().getHours();
    return hour < 12 ? "Good Morning!" : hour < 18 ? "Good Afternoon!" : "Good Evening!";
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
      console.log("Patient data:", patientData);
      document.getElementById("patient-id").textContent = uid;
      document.getElementById("next-visit").textContent = patientData.next_visit || "Not scheduled";
      document.getElementById("consultant-name").textContent = patientData.consultant_id
        ? await db.doc(`consultant_registrations/${patientData.consultant_id}`).get()
            .then(snap => snap.exists ? snap.data().full_name : "Unknown Consultant")
            .catch(() => "Unknown Consultant")
        : "Not assigned";
    } catch (error) {
      console.error("Error loading patient data:", error);
    }
  }

  async function loadPrescriptions(uid) {
    const container = document.getElementById("prescription-summary");
    if (!container) return;

    container.innerHTML = "<div class='summary-title' style='font-size: 1.5em; margin-bottom: 15px; text-align: left; border-bottom: 2px solid #007bff; padding-bottom: 5px; color: #007bff;'>Prescriptions</div>";
    try {
      const querySnapshot = await db.collection('prescriptions')
        .where('uid', '==', uid)
        .orderBy('timestamp', 'desc')
        .get();

      if (querySnapshot.empty) {
        container.innerHTML += "<p>No prescriptions found.</p>";
        return;
      }

      container.innerHTML += "<div class='report-container' style='margin-top: 20px;'>";
      for (const doc of querySnapshot.docs) {
        const data = doc.data();
        const consultantName = data.consultant_id
          ? await db.doc(`consultant_registrations/${data.consultant_id}`).get()
              .then(snap => snap.exists ? snap.data().full_name : "Unknown")
              .catch(() => "Unknown")
          : "Not assigned";

        const metadata = `
          <div style="margin-bottom: 15px; text-align: left; font-family: 'Times New Roman', serif; color: #444;">
            <p style="margin-bottom: 10px;"><strong>Patient Name:</strong> ${data.patient_name || 'Unknown'}</p>
            <p style="margin-bottom: 10px;"><strong>Date:</strong> ${new Date(data.timestamp?.seconds * 1000 || Date.now()).toLocaleDateString()}</p>
            <p><strong>Consultant:</strong> ${consultantName}</p>
          </div>
        `;

        const summaryLines = (data.summary || 'No summary available').split('\n')
          .filter(line => line.trim())
          .map(line => {
            line = line.replace(/\*+/g, '').trim();
            if (line.startsWith('**') && line.includes(':**')) {
              const heading = line.replace(/^(\*\*[^:]+:\*\*)/, '$1').replace(/\*\*/g, '');
              return `<div class="summary-heading" style="margin-bottom: 8px;"><strong>${heading}</strong></div>`;
            }
            return `<div class="summary-text" style="margin-bottom: 8px;">${line}</div>`;
          })
          .join('');

        container.innerHTML += `
          <div class="report-section" style="border: 1px solid #ccc; padding: 20px; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); background: #fff; text-align: left; transition: transform 0.3s;">
            ${metadata}
            <div class="summary-container" style="background: #f9f9f9; border: 1px solid #ddd; border-radius: 5px; padding: 15px; line-height: 1.8; font-family: 'Courier New', monospace; color: #333;">
              ${summaryLines}
            </div>
          </div>
        `;
      }
      container.innerHTML += "</div>";
    } catch (error) {
      console.error("Error loading prescriptions:", error);
      container.innerHTML += "<p>Failed to load prescriptions.</p>";
    }
  }

  async function loadLabRecords(uid) {
    const container = document.getElementById("lab-records-summary");
    if (!container) return;

    container.innerHTML = "<div class='summary-title' style='font-size: 1.5em; margin-bottom: 15px; text-align: left; border-bottom: 2px solid #007bff; padding-bottom: 5px; color: #007bff;'>Lab Records</div>";
    try {
      const querySnapshot = await db.collection('lab_records')
        .where('uid', '==', uid)
        .orderBy('timestamp', 'desc')
        .get();

      if (querySnapshot.empty) {
        container.innerHTML += "<p>No lab records found.</p>";
        return;
      }

      container.innerHTML += "<div class='report-container' style='margin-top: 20px;'>";
      for (const doc of querySnapshot.docs) {
        const data = doc.data();
        const consultantName = data.consultant_id
          ? await db.doc(`consultant_registrations/${data.consultant_id}`).get()
              .then(snap => snap.exists ? snap.data().full_name : "Unknown")
              .catch(() => "Unknown")
          : "Not assigned";

        const metadata = `
          <div style="margin-bottom: 15px; text-align: left; font-family: 'Times New Roman', serif; color: #444;">
            <p style="margin-bottom: 10px;"><strong>Patient Name:</strong> ${data.patient_name || 'Unknown'}</p>
            <p style="margin-bottom: 10px;"><strong>Date:</strong> ${new Date(data.timestamp?.seconds * 1000 || Date.now()).toLocaleDateString()}</p>
            <p><strong>Consultant:</strong> ${consultantName}</p>
          </div>
        `;

        const summaryLines = (data.summary || 'No summary available').split('\n')
          .filter(line => line.trim())
          .map(line => {
            line = line.replace(/\*+/g, '').trim();
            if (line.startsWith('**') && line.includes(':**')) {
              const heading = line.replace(/^(\*\*[^:]+:\*\*)/, '$1').replace(/\*\*/g, '');
              return `<div class="summary-heading" style="margin-bottom: 8px;"><strong>${heading}</strong></div>`;
            }
            return `<div class="summary-text" style="margin-bottom: 8px;">${line}</div>`;
          })
          .join('');

        container.innerHTML += `
          <div class="report-section" style="border: 1px solid #ccc; padding: 20px; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); background: #fff; text-align: left; transition: transform 0.3s;">
            ${metadata}
            <div class="summary-container" style="background: #f9f9f9; border: 1px solid #ddd; border-radius: 5px; padding: 15px; line-height: 1.8; font-family: 'Courier New', monospace; color: #333;">
              ${summaryLines}
            </div>
          </div>
        `;
      }
      container.innerHTML += "</div>";
    } catch (error) {
      console.error("Error loading lab records:", error);
      container.innerHTML += "<p>Failed to load lab records.</p>";
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
        alert(`${category === 'prescriptions' ? 'Prescription' : 'Lab Record'} processed successfully!`);
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

  function renderVisitsChart() {
    const ctx = document.getElementById("visitsChart")?.getContext("2d");
    if (!ctx) return;

    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const visitsData = Array(12).fill().map(() => Math.floor(Math.random() * 10 + 1));

    new Chart(ctx, {
      type: "bar",
      data: {
        labels: months,
        datasets: [{
          label: "Number of Visits",
          data: visitsData,
          backgroundColor: "#1a73e8",
        }],
      },
      options: {
        scales: {
          y: { beginAtZero: true, title: { display: true, text: "Visits" } },
          x: { title: { display: true, text: "Months" } },
        },
      },
    });
  }

  window.onload = () => {
    auth.onAuthStateChanged((user) => {
      if (!user) {
        console.log("No user authenticated, redirecting to login");
        window.location.href = "/login";
        return;
      }
      console.log("User authenticated, initializing dashboard");
      initializeDashboard();
    });
  };

  async function initializeDashboard() {
    try {
      const response = await fetchWithAuth('/dashboard');
      if (!response.ok) {
        throw new Error(`Dashboard fetch failed: ${response.status}`);
      }
      const html = await response.text();
      document.body.innerHTML = html;

      const user = auth.currentUser;
      if (!user) {
        console.log("No user after dashboard load, redirecting to /login");
        window.location.href = "/login";
        return;
      }

      const uid = user.uid;
      await Promise.all([
        loadPatientData(uid),
        loadPrescriptions(uid),
        loadLabRecords(uid)
      ]);
      renderVisitsChart();

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
        labRecordBtn: document.getElementById("labRecordBtn")
      };

      if (Object.values(elements).some(el => !el)) {
        console.error("Missing required DOM elements:", elements);
        return;
      }

      let selectedFile = null;
      let selectedCategory = "prescriptions";

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
        document.querySelectorAll('.toggle').forEach(t => t.classList.remove('active'));
        document.querySelector('.toggle[data-section="prescriptions"]').classList.add('active');
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        document.getElementById("prescriptions")?.classList.add('active');
      });

      elements.labRecordBtn.addEventListener("click", () => {
        selectedCategory = "lab_records";
        elements.labRecordBtn.classList.add("active");
        elements.prescriptionBtn.classList.remove("active");
        document.querySelectorAll('.toggle').forEach(t => t.classList.remove('active'));
        document.querySelector('.toggle[data-section="lab-records"]').classList.add('active');
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        document.getElementById("lab-records")?.classList.add('active');
      });

      document.querySelectorAll('.toggle').forEach(toggle => {
        toggle.addEventListener('click', () => {
          const section = toggle.getAttribute('data-section');
          document.querySelectorAll('.toggle').forEach(t => t.classList.remove('active'));
          toggle.classList.add('active');
          document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
          const targetSection = document.getElementById(section);
          if (targetSection) {
            targetSection.classList.add('active');
          }
          if (section === 'prescriptions') {
            selectedCategory = "prescriptions";
            elements.prescriptionBtn.classList.add('active');
            elements.labRecordBtn.classList.remove('active');
          } else if (section === 'lab-records') {
            selectedCategory = "lab_records";
            elements.labRecordBtn.classList.add('active');
            elements.prescriptionBtn.classList.remove('active');
          } else {
            elements.prescriptionBtn.classList.remove('active');
            elements.labRecordBtn.classList.remove('active');
          }
        });
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
      window.location.href = "/";
    }
  }
})();