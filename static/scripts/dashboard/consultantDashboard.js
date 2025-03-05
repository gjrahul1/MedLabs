const db = window.db;
const storage = window.storage;

function loadConsultantDashboard() {
  firebase.auth().onAuthStateChanged((user) => {
    if (user) {
      console.log("Authenticated user found:", user.uid);
      loadDashboardData(user);
    } else {
      console.error("No authenticated user found.");
      const container = document.getElementById("patient-list");
      if (container) container.innerHTML = "<p>Please log in to view the dashboard.</p>";
    }
  });
}

function formatSummary(summaryText) {
  if (!summaryText) return '<ul class="summary-list"><li>No summary</li></ul>';
  const lines = summaryText.split('\n')
    .filter(line => line.trim().length > 0)
    .map(line => line.replace(/\*/g, '').replace(/\[Age\]/g, '').trim());
  if (lines.length > 1) {
    return '<ul class="summary-list">' + lines.map(line => `<li>${line}</li>`).join('') + '</ul>';
  }
  return `<ul class="summary-list"><li>${lines[0]}</li></ul>`;
}

async function loadDashboardData(user) {
  console.log("Starting loadConsultantDashboard");
  const consultantId = window.consultantId || user.uid;
  console.log("Using consultantId:", consultantId);
  if (!consultantId) {
    console.error("No valid consultantId or user UID found.");
    document.getElementById("patient-list").innerHTML = "<p>Invalid consultant or user ID.</p>";
    return;
  }

  const container = document.getElementById("patient-list");
  const nextPatientContainer = document.getElementById("next-patient");
  const prescriptionContainer = document.getElementById("prescription-summary");
  const labRecordsContainer = document.getElementById("lab-records-summary");

  if (!container || !nextPatientContainer || !prescriptionContainer || !labRecordsContainer) {
    console.error("Required DOM elements not found in consultantDashboard.html");
    container.innerHTML = "<p>Error: Dashboard containers missing. Contact support.</p>";
    return;
  }

  try {
    const consultantQuery = await db.collection("consultant_registrations").where("uid", "==", user.uid).limit(1).get();
    if (consultantQuery.empty) {
      console.error("Consultant not found for Firebase UID:", user.uid);
      container.innerHTML = `<p>Consultant data not found for UID: ${user.uid}.</p>`;
      nextPatientContainer.innerHTML = "<p>No upcoming patients</p>";
      prescriptionContainer.innerHTML = "<p>No prescriptions available</p>";
      labRecordsContainer.innerHTML = "<p>No lab records available</p>";
      return;
    }

    const consultantData = consultantQuery.docs[0].data();
    console.log("Consultant data:", consultantData);
    if (!window.consultantId) {
      window.consultantId = consultantData.consultant_id || consultantQuery.docs[0].id;
      console.log("Updated consultantId from Firestore:", window.consultantId);
    }

    const prescriptionQuery = await db.collection("prescriptions").where("consultant_id", "==", consultantId).get();
    let prescriptionHTML = '<div class="summary-title">Prescription Summary</div>';
    if (prescriptionQuery.empty) {
      prescriptionHTML += "<p>No prescriptions available</p>";
    } else {
      prescriptionQuery.forEach((doc) => {
        const data = doc.data();
        const summaryContent = formatSummary(data.professional_summary);
        prescriptionHTML += `
          <div class="report-section paper-summary">
            <h3 class="report-title">Prescription Report</h3>
            <ul class="summary-list">
              <li><strong>Patient:</strong> ${data.patient_name || 'Unknown'}</li>
              <li><strong>Date:</strong> ${new Date(data.timestamp?.seconds * 1000).toLocaleDateString()}</li>
              <li><strong>Summary:</strong> ${summaryContent}</li>
            </ul>
          </div>
        `;
      });
    }
    prescriptionContainer.innerHTML = prescriptionHTML;

    const labRecordsQuery = await db.collection("lab_records").where("consultant_id", "==", consultantId).get();
    let labRecordsHTML = '<div class="summary-title">Lab Records Summary</div>';
    if (labRecordsQuery.empty) {
      labRecordsHTML += "<p>No lab records available</p>";
    } else {
      labRecordsQuery.forEach((doc) => {
        const data = doc.data();
        const labSummary = formatSummary(data.professional_summary);
        labRecordsHTML += `
          <div class="report-section paper-summary">
            <h3 class="report-title">Lab Report</h3>
            <ul class="summary-list">
              <li><strong>Patient:</strong> ${data.patient_name || 'Unknown'}</li>
              <li><strong>Date:</strong> ${new Date(data.timestamp?.seconds * 1000).toLocaleDateString()}</li>
              <li><strong>Summary:</strong> ${labSummary}</li>
            </ul>
          </div>
        `;
      });
    }
    labRecordsContainer.innerHTML = labRecordsHTML;

    const patientIds = new Set();
    prescriptionQuery.forEach((doc) => patientIds.add(doc.data().uid));
    container.innerHTML = `
      <div class="summary-title">Patient List</div>
      <select id="patient-dropdown" class="patient-dropdown" onchange="handlePatientSelection(this.value)">
        <option value="">Select a patient...</option>
      </select>
      <div id="patient-details" style="margin-top: 20px; display: none;"></div>
    `;
    for (const patientUid of patientIds) {
      const patientDoc = await db.collection("patient_registrations").doc(patientUid).get();
      if (patientDoc.exists) {
        const patientData = patientDoc.data();
        const dropdown = container.querySelector("#patient-dropdown");
        if (dropdown) {
          const option = new Option(`${patientData.full_name || 'Unknown'} (ID: ${patientUid})`, patientUid);
          dropdown.appendChild(option);
        }
      }
    }

    let nextPatientHTML = '<div class="summary-title">Next Patient</div>';
    if (patientIds.size > 0) {
      const patients = [];
      for (const patientUid of patientIds) {
        const patientDoc = await db.collection("patient_registrations").doc(patientUid).get();
        if (patientDoc.exists) {
          const patientData = patientDoc.data();
          patients.push({ uid: patientUid, name: patientData.full_name || "Unknown", next_visit: patientData.next_visit });
        }
      }
      const nextPatient = patients.reduce((earliest, patient) => {
        if (!earliest || (patient.next_visit && new Date(patient.next_visit.seconds * 1000) < new Date(earliest.next_visit.seconds * 1000))) {
          return patient;
        }
        return earliest;
      }, null);
      nextPatientHTML += nextPatient && nextPatient.next_visit ? `
        <div class="next-patient-card">
          <h3>Next Patient</h3>
          <p>Name: ${nextPatient.name}</p>
          <p>Next Visit: ${new Date(nextPatient.next_visit.seconds * 1000).toLocaleDateString()}</p>
        </div>
      ` : "<p>No upcoming patients</p>";
    } else {
      nextPatientHTML += "<p>No upcoming patients</p>";
    }
    nextPatientContainer.innerHTML = nextPatientHTML;

  } catch (error) {
    console.error("Error loading consultant dashboard:", error);
    container.innerHTML = "<p>Failed to load dashboard: " + error.message + "</p>";
    nextPatientContainer.innerHTML = "<p>No upcoming patients</p>";
    prescriptionContainer.innerHTML = "<p>No prescriptions available</p>";
    labRecordsContainer.innerHTML = "<p>No lab records available</p>";
  }
}

async function handlePatientSelection(patientUid) {
  if (!patientUid) {
    document.getElementById("patient-details").innerHTML = "";
    const overlay = document.getElementById("patient-overlay");
    overlay.classList.remove("active");
    setTimeout(() => overlay.style.display = "none", 300); // Match transition duration
    return;
  }
  console.log("Selected patient UID:", patientUid);

  const patientDetailsContainer = document.getElementById("overlay-patient-details");
  const overlay = document.getElementById("patient-overlay");
  overlay.style.display = "flex"; // Ensure overlay is visible before animation
  setTimeout(() => overlay.classList.add("active"), 10); // Slight delay for transition

  try {
    const patientDoc = await db.collection("patient_registrations").doc(patientUid).get();
    if (!patientDoc.exists) {
      patientDetailsContainer.innerHTML = "<h3>Patient Details</h3><p>Patient data not found.</p>";
      return;
    }
    const patientData = patientDoc.data();

    patientDetailsContainer.innerHTML = `
      <h3>Patient Details</h3>
      <div style="margin-bottom: 15px;">
        <p><strong>Name:</strong> ${patientData.full_name || 'Unknown'}</p>
        <p><strong>Next Visit:</strong> ${patientData.next_visit ? new Date(patientData.next_visit.seconds * 1000).toLocaleDateString() : 'Not scheduled'}</p>
      </div>
    `;

    const prescriptionQuery = await db.collection("prescriptions").where("uid", "==", patientUid).get();
    if (prescriptionQuery.empty) {
      patientDetailsContainer.innerHTML += "<p>No prescriptions found for this patient.</p>";
    } else {
      patientDetailsContainer.innerHTML += "<h4>Prescriptions</h4>";
      prescriptionQuery.forEach(doc => {
        const data = doc.data();
        const summaryContent = formatSummary(data.professional_summary);
        patientDetailsContainer.innerHTML += `
          <div class="report-section" style="border: 1px solid #ccc; padding: 10px; margin-bottom: 10px;">
            <p><strong>Date:</strong> ${new Date(data.timestamp?.seconds * 1000).toLocaleDateString()}</p>
            <p><strong>Summary:</strong> ${summaryContent}</p>
          </div>
        `;
      });
    }

    const labRecordsQuery = await db.collection("lab_records").where("uid", "==", patientUid).get();
    if (labRecordsQuery.empty) {
      patientDetailsContainer.innerHTML += "<p>No lab records found for this patient.</p>";
    } else {
      patientDetailsContainer.innerHTML += "<h4>Lab Records</h4>";
      labRecordsQuery.forEach(doc => {
        const data = doc.data();
        const labSummary = formatSummary(data.professional_summary);
        patientDetailsContainer.innerHTML += `
          <div class="report-section" style="border: 1px solid #ccc; padding: 10px; margin-bottom: 10px;">
            <p><strong>Date:</strong> ${new Date(data.timestamp?.seconds * 1000).toLocaleDateString()}</p>
            <p><strong>Summary:</strong> ${labSummary}</p>
          </div>
        `;
      });
    }
  } catch (error) {
    console.error("Error loading patient details:", error);
    patientDetailsContainer.innerHTML = "<h3>Patient Details</h3><p>Failed to load patient details: " + error.message + "</p>";
  }
}

function closePatientOverlay() {
  const overlay = document.getElementById("patient-overlay");
  overlay.classList.remove("active");
  setTimeout(() => overlay.style.display = "none", 300); // Match transition duration
  document.getElementById("overlay-patient-details").innerHTML = "";
}

document.getElementById('logout-btn').addEventListener('click', async () => {
  console.log("Logout button clicked, current user:", window.auth.currentUser);
  if (confirm("Are you sure you want to log out?")) {
    if (!window.auth.currentUser) {
      console.warn("No active user session found, redirecting to /login");
      window.location.href = "/login";
      return;
    }
    try {
      console.log("Calling server logout");
      const logoutResponse = await fetch('/logout?confirm=yes', { method: 'GET', redirect: 'follow' });
      console.log("Server logout response status:", logoutResponse.status);
      if (logoutResponse.status === 302 || logoutResponse.ok) {
        console.log("Server logout successful");
        await window.auth.signOut();
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

window.onload = function() {
  console.log("Window loaded, calling loadConsultantDashboard");
  loadConsultantDashboard();
};