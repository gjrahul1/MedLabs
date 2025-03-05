// patientDataFetcher.js

// Import Firebase SDK v9.23.0 modular functions
import { getAuth } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";
import { getFirestore, doc, getDoc } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-firestore.js";

// Initialize Firebase services
const auth = getAuth();
const db = getFirestore();

// Fetch and display patient data
async function fetchAndDisplayPatientData() {
  try {
    const user = auth.currentUser;
    if (!user) {
      console.error("User not authenticated.");
      window.location.href = '/login';
      return;
    }

    // Fetch patient data from Firestore using email as document ID
    const docRef = doc(db, 'patient_registrations', user.email);
    const docSnap = await getDoc(docRef);

    if (docSnap.exists()) {
      const data = docSnap.data();
      displayPatientInfo(data);
    } else {
      console.error("No patient data found.");
      displayPatientInfo({}); // Display "NaN" for all fields if no data exists
    }
  } catch (error) {
    console.error("Error fetching patient data:", error);
    alert("Failed to fetch patient data. Please try again.");
    displayPatientInfo({}); // Fallback to "NaN" on error
  }
}

// Display patient information with "NaN" fallback
function displayPatientInfo(data) {
  document.getElementById('greeting').textContent = `Hello, ${data.full_name || 'Patient'}`;
  setFieldContent('patient-id', data.patient_id || data.uid || 'NaN'); // Use uid as fallback if patient_id is missing
  setFieldContent('next-visit', data.next_visit || 'NaN');
  setFieldContent('consultant-name', data.consultant || data.consultant_id || 'NaN'); // Use consultant_id if consultant name is missing
  setFieldContent('disease', data.disease || 'NaN');
  setFieldContent('last-visit', data.last_visit || 'NaN');
}

// Helper function to set field content
function setFieldContent(id, content) {
  const element = document.getElementById(id);
  element.textContent = content === 'NaN' ? 'NaN' : content;
}

// Initialize data fetch on page load
window.onload = fetchAndDisplayPatientData;