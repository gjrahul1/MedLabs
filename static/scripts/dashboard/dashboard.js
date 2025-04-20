import { getAuth } from "https://www.gstatic.com/firebasejs/9.23.0/firebase-auth.js";

const auth = getAuth();

// Function to get a fresh ID token
async function getFreshToken() {
  const user = auth.currentUser;
  if (!user) {
    console.error("User not authenticated.");
    window.location.href = '/login';
    return null;
  }
  try {
    const idToken = await user.getIdToken(true); // Force refresh
    sessionStorage.setItem('idToken', idToken);
    console.log('Refreshed ID Token:', idToken.substring(0, 20) + '...');
    return idToken;
  } catch (error) {
    console.error("Error refreshing token:", error);
    window.location.href = '/login';
    return null;
  }
}

// Function to load dashboard data
async function loadDashboardData() {
  const idToken = await getFreshToken();
  if (!idToken) return;

  try {
    const response = await fetch('/dashboard', {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${idToken}`,
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      console.error('Dashboard access failed:', response.status, await response.text());
      window.location.href = '/login';
      return;
    }

    const data = await response.text(); // Expect HTML content
    document.body.innerHTML = data; // Render the dashboard HTML
    console.log('Dashboard loaded successfully');
    // Initialize patientDashboard.js or other scripts if needed
    const script = document.createElement('script');
    script.src = '/static/scripts/js/patientDashboard.js';
    document.body.appendChild(script);
  } catch (error) {
    console.error('Error loading dashboard:', error);
    window.location.href = '/login';
  }
}

// Load dashboard when the page loads
window.onload = () => {
  auth.onAuthStateChanged((user) => {
    if (user) {
      loadDashboardData();
    } else {
      console.log("No user authenticated, redirecting to login");
      window.location.href = '/login';
    }
  });
};