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
    console.log('Refreshed ID Token:', idToken);
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

  const response = await fetch('/dashboard', {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${idToken}`,
      'Content-Type': 'application/json'
    }
  });

  if (!response.ok) {
    console.error('Dashboard access failed:', response.status);
    window.location.href = '/login';
    return;
  }

  const data = await response.json();
  console.log('Dashboard data loaded:', data);
  // Render dashboard content here
}

// Load dashboard when the page loads
window.onload = loadDashboardData;