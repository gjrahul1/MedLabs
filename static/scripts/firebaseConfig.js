// firebaseConfig.js (Compat Version)

// Check that the global firebase object exists
if (!window.firebase) {
  console.error("Firebase SDK is not loaded. Ensure firebase-app-compat.js is included.");
} else {
  console.log("Firebase SDK loaded. Version:", firebase.SDK_VERSION);
}

var firebaseConfig = {
  apiKey: "API-KEY",
  authDomain: "med-labs-42f13.firebaseapp.com",
  projectId: "med-labs-42f13",
  storageBucket: "med-labs-42f13",
  messagingSenderId: "102867365288",
  appId: "1:102867365288:web:38cb06f82566a19e9c819d",
  measurementId: "G-W0B3MSZJXE"
};

// Initialize Firebase using the Compat API
try {
  firebase.initializeApp(firebaseConfig);
  window.auth = firebase.auth();
  window.db = firebase.firestore();
  window.storage = firebase.storage(); // Ensure storage is initialized
  console.log("✅ Firebase initialized successfully!");
} catch (err) {
  console.error("Firebase initialization error:", err);
}

// Export for potential module usage (optional)
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    auth: window.auth,
    db: window.db,
    storage: window.storage
  };
}
