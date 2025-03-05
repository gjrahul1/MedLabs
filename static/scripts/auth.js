// auth.js

// Basic auth utility functions
const auth = firebase.auth();

// Login function (to be used independently if needed)
async function login(email, password) {
    try {
        await auth.signInWithEmailAndPassword(email, password);
        alert("Login successful!");
        window.location.href = "/login"; // Redirect to backend login endpoint
    } catch (error) {
        alert(`Login Failed: ${error.message}`);
    }
}

// Logout function
async function logout() {
    try {
        await auth.signOut();
        window.location.href = "/";
    } catch (error) {
        console.error('Logout Error:', error);
        alert('Logout failed. Please try again.');
    }
}

// Track auth state (optional, can be handled by login.js)
firebase.auth().onAuthStateChanged((user) => {
    if (!user && window.location.pathname !== '/login' && window.location.pathname !== '/register') {
        window.location.href = '/login';
    }
});

// Export functions for potential reuse
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { login, logout };
}